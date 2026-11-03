"""A dependency-free publish/subscribe bus for cognitive lifecycle events.

The cognitive architecture is wired by direct calls: the controller calls the
reasoner, the assistant calls the retriever. That is deliberate -- a *decision*
must never depend on an event that may or may not have been delivered. But it
also means nothing can *observe* the system without being called from inside it,
which is what the resource governor (Phase 1) and the self-improvement monitor
(Phase 9) both need.

``EventBus`` fills that gap and nothing more. It is **notification only**:

* an event never carries control flow -- no subscriber's result is returned to
  the publisher, and the bus has no way to change a decision;
* a raising subscriber is **isolated** and recorded through
  :mod:`core.best_effort`, so one bad observer cannot break a cognitive cycle or
  disappear without a trace;
* delivery is **synchronous by default** (deterministic ordering, no threads, no
  hidden state), with an optional *bounded* queue mode for a producer that must
  not be slowed;
* every published event is kept in a bounded, ordered record so a caller (the
  audit, a test, a diagnostic) can read what happened.

Nothing here imports a model, Tk, or the network, and nothing here writes to the
knowledge graph -- the bus only *reports* what the core already did.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.best_effort import best_effort

#: How many events the bus remembers by default. An ordinary cognitive run
#: publishes a few dozen, so the ring holds a whole session without growing.
DEFAULT_HISTORY = 512

#: The bound on a queue-mode bus's pending events, so a producer that outruns its
#: consumers cannot grow the process without limit.
DEFAULT_QUEUE_LIMIT = 1024


@dataclass(frozen=True)
class Event:
    """One notification: a named thing that happened, and its data.

    ``name`` is a dotted, stable identifier (``cycle.started``,
    ``belief.revised``) so a subscriber can subscribe to a family by prefix.
    ``data`` is deliberately plain (JSON-compatible values) -- an event is a
    report, not a handle to a live object.
    """

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "core"
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Event name cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data,
            "source": self.source,
            "at": self.at.isoformat(),
        }


Subscriber = Callable[[Event], None]


@dataclass(frozen=True)
class Subscription:
    """A registered subscriber, with the prefix it asked for."""

    prefix: str
    callback: Subscriber
    id: int

    def matches(self, name: str) -> bool:
        """Whether this subscription wants an event with this name.

        A prefix matches the event's whole leading segments, so ``cycle``
        matches ``cycle.started`` but not ``cyclone`` -- a boundary check, not a
        string ``startswith``.
        """
        return _prefix_matches(self.prefix, name)


class EventBus:
    """Publish/subscribe over :class:`Event`, synchronous by default.

    The bus is intentionally small: ``subscribe``/``unsubscribe``/``publish``,
    a bounded ``history``, and an optional queued mode. It never raises from
    ``publish`` because of a subscriber, and it never blocks on a slow one in
    synchronous mode (the caller chose synchronous delivery).

    ``queue=True`` buffers events and delivers them on ``drain()``, bounded by
    ``queue_limit`` (oldest dropped first, and the drop is *counted* so it is
    visible). This is the mode a background producer (a training loop) uses so
    its publishing cost is O(1) regardless of the subscribers.
    """

    def __init__(self, *, history: int = DEFAULT_HISTORY, queue: bool = False, queue_limit: int = DEFAULT_QUEUE_LIMIT) -> None:
        if history < 0:
            raise ValueError("History size cannot be negative.")
        if queue_limit <= 0:
            raise ValueError("Queue limit must be positive.")
        self._subscribers: list[Subscription] = []
        self._history: deque[Event] = deque(maxlen=history if history else 0)
        self._queue: deque[Event] | None = deque(maxlen=queue_limit) if queue else None
        self._next_id = 0
        self._dropped = 0

    # -- subscription ------------------------------------------------------

    def subscribe(self, callback: Subscriber, prefix: str = "") -> Subscription:
        """Register ``callback`` for events whose name starts with ``prefix``.

        The empty prefix (the default) subscribes to everything. The returned
        :class:`Subscription` is what ``unsubscribe`` takes, so a caller can
        never remove the wrong one.
        """
        if not callable(callback):
            raise TypeError("Subscriber must be callable.")
        subscription = Subscription(prefix=prefix, callback=callback, id=self._next_id)
        self._next_id += 1
        self._subscribers.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription, returning whether it was present."""
        for index, existing in enumerate(self._subscribers):
            if existing is subscription:
                del self._subscribers[index]
                return True
        return False

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # -- publishing --------------------------------------------------------

    def publish(self, name: str, data: dict[str, Any] | None = None, *, source: str = "core") -> Event:
        """Deliver one event and return it (the event is never changed by a subscriber).

        A subscriber that raises is recorded through ``best_effort`` and skipped;
        the remaining subscribers still receive the event, and the publisher
        never sees the error.
        """
        event = Event(name=name, data=dict(data or {}), source=source)
        self._history.append(event)
        if self._queue is not None:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            self._queue.append(event)
            return event
        self._deliver(event)
        return event

    def _deliver(self, event: Event) -> int:
        """Send one event to every matching subscriber; return how many saw it."""
        delivered = 0
        for subscription in tuple(self._subscribers):
            if not subscription.matches(event.name):
                continue
            with best_effort(f"event subscriber for {event.name}"):
                subscription.callback(event)
                delivered += 1
        return delivered

    def drain(self, limit: int | None = None) -> int:
        """Deliver queued events (queue mode only); return how many were delivered.

        Draining in FIFO order preserves the order the producer published in.
        ``limit`` bounds one drain so a caller can spread the work across turns.
        """
        if self._queue is None:
            return 0
        delivered = 0
        while self._queue and (limit is None or delivered < limit):
            self._deliver(self._queue.popleft())
            delivered += 1
        return delivered

    @property
    def pending(self) -> int:
        """How many events are waiting in queue mode (0 in synchronous mode)."""
        return len(self._queue) if self._queue is not None else 0

    @property
    def dropped(self) -> int:
        """How many queued events were dropped because the queue was full."""
        return self._dropped

    # -- reading -----------------------------------------------------------

    @property
    def history(self) -> tuple[Event, ...]:
        """The events published so far, oldest first (bounded)."""
        return tuple(self._history)

    def recent(self, limit: int = 20, *, prefix: str = "") -> tuple[Event, ...]:
        """The most recent events, newest first, optionally filtered by prefix."""
        if limit <= 0:
            return ()
        items = [event for event in reversed(self._history) if _prefix_matches(prefix, event.name)]
        return tuple(items[:limit])

    def count(self, name: str | None = None) -> int:
        """How many events were published, in total or for one exact name."""
        if name is None:
            return len(self._history)
        return sum(1 for event in self._history if event.name == name)

    def clear(self) -> None:
        """Forget the history and any queued events (a fresh diagnostic window)."""
        self._history.clear()
        if self._queue is not None:
            self._queue.clear()
        self._dropped = 0

    def describe(self) -> str:
        mode = "queued" if self._queue is not None else "synchronous"
        return (
            f"EventBus({mode}, {len(self._subscribers)} subscriber(s), "
            f"{len(self._history)} recorded, {self.pending} pending)"
        )

    def __iter__(self) -> Iterator[Event]:
        return iter(self._history)

    def __len__(self) -> int:
        return len(self._history)


def _prefix_matches(prefix: str, name: str) -> bool:
    """Whether an event name belongs to a subscription prefix.

    The empty prefix (or ``*``) is "everything"; otherwise the prefix must match
    the name's whole leading dot-separated segments, so ``cycle`` matches
    ``cycle.started`` but not ``cyclone``.
    """
    if prefix in ("", "*"):
        return True
    return name == prefix or name.startswith(f"{prefix}.")


#: The event names the cognitive core publishes. Declared as constants so a
#: publisher and a subscriber cannot drift on a string literal.
EVENT_CYCLE_STARTED = "cycle.started"
EVENT_CYCLE_FINISHED = "cycle.finished"
EVENT_BELIEF_REVISED = "belief.revised"
EVENT_RELATION_LEARNED = "relation.learned"
EVENT_KNOWLEDGE_INTEGRATED = "knowledge.integrated"
EVENT_KNOWLEDGE_REJECTED = "knowledge.rejected"
EVENT_CONFLICT_DETECTED = "conflict.detected"
EVENT_RESOURCE_DEFERRED = "resource.deferred"
EVENT_MODEL_LOADED = "model.loaded"
EVENT_MODEL_UNLOADED = "model.unloaded"
EVENT_MEMORY_STORED = "memory.stored"
EVENT_MEMORY_CONSOLIDATED = "memory.consolidated"
EVENT_MEMORY_ARCHIVED = "memory.archived"

EVENT_NAMES: tuple[str, ...] = (
    EVENT_CYCLE_STARTED,
    EVENT_CYCLE_FINISHED,
    EVENT_BELIEF_REVISED,
    EVENT_RELATION_LEARNED,
    EVENT_KNOWLEDGE_INTEGRATED,
    EVENT_KNOWLEDGE_REJECTED,
    EVENT_CONFLICT_DETECTED,
    EVENT_RESOURCE_DEFERRED,
    EVENT_MODEL_LOADED,
    EVENT_MODEL_UNLOADED,
    EVENT_MEMORY_STORED,
    EVENT_MEMORY_CONSOLIDATED,
    EVENT_MEMORY_ARCHIVED,
)
"""Every lifecycle event the core declares, so a reader sees the vocabulary."""


def default_bus() -> EventBus:
    """A fresh synchronous bus -- the one a controller builds when none is given."""
    return EventBus()


__all__ = [
    "DEFAULT_HISTORY",
    "DEFAULT_QUEUE_LIMIT",
    "EVENT_BELIEF_REVISED",
    "EVENT_CONFLICT_DETECTED",
    "EVENT_CYCLE_FINISHED",
    "EVENT_CYCLE_STARTED",
    "EVENT_KNOWLEDGE_INTEGRATED",
    "EVENT_KNOWLEDGE_REJECTED",
    "EVENT_MEMORY_ARCHIVED",
    "EVENT_MEMORY_CONSOLIDATED",
    "EVENT_MEMORY_STORED",
    "EVENT_MODEL_LOADED",
    "EVENT_MODEL_UNLOADED",
    "EVENT_NAMES",
    "EVENT_RELATION_LEARNED",
    "EVENT_RESOURCE_DEFERRED",
    "Event",
    "EventBus",
    "Subscriber",
    "Subscription",
    "default_bus",
]
