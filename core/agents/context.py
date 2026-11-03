"""Independent working context per personality.

The task requires each personality to keep its **own** working context: the
current conversation's short-term state, its pending goals, the task it is
executing, and the facts it is holding for the moment. Two personalities talking
at once must not see each other's working state -- "independent working contexts"
is one of the acceptance criteria.

A :class:`WorkingContext` is deliberately small and in-memory: it is the *working*
memory (the project's durable memory is Phase 2's), bounded so it cannot grow
without limit, and it records the owner it belongs to so a context can never be
handed to the wrong personality. :class:`ContextManager` keeps one context per
owner, resolved through the isolation policy, and refuses to hand one personality
another's context.

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.learning_isolation import is_owner, owner_for

#: How many items a working context keeps before the oldest is dropped. Bounded on
#: purpose: a working context is short-term state, not a transcript.
DEFAULT_CONTEXT_LIMIT = 50


@dataclass(frozen=True)
class ContextItem:
    """One item of working state: what was said, thought, or is pending."""

    kind: str
    text: str
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, str] = field(default_factory=dict)

    def to_primitive(self) -> dict[str, object]:
        return {"kind": self.kind, "text": self.text, "at": self.at, "metadata": dict(self.metadata)}


@dataclass
class WorkingContext:
    """One personality's bounded, independent working state.

    ``owner`` is fixed at construction: the context knows whose it is, so a caller
    can never *give* it to another personality, and :class:`ContextManager` can
    refuse a mismatched read.
    """

    owner: str
    limit: int = DEFAULT_CONTEXT_LIMIT
    items: list[ContextItem] = field(default_factory=list)
    #: The task currently being executed, if any (a task id from the goal manager).
    current_task: str = ""

    def __post_init__(self) -> None:
        self.owner = owner_for(self.owner)
        if self.limit < 1:
            raise ValueError("A working context needs a positive limit.")

    def add(self, kind: str, text: str, **metadata: str) -> ContextItem:
        """Append an item, dropping the oldest when the bound is reached."""
        item = ContextItem(kind=kind, text=text, metadata=dict(metadata))
        self.items.append(item)
        if len(self.items) > self.limit:
            del self.items[: len(self.items) - self.limit]
        return item

    def recent(self, *, kind: str | None = None, limit: int | None = None) -> tuple[ContextItem, ...]:
        """The most recent items (optionally of one kind), newest last."""
        chosen = [item for item in self.items if kind is None or item.kind == kind]
        if limit is not None and limit >= 0:
            chosen = chosen[-limit:]
        return tuple(chosen)

    def clear(self) -> None:
        """Empty the context (a new conversation), keeping the bound and owner."""
        self.items.clear()
        self.current_task = ""

    def size(self) -> int:
        return len(self.items)

    def describe(self) -> str:
        task = self.current_task or "(none)"
        return f"{self.owner}: {len(self.items)} item(s), current task {task}"


@dataclass
class ContextManager:
    """One working context per personality, kept apart and policy-checked.

    ``contexts`` is injectable, so a test can inspect or pre-seed contexts; the
    manager resolves an owner through the isolation policy and refuses an unknown
    one, so a context is never created for a personality this build does not know.
    """

    limit: int = DEFAULT_CONTEXT_LIMIT
    contexts: dict[str, WorkingContext] = field(default_factory=dict)

    def context_for(self, owner: str) -> WorkingContext:
        """The working context of ``owner`` (created on first use)."""
        resolved = owner_for(owner)
        if not is_owner(resolved):
            raise ValueError(f"{owner!r} is not a known personality.")
        existing = self.contexts.get(resolved)
        if existing is None:
            existing = WorkingContext(resolved, limit=self.limit)
            self.contexts[resolved] = existing
        return existing

    def reset(self, owner: str) -> WorkingContext:
        """Clear and return ``owner``'s context (a new conversation for it)."""
        context = self.context_for(owner)
        context.clear()
        return context

    def owners(self) -> tuple[str, ...]:
        """The owners that currently have a context."""
        return tuple(sorted(self.contexts))

    def describe(self) -> str:
        if not self.contexts:
            return "no working contexts yet"
        return "\n".join(
            self.contexts[owner].describe() for owner in sorted(self.contexts)
        )


__all__ = [
    "DEFAULT_CONTEXT_LIMIT",
    "ContextItem",
    "ContextManager",
    "WorkingContext",
]
