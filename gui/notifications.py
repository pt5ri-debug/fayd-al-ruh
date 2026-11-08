"""The notification feed: a bounded record of what the background actually did.

Every window in the project runs long work on a worker thread and hands the
result back to the UI thread through a queue. Before this, a *finished* training
run or a *failed* download was shown once in the transcript and then scrolled
away, and a result that arrived while the user was looking elsewhere was easy to
miss. This module is the one place those results are *recorded*, so the Home
dashboard can show them and the window can tell the user how many are new.

It is deliberately Tk-free: it holds plain :class:`Notification` records (the same
shape :class:`gui.components.NotificationCenter` draws) and never touches a
widget, so it is exercised with no display. It is bounded, so a long session
cannot grow it without limit, and it holds *what actually happened* -- never a
fabricated progress line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: The notification levels, in the order a reader should weight them.
LEVELS: tuple[str, ...] = ("info", "success", "warning", "error")

#: How many notifications are kept (the oldest are dropped).
DEFAULT_LIMIT = 50


@dataclass(frozen=True)
class Notification:
    """One background result: what happened, how bad, and when."""

    title: str
    detail: str = ""
    level: str = "info"
    at: float = 0.0

    def describe(self) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(self.at)) if self.at else ""
        head = f"[{self.level}] {self.title}"
        if when:
            head = f"{head} ({when})"
        return f"{head}\n  {self.detail}" if self.detail else head


class NotificationFeed:
    """A bounded, Tk-free list of background results, newest first.

    ``post`` records one result; ``post_result`` is the helper a worker pump uses
    (a success or an error, from a label and a detail). ``unread`` counts what has
    arrived since the last :meth:`mark_seen`, so the window can badge the Home
    section -- and the count is honest: a notification is "unread" because it
    arrived, never because it is important.
    """

    def __init__(self, *, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = max(1, int(limit))
        self._items: list[Notification] = []
        self._seen = 0

    # -- writing -----------------------------------------------------------

    def post(self, title: str, detail: str = "", level: str = "info") -> Notification:
        """Record one notification and return it (newest first)."""
        if level not in LEVELS:
            raise ValueError(f"unknown notification level {level!r}")
        notification = Notification(
            title=str(title), detail=str(detail), level=level, at=time.time()
        )
        self._items.insert(0, notification)
        del self._items[self._limit :]
        return notification

    def post_result(self, label: str, *, ok: bool, detail: str = "") -> Notification:
        """Record a background job's outcome: a success or an error, never both.

        ``label`` names the action ("Training the language model"); ``detail`` is
        the job's own message (an error's text, or a short summary). A success is
        ``success`` and a failure is ``error`` -- the level is the outcome, so a
        failed job can never be shown as a success.
        """
        return self.post(
            label,
            detail=detail,
            level="success" if ok else "error",
        )

    def clear(self) -> None:
        """Drop every recorded notification and reset the unread count."""
        self._items.clear()
        self._seen = 0

    # -- reading -----------------------------------------------------------

    @property
    def items(self) -> tuple[Notification, ...]:
        """Every recorded notification, newest first."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def unread(self) -> int:
        """How many notifications have arrived since :meth:`mark_seen`."""
        return max(0, len(self._items) - self._seen)

    def mark_seen(self) -> None:
        """Mark every current notification as seen (the badge clears)."""
        self._seen = len(self._items)

    def recent(self, count: int = 5) -> tuple[Notification, ...]:
        """The newest ``count`` notifications (bounded), newest first."""
        return tuple(self._items[: max(0, int(count))])

    def describe(self) -> str:
        """A readable block of the recorded notifications, newest first."""
        if not self._items:
            return "No notifications yet."
        return "\n".join(item.describe() for item in self._items)


__all__ = [
    "DEFAULT_LIMIT",
    "LEVELS",
    "Notification",
    "NotificationFeed",
]
