"""The evolving *story* of the user's relationship with Natasha -- told honestly.

This is the narrative layer: the small, human things that make a relationship feel
continuous -- a preference about how to talk, a topic to come back to, a date the
user wants remembered, a moment that happened in the conversation. It is stored as
dated *events*, each with a kind that says exactly how it should be treated:

* ``fact``       -- something the user stated about themselves;
* ``inference``  -- a guess the program made and labelled as a guess;
* ``fiction``    -- something that happened *inside* the role-play, not in reality;
* ``preference`` -- a stated liking about how the two should talk;
* ``occasion``   -- a date or topic the user wants to return to;
* ``milestone``  -- a moment the user asked to mark.

Two honesty rules are built in. There is **no numeric love or loyalty score**: the
"development" is described in plain words (how many events of each kind, when the
story began, what it has been about), never as a fake measurement. And the story
only moves when the user *approves* an event -- ``add_event`` refuses unless
``approved=True`` -- so nothing is recorded behind their back. The record can be
corrected, deleted, or reset at any time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from companion.database import CompanionDatabase, _utc_now

#: The event kinds. ``fiction`` is the only one that is automatically marked as
#: fictional, because it happened inside the role-play rather than in reality.
KIND_FACT = "fact"
KIND_INFERENCE = "inference"
KIND_FICTION = "fiction"
KIND_PREFERENCE = "preference"
KIND_OCCASION = "occasion"
KIND_MILESTONE = "milestone"
EVENT_KINDS: tuple[str, ...] = (
    KIND_FACT,
    KIND_INFERENCE,
    KIND_FICTION,
    KIND_PREFERENCE,
    KIND_OCCASION,
    KIND_MILESTONE,
)

#: How each kind should be described when it is recalled, so a guess never reads
#: like a fact and a story never reads like reality.
KIND_LABELS: dict[str, str] = {
    KIND_FACT: "you told me this",
    KIND_INFERENCE: "my guess (not confirmed)",
    KIND_FICTION: "something that happened in our story",
    KIND_PREFERENCE: "a preference you told me",
    KIND_OCCASION: "a date or topic to come back to",
    KIND_MILESTONE: "a moment you asked me to mark",
}

#: A cue that the user wants something remembered, used by :meth:`propose_event`.
_APPROVE_CUES = (
    "remember this", "please remember", "don't forget", "keep this",
    "احفظ هذا", "تذكر هذا", "تذكري هذا", "خليك فاكرة", "لا تنسى",
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


@dataclass(frozen=True)
class RelationshipEvent:
    """One dated event in the relationship's story."""

    id: str
    kind: str
    title: str
    detail: str
    occurred_at: str
    created_at: str
    active: bool = True
    fictional: bool = False

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    def describe(self) -> str:
        marker = " (in our story)" if self.fictional else ""
        return f"{self.title} -- {self.label}{marker}"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "detail": self.detail,
            "occurred_at": self.occurred_at,
            "created_at": self.created_at,
            "active": self.active,
            "fictional": self.fictional,
        }


@dataclass(frozen=True)
class EventCandidate:
    """A *proposed* relationship event: suggested, not stored."""

    kind: str
    title: str
    detail: str
    reason: str


@dataclass(frozen=True)
class RelationshipSummary:
    """A plain-words account of the relationship so far (never a score)."""

    total: int
    by_kind: dict[str, int]
    first_at: str | None
    last_at: str | None
    recent_titles: tuple[str, ...]

    def describe(self) -> str:
        if self.total == 0:
            return "We are at the beginning -- there is nothing saved in our story yet."
        parts = ", ".join(f"{count} {kind}" for kind, count in sorted(self.by_kind.items()))
        recent = "; ".join(self.recent_titles) if self.recent_titles else "nothing yet"
        return (
            f"Our story has {self.total} saved moments ({parts}). "
            f"It began on {self.first_at[:10] if self.first_at else 'unknown'}. "
            f"Recently: {recent}."
        )


def _row_to_event(row: Any) -> RelationshipEvent:
    return RelationshipEvent(
        id=row["id"],
        kind=row["kind"],
        title=row["title"],
        detail=row["detail"],
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
        active=bool(row["active"]),
        fictional=bool(row["fictional"]),
    )


class RelationshipManager:
    """Stores and manages the relationship's story, with the user's approval."""

    def __init__(self, database: CompanionDatabase) -> None:
        self._db = database

    # -- writing ----------------------------------------------------------
    def add_event(
        self,
        kind: str,
        title: str,
        detail: str = "",
        *,
        occurred_at: str | None = None,
        approved: bool = False,
    ) -> RelationshipEvent:
        """Store one event, but only with the user's approval.

        ``approved=False`` is refused on purpose: nothing is added to the story
        unless the user agreed to keep it. A ``fiction`` event is automatically
        marked fictional so it is never recalled as reality.
        """
        if kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}.")
        if not approved:
            raise ValueError(
                "A relationship event is only stored with the user's approval "
                "(pass approved=True once they agree)."
            )
        clean_title = (title or "").strip()
        if not clean_title:
            raise ValueError("A relationship event needs a title.")
        identifier = _new_id("REL")
        now = _utc_now()
        when = (occurred_at or now).strip()
        self._db.execute(
            "INSERT INTO relationship_events(id, kind, title, detail, occurred_at, created_at, "
            "active, fictional) VALUES(?, ?, ?, ?, ?, ?, 1, ?)",
            (identifier, kind, clean_title, (detail or "").strip(), when, now, 1 if kind == KIND_FICTION else 0),
        )
        return RelationshipEvent(
            id=identifier,
            kind=kind,
            title=clean_title,
            detail=(detail or "").strip(),
            occurred_at=when,
            created_at=now,
            fictional=kind == KIND_FICTION,
        )

    def correct_event(
        self,
        event_id: str,
        *,
        title: str | None = None,
        detail: str | None = None,
        kind: str | None = None,
    ) -> RelationshipEvent:
        """Correct an event in place (the story stays, its wording is fixed)."""
        original = self.get(event_id)
        if original is None:
            raise KeyError(f"Unknown relationship event: {event_id}")
        new_kind = kind or original.kind
        if new_kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}.")
        new_title = original.title if title is None else (title or "").strip()
        if not new_title:
            raise ValueError("A relationship event needs a title.")
        new_detail = original.detail if detail is None else (detail or "").strip()
        self._db.execute(
            "UPDATE relationship_events SET kind = ?, title = ?, detail = ?, fictional = ? WHERE id = ?",
            (new_kind, new_title, new_detail, 1 if new_kind == KIND_FICTION else 0, event_id),
        )
        updated = self.get(event_id)
        assert updated is not None
        return updated

    def delete_event(self, event_id: str) -> bool:
        cursor = self._db.execute("DELETE FROM relationship_events WHERE id = ?", (event_id,))
        return cursor.rowcount > 0

    def reset(self) -> int:
        """Remove every relationship event (an explicit reset of the story)."""
        cursor = self._db.execute("DELETE FROM relationship_events")
        return cursor.rowcount

    # -- reading ----------------------------------------------------------
    def get(self, event_id: str) -> RelationshipEvent | None:
        row = self._db.query_one("SELECT * FROM relationship_events WHERE id = ?", (event_id,))
        return _row_to_event(row) if row is not None else None

    def events(
        self,
        *,
        kind: str | None = None,
        include_fiction: bool = True,
        include_inactive: bool = False,
    ) -> list[RelationshipEvent]:
        """The stored events, newest first, optionally filtered by kind."""
        sql = "SELECT * FROM relationship_events"
        clauses: list[str] = []
        params: list[object] = []
        if not include_inactive:
            clauses.append("active = 1")
        if kind is not None:
            if kind not in EVENT_KINDS:
                raise ValueError(f"kind must be one of {EVENT_KINDS}.")
            clauses.append("kind = ?")
            params.append(kind)
        if not include_fiction:
            clauses.append("fictional = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY occurred_at DESC, created_at DESC"
        return [_row_to_event(row) for row in self._db.query(sql, params)]

    def summary(self, *, recent: int = 3) -> RelationshipSummary:
        """A plain-words account of the story, never a numeric love score."""
        events = self.events(include_inactive=False)
        by_kind: dict[str, int] = {}
        for event in events:
            by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
        ordered = sorted(events, key=lambda e: (e.occurred_at, e.created_at))
        return RelationshipSummary(
            total=len(events),
            by_kind=by_kind,
            first_at=ordered[0].occurred_at if ordered else None,
            last_at=ordered[-1].occurred_at if ordered else None,
            recent_titles=tuple(event.title for event in events[:recent]),
        )

    # -- proposals --------------------------------------------------------
    def propose_event(self, text: str) -> list[EventCandidate]:
        """Suggest events from a message -- never store them without approval."""
        message = (text or "").strip()
        if not message:
            return []
        lowered = message.casefold()
        if any(cue.casefold() in lowered for cue in _APPROVE_CUES):
            return [
                EventCandidate(
                    kind=KIND_FACT,
                    title=message[:80],
                    detail=message,
                    reason="the user asked to keep this",
                )
            ]
        return []

    def export(self) -> list[dict[str, Any]]:
        """Every event (including any inactive) as plain data, for a backup."""
        return [event.to_primitive() for event in self.events(include_inactive=True)]


__all__ = [
    "EVENT_KINDS",
    "KIND_FACT",
    "KIND_FICTION",
    "KIND_INFERENCE",
    "KIND_LABELS",
    "KIND_MILESTONE",
    "KIND_OCCASION",
    "KIND_PREFERENCE",
    "EventCandidate",
    "RelationshipEvent",
    "RelationshipManager",
    "RelationshipSummary",
]
