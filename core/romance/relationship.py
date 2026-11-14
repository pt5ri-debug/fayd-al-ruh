"""The relationship engine: an independent, evidence-driven relationship per character.

Each character has its own relationship state with the user, scored on seven
transparent dimensions, and the state moves **only** when an explicit event is
observed -- never at random, and never by more than the event's own bounded delta.
Every event is stored (append-only) with its timestamp, its message reference, and
the reason it moved the state, so :meth:`RelationshipEngine.explain` can say *why*
the relationship is where it is.

The three rules this module exists to keep:

* **No random change.** :meth:`observe` requires an explicit event kind and derives
  its deltas from the event, bounded by :data:`MAX_STEP`; a dimension can never
  jump, and calling :meth:`observe` with the same event twice changes nothing
  (idempotent by event key).
* **Evidence is required.** An event carries the message it came from; a "milestone"
  with no evidence is recorded but moves nothing.
* **Never coercive.** The engine records *connection* and *trust*; it has no
  "dependency", "possessiveness", or "jealousy" dimension, and a conflict moves
  closeness **down**, so the model cannot be rewarded for pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.romance.characters import CharacterManager
from core.romance.model import utc_now
from core.romance.storage import CompanionStorageService

#: The seven scored dimensions, in a stable order.
DIMENSIONS: tuple[str, ...] = (
    "familiarity",
    "trust",
    "closeness",
    "comfort",
    "affection",
    "shared_experiences",
    "stability",
)

#: The most any single event may move a dimension (a bound, not a target).
MAX_STEP = 0.12

#: The default per-event effect on each dimension. Positive moves the relationship
#: forward; a conflict moves closeness and trust down. These are *defaults*; a
#: caller may supply its own deltas to :meth:`RelationshipEngine.observe`.
EVENT_EFFECTS: dict[str, dict[str, float]] = {
    "first_conversation": {"familiarity": 0.06, "comfort": 0.03},
    "introduction": {"familiarity": 0.05, "comfort": 0.02},
    "shared_interest": {"closeness": 0.04, "comfort": 0.04},
    "personal_share": {"trust": 0.06, "closeness": 0.04},
    "meaningful_conversation": {"closeness": 0.05, "trust": 0.03},
    "shared_experience": {"shared_experiences": 0.08, "closeness": 0.03},
    "affection_expressed": {"affection": 0.06, "closeness": 0.03},
    "conflict": {"closeness": -0.08, "trust": -0.06, "stability": -0.05},
    "reconciliation": {"trust": 0.05, "closeness": 0.04, "stability": 0.06},
    "milestone": {"closeness": 0.04, "shared_experiences": 0.03},
}

#: The relationship stages, from stranger to romantic partner. Configurable: a
#: caller may pass its own tuple of ``(key, title, threshold)`` to the engine.
STAGES: tuple[tuple[str, str, float], ...] = (
    ("stranger", "Stranger", 0.0),
    ("acquaintance", "Acquaintance", 0.15),
    ("friend", "Friend", 0.35),
    ("close_friend", "Close friend", 0.55),
    ("romantic_interest", "Romantic interest", 0.7),
    ("romantic_partner", "Romantic partner", 0.85),
)

#: The events a caller may observe.
EVENT_KINDS: tuple[str, ...] = tuple(EVENT_EFFECTS)

STAGE_UNKNOWN = "stranger"


class RelationshipError(RuntimeError):
    """Raised when a relationship operation cannot be carried out as asked."""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded(delta: float) -> float:
    return max(-MAX_STEP, min(MAX_STEP, delta))


@dataclass(frozen=True)
class RelationshipSnapshot:
    """A read-only view of one character's relationship state."""

    character_id: str
    user_id: str
    dimensions: dict[str, float]
    stage: str
    stage_title: str
    overall: float
    event_count: int

    def describe(self) -> str:
        parts = ", ".join(f"{name} {self.dimensions[name]:.2f}" for name in DIMENSIONS)
        return f"{self.stage_title} ({self.overall:.2f}) — {parts}"


@dataclass(frozen=True)
class RelationshipChange:
    """Why a relationship moved: the event, the deltas, and the stage it reached."""

    kind: str
    title: str
    deltas: dict[str, float] = field(default_factory=dict)
    stage_before: str = STAGE_UNKNOWN
    stage_after: str = STAGE_UNKNOWN
    message_id: str = ""
    reason: str = ""

    def describe(self) -> str:
        moves = ", ".join(f"{name} {delta:+.2f}" for name, delta in self.deltas.items())
        stage = (
            f"stage {self.stage_before} → {self.stage_after}"
            if self.stage_before != self.stage_after
            else f"stage unchanged ({self.stage_after})"
        )
        return f"{self.title or self.kind}: {moves or 'no move'}; {stage}"


class RelationshipEngine:
    """Read, observe, explain, reset, and edit one character's relationship."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
        stages: tuple[tuple[str, str, float], ...] = STAGES,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)
        self.stages = stages
        self._seen: set[str] = set()

    # -- state ------------------------------------------------------------

    def state(self, character_id: str, *, user_id: str = "local") -> Any:
        """The stored relationship row, creating a fresh one on first read."""
        existing = self.storage.domain.list_records(
            "relationship_states",
            where="character_id = ? AND user_id = ?",
            params=(character_id, user_id),
        )
        if existing:
            return existing[0]
        return self.storage.domain.create(
            "relationship_states", character_id=character_id, user_id=user_id
        )

    def snapshot(self, character_id: str, *, user_id: str = "local") -> RelationshipSnapshot:
        state = self.state(character_id, user_id=user_id)
        dimensions = {name: float(getattr(state, name)) for name in DIMENSIONS}
        events = self.events(character_id, user_id=user_id)
        stage = self.stage_for(dimensions)
        return RelationshipSnapshot(
            character_id=character_id,
            user_id=user_id,
            dimensions=dimensions,
            stage=stage[0],
            stage_title=stage[1],
            overall=self._overall(dimensions),
            event_count=len(events),
        )

    def events(self, character_id: str, *, user_id: str = "local") -> tuple[Any, ...]:
        """Every recorded event for the character, oldest first."""
        return tuple(
            self.storage.domain.list_records(
                "relationship_events",
                where="character_id = ?",
                params=(character_id,),
                order_by="occurred_at, record_id",
            )
        )

    # -- observation ------------------------------------------------------

    def observe(
        self,
        character_id: str,
        kind: str,
        *,
        message_id: str = "",
        conversation_id: str = "",
        title: str = "",
        detail: str = "",
        deltas: dict[str, float] | None = None,
        user_id: str = "local",
    ) -> RelationshipChange:
        """Record an explicit event and move the state by its bounded deltas.

        The event must be a known kind. ``message_id`` is the evidence: an event
        with no evidence is still recorded, but it moves nothing (a milestone the
        user asked to mark is a record, not a change). Calling with the same
        ``(kind, message_id)`` twice is idempotent -- the second call changes
        nothing, so a message replayed does not double-count.
        """
        if kind not in EVENT_EFFECTS:
            raise RelationshipError(f"unknown relationship event: {kind!r}")
        key = f"{character_id}:{kind}:{message_id or conversation_id or title}"
        if message_id and key in self._seen:
            return RelationshipChange(
                kind=kind,
                title=title or kind,
                reason="already recorded; no change",
            )
        if message_id:
            self._seen.add(key)

        state = self.state(character_id, user_id=user_id)
        before_stage = self.stage_for({n: float(getattr(state, n)) for n in DIMENSIONS})[0]

        applied: dict[str, float] = {}
        if message_id:  # evidence required for a move
            source = deltas if deltas is not None else EVENT_EFFECTS[kind]
            for name, delta in source.items():
                if name not in DIMENSIONS:
                    continue
                current = float(getattr(state, name))
                step = _bounded(float(delta))
                new_value = _clamp(current + step)
                applied[name] = round(new_value - current, 6)
            if applied:
                state = self.storage.domain.update(
                    "relationship_states", state.record_id, **{
                        name: _clamp(float(getattr(state, name)) + value)
                        for name, value in applied.items()
                    }
                )

        dimensions = {name: float(getattr(state, name)) for name in DIMENSIONS}
        after_stage = self.stage_for(dimensions)
        if after_stage[0] != before_stage:
            state = self.storage.domain.update(
                "relationship_states", state.record_id, stage=after_stage[0]
            )

        self.storage.domain.create(
            "relationship_events",
            character_id=character_id,
            kind=kind,
            title=title or kind,
            detail=detail,
            dimension=", ".join(applied) if applied else "",
            delta=sum(applied.values()),
            message_id=message_id,
            conversation_id=conversation_id,
            occurred_at=utc_now(),
        )
        return RelationshipChange(
            kind=kind,
            title=title or kind,
            deltas=applied,
            stage_before=before_stage,
            stage_after=after_stage[0],
            message_id=message_id,
            reason=detail or f"{kind} observed",
        )

    # -- explanation ------------------------------------------------------

    def explain(self, character_id: str, *, user_id: str = "local") -> str:
        """Why the relationship is where it is: the events, in order."""
        snapshot = self.snapshot(character_id, user_id=user_id)
        events = self.events(character_id, user_id=user_id)
        lines = [
            f"Relationship with {character_id}: {snapshot.stage_title} "
            f"({snapshot.overall:.2f}), {len(events)} event(s)."
        ]
        if not events:
            lines.append("No events yet — the relationship is at its starting point.")
        for event in events:
            moved = f" ({event.dimension}: {event.delta:+.2f})" if event.dimension else ""
            lines.append(f"- {event.occurred_at} — {event.title or event.kind}{moved}")
        return "\n".join(lines)

    # -- reset / edit -----------------------------------------------------

    def reset(self, character_id: str, *, user_id: str = "local") -> RelationshipSnapshot:
        """Return the state to its starting point (the event log is preserved)."""
        state = self.state(character_id, user_id=user_id)
        self.storage.domain.update(
            "relationship_states",
            state.record_id,
            familiarity=0.0,
            trust=0.0,
            closeness=0.0,
            comfort=0.0,
            affection=0.0,
            shared_experiences=0.0,
            stability=0.5,
            stage=STAGE_UNKNOWN,
        )
        self._seen.clear()
        return self.snapshot(character_id, user_id=user_id)

    def edit(
        self, character_id: str, *, user_id: str = "local", **dimensions: float
    ) -> RelationshipSnapshot:
        """Set one or more dimensions directly (a user's explicit correction)."""
        state = self.state(character_id, user_id=user_id)
        changes = {name: _clamp(float(value)) for name, value in dimensions.items() if name in DIMENSIONS}
        if not changes:
            raise RelationshipError("no known dimension to edit")
        state = self.storage.domain.update("relationship_states", state.record_id, **changes)
        stage = self.stage_for({n: float(getattr(state, n)) for n in DIMENSIONS})
        self.storage.domain.update("relationship_states", state.record_id, stage=stage[0])
        return self.snapshot(character_id, user_id=user_id)

    # -- stages -----------------------------------------------------------

    def stage_for(self, dimensions: dict[str, float]) -> tuple[str, str]:
        """The stage the dimensions reach: the highest threshold they pass."""
        score = self._overall(dimensions)
        reached = self.stages[0]
        for entry in self.stages:
            if score >= entry[2]:
                reached = entry
        return reached[0], reached[1]

    @staticmethod
    def _overall(dimensions: dict[str, float]) -> float:
        """The overall closeness: the mean of the seven bounded dimensions."""
        if not dimensions:
            return 0.0
        return sum(dimensions.values()) / len(dimensions)


__all__ = [
    "DIMENSIONS",
    "EVENT_EFFECTS",
    "EVENT_KINDS",
    "MAX_STEP",
    "STAGES",
    "STAGE_UNKNOWN",
    "RelationshipChange",
    "RelationshipEngine",
    "RelationshipError",
    "RelationshipSnapshot",
]
