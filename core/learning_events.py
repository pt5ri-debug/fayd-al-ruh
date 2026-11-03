"""The learning-event contract: one durable, auditable record per learning act.

Phase 3 requires every meaningful learning update to be auditable and reversible.
This module is the *shape* of that record, so the knowledge pipeline, the
experience pipeline, the feedback processor, and the pattern learner all emit the
same thing and the durable store, the evaluation, and the gate all read the same
thing -- without any of them inventing their own dictionary.

**Honest by construction.** A learning event always carries:

* a unique, monotonic **event id** and a **timestamp**;
* its **kind** (knowledge, experience, feedback, pattern, procedure, rollback);
* its **source** and **source reference** (where the learning came from);
* the **target** it acted on and a **change summary**;
* a **before / after** state reference where a change happened;
* the **outcome** (accepted, rejected, deferred, conflicted, superseded, rolled
  back) and whether the change was actually **applied**;
* an optional **evaluation** reference and **version** identifier;
* the **owner** whose learning it is, so per-personality isolation is structural.

Nothing here opens a file, loads a model, or reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from hashlib import blake2b
from typing import Any

#: The kinds of learning act the architecture records.
KIND_KNOWLEDGE = "knowledge"
KIND_EXPERIENCE = "experience"
KIND_FEEDBACK = "feedback"
KIND_PATTERN = "pattern"
KIND_PROCEDURE = "procedure"
KIND_ROLLBACK = "rollback"

LEARNING_KINDS: tuple[str, ...] = (
    KIND_KNOWLEDGE,
    KIND_EXPERIENCE,
    KIND_FEEDBACK,
    KIND_PATTERN,
    KIND_PROCEDURE,
    KIND_ROLLBACK,
)

#: The outcome of a learning act. ``applied`` is a *separate* fact: an accepted
#: act may still be deferred, and a rejected act is never applied.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_DEFERRED = "deferred"
OUTCOME_CONFLICTED = "conflicted"
OUTCOME_SUPERSEDED = "superseded"
OUTCOME_ROLLED_BACK = "rolled_back"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_ACCEPTED,
    OUTCOME_REJECTED,
    OUTCOME_DEFERRED,
    OUTCOME_CONFLICTED,
    OUTCOME_SUPERSEDED,
    OUTCOME_ROLLED_BACK,
)

#: Provenance kinds a learning event may carry (kept in step with the memory and
#: learning-record vocabularies, so the three never disagree).
SOURCE_USER = "user"
SOURCE_DOCUMENT = "document"
SOURCE_INTERNET = "internet"
SOURCE_MODEL = "model"
SOURCE_CHARACTER = "character"
SOURCE_SYSTEM = "system"
SOURCE_REASONING = "reasoning"
SOURCE_IMPORT = "import"

SOURCES: tuple[str, ...] = (
    SOURCE_USER,
    SOURCE_DOCUMENT,
    SOURCE_INTERNET,
    SOURCE_MODEL,
    SOURCE_CHARACTER,
    SOURCE_SYSTEM,
    SOURCE_REASONING,
    SOURCE_IMPORT,
)


def now_utc() -> datetime:
    """The current moment, timezone-aware (UTC)."""
    return datetime.now(UTC)


def _aware(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment


def make_event_id(kind: str, target: str, *, moment: datetime | None = None, owner: str = "") -> str:
    """A deterministic, content-addressed event id.

    Two events with the same kind, target, owner, and timestamp collapse to one
    id, which is what makes a replayed learning act idempotent.
    """
    when = _aware(moment) or now_utc()
    digest = blake2b(
        f"{kind}|{owner}|{target}|{when.isoformat()}".encode(), digest_size=8
    ).hexdigest()
    return f"LE-{digest}"


@dataclass(frozen=True)
class LearningEvent:
    """One auditable learning act, with its provenance and its outcome.

    Frozen on purpose: a later act (a correction, a rollback) is a **new** event
    that points back at the one it revises, so the history is never rewritten.
    """

    kind: str
    target: str
    owner: str = "fayd_al_ruh"
    event_id: str = ""
    source: str = SOURCE_SYSTEM
    source_reference: str = ""
    summary: str = ""
    before: str = ""
    after: str = ""
    outcome: str = OUTCOME_ACCEPTED
    applied: bool = False
    confidence: float = 0.0
    version: str = ""
    evaluation: str = ""
    supersedes: str = ""
    created_at: datetime = field(default_factory=now_utc)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in LEARNING_KINDS:
            raise ValueError(f"Unknown learning kind {self.kind!r}.")
        if not self.target.strip():
            raise ValueError("A learning event needs a target.")
        if not self.owner.strip():
            raise ValueError("A learning event needs an owner.")
        if self.source not in SOURCES:
            raise ValueError(f"Unknown learning source {self.source!r}.")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Unknown learning outcome {self.outcome!r}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Learning confidence must be between 0 and 1.")
        object.__setattr__(self, "created_at", _aware(self.created_at) or now_utc())
        if not self.event_id:
            object.__setattr__(
                self,
                "event_id",
                make_event_id(self.kind, self.target, moment=self.created_at, owner=self.owner),
            )

    @property
    def changed(self) -> bool:
        """True when the act actually changed stored state."""
        return self.applied and self.before != self.after

    def describe(self) -> str:
        """One honest line: kind, target, outcome, and whether it was applied."""
        marker = "applied" if self.applied else "not applied"
        return (
            f"[{self.kind}] {self.target} -> {self.outcome} ({marker}, "
            f"owner={self.owner}, source={self.source})"
        )

    def to_primitive(self) -> dict[str, Any]:
        """The event as plain JSON-able data, for the durable store."""
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "target": self.target,
            "owner": self.owner,
            "source": self.source,
            "source_reference": self.source_reference,
            "summary": self.summary,
            "before": self.before,
            "after": self.after,
            "outcome": self.outcome,
            "applied": self.applied,
            "confidence": self.confidence,
            "version": self.version,
            "evaluation": self.evaluation,
            "supersedes": self.supersedes,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> LearningEvent:
        """Rebuild an event from stored data (a corrupt field falls back)."""
        return cls(
            kind=str(data.get("kind", KIND_KNOWLEDGE)),
            target=str(data.get("target", "")),
            owner=str(data.get("owner", "fayd_al_ruh")),
            event_id=str(data.get("event_id", "")),
            source=str(data.get("source", SOURCE_SYSTEM)),
            source_reference=str(data.get("source_reference", "")),
            summary=str(data.get("summary", "")),
            before=str(data.get("before", "")),
            after=str(data.get("after", "")),
            outcome=str(data.get("outcome", OUTCOME_ACCEPTED)),
            applied=bool(data.get("applied", False)),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            version=str(data.get("version", "")),
            evaluation=str(data.get("evaluation", "")),
            supersedes=str(data.get("supersedes", "")),
            created_at=_parse_time(data.get("created_at")),
            metadata=dict(data.get("metadata") or {}),
        )


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware(value) or now_utc()
    if isinstance(value, str) and value:
        try:
            return _aware(datetime.fromisoformat(value)) or now_utc()
        except ValueError:
            return now_utc()
    return now_utc()


class LearningKind(str, Enum):
    """A string enum of the kinds, for callers that prefer a typed value.

    A genuine string enum, so ``str(LearningKind.KNOWLEDGE)`` is
    ``"LearningKind.KNOWLEDGE"`` and a persisted value round-trips.
    """

    KNOWLEDGE = KIND_KNOWLEDGE
    EXPERIENCE = KIND_EXPERIENCE
    FEEDBACK = KIND_FEEDBACK
    PATTERN = KIND_PATTERN
    PROCEDURE = KIND_PROCEDURE
    ROLLBACK = KIND_ROLLBACK


__all__ = [
    "KIND_EXPERIENCE",
    "KIND_FEEDBACK",
    "KIND_KNOWLEDGE",
    "KIND_PATTERN",
    "KIND_PROCEDURE",
    "KIND_ROLLBACK",
    "LEARNING_KINDS",
    "OUTCOMES",
    "OUTCOME_ACCEPTED",
    "OUTCOME_CONFLICTED",
    "OUTCOME_DEFERRED",
    "OUTCOME_REJECTED",
    "OUTCOME_ROLLED_BACK",
    "OUTCOME_SUPERSEDED",
    "SOURCES",
    "SOURCE_CHARACTER",
    "SOURCE_DOCUMENT",
    "SOURCE_IMPORT",
    "SOURCE_INTERNET",
    "SOURCE_MODEL",
    "SOURCE_REASONING",
    "SOURCE_SYSTEM",
    "SOURCE_USER",
    "LearningEvent",
    "LearningKind",
    "make_event_id",
    "now_utc",
]
