"""Pattern and procedure learning: recurring approaches, versioned and evaluated.

Phase 3 requires the system to identify recurring patterns from stored
experiences, track supporting and contradicting experiences, evaluate a
procedure's reliability, record its applicability conditions and known failure
modes, version learned procedures, and -- crucially -- **never** promote a weak
pattern into a trusted procedure automatically.

This module is that process. It reads the durable episodic memories (the Phase-2
store) and folds them into versioned :class:`LearnedProcedure` records that live
in the Phase-3 learning store.

**Honest by construction.** A procedure is a **proposal** until it is evaluated:

* it starts at ``STATUS_PROPOSED`` with a reliability that must be earned from
  real support, never asserted;
* a pattern seen once is reported but never promoted;
* supporting and contradicting experiences are counted **separately**, so a
  procedure that fails half the time cannot read as reliable;
* its applicability conditions and known failure modes are recorded, so a caller
  knows when it holds and when it has failed;
* each version is kept, and a revision supersedes the old one without deleting it.

Nothing here executes a procedure: it *records* one. Execution stays behind the
project's existing tool sandbox and approval gates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from core.learning_events import (
    KIND_PROCEDURE,
    OUTCOME_ACCEPTED,
    OUTCOME_DEFERRED,
    SOURCE_REASONING,
    LearningEvent,
    now_utc,
)
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_record import MemoryKind

#: A procedure's lifecycle. It is a proposal until evaluated; only an evaluated
#: procedure with real support becomes trusted.
STATUS_PROPOSED = "proposed"
STATUS_EVALUATED = "evaluated"
STATUS_TRUSTED = "trusted"
STATUS_DEPRECATED = "deprecated"

PROCEDURE_STATUSES: tuple[str, ...] = (
    STATUS_PROPOSED,
    STATUS_EVALUATED,
    STATUS_TRUSTED,
    STATUS_DEPRECATED,
)

#: The number of supporting experiences a procedure needs before it may be
#: trusted. A single positive result never proves reliability.
DEFAULT_MIN_SUPPORT = 3

#: The success rate a procedure needs before it may be trusted.
DEFAULT_MIN_SUCCESS_RATE = 0.7


@dataclass
class LearnedProcedure:
    """A reusable procedure learned from experience, with its evidence and version.

    Frozen on purpose: a revision is a **new** version that ``supersedes`` the old
    one, so every version is kept.
    """

    procedure_id: str
    name: str
    steps: tuple[str, ...] = ()
    status: str = STATUS_PROPOSED
    version: int = 1
    support: int = 0
    contradictions: int = 0
    confidence: float = 0.0
    applicability: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    experience_ids: tuple[str, ...] = ()
    owner: str = "fayd_al_ruh"
    supersedes: str = ""
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        if self.status not in PROCEDURE_STATUSES:
            raise ValueError(f"Unknown procedure status {self.status!r}.")
        if not self.name.strip():
            raise ValueError("A learned procedure needs a name.")
        if self.version < 1:
            raise ValueError("A procedure version starts at 1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Procedure confidence must be in [0, 1].")

    @property
    def kind(self) -> str:
        """The learned-item kind (used by the durable store's item table)."""
        return KIND_PROCEDURE

    @property
    def item_id(self) -> str:
        """The store key: one row per procedure name, holding its latest version."""
        return f"procedure:{self.owner}:{self.name}"

    @property
    def success_rate(self) -> float:
        total = self.support + self.contradictions
        return self.support / total if total else 0.0

    @property
    def trusted(self) -> bool:
        return self.status == STATUS_TRUSTED

    def describe(self) -> str:
        return (
            f"[{self.status} v{self.version}] {self.name}: "
            f"{len(self.steps)} step(s), support={self.support}, "
            f"contradictions={self.contradictions}, rate={self.success_rate:.2f}"
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "steps": list(self.steps),
            "status": self.status,
            "version": self.version,
            "support": self.support,
            "contradictions": self.contradictions,
            "confidence": self.confidence,
            "applicability": list(self.applicability),
            "failure_modes": list(self.failure_modes),
            "experience_ids": list(self.experience_ids),
            "owner": self.owner,
            "supersedes": self.supersedes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> LearnedProcedure:
        return cls(
            procedure_id=str(data.get("procedure_id", "")),
            name=str(data.get("name", "")),
            steps=tuple(str(step) for step in data.get("steps", ())),
            status=str(data.get("status", STATUS_PROPOSED)),
            version=int(data.get("version", 1) or 1),
            support=int(data.get("support", 0) or 0),
            contradictions=int(data.get("contradictions", 0) or 0),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            applicability=tuple(str(item) for item in data.get("applicability", ())),
            failure_modes=tuple(str(item) for item in data.get("failure_modes", ())),
            experience_ids=tuple(str(item) for item in data.get("experience_ids", ())),
            owner=str(data.get("owner", "fayd_al_ruh")),
            supersedes=str(data.get("supersedes", "")),
        )


@dataclass
class PatternReport:
    """What one extraction pass found and what it did about it."""

    found: tuple[str, ...] = ()
    promoted: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    events: tuple[LearningEvent, ...] = ()
    reason: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.promoted or self.updated)

    def describe(self) -> str:
        return (
            f"patterns: {len(self.found)} found, {len(self.promoted)} promoted, "
            f"{len(self.updated)} updated"
            + (f"; {self.reason}" if self.reason else "")
        )


class PatternLearning:
    """The one path from stored experiences to versioned, evaluated procedures."""

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        store: LearningStore | None = None,
        owner: str = "fayd_al_ruh",
        min_support: int = DEFAULT_MIN_SUPPORT,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ) -> None:
        if min_support < 1:
            raise ValueError("Minimum support must be at least 1.")
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("Minimum success rate must be in [0, 1].")
        self.memory = memory
        self.store = store
        self.owner = owner
        self.min_support = min_support
        self.min_success_rate = min_success_rate
        self._procedures: dict[str, LearnedProcedure] = {}

    # -- extraction --------------------------------------------------------
    def extract(self, *, limit: int = 500, record: bool = True) -> PatternReport:
        """Read the durable episodic memories and fold them into procedures."""
        records, _total = self.memory.page(kind=MemoryKind.EPISODIC, limit=limit)
        groups: dict[str, list[Any]] = {}
        for memory in records:
            key = self._group_key(memory)
            if key:
                groups.setdefault(key, []).append(memory)

        found: list[str] = []
        promoted: list[str] = []
        updated: list[str] = []
        events: list[LearningEvent] = []

        for key, group in sorted(groups.items()):
            if len(group) < 2:  # a single experience is not yet a pattern
                continue
            found.append(key)
            procedure, changed, was_promoted = self._fold(key, group)
            if procedure is None:
                continue
            if was_promoted:
                promoted.append(key)
            elif changed:
                updated.append(key)
            if record and changed:
                events.append(self._event(procedure, was_promoted))

        if record and self.store is not None and events:
            version = self.store.next_version()
            events = [replace(event, version=str(version)) for event in events]
            self.store.record_many(events)

        return PatternReport(
            found=tuple(found),
            promoted=tuple(promoted),
            updated=tuple(updated),
            events=tuple(events),
            reason=self._reason(found, promoted),
        )

    def procedures(self) -> tuple[LearnedProcedure, ...]:
        return tuple(self._procedures.values())

    def trusted_procedures(self) -> tuple[LearnedProcedure, ...]:
        return tuple(procedure for procedure in self._procedures.values() if procedure.trusted)

    def procedure(self, name: str) -> LearnedProcedure | None:
        return self._procedures.get(name)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _group_key(record: Any) -> str:
        """The topic a memory belongs to: its goal summary, or its entities."""
        from core.tokenize import tokenize

        summary = str(getattr(record, "summary", "") or "").strip()
        if summary:
            return " ".join(sorted(tokenize(summary)))
        entities = getattr(record, "entities", ())
        return " ".join(sorted(str(entity) for entity in entities))

    def _fold(self, key: str, group: list[Any]) -> tuple[LearnedProcedure | None, bool, bool]:
        supports = [record for record in group if bool(record.metadata.get("success"))]
        contradicts = [record for record in group if not bool(record.metadata.get("success"))]
        if not supports:
            return None, False, False

        existing = self._procedures.get(key)
        version = (existing.version + 1) if existing is not None else 1
        steps = tuple(self._steps(supports))
        failure_modes = tuple(sorted({self._failure(record) for record in contradicts}))
        applicability = tuple(sorted({str(record.metadata.get("scope", "")) for record in supports if record.metadata.get("scope")}))
        support = len(supports)
        contradictions = len(contradicts)
        confidence = round((support + 1) / (support + contradictions + 2), 6)
        status = self._status(support, contradictions, confidence)
        procedure = LearnedProcedure(
            procedure_id=f"PROC-{key[:12]}-v{version}",
            name=key,
            steps=steps,
            status=status,
            version=version,
            support=support,
            contradictions=contradictions,
            confidence=confidence,
            applicability=applicability,
            failure_modes=failure_modes,
            experience_ids=tuple(str(record.record_id) for record in group),
            owner=self.owner,
            supersedes=existing.procedure_id if existing is not None else "",
        )
        changed = existing is None or existing.to_primitive() != procedure.to_primitive()
        if not changed:
            return existing, False, False
        was_promoted = status == STATUS_TRUSTED and (existing is None or existing.status != STATUS_TRUSTED)
        self._procedures[key] = procedure
        if self.store is not None:
            self.store.save_item(procedure)
        return procedure, True, was_promoted

    @staticmethod
    def _steps(records: Iterable[Any]) -> list[str]:
        """The approach steps the supporting experiences agree on."""
        steps: list[str] = []
        for record in records:
            approach = str(record.metadata.get("approach", "")).strip()
            if approach and approach not in steps:
                steps.append(approach)
        return steps

    @staticmethod
    def _failure(record: Any) -> str:
        errors = record.metadata.get("errors") or ()
        if errors:
            return str(errors[0])
        return str(record.content)[:120]

    def _status(self, support: int, contradictions: int, confidence: float) -> str:
        rate = support / (support + contradictions) if (support + contradictions) else 0.0
        if support >= self.min_support and rate >= self.min_success_rate:
            return STATUS_TRUSTED
        if support + contradictions >= 2:
            return STATUS_EVALUATED
        return STATUS_PROPOSED

    def _event(self, procedure: LearnedProcedure, promoted: bool) -> LearningEvent:
        return LearningEvent(
            kind=KIND_PROCEDURE,
            target=procedure.name,
            owner=self.owner,
            source=SOURCE_REASONING,
            summary=procedure.describe(),
            after=procedure.procedure_id,
            outcome=OUTCOME_ACCEPTED if promoted else OUTCOME_DEFERRED,
            applied=True,
            confidence=procedure.confidence,
            metadata={"version": procedure.version, "status": procedure.status},
        )

    def _reason(self, found: list[str], promoted: list[str]) -> str:
        if not found:
            return "no repeated pattern found yet"
        if not promoted:
            return "patterns found but none met the support and success-rate floors"
        return f"{len(promoted)} procedure(s) reached trusted status"

    # -- history -----------------------------------------------------------
    def revisions(self, name: str) -> tuple[dict[str, Any], ...]:
        """The stored versions of a procedure (its revision history)."""
        if self.store is None:
            return ()
        events, _ = self.store.events(kind=KIND_PROCEDURE, limit=500)
        return tuple(
            event.to_primitive() for event in events if event.target == name
        )

    def describe(self) -> str:
        trusted = len(self.trusted_procedures())
        return (
            f"pattern learning for {self.owner}: {len(self._procedures)} procedure(s), "
            f"{trusted} trusted"
        )


__all__ = [
    "DEFAULT_MIN_SUCCESS_RATE",
    "DEFAULT_MIN_SUPPORT",
    "PROCEDURE_STATUSES",
    "STATUS_DEPRECATED",
    "STATUS_EVALUATED",
    "STATUS_PROPOSED",
    "STATUS_TRUSTED",
    "LearnedProcedure",
    "PatternLearning",
    "PatternReport",
]
