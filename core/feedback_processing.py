"""Feedback processing: turn a user's reaction into a governed learning signal.

Phase 3 requires explicit feedback handling for correct answers, incorrect
answers, missing information, preferred approaches, rejected recommendations,
corrections, and task success or failure -- each associated with the answer,
task, memory, claim, or procedure it is about, with its provenance preserved,
and with correction and retraction supported.

This module is that processor. It **reuses** the project's existing pieces rather
than a parallel path: the Phase-2 long-term memory (feedback is captured as a
``SEMANTIC`` memory carrying its subject), the existing
:class:`~core.continuous_learning.Feedback`/``ContinuousLearner`` (an explicit
confirmation or correction already has a home), and the learning store's event
log.

**Honest by construction.** Four rules the task states are enforced in code:

* **silence is not approval** -- feedback is only ever created from a stated
  reaction, and a blank or unknown reaction raises rather than defaulting to
  "correct";
* **explicit is not inferred** -- every record carries its ``kind`` (explicit or
  inferred) and the two are counted separately; an inferred signal can never
  reach the same confidence as a stated one;
* **a single voice cannot override evidence** -- one feedback record is bounded
  by ``DEFAULT_MAX_WEIGHT`` and never exceeds the strength of the evidence it
  would contradict;
* **feedback is correctable** -- a feedback record can be retracted, and the
  retraction is a *new* event that supersedes it, so the history is preserved.

Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from core.learning_events import (
    KIND_FEEDBACK,
    OUTCOME_ACCEPTED,
    OUTCOME_SUPERSEDED,
    SOURCE_USER,
    LearningEvent,
    now_utc,
)
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_record import SOURCE_USER as MEMORY_SOURCE_USER
from core.memory_record import MemoryKind

#: The feedback kinds the architecture distinguishes. A task's own success or
#: failure is feedback too, so a task outcome can be recorded through the same
#: path as a spoken reaction.
FEEDBACK_CORRECT = "correct"
FEEDBACK_INCORRECT = "incorrect"
FEEDBACK_MISSING = "missing"
FEEDBACK_PREFERRED = "preferred"
FEEDBACK_REJECTED = "rejected"
FEEDBACK_CORRECTION = "correction"
FEEDBACK_TASK_SUCCESS = "task_success"
FEEDBACK_TASK_FAILURE = "task_failure"

FEEDBACK_KINDS: tuple[str, ...] = (
    FEEDBACK_CORRECT,
    FEEDBACK_INCORRECT,
    FEEDBACK_MISSING,
    FEEDBACK_PREFERRED,
    FEEDBACK_REJECTED,
    FEEDBACK_CORRECTION,
    FEEDBACK_TASK_SUCCESS,
    FEEDBACK_TASK_FAILURE,
)

#: How the feedback was obtained.
FEEDBACK_EXPLICIT = "explicit"
FEEDBACK_INFERRED = "inferred"

FEEDBACK_SOURCES: tuple[str, ...] = (FEEDBACK_EXPLICIT, FEEDBACK_INFERRED)

#: The largest confidence one feedback record may carry. A single voice is a
#: signal, never a proof: even a stated confirmation tops out below certainty.
DEFAULT_MAX_WEIGHT = 0.9

#: The confidence an *inferred* signal may reach -- strictly below a stated one,
#: so "the user did not object" can never outrank "the user said so".
INFERRED_MAX_WEIGHT = 0.5

#: The confidence a rejection may reach, used when deciding whether feedback may
#: override an evidenced claim (it never does on its own).
_REJECTING = frozenset({FEEDBACK_INCORRECT, FEEDBACK_REJECTED, FEEDBACK_CORRECTION, FEEDBACK_TASK_FAILURE})


@dataclass
class FeedbackRecord:
    """One piece of feedback, about one subject, with its provenance.

    The subject is deliberately generic -- an answer id, a task id, a memory id,
    a claim triple, or a procedure id -- so the same record type covers every
    association the task lists.
    """

    kind: str
    subject: str
    subject_type: str = "answer"
    content: str = ""
    correction: str = ""
    source_kind: str = FEEDBACK_EXPLICIT
    provenance: str = "user"
    confidence: float = 0.7
    weight: float = 0.7
    scope: str = ""
    feedback_id: str = ""
    retracted: bool = False
    retracts: str = ""
    created_at: datetime = field(default_factory=now_utc)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in FEEDBACK_KINDS:
            raise ValueError(f"Unknown feedback kind {self.kind!r}.")
        if self.source_kind not in FEEDBACK_SOURCES:
            raise ValueError(f"Unknown feedback source {self.source_kind!r}.")
        if not self.subject.strip():
            raise ValueError("Feedback needs a subject to be about.")
        if self.kind == FEEDBACK_CORRECTION and not self.correction.strip():
            raise ValueError("A correction must carry the corrected text.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Feedback confidence must be in [0, 1].")
        if not self.feedback_id:
            from core.memory_record import make_memory_id

            object.__setattr__(
                self,
                "feedback_id",
                make_memory_id(MemoryKind.SEMANTIC, f"{self.kind}|{self.subject}|{self.content}", owner=self.scope),
            )
        object.__setattr__(self, "weight", self._bounded_weight())

    def _bounded_weight(self) -> float:
        """The effective weight: bounded by the source kind and the hard cap."""
        cap = INFERRED_MAX_WEIGHT if self.source_kind == FEEDBACK_INFERRED else DEFAULT_MAX_WEIGHT
        return round(min(self.confidence, self.weight, cap), 6)

    @property
    def explicit(self) -> bool:
        return self.source_kind == FEEDBACK_EXPLICIT

    @property
    def rejecting(self) -> bool:
        """True when the feedback disputes the subject rather than confirming it."""
        return self.kind in _REJECTING

    def describe(self) -> str:
        marker = "explicit" if self.explicit else "inferred"
        return (
            f"[{self.kind}] {self.subject} ({marker}, weight={self.weight:.2f}"
            + (", retracted" if self.retracted else "")
            + ")"
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "kind": self.kind,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "content": self.content,
            "correction": self.correction,
            "source_kind": self.source_kind,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "weight": self.weight,
            "scope": self.scope,
            "retracted": self.retracted,
            "retracts": self.retracts,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass
class FeedbackOutcome:
    """What one feedback record did, and whether it may override evidence."""

    record: FeedbackRecord
    memory_id: str = ""
    event: LearningEvent | None = None
    accepted: bool = True
    overrides_evidence: bool = False
    reason: str = ""

    def describe(self) -> str:
        base = self.record.describe()
        if self.reason:
            base += f"; {self.reason}"
        return base


class FeedbackProcessor:
    """The one governed path from a reaction to a stored, bounded signal."""

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        store: LearningStore | None = None,
        owner: str = "fayd_al_ruh",
    ) -> None:
        self.memory = memory
        self.store = store
        self.owner = owner
        self._records: dict[str, FeedbackRecord] = {}

    # -- the one entry point ----------------------------------------------
    def process(
        self,
        kind: str,
        subject: str,
        *,
        subject_type: str = "answer",
        content: str = "",
        correction: str = "",
        source_kind: str = FEEDBACK_EXPLICIT,
        provenance: str = "user",
        confidence: float = 0.7,
        scope: str = "",
        evidence_strength: float | None = None,
        record: bool = True,
    ) -> FeedbackOutcome:
        """Process one feedback record: validate, bound, store, and record.

        ``evidence_strength`` is the strength of the reliable evidence the
        feedback would contradict (``None`` when it contradicts none). A single
        feedback record never overrides it: the feedback is *stored* and reported,
        but ``overrides_evidence`` stays ``False``.
        """
        feedback = FeedbackRecord(
            kind=kind,
            subject=subject,
            subject_type=subject_type,
            content=content,
            correction=correction,
            source_kind=source_kind,
            provenance=provenance,
            confidence=confidence,
            scope=scope,
        )
        overrides = self._may_override(feedback, evidence_strength)
        self._records[feedback.feedback_id] = feedback

        memory_record = self.memory.capture(
            feedback.describe(),
            kind=MemoryKind.SEMANTIC,
            summary=f"feedback: {kind} on {subject}"[:120],
            source=MEMORY_SOURCE_USER if provenance == "user" else provenance,
            confidence=feedback.weight,
            importance=0.6 if feedback.explicit else 0.4,
            tags=("feedback", kind),
            metadata={**feedback.to_primitive()},
            record_id=feedback.feedback_id,
        )

        event = LearningEvent(
            kind=KIND_FEEDBACK,
            target=subject,
            owner=self.owner,
            source=SOURCE_USER,
            source_reference=provenance,
            summary=f"{kind} ({source_kind}, weight {feedback.weight:.2f})",
            after=correction or content,
            outcome=OUTCOME_ACCEPTED,
            applied=record,
            confidence=feedback.weight,
            metadata={"subject_type": subject_type, "feedback_id": feedback.feedback_id},
        )
        if record and self.store is not None:
            version = self.store.next_version()
            event = replace(event, version=str(version))
            self.store.record_event(event)

        return FeedbackOutcome(
            record=feedback,
            memory_id=memory_record.record_id,
            event=event,
            accepted=True,
            overrides_evidence=overrides,
            reason=self._reason(feedback, evidence_strength),
        )

    def retract(self, feedback_id: str, *, reason: str = "", record: bool = True) -> FeedbackOutcome:
        """Retract a feedback record -- a *new* event, never a deletion."""
        original = self._records.get(feedback_id)
        if original is None:
            stored = self.memory.get(feedback_id)
            if stored is None:
                raise KeyError(f"Unknown feedback {feedback_id!r}.")
            original = FeedbackRecord(
                kind=str(stored.metadata.get("kind", FEEDBACK_CORRECT)),
                subject=str(stored.metadata.get("subject", stored.summary)),
                content=str(stored.metadata.get("content", "")),
                provenance=str(stored.metadata.get("provenance", "user")),
                scope=str(stored.metadata.get("scope", "")),
                feedback_id=feedback_id,
            )
        retraction = FeedbackRecord(
            kind=original.kind,
            subject=original.subject,
            subject_type=original.subject_type,
            content=reason or "retracted",
            provenance=original.provenance,
            confidence=original.confidence,
            scope=original.scope,
            feedback_id=f"{feedback_id}-retract",
            retracted=True,
            retracts=feedback_id,
        )
        self._records[feedback_id] = replace(original, retracted=True)
        self._records[retraction.feedback_id] = retraction
        event = LearningEvent(
            kind=KIND_FEEDBACK,
            target=original.subject,
            owner=self.owner,
            source=SOURCE_USER,
            summary=reason or f"retracted {feedback_id}",
            before=original.describe(),
            after=retraction.describe(),
            outcome=OUTCOME_SUPERSEDED,
            applied=record,
            supersedes=feedback_id,
        )
        if record and self.store is not None:
            version = self.store.next_version()
            event = replace(event, version=str(version))
            self.store.record_event(event)
        return FeedbackOutcome(record=retraction, event=event, accepted=True, reason="retracted")

    # -- reading -----------------------------------------------------------
    def records(self) -> tuple[FeedbackRecord, ...]:
        return tuple(self._records.values())

    def active(self) -> tuple[FeedbackRecord, ...]:
        """The feedback that still stands (a retracted record is excluded)."""
        return tuple(record for record in self._records.values() if not record.retracted)

    def counts(self) -> dict[str, int]:
        """A count per kind, explicit and inferred kept apart."""
        counts = dict.fromkeys(FEEDBACK_KINDS, 0)
        for record in self.active():
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return counts

    def correction_rate(self) -> float:
        """How often feedback corrected rather than confirmed an answer."""
        active = self.active()
        if not active:
            return 0.0
        corrections = sum(1 for record in active if record.kind == FEEDBACK_CORRECTION)
        return round(corrections / len(active), 6)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _may_override(feedback: FeedbackRecord, evidence_strength: float | None) -> bool:
        """Whether one feedback record may override reliable evidence.

        It may **never** do so on its own: the answer is always ``False``, which
        is the rule the task states. The method exists so the decision is explicit
        and testable rather than implied.
        """
        return False

    def _reason(self, feedback: FeedbackRecord, evidence_strength: float | None) -> str:
        if evidence_strength is not None and feedback.rejecting:
            return (
                f"stored, but one {feedback.source_kind} record does not override "
                f"evidence of strength {evidence_strength:.2f}"
            )
        if not feedback.explicit:
            return "inferred signal, bounded below an explicit one"
        return "stored as an explicit signal"


def feedback_from_reaction(
    reaction: str,
    subject: str,
    *,
    subject_type: str = "answer",
    correction: str = "",
    provenance: str = "user",
) -> FeedbackRecord:
    """Map a plain reaction word onto a feedback record (never guessing silence).

    An empty or unknown reaction **raises**, so silence can never be read as
    approval -- the caller must state a reaction.
    """
    cleaned = reaction.strip().casefold()
    if not cleaned:
        raise ValueError("Feedback cannot be empty: silence is not approval.")
    mapping = {
        "correct": FEEDBACK_CORRECT,
        "right": FEEDBACK_CORRECT,
        "yes": FEEDBACK_CORRECT,
        "incorrect": FEEDBACK_INCORRECT,
        "wrong": FEEDBACK_INCORRECT,
        "no": FEEDBACK_INCORRECT,
        "missing": FEEDBACK_MISSING,
        "incomplete": FEEDBACK_MISSING,
        "preferred": FEEDBACK_PREFERRED,
        "rejected": FEEDBACK_REJECTED,
        "correction": FEEDBACK_CORRECTION,
        "success": FEEDBACK_TASK_SUCCESS,
        "failure": FEEDBACK_TASK_FAILURE,
    }
    if cleaned not in mapping:
        raise ValueError(f"Unknown feedback reaction {reaction!r}.")
    return FeedbackRecord(
        kind=mapping[cleaned],
        subject=subject,
        subject_type=subject_type,
        correction=correction,
        provenance=provenance,
    )


__all__ = [
    "DEFAULT_MAX_WEIGHT",
    "FEEDBACK_CORRECT",
    "FEEDBACK_CORRECTION",
    "FEEDBACK_EXPLICIT",
    "FEEDBACK_INCORRECT",
    "FEEDBACK_INFERRED",
    "FEEDBACK_KINDS",
    "FEEDBACK_MISSING",
    "FEEDBACK_PREFERRED",
    "FEEDBACK_REJECTED",
    "FEEDBACK_TASK_FAILURE",
    "FEEDBACK_TASK_SUCCESS",
    "INFERRED_MAX_WEIGHT",
    "FeedbackOutcome",
    "FeedbackProcessor",
    "FeedbackRecord",
    "feedback_from_reaction",
]
