"""Shared records for the autonomous cognitive development system.

Every subsystem of :mod:`cognition` speaks in these few types, so the Proactive
Mind, the Cognitive Learning System, the Teacher-Student system, the incremental
trainer, and the evaluation harness can be wired together through clear
interfaces without any of them reaching into another's internals.

The most important rule the whole package keeps is the **epistemic ladder**:
an auto-generated :class:`Idea` is never a fact. The four rungs are kept as
explicit, ordered kinds --

``IDEA`` < ``HYPOTHESIS`` < ``SUPPORTED`` < ``VERIFIED``

-- and a rung can only be raised by an explicit, recorded act (a hypothesis is
made testable, evidence is attached, a verification passes), never by the
generator itself. :func:`promote` is the only function that moves an idea up a
rung, and it refuses to skip a rung.

The module is dependency-free and Tk-free, so it imports on a fresh clone with
no display and no weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4


def _now() -> datetime:
    return datetime.now(UTC)


def _identifier(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


class IdeaKind(str, Enum):
    """The epistemic rung an idea sits on.

    ``IDEA`` is a freshly generated thought; ``HYPOTHESIS`` is an idea stated so
    it *can* be tested; ``SUPPORTED`` has evidence attached but is not yet
    verified; ``VERIFIED`` has passed an explicit check. The order is the ladder
    :func:`promote` walks one rung at a time.
    """

    IDEA = "idea"
    HYPOTHESIS = "hypothesis"
    SUPPORTED = "supported"
    VERIFIED = "verified"

    @property
    def rank(self) -> int:
        return _KIND_RANK[self]


_KIND_RANK: dict[IdeaKind, int] = {
    IdeaKind.IDEA: 0,
    IdeaKind.HYPOTHESIS: 1,
    IdeaKind.SUPPORTED: 2,
    IdeaKind.VERIFIED: 3,
}


class SourceKind(str, Enum):
    """Where a piece of knowledge came from, for the reliability model."""

    CURRICULUM = "curriculum"
    CORPUS = "corpus"
    ONTOLOGY = "ontology"
    KNOWLEDGE_BASE = "knowledge_base"
    TEACHER = "teacher"
    STUDENT = "student"
    INTERNET = "internet"
    USER = "user"
    SELF = "self"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Source:
    """A named origin for a statement, with the reliability it is credited."""

    kind: SourceKind
    reference: str = ""
    reliability: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Source reliability must be between 0 and 1.")

    def describe(self) -> str:
        label = self.reference or self.kind.value
        return f"{label} (reliability {self.reliability:.2f})"

    def to_primitive(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "reference": self.reference,
            "reliability": self.reliability,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Source:
        kind = data.get("kind", SourceKind.UNKNOWN.value)
        try:
            parsed = SourceKind(str(kind))
        except ValueError:
            parsed = SourceKind.UNKNOWN
        return cls(kind=parsed, reference=str(data.get("reference", "")))


@dataclass
class Idea:
    """One generated thought, and everything needed to judge it honestly.

    ``kind`` is the epistemic rung; ``sources`` are the concepts/knowledge the
    idea was built from (so a reviewer can trace it); ``confidence`` is the
    generator's own belief, deliberately allowed to be low; ``critique`` holds
    the self-criticism engine's notes; ``evidence`` holds the ids of any
    supporting records once the idea reaches ``SUPPORTED``.
    """

    text: str
    kind: IdeaKind = IdeaKind.IDEA
    engine: str = ""
    sources: tuple[str, ...] = ()
    rationale: str = ""
    confidence: float = 0.3
    novelty: float = 0.5
    critique: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    origin: Source | None = None
    id: str = field(default_factory=lambda: _identifier("IDEA"))
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Idea text cannot be empty.")
        for value in (self.confidence, self.novelty):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Idea confidence and novelty must be between 0 and 1.")

    @property
    def is_fact(self) -> bool:
        """True only for a verified idea -- never for a generated one."""
        return self.kind is IdeaKind.VERIFIED

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind.value,
            "engine": self.engine,
            "sources": list(self.sources),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "critique": list(self.critique),
            "evidence": list(self.evidence),
            "origin": None
            if self.origin is None
            else {
                "kind": self.origin.kind.value,
                "reference": self.origin.reference,
                "reliability": self.origin.reliability,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Idea:
        origin_data = data.get("origin")
        origin = None
        if isinstance(origin_data, dict):
            try:
                origin = Source(
                    kind=SourceKind(str(origin_data.get("kind", "unknown"))),
                    reference=str(origin_data.get("reference", "")),
                    reliability=float(origin_data.get("reliability", 0.5)),
                )
            except (ValueError, TypeError):
                origin = None
        return cls(
            id=str(data.get("id") or _identifier("IDEA")),
            text=str(data.get("text", "")),
            kind=_kind_from(str(data.get("kind", "idea"))),
            engine=str(data.get("engine", "")),
            sources=_as_string_tuple(data.get("sources")),
            rationale=str(data.get("rationale", "")),
            confidence=_as_float(data.get("confidence"), 0.3),
            novelty=_as_float(data.get("novelty"), 0.5),
            critique=_as_string_tuple(data.get("critique")),
            evidence=_as_string_tuple(data.get("evidence")),
            origin=origin,
            created_at=_parse_time(data.get("created_at")),
            updated_at=_parse_time(data.get("updated_at")),
        )

    def describe(self) -> str:
        marker = {
            IdeaKind.IDEA: "idea",
            IdeaKind.HYPOTHESIS: "hypothesis",
            IdeaKind.SUPPORTED: "supported",
            IdeaKind.VERIFIED: "verified",
        }[self.kind]
        return f"[{marker}] {self.text} (confidence {self.confidence:.2f})"


def _kind_from(value: str) -> IdeaKind:
    try:
        return IdeaKind(value)
    except ValueError:
        return IdeaKind.IDEA


def _as_float(value: object, fallback: float = 0.0) -> float:
    """Read a stored float, falling back rather than raising on bad user data."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _as_string_tuple(value: object) -> tuple[str, ...]:
    """Read a stored list of strings, ignoring anything that is not a list."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _parse_time(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return _now()
    return _now()


class PromotionError(ValueError):
    """Raised when an idea is asked to skip a rung of the epistemic ladder."""


def promote(idea: Idea, to: IdeaKind, *, evidence: tuple[str, ...] = ()) -> Idea:
    """Move an idea exactly one rung up the ladder, recording why.

    Refuses to skip a rung (an ``IDEA`` cannot become ``VERIFIED`` directly) and
    refuses to attach evidence before the ``SUPPORTED`` rung, so the ladder is
    never decorative. Returns a **new** :class:`Idea`; the original is untouched.
    """
    if to.rank <= idea.kind.rank:
        raise PromotionError(f"Cannot promote {idea.kind.value} to {to.value}.")
    if to.rank - idea.kind.rank > 1:
        raise PromotionError(
            f"Cannot skip from {idea.kind.value} to {to.value}; promote one rung at a time."
        )
    if evidence and to is not IdeaKind.SUPPORTED:
        raise PromotionError("Evidence may only be attached when reaching 'supported'.")
    merged = idea.evidence + tuple(e for e in evidence if e not in idea.evidence)
    return replace(
        idea,
        kind=to,
        evidence=merged,
        updated_at=_now(),
        confidence=max(idea.confidence, 0.5) if to is IdeaKind.VERIFIED else idea.confidence,
    )


@dataclass(frozen=True)
class Experience:
    """One thing that happened: an expectation, the outcome, and the lesson.

    The learning system records these when the proactive mind forms an
    expectation and later observes what actually happened, so an error can be
    analysed against the belief that produced it rather than in a vacuum.
    """

    expectation: str
    outcome: str
    lesson: str = ""
    success: bool = False
    surprise: float = 0.0
    source: Source | None = None
    tags: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: _identifier("EXP"))
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.expectation.strip():
            raise ValueError("Experience expectation cannot be empty.")
        if not 0.0 <= self.surprise <= 1.0:
            raise ValueError("Surprise must be between 0 and 1.")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "expectation": self.expectation,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "success": self.success,
            "surprise": self.surprise,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Experience:
        return cls(
            id=str(data.get("id") or _identifier("EXP")),
            expectation=str(data.get("expectation", "")),
            outcome=str(data.get("outcome", "")),
            lesson=str(data.get("lesson", "")),
            success=bool(data.get("success", False)),
            surprise=_as_float(data.get("surprise"), 0.0),
            tags=_as_string_tuple(data.get("tags")),
            created_at=_parse_time(data.get("created_at")),
        )

    def describe(self) -> str:
        verdict = "worked" if self.success else "did not work"
        return f"{self.expectation} -> {self.outcome} ({verdict})"


@dataclass(frozen=True)
class Reliability:
    """A scored judgement about how much a statement can be trusted.

    ``score`` is the reliability, ``source`` is where it came from, and
    ``verified`` records whether an explicit check raised it to a fact. A record
    with ``verified=False`` is never stored as an established fact.
    """

    score: float
    source: Source | None = None
    verified: bool = False
    notes: tuple[str, ...] = ()

    def to_primitive(self) -> dict[str, object]:
        return {
            "score": self.score,
            "source": None if self.source is None else self.source.to_primitive(),
            "verified": self.verified,
            "notes": list(self.notes),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> Reliability:
        raw_source = data.get("source")
        source = Source.from_primitive(raw_source) if isinstance(raw_source, dict) else None
        return cls(
            score=_as_float(data.get("score"), 0.0),
            source=source,
            verified=bool(data.get("verified", False)),
            notes=_as_string_tuple(data.get("notes")),
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Reliability score must be between 0 and 1.")

    def describe(self) -> str:
        state = "verified" if self.verified else "unverified"
        return f"{self.score:.2f} ({state})"


@dataclass(frozen=True)
class ThoughtCycle:
    """One complete proactive-mind cycle, recorded so it can be inspected.

    ``questions``, ``ideas``, and ``hypotheses`` are ids into the idea store;
    ``gaps`` are the topics the cycle judged most worth pursuing; ``duration_ms``
    is the measured wall time of the cycle.
    """

    questions: tuple[str, ...] = ()
    ideas: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    duration_ms: float = 0.0
    id: str = field(default_factory=lambda: _identifier("CYCLE"))
    created_at: datetime = field(default_factory=_now)

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "questions": list(self.questions),
            "ideas": list(self.ideas),
            "hypotheses": list(self.hypotheses),
            "gaps": list(self.gaps),
            "notes": list(self.notes),
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> ThoughtCycle:
        return cls(
            id=str(data.get("id") or _identifier("CYCLE")),
            questions=_as_string_tuple(data.get("questions")),
            ideas=_as_string_tuple(data.get("ideas")),
            hypotheses=_as_string_tuple(data.get("hypotheses")),
            gaps=_as_string_tuple(data.get("gaps")),
            notes=_as_string_tuple(data.get("notes")),
            duration_ms=_as_float(data.get("duration_ms"), 0.0),
            created_at=_parse_time(data.get("created_at")),
        )

    def describe(self) -> str:
        return (
            f"cycle {self.id}: {len(self.questions)} question(s), "
            f"{len(self.ideas)} idea(s), {len(self.hypotheses)} hypothesis(es) "
            f"in {self.duration_ms:.0f} ms"
        )


class TrainingExampleKind(str, Enum):
    """What a training example teaches: an answer, a reasoning step, or a style."""

    ANSWER = "answer"
    REASONING = "reasoning"
    STYLE = "style"


@dataclass(frozen=True)
class TrainingExample:
    """One qualified example, ready to become training data -- with provenance.

    The task requires that a teacher's answer is never accepted automatically and
    that every example keeps its origin, the task, the proposed answer, the
    verification result, the confidence, the reason it was accepted or rejected,
    and the date it was approved. Those are exactly this record's fields.

    ``qualified`` is the gate the dataset builder reads: an example is usable
    only when it was **verified** (or accepted by the reliability rule) and its
    confidence clears the threshold. An unqualified example is still stored, so a
    rejection is auditable rather than silent.
    """

    task: str
    answer: str
    kind: TrainingExampleKind = TrainingExampleKind.ANSWER
    source: Source | None = None
    verified: bool = False
    confidence: float = 0.5
    qualified: bool = False
    reason: str = ""
    id: str = field(default_factory=lambda: _identifier("EX"))
    created_at: datetime = field(default_factory=_now)
    approved_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("Training example task cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Training example confidence must be between 0 and 1.")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "task": self.task,
            "answer": self.answer,
            "kind": self.kind.value,
            "source": None
            if self.source is None
            else {
                "kind": self.source.kind.value,
                "reference": self.source.reference,
                "reliability": self.source.reliability,
            },
            "verified": self.verified,
            "confidence": self.confidence,
            "qualified": self.qualified,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "approved_at": None if self.approved_at is None else self.approved_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> TrainingExample:
        source_data = data.get("source")
        source = None
        if isinstance(source_data, dict):
            try:
                source = Source(
                    kind=SourceKind(str(source_data.get("kind", "unknown"))),
                    reference=str(source_data.get("reference", "")),
                    reliability=float(source_data.get("reliability", 0.5)),
                )
            except (ValueError, TypeError):
                source = None
        approved = data.get("approved_at")
        try:
            approved_at = datetime.fromisoformat(str(approved)) if approved else None
        except ValueError:
            approved_at = None
        return cls(
            id=str(data.get("id") or _identifier("EX")),
            task=str(data.get("task", "")),
            answer=str(data.get("answer", "")),
            kind=_example_kind_from(str(data.get("kind", "answer"))),
            source=source,
            verified=bool(data.get("verified", False)),
            confidence=_as_float(data.get("confidence"), 0.5),
            qualified=bool(data.get("qualified", False)),
            reason=str(data.get("reason", "")),
            created_at=_parse_time(data.get("created_at")),
            approved_at=approved_at,
        )

    def describe(self) -> str:
        verdict = "qualified" if self.qualified else "not qualified"
        return f"[{verdict}] {self.task} -> {self.answer[:60]} ({self.reason})"


def _example_kind_from(value: str) -> TrainingExampleKind:
    try:
        return TrainingExampleKind(value)
    except ValueError:
        return TrainingExampleKind.ANSWER


__all__ = [
    "Experience",
    "Idea",
    "IdeaKind",
    "PromotionError",
    "Reliability",
    "Source",
    "SourceKind",
    "ThoughtCycle",
    "TrainingExample",
    "TrainingExampleKind",
    "promote",
]
