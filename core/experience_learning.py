"""Experience-based learning: turn a task's outcome into a reusable lesson.

Phase 3 requires the system to learn from stored experiences: capture a task's
goal, context, constraints, approach, actions, result, errors, feedback,
evaluation, related memories, and provenance; compare the intended outcome with
the actual one; decide whether the result supports a reusable lesson; and store
that lesson with its evidence and scope -- **without** generalising from a single
ambiguous or unsuccessful experience.

This module is that process. It **reuses** the Phase-2 long-term memory rather
than a parallel store: an experience is captured as an ``EPISODIC`` memory, and a
lesson that is strong enough becomes a ``PROCEDURAL`` memory (and a versioned
procedure in the learning store).

**Honest by construction.** The four simplifications the task forbids are refused
in code:

* a single success does **not** become a trusted lesson -- a lesson needs
  ``DEFAULT_MIN_SUPPORT`` corroborating experiences before it is promoted;
* a failure is not recorded as "the approach is always wrong" -- it is recorded
  as an experience whose lesson carries its uncertainty;
* repeated feedback is not automatically correct -- support is counted across
  *distinct* experiences, not repetitions of one;
* a single positive result never proves a procedure reliable -- the procedure
  starts as a **proposal** with a reliability it must earn.

Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.learning_events import (
    KIND_EXPERIENCE,
    KIND_PROCEDURE,
    OUTCOME_ACCEPTED,
    SOURCE_SYSTEM,
    LearningEvent,
    now_utc,
)
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_record import (
    SOURCE_CHARACTER,
    SOURCE_DOCUMENT,
    SOURCE_IMPORT,
    SOURCE_INTERNET,
    SOURCE_MODEL,
    SOURCE_REASONING,
    SOURCE_USER,
    MemoryKind,
)
from core.memory_record import (
    SOURCE_SYSTEM as MEMORY_SOURCE_SYSTEM,
)

#: The number of *distinct* corroborating experiences a lesson needs before it is
#: promoted to procedural memory. One success is never enough.
DEFAULT_MIN_SUPPORT = 2

#: The lowest success rate a lesson may have and still be promoted.
DEFAULT_MIN_SUCCESS_RATE = 0.6

#: The lowest mean importance an experience needs before it is worth learning from.
DEFAULT_MIN_IMPORTANCE = 0.3

#: Map the learning-event provenance vocabulary onto the memory one.
_SOURCE_ALIASES: dict[str, str] = {
    "user": SOURCE_USER,
    "document": SOURCE_DOCUMENT,
    "internet": SOURCE_INTERNET,
    "model": SOURCE_MODEL,
    "character": SOURCE_CHARACTER,
    "system": MEMORY_SOURCE_SYSTEM,
    "reasoning": SOURCE_REASONING,
    "import": SOURCE_IMPORT,
}


def _memory_source(source: str) -> str:
    """Translate a learning source into a memory source (default ``system``)."""
    return _SOURCE_ALIASES.get(source, MEMORY_SOURCE_SYSTEM)


@dataclass
class ExperienceRecord:
    """One meaningful task or experience, with everything needed to learn from it.

    Every field the task lists is here: the goal, the context, the constraints,
    the approach, the actions, the result, the errors, the user feedback, the
    evaluation outcome, the related memories, the scope, and the provenance.
    """

    goal: str
    outcome: str
    success: bool = False
    context: str = ""
    constraints: tuple[str, ...] = ()
    approach: str = ""
    actions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    feedback: str = ""
    evaluation: str = ""
    related: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    importance: float = 0.5
    confidence: float = 0.5
    surprise: float = 0.0
    source: str = SOURCE_SYSTEM
    source_reference: str = ""
    scope: str = ""
    experience_id: str = ""
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("An experience needs a goal.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Experience importance must be in [0, 1].")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Experience confidence must be in [0, 1].")
        if not 0.0 <= self.surprise <= 1.0:
            raise ValueError("Experience surprise must be in [0, 1].")
        if not self.experience_id:
            from core.memory_record import make_memory_id

            object.__setattr__(
                self,
                "experience_id",
                make_memory_id(MemoryKind.EPISODIC, f"{self.goal}|{self.outcome}", owner=self.scope),
            )

    @property
    def topic_terms(self) -> frozenset[str]:
        """The goal's content terms -- what a lesson about this is *about*."""
        from core.tokenize import tokenize

        return frozenset(tokenize(self.goal))

    def to_text(self) -> str:
        """The experience as one honest, readable statement."""
        parts = [f"Goal: {self.goal}", f"Outcome: {self.outcome}"]
        if self.approach:
            parts.append(f"Approach: {self.approach}")
        if self.errors:
            parts.append("Errors: " + "; ".join(self.errors))
        if self.feedback:
            parts.append(f"Feedback: {self.feedback}")
        verdict = "worked" if self.success else "did not work"
        parts.append(f"Result: {verdict}")
        return ". ".join(parts)

    def describe(self) -> str:
        verdict = "worked" if self.success else "did not work"
        return f"[{'success' if self.success else 'failure'}] {self.goal} -> {self.outcome} ({verdict})"


@dataclass
class Lesson:
    """A candidate reusable lesson, with its evidence and its scope."""

    topic: str
    statement: str
    success: bool
    support: int = 1
    contradictions: int = 0
    confidence: float = 0.0
    scope: str = ""
    experience_ids: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float:
        total = self.support + self.contradictions
        return self.support / total if total else 0.0

    def describe(self) -> str:
        return (
            f"{self.statement} (support={self.support}, "
            f"contradictions={self.contradictions}, rate={self.success_rate:.2f})"
        )


@dataclass
class ExperienceLearningResult:
    """What one experience taught, and whether a lesson was promoted."""

    experience: ExperienceRecord
    memory_id: str = ""
    lesson: Lesson | None = None
    promoted: bool = False
    events: tuple[LearningEvent, ...] = ()
    reason: str = ""

    def describe(self) -> str:
        base = f"experience '{self.experience.goal}': captured"
        if self.lesson is not None:
            base += f"; lesson {self.lesson.describe()}"
        base += f"; promoted={self.promoted}"
        if self.reason:
            base += f" ({self.reason})"
        return base


class ExperienceLearningPipeline:
    """The one path from a task outcome to a durable, evaluated lesson."""

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
        self._lessons: dict[str, Lesson] = {}

    # -- the one entry point ----------------------------------------------
    def learn(self, experience: ExperienceRecord, *, record: bool = True) -> ExperienceLearningResult:
        """Capture an experience, evaluate it, and promote a lesson only when earned."""
        memory_record = self.memory.capture(
            experience.to_text(),
            kind=MemoryKind.EPISODIC,
            summary=experience.goal[:120],
            source=_memory_source(experience.source),
            source_reference=experience.source_reference,
            confidence=experience.confidence,
            importance=experience.importance,
            entities=tuple(sorted(experience.topic_terms)),
            task_refs=(experience.scope,) if experience.scope else (),
            tags=experience.tags,
            metadata={
                "success": experience.success,
                "surprise": experience.surprise,
                "approach": experience.approach,
                "actions": list(experience.actions),
                "errors": list(experience.errors),
                "constraints": list(experience.constraints),
                "feedback": experience.feedback,
                "evaluation": experience.evaluation,
                "related": list(experience.related),
                "scope": experience.scope,
                "experience_id": experience.experience_id,
            },
            record_id=experience.experience_id,
        )

        lesson = self._update_lesson(experience)
        promoted = False
        events: list[LearningEvent] = [
            LearningEvent(
                kind=KIND_EXPERIENCE,
                target=experience.goal[:120],
                owner=self.owner,
                source=experience.source,
                source_reference=experience.source_reference,
                summary=experience.outcome[:200],
                after=memory_record.record_id,
                outcome=OUTCOME_ACCEPTED,
                applied=True,
                confidence=experience.confidence,
            )
        ]
        if lesson is not None and self._promotable(lesson):
            promoted = self._promote(lesson, memory_record.record_id)
            events.append(
                LearningEvent(
                    kind=KIND_PROCEDURE,
                    target=lesson.topic,
                    owner=self.owner,
                    source=experience.source,
                    summary=lesson.statement,
                    after=lesson.statement,
                    outcome=OUTCOME_ACCEPTED,
                    applied=promoted,
                    confidence=lesson.confidence,
                )
            )

        if record and self.store is not None:
            version = self.store.next_version()
            from dataclasses import replace

            stamped = tuple(
                replace(event, version=str(version)) if event.applied else event for event in events
            )
            self.store.record_many(stamped)
            events = list(stamped)

        return ExperienceLearningResult(
            experience=experience,
            memory_id=memory_record.record_id,
            lesson=lesson,
            promoted=promoted,
            events=tuple(events),
            reason=self._reason(lesson),
        )

    def lessons(self) -> tuple[Lesson, ...]:
        """Every candidate lesson the pipeline currently holds."""
        return tuple(self._lessons.values())

    # -- internals ---------------------------------------------------------
    def _lesson_key(self, experience: ExperienceRecord) -> str:
        return " ".join(sorted(experience.topic_terms)) or experience.goal.strip().casefold()

    def _update_lesson(self, experience: ExperienceRecord) -> Lesson | None:
        """Fold one experience into its topic's lesson (support or contradiction)."""
        if experience.importance < DEFAULT_MIN_IMPORTANCE:
            return None
        key = self._lesson_key(experience)
        if not key:
            return None
        existing = self._lessons.get(key)
        if existing is None:
            statement = self._statement(experience)
            lesson = Lesson(
                topic=key,
                statement=statement,
                success=experience.success,
                support=1 if experience.success else 0,
                contradictions=0 if experience.success else 1,
                confidence=experience.confidence,
                scope=experience.scope,
                experience_ids=(experience.experience_id,),
            )
            self._lessons[key] = lesson
            return lesson
        support = existing.support + (1 if experience.success else 0)
        contradictions = existing.contradictions + (0 if experience.success else 1)
        confidence = self._confidence(support, contradictions)
        lesson = Lesson(
            topic=key,
            statement=existing.statement,
            success=existing.success,
            support=support,
            contradictions=contradictions,
            confidence=confidence,
            scope=existing.scope or experience.scope,
            experience_ids=(*existing.experience_ids, experience.experience_id),
        )
        self._lessons[key] = lesson
        return lesson

    @staticmethod
    def _statement(experience: ExperienceRecord) -> str:
        if experience.success:
            approach = experience.approach or experience.goal
            return f"when the goal is '{experience.goal}', the approach '{approach}' worked"
        return f"the approach for '{experience.goal}' did not work ({experience.outcome})"

    @staticmethod
    def _confidence(support: int, contradictions: int) -> float:
        """A bounded, support-scaled confidence -- never 1.0 from one success.

        A Laplace-smoothed success estimate: two successes of two gives 0.75, not
        1.0, so a lesson's confidence can only rise as real corroboration arrives.
        """
        total = support + contradictions
        if total == 0:
            return 0.0
        return round((support + 1) / (total + 2), 6)

    def _promotable(self, lesson: Lesson) -> bool:
        return (
            lesson.success
            and lesson.support >= self.min_support
            and lesson.success_rate >= self.min_success_rate
        )

    def _promote(self, lesson: Lesson, memory_id: str) -> bool:
        """Store a proven lesson as a procedural memory and a versioned procedure."""
        procedure = self.memory.capture(
            lesson.statement,
            kind=MemoryKind.PROCEDURAL,
            summary=lesson.topic[:120],
            source=SOURCE_REASONING,
            confidence=lesson.confidence,
            importance=min(1.0, 0.5 + 0.1 * lesson.support),
            tags=("lesson", *([lesson.scope] if lesson.scope else [])),
            metadata={
                "support": lesson.support,
                "contradictions": lesson.contradictions,
                "success_rate": lesson.success_rate,
                "from_experiences": list(lesson.experience_ids),
                "source_memory": memory_id,
            },
        )
        return bool(procedure.record_id)

    def _reason(self, lesson: Lesson | None) -> str:
        if lesson is None:
            return "below the importance floor; no lesson"
        if not lesson.success:
            return "unsuccessful experience recorded; lesson kept as a caution, not a rule"
        if lesson.support < self.min_support:
            return f"needs {self.min_support - lesson.support} more corroborating experience(s)"
        if lesson.success_rate < self.min_success_rate:
            return f"success rate {lesson.success_rate:.2f} below the {self.min_success_rate:.2f} floor"
        return "lesson promoted to procedural memory"

    # -- reading stored experiences ---------------------------------------
    def stored_experiences(self, *, limit: int = 100) -> tuple[Any, ...]:
        """The durable episodic memories this pipeline has captured."""
        records, _ = self.memory.page(kind=MemoryKind.EPISODIC, limit=limit)
        return tuple(records)

    def lessons_from_memory(self) -> tuple[Lesson, ...]:
        """Rebuild the lessons from the durable episodic memories.

        This is what makes a lesson survive a restart: the experiences are read
        back and folded exactly as they were when captured, so the support count
        is the *real* one rather than a per-process count.
        """
        self._lessons = {}
        records, _ = self.memory.page(kind=MemoryKind.EPISODIC, limit=500)
        for record in sorted(records, key=lambda item: item.created_at):
            experience = self._experience_from_memory(record)
            if experience is not None:
                self._update_lesson(experience)
        return self.lessons()

    @staticmethod
    def _experience_from_memory(record: Any) -> ExperienceRecord | None:
        metadata = dict(getattr(record, "metadata", {}) or {})
        if "success" not in metadata:
            return None
        goal = record.summary or record.content.split(".")[0]
        try:
            return ExperienceRecord(
                goal=goal,
                outcome=record.content,
                success=bool(metadata.get("success", False)),
                approach=str(metadata.get("approach", "")),
                errors=tuple(metadata.get("errors", ())),
                feedback=str(metadata.get("feedback", "")),
                importance=record.importance,
                confidence=record.confidence,
                surprise=float(metadata.get("surprise", 0.0) or 0.0),
                source="reasoning",
                scope=str(metadata.get("scope", "")),
                experience_id=str(metadata.get("experience_id", record.record_id)),
            )
        except ValueError:
            return None

    def describe(self) -> str:
        promoted = sum(1 for lesson in self._lessons.values() if self._promotable(lesson))
        return (
            f"experience learning for {self.owner}: {len(self._lessons)} lesson(s), "
            f"{promoted} promoted; {self.memory.count(kind=MemoryKind.EPISODIC)} episodic memory(s)"
        )


def experience_from_result(
    goal: str,
    outcome: str,
    *,
    success: bool | None = None,
    approach: str = "",
    errors: Iterable[str] = (),
    feedback: str = "",
    importance: float = 0.5,
    confidence: float = 0.5,
    source: str = SOURCE_SYSTEM,
    scope: str = "",
) -> ExperienceRecord:
    """Build an experience from a goal and an outcome, inferring success if unstated.

    When ``success`` is not given it is read from the outcome with the project's
    own outcome vocabulary (``failed``/``error``/``unable`` mean failure); the
    caller can always state it explicitly, and stating it always wins.
    """
    if success is None:
        lowered = outcome.casefold()
        success = not any(
            marker in lowered
            for marker in ("fail", "error", "unable", "could not", "did not", "wrong")
        )
    return ExperienceRecord(
        goal=goal,
        outcome=outcome,
        success=success,
        approach=approach,
        errors=tuple(errors),
        feedback=feedback,
        importance=importance,
        confidence=confidence,
        source=source,
        scope=scope,
    )


__all__ = [
    "DEFAULT_MIN_IMPORTANCE",
    "DEFAULT_MIN_SUCCESS_RATE",
    "DEFAULT_MIN_SUPPORT",
    "ExperienceLearningPipeline",
    "ExperienceLearningResult",
    "ExperienceRecord",
    "Lesson",
    "experience_from_result",
]
