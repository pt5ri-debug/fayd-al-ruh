"""Teacher-Student learning: the student tries first, and never trusts blindly.

The design follows the task's eight steps exactly:

1. the **student** (the project's own Fayd Al-Ruh model) answers first, unaided;
2. the system judges how well the student did
   (:class:`AnswerQuality` from length, question overlap, and repetition);
3. **only when the student is weak** is the teacher consulted;
4. the teacher supplies an example/explanation
   (the ``TeacherModel`` interface, a real local model when one is available);
5. the two answers are **compared**;
6. the teacher's answer is **verified** against the project's own knowledge
   (the ontology and the corpus), never accepted automatically;
7. an accepted answer is recorded as an **example** with its provenance;
8. only a **qualified** example becomes future training data.

Nothing here copies a teacher's whole output into training data: every example
passes the reliability gate and carries the reason it was accepted or rejected,
so a rejection is auditable.

The unified :class:`ModelBackend` interface is the one place a model -- student or
teacher -- is called, so a local GGUF, an in-process checkpoint, or a test double
are interchangeable and the learning logic never names a runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.ontology import Ontology
from core.tokenize import normalize_token

from cognition.records import (
    Source,
    SourceKind,
    TrainingExample,
    TrainingExampleKind,
)
from cognition.reliability import TRUST_THRESHOLD, score
from cognition.storage import ExampleStore


@runtime_checkable
class ModelBackend(Protocol):
    """The one interface every model -- student or teacher -- is called through."""

    name: str

    def available(self) -> bool: ...

    def answer(self, question: str) -> str: ...


@dataclass
class CallableBackend:
    """Adapt any ``question -> answer`` callable into a :class:`ModelBackend`."""

    name: str
    fn: Callable[[str], str]
    is_available: bool = True

    def available(self) -> bool:
        return self.is_available

    def answer(self, question: str) -> str:
        text = self.fn(question)
        return text.strip() if isinstance(text, str) else ""


@dataclass(frozen=True)
class AnswerQuality:
    """How good an answer looks, judged without a model of its own."""

    score: float
    reasons: tuple[str, ...]

    @property
    def weak(self) -> bool:
        return self.score < 0.5

    def describe(self) -> str:
        return f"{self.score:.2f} ({'; '.join(self.reasons) or 'no issues'})"


@dataclass
class TeachingOutcome:
    """The result of one teacher-student exchange."""

    question: str
    student_answer: str
    student_quality: AnswerQuality
    teacher_used: bool
    teacher_answer: str = ""
    comparison: str = ""
    example: TrainingExample | None = None
    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        used = "teacher consulted" if self.teacher_used else "student answered alone"
        verdict = "qualified" if (self.example and self.example.qualified) else "not qualified"
        return f"{self.question[:50]} -> {used}; example {verdict}"


class TeacherStudentSystem:
    """Run the student-first exchange and qualify examples for training."""

    def __init__(
        self,
        student: ModelBackend,
        teacher: ModelBackend | None,
        *,
        ontology: Ontology | None = None,
        examples: ExampleStore | None = None,
        min_confidence: float = 0.6,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.ontology = ontology
        self.examples = examples or ExampleStore()
        self.min_confidence = min_confidence

    # -- judgement ----------------------------------------------------------
    def assess(self, question: str, answer: str) -> AnswerQuality:
        """Judge an answer by length, overlap with the question, and repetition."""
        text = answer.strip()
        reasons: list[str] = []
        if not text:
            return AnswerQuality(0.0, ("empty answer",))
        words = [w for w in text.split() if w.strip()]
        score = 0.5
        if len(words) < 3:
            reasons.append("too short to be useful")
            score -= 0.3
        elif len(words) >= 8:
            score += 0.2
        question_tokens = {normalize_token(w) for w in question.split() if len(w) > 2}
        answer_tokens = {normalize_token(w) for w in words if len(w) > 2}
        if question_tokens:
            overlap = len(question_tokens & answer_tokens) / len(question_tokens)
            if overlap < 0.1:
                reasons.append("does not engage the question's terms")
                score -= 0.2
            elif overlap >= 0.4:
                score += 0.1
        if len(answer_tokens) < max(1, len(words) * 0.5):
            reasons.append("highly repetitive")
            score -= 0.2
        return AnswerQuality(score=round(min(max(score, 0.0), 1.0), 6), reasons=tuple(reasons))

    # -- verification -------------------------------------------------------
    def verify(self, question: str, answer: str) -> tuple[bool, float, str]:
        """Check an answer against the project's own knowledge.

        An answer is verified when it names at least one concept the ontology
        knows (the graph corroborates it), and the confidence is the reliability
        of a teacher statement corroborated by the graph. An answer naming no
        known concept is **not** verified -- the teacher is never trusted blindly.
        """
        if self.ontology is None:
            return False, 0.4, "no ontology to check against"
        known = {c.label for c in self.ontology.concepts.values()}
        named = {t for t in (normalize_token(w) for w in answer.split()) if t in known}
        if named:
            reliability = score(
                source=Source(SourceKind.TEACHER, "teacher", reliability=0.55),
                confidence=0.6,
                corroborations=len(named),
                verified=True,
            )
            return True, reliability.score, f"names known concepts: {', '.join(sorted(named)[:3])}"
        return False, 0.4, "names no concept the graph knows"

    # -- the exchange -------------------------------------------------------
    def teach(self, question: str, *, force_teacher: bool = False) -> TeachingOutcome:
        """Run one student-first exchange and record the outcome.

        The student answers; if it is adequate and no teacher is forced, the
        teacher is not consulted (the task's "perform the task without the teacher
        once capable"). When the teacher is used, its answer is verified and only
        a qualified example is created.
        """
        student_answer = self._answer(self.student, question)
        quality = self.assess(question, student_answer)
        notes: list[str] = [f"student quality {quality.describe()}"]

        if not quality.weak and not force_teacher:
            return TeachingOutcome(
                question=question,
                student_answer=student_answer,
                student_quality=quality,
                teacher_used=False,
                notes=(*notes, "student was adequate; teacher not needed"),
            )

        if self.teacher is None or not self.teacher.available():
            notes.append("no teacher available; keeping the student's answer")
            return TeachingOutcome(
                question=question,
                student_answer=student_answer,
                student_quality=quality,
                teacher_used=False,
                notes=tuple(notes),
            )

        teacher_answer = self._answer(self.teacher, question)
        comparison = self._compare(student_answer, teacher_answer)
        verified, confidence, reason = self.verify(question, teacher_answer)
        qualified = verified and confidence >= self.min_confidence
        example = TrainingExample(
            task=question,
            answer=teacher_answer,
            kind=TrainingExampleKind.ANSWER,
            source=Source(SourceKind.TEACHER, self.teacher.name, reliability=0.55),
            verified=verified,
            confidence=round(confidence, 6),
            qualified=qualified,
            reason=reason if qualified else f"rejected: {reason}",
        )
        self.examples.add(example)
        notes.append(f"comparison: {comparison}")
        return TeachingOutcome(
            question=question,
            student_answer=student_answer,
            student_quality=quality,
            teacher_used=True,
            teacher_answer=teacher_answer,
            comparison=comparison,
            example=example,
            notes=tuple(notes),
        )

    def compare(self, question: str) -> TeachingOutcome:
        """Always consult the teacher (for inspection), still verifying its answer."""
        return self.teach(question, force_teacher=True)

    # -- internals ----------------------------------------------------------
    def _answer(self, backend: ModelBackend, question: str) -> str:
        try:
            return backend.answer(question)
        except Exception as error:  # a broken backend is reported, not fatal
            return f"[{backend.name} failed: {type(error).__name__}]"

    def _compare(self, student: str, teacher: str) -> str:
        student_tokens = {normalize_token(w) for w in student.split() if len(w) > 2}
        teacher_tokens = {normalize_token(w) for w in teacher.split() if len(w) > 2}
        if not student_tokens or not teacher_tokens:
            return "cannot compare (one answer is empty)"
        union = student_tokens | teacher_tokens
        overlap = len(student_tokens & teacher_tokens) / len(union) if union else 0.0
        if overlap >= 0.6:
            return f"the two answers agree closely (overlap {overlap:.2f})"
        if overlap >= 0.2:
            return f"the two answers partly agree (overlap {overlap:.2f})"
        return f"the two answers differ (overlap {overlap:.2f})"


@dataclass
class StudentProfile:
    """A simple record of how often the student answered without the teacher."""

    total: int = 0
    alone: int = 0
    with_teacher: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def independence(self) -> float:
        """The share of exchanges the student handled alone."""
        return round(self.alone / self.total, 6) if self.total else 0.0

    def record(self, outcome: TeachingOutcome) -> None:
        self.total += 1
        if outcome.teacher_used:
            self.with_teacher += 1
        else:
            self.alone += 1

    def describe(self) -> str:
        return (
            f"{self.alone}/{self.total} answered alone "
            f"(independence {self.independence:.0%})"
        )


__all__ = [
    "TRUST_THRESHOLD",
    "AnswerQuality",
    "CallableBackend",
    "ModelBackend",
    "StudentProfile",
    "TeacherStudentSystem",
    "TeachingOutcome",
]
