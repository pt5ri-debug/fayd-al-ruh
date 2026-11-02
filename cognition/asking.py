"""The student asks the teacher: a continuous, non-repeating question loop.

The project already has a **teacher-student** exchange
(:mod:`cognition.teacher_student`): the student answers first, and only a weak
answer reaches a teacher. This module is the *other* direction, and it is the one
the task asks for: the project's own model -- **Fayd Al-Ruh the learner** --
**invents** a question, the **linked model** -- **Fayd Al-Ruh the teacher** --
answers it, the learner folds the answer into its knowledge, and then asks the
next, *different* question, for as long as the caller lets it run. The caller
stops the loop by asking it to stop (the window's toggle, a Ctrl-C, a bound).

Three honesty rules shape the whole design:

* **The answer is always the teacher's.** The loop never lets the learner answer
  its own question: the text a turn carries is exactly what the teacher returned,
  and a turn records the teacher that produced it. A test injects a learner that
  would answer differently, and the gate asserts the turn's answer is the
  teacher's, so a regression that let the learner answer fails here.
* **No question repeats.** Every question's normalised key is remembered (in the
  run and, when a :class:`~cognition.question_bank.QuestionLog` is supplied,
  across runs), and an invented question that is *too similar* to a recent one is
  refused, so the loop works through genuinely distinct material instead of
  looping on a handful of phrasings.
* **An answer is never blindly trusted.** Each teacher answer is verified against
  the project's own knowledge (the ontology) and becomes a *qualified* training
  example only when it names something the graph knows and clears the confidence
  threshold -- exactly the rule :mod:`cognition.teacher_student` already applies.
  A rejected answer is still recorded, with the reason, so a rejection is
  auditable.

Where the questions come from is stated per turn (:attr:`AskingTurn.source`): the
learner's own model may propose one (``student``), the mind's own question bank
built from the graph and the authored topics may supply one (``bank``), or the
learner invents one combinatorially from its own concepts (``invented``). Nothing
is invented as a *fact*: a question only names concepts the project already
holds, and a question is never treated as knowledge.

Dependency-free and Tk-free, so the whole loop is exercised with no model, no
display, and no network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import blake2b

from cognition.question_bank import (
    BankQuestion,
    QuestionLog,
    question_key,
)
from cognition.records import Source, SourceKind
from cognition.storage import ExampleStore
from cognition.teacher_student import ModelBackend, TeacherStudentSystem

#: How many of the most recent questions the similarity guard compares against.
SIMILARITY_WINDOW = 60

#: The Jaccard similarity at or above which a proposed question is a near-repeat.
SIMILARITY_LIMIT = 0.8

#: The confidence a verified answer must clear to become a qualified example.
DEFAULT_MIN_CONFIDENCE = 0.6

#: The name used for the teacher when the backend does not carry one.
DEFAULT_TEACHER_NAME = "teacher"

#: The prompt a learner's own model is given when it is asked to invent a
#: question. It is deliberately a *question-asking* frame (not the answer frame
#: the model was trained on), so a proposal reads as a question rather than a
#: continuation.
QUESTION_PROMPT = (
    "You are Fayd Al-Ruh, a curious petroleum-engineering student. "
    "Ask one short, specific question about {topic} that you do not yet know "
    "the answer to. Write the question only.\nQuestion:"
)

#: Question templates over a *pair* of the learner's own concepts, so the learner
#: can invent a fresh question after its fixed bank is spent -- without inventing
#: a fact. ``{a}`` and ``{b}`` are two distinct concept labels.
PAIR_TEMPLATES_EN: tuple[str, ...] = (
    "How does {a} relate to {b}?",
    "What does {a} do to {b}?",
    "Why is {a} important for {b}?",
    "What happens to {b} when {a} changes?",
    "How is {a} measured or judged in terms of {b}?",
)

#: The Arabic twins of :data:`PAIR_TEMPLATES_EN`, so the loop asks some questions
#: in the language the project is Arabic-first for.
PAIR_TEMPLATES_AR: tuple[str, ...] = (
    "ما العلاقة بين {a} و{b}؟",
    "ماذا يفعل {a} بـ{b}؟",
    "لماذا يُعدّ {a} مهمًّا لـ{b}؟",
    "ماذا يحدث لـ{b} عندما يتغيّر {a}؟",
    "كيف يمكن قياس {a} من حيث {b}؟",
)

#: A question's token set is the similarity guard's unit.
_QUESTION_WORDS = ("what", "why", "how", "when", "where", "which", "who", "is", "does", "can")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(word for word in text.casefold().split() if len(word) > 2)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _fact_id(statement: str) -> str:
    """A stable, content-addressed id for a learned statement.

    ``hash()`` is salted per process, so an id built from it would change on every
    run and the same answer would be stored twice; a digest of the normalised text
    is the same across runs.
    """
    from core.tokenize import normalize_phrase

    digest = blake2b(normalize_phrase(statement).encode("utf-8"), digest_size=6)
    return f"ask-{digest.hexdigest()}"


# --------------------------------------------------------------------- turns


@dataclass(frozen=True)
class AskingTurn:
    """One question the learner asked and the answer the teacher gave."""

    index: int
    question: str
    answer: str
    source: str = "bank"
    """Where the question came from: ``student``, ``bank``, or ``invented``."""

    teacher_name: str = DEFAULT_TEACHER_NAME
    verified: bool = False
    confidence: float = 0.0
    qualified: bool = False
    reason: str = ""
    relations_learned: tuple[str, ...] = ()
    learned: bool = False
    error: str = ""
    at: str = field(default_factory=_now)

    @property
    def answered(self) -> bool:
        """True when the teacher produced a non-empty answer."""
        return bool(self.answer.strip()) and not self.error

    def describe(self) -> str:
        if self.error:
            return f"Q{self.index}: {self.question}\n  (no answer: {self.error})"
        verdict = "qualified" if self.qualified else "kept (unverified)"
        return (
            f"Q{self.index}: {self.question}\n"
            f"  A ({self.teacher_name}): {self.answer}\n"
            f"  [{verdict}; {self.reason}]"
        )

    def to_primitive(self) -> dict[str, object]:
        return {
            "index": self.index,
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "teacher_name": self.teacher_name,
            "verified": self.verified,
            "confidence": self.confidence,
            "qualified": self.qualified,
            "reason": self.reason,
            "relations_learned": list(self.relations_learned),
            "learned": self.learned,
            "error": self.error,
            "at": self.at,
        }


@dataclass(frozen=True)
class AskingReport:
    """What one asking run actually did, measured."""

    asked: int
    answered: int
    qualified: int
    student_questions: int
    bank_questions: int
    invented_questions: int
    relations_learned: int
    facts_gained: int
    stopped: bool
    exhausted: bool
    duration_ms: float
    teacher_name: str = DEFAULT_TEACHER_NAME

    @property
    def answer_rate(self) -> float:
        return round(self.answered / self.asked, 6) if self.asked else 0.0

    @property
    def qualified_rate(self) -> float:
        return round(self.qualified / self.answered, 6) if self.answered else 0.0

    def describe(self) -> str:
        lines = [
            "The learner asked; the teacher answered.",
            f"  teacher: {self.teacher_name}",
            f"  questions asked: {self.asked} (answered: {self.answered}, "
            f"{self.answer_rate:.0%})",
            f"  qualified answers learned: {self.qualified} ({self.qualified_rate:.0%} of answers)",
            f"  questions from the learner's model: {self.student_questions}, "
            f"its bank: {self.bank_questions}, invented: {self.invented_questions}",
            f"  relations learned: {self.relations_learned}",
            f"  facts gained: {self.facts_gained}",
            f"  ran for {self.duration_ms / 1000:.1f}s",
        ]
        if self.stopped:
            lines.append("  stopped: the caller asked the loop to stop")
        elif self.exhausted:
            lines.append(
                "  stopped: no distinct question remained (the learner's bank and "
                "its invented questions are spent)"
            )
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "asked": self.asked,
            "answered": self.answered,
            "qualified": self.qualified,
            "student_questions": self.student_questions,
            "bank_questions": self.bank_questions,
            "invented_questions": self.invented_questions,
            "relations_learned": self.relations_learned,
            "facts_gained": self.facts_gained,
            "stopped": self.stopped,
            "exhausted": self.exhausted,
            "duration_ms": self.duration_ms,
            "teacher_name": self.teacher_name,
        }


#: The callback a streaming surface passes: it receives each :class:`AskingTurn`
#: the moment it is produced, so a window can show the question and answer live.
TurnCallback = Callable[[AskingTurn], None]


class InventedQuestionSource:
    """A deterministic, effectively unbounded stream of *pair* questions.

    When the learner's fixed question bank is spent, the learner keeps inventing:
    it asks how one of its own concepts relates to another, over every distinct
    pair and template, in a fixed order. The questions name only concepts the
    project already holds, so nothing is invented as a fact -- and the space
    (concepts x concepts x templates) is far larger than any run will reach, which
    is what makes the loop able to continue "to infinity" without repeating.
    """

    def __init__(self, ontology) -> None:
        labels = sorted(
            {str(getattr(c, "label", "")).strip() for c in ontology.concepts.values()}
        )
        self._labels = [label for label in labels if label]
        self._templates = (*PAIR_TEMPLATES_EN, *PAIR_TEMPLATES_AR)
        self._cursor = 0

    def _total(self) -> int:
        n = len(self._labels)
        return (n * (n - 1) // 2) * len(self._templates)

    def next(self, asked: set[str]) -> BankQuestion | None:
        """The next invented question not in ``asked``, or ``None`` when spent."""
        labels = self._labels
        templates = self._templates
        per_pair = len(templates)
        while self._cursor < self._total():
            pair_index, template_index = divmod(self._cursor, per_pair)
            self._cursor += 1
            # Map a flat pair index onto the (i < j) pair it names.
            i, j = self._pair(pair_index)
            a, b = labels[i], labels[j]
            text = templates[template_index].format(a=a, b=b)
            if question_key(text) in asked:
                continue
            return BankQuestion(text=text, topic=a, kind="invented")
        return None

    def _pair(self, pair_index: int) -> tuple[int, int]:
        n = len(self._labels)
        for i in range(n - 1):
            row = n - 1 - i
            if pair_index < row:
                return i, i + 1 + pair_index
            pair_index -= row
        return n - 2, n - 1


# ----------------------------------------------------------------- the loop


class AskingLoop:
    """Run the learner-asks / teacher-answers loop until the caller stops it.

    ``student`` is the learner's own model (the project's own Fayd Al-Ruh model);
    ``teacher`` is the linked model that **always** answers. ``ontology`` is the
    project's own knowledge the answers are verified against. ``learning`` (a
    :class:`cognition.learning_system.CognitiveLearningSystem`) and
    ``learn_relations`` (a ``text -> OntologyLearning`` callable, the controller's
    own ``learn_extracted_relations``) are how an answer is folded into the
    learner's knowledge -- both optional, so the loop runs with neither.

    ``questions`` may be an explicit sequence (the mind's bank); when it is
    ``None`` the loop asks the learner's own model first, then the bank the caller
    supplies through ``questions``, then invents combinatorially. ``count`` bounds
    the run; ``None`` means "until stopped".
    """

    def __init__(
        self,
        student: ModelBackend,
        teacher: ModelBackend,
        *,
        ontology=None,
        examples: ExampleStore | None = None,
        learning=None,
        learn_relations: Callable[[str], object] | None = None,
        questions: Sequence[BankQuestion] | None = None,
        log: QuestionLog | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        invent_with_model: bool = True,
        invent_questions: bool = True,
        teacher_name: str = "",
        on_turn: TurnCallback | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.ontology = ontology
        self.examples = examples if examples is not None else ExampleStore()
        self.learning = learning
        self.learn_relations = learn_relations
        self._verifier = TeacherStudentSystem(
            student, teacher, ontology=ontology, examples=self.examples,
            min_confidence=min_confidence,
        )
        self._log = log
        self._questions: list[BankQuestion] = list(questions or ())
        self._bank_index = 0
        self._inventor = (
            InventedQuestionSource(ontology) if ontology is not None and invent_questions else None
        )
        self._invent_with_model = invent_with_model
        self.teacher_name = teacher_name or str(getattr(teacher, "name", "")) or DEFAULT_TEACHER_NAME
        self.on_turn = on_turn
        self._stop = False
        self._asked: set[str] = set()
        self._recent: list[frozenset[str]] = []
        self._concept_labels = self._ordered_labels()

    # -- control -----------------------------------------------------------

    def stop(self) -> None:
        """Ask the loop to stop; it takes effect at the next question boundary."""
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop

    def reset(self) -> None:
        """Clear the stop flag (a fresh run of the same loop object)."""
        self._stop = False

    # -- the run -----------------------------------------------------------

    def run(self, *, count: int | None = None, on_turn: TurnCallback | None = None) -> AskingReport:
        """Ask, answer, and learn, until stopped or ``count`` questions are asked.

        Each turn is emitted through ``on_turn`` (or the loop's own callback) as
        soon as it is produced, so a caller can stream the questions and answers.
        A turn whose teacher returned nothing is still recorded, honestly, so the
        loop never pretends an answer it did not get.
        """
        if count is not None and count < 1:
            raise ValueError("count must be at least 1, or None for unbounded.")
        emit = on_turn if on_turn is not None else self.on_turn
        started = time.monotonic()
        turns: list[AskingTurn] = []
        before_facts = self._fact_count()
        index = 0
        exhausted = False
        while not self._stop and (count is None or index < count):
            question = self._next_question()
            if question is None:
                exhausted = True
                break
            index += 1
            turn = self._ask_one(index, question)
            turns.append(turn)
            if emit is not None:
                emit(turn)
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        after_facts = self._fact_count()
        return self._report(turns, duration_ms, before_facts, after_facts, exhausted)

    # -- one turn ----------------------------------------------------------

    def _ask_one(self, index: int, question: BankQuestion) -> AskingTurn:
        self._mark_asked(question)
        answer, failure = self._teacher_answer(question.text)
        if failure:
            return AskingTurn(
                index=index,
                question=question.text,
                answer="",
                source=question.kind,
                teacher_name=self.teacher_name,
                error=failure,
            )
        if not answer.strip():
            return AskingTurn(
                index=index,
                question=question.text,
                answer="",
                source=question.kind,
                teacher_name=self.teacher_name,
                error="the teacher produced no answer",
            )
        verified, confidence, reason = self._verifier.verify(question.text, answer)
        example = self._verifier.record_answer(
            question.text, answer, teacher_name=self.teacher_name
        )
        learned = self._learn(question, answer, verified=verified, confidence=confidence)
        return AskingTurn(
            index=index,
            question=question.text,
            answer=answer,
            source=question.kind,
            teacher_name=self.teacher_name,
            verified=verified,
            confidence=round(confidence, 6),
            qualified=bool(example.qualified),
            reason=reason,
            relations_learned=learned,
            learned=bool(learned),
        )

    def _teacher_answer(self, question: str) -> tuple[str, str]:
        """The teacher's answer -- the *only* source of a turn's answer text.

        Returns ``(answer, failure)``: a working teacher gives ``(text, "")``; a
        teacher that raises gives ``("", "<reason>")``, so the turn is honestly
        reported as unanswered rather than carrying a stand-in as if it were the
        teacher's words.
        """
        try:
            text = self.teacher.answer(question)
        except Exception as error:  # a broken teacher is reported, never fatal
            return "", f"the teacher failed: {type(error).__name__}"
        return (text.strip() if isinstance(text, str) else ""), ""

    def _learn(
        self, question: BankQuestion, answer: str, *, verified: bool, confidence: float
    ) -> tuple[str, ...]:
        """Fold the teacher's answer into the learner's knowledge, honestly.

        A fact (the answer's first complete sentence, keyed by the question) is
        stored through the learning system's own bounded memory and is marked
        *verified* only when the graph corroborated it; the typed relations the
        answer states are folded through the same bounded learner every other
        write uses. Both are best-effort: a failure never stops the loop.
        """
        if self.learning is not None:
            statement = self._statement(answer)
            try:
                self.learning.learn_fact(
                    _fact_id(f"{question.text}\n{statement}"),
                    f"{question.topic}: {statement}",
                    source=Source(SourceKind.TEACHER, self.teacher_name, reliability=0.55),
                    confidence=confidence if verified else 0.4,
                    verified=verified,
                )
            except Exception:
                pass
            try:
                from cognition.records import Experience

                self.learning.record_experience(
                    Experience(
                        expectation=f"an answer to: {question.text}",
                        outcome=statement,
                        lesson=f"learned from the teacher {self.teacher_name}",
                        success=verified,
                    )
                )
            except Exception:
                pass
        if self.learn_relations is None:
            return ()
        try:
            learning = self.learn_relations(answer)
        except Exception:
            return ()
        return tuple(
            f"{source} {relation} {target}"
            for source, relation, target in getattr(learning, "added", ())
        )

    @staticmethod
    def _statement(answer: str) -> str:
        try:
            from core.composition import first_sentences

            return first_sentences(answer, limit=1) or answer[:200]
        except Exception:
            return answer[:200]

    # -- question selection ------------------------------------------------

    def _next_question(self) -> BankQuestion | None:
        if self._invent_with_model:
            proposed = self._invent_from_model()
            if proposed is not None:
                return proposed
        bank = self._next_bank_question()
        if bank is not None:
            return bank
        if self._inventor is not None:
            return self._inventor.next(self._asked)
        return None

    def _next_bank_question(self) -> BankQuestion | None:
        while self._bank_index < len(self._questions):
            question = self._questions[self._bank_index]
            self._bank_index += 1
            if question.key not in self._asked:
                return question
        return None

    def _invent_from_model(self) -> BankQuestion | None:
        propose = getattr(self.student, "propose_question", None)
        if not callable(propose):
            return None
        try:
            if not self.student.available():
                return None
        except Exception:
            return None
        topic = self._next_topic()
        try:
            text = propose(topic)
        except Exception:
            return None
        text = self._clean_question(text)
        if not text or not self._is_question(text):
            return None
        if self._too_similar(text) or self._groundless(text):
            return None
        if question_key(text) in self._asked:
            return None
        return BankQuestion(text=text, topic=topic, kind="student")

    def _next_topic(self) -> str:
        if not self._concept_labels:
            return "petroleum engineering"
        return self._concept_labels[len(self._asked) % len(self._concept_labels)]

    def _ordered_labels(self) -> list[str]:
        if self.ontology is None:
            return []
        labels = sorted(
            {str(getattr(c, "label", "")).strip() for c in self.ontology.concepts.values()}
        )
        return [label for label in labels if label]

    @staticmethod
    def _clean_question(text: str) -> str:
        if not isinstance(text, str):
            return ""
        line = text.strip().splitlines()[0].strip() if text.strip() else ""
        for prefix in ("Question:", "Q:", "السؤال:"):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        return line.strip().strip('"').strip()

    @staticmethod
    def _is_question(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith(("?", "؟")):
            return True
        lowered = stripped.casefold()
        return any(lowered.startswith(word) for word in _QUESTION_WORDS)

    def _groundless(self, text: str) -> bool:
        """True when an invented question names no concept the graph knows.

        Keeps an invented question anchored in the project's own knowledge rather
        than a floating phrase the learner could not really study. With no
        ontology the check is skipped (nothing to check against).
        """
        if self.ontology is None:
            return False
        try:
            return not self.ontology.concepts_in(text)
        except Exception:
            return False

    def _too_similar(self, text: str) -> bool:
        tokens = _tokens(text)
        return any(_jaccard(tokens, recent) >= SIMILARITY_LIMIT for recent in self._recent)

    def _mark_asked(self, question: BankQuestion) -> None:
        self._asked.add(question.key)
        self._recent.append(_tokens(question.text))
        del self._recent[:-SIMILARITY_WINDOW]
        if self._log is not None:
            try:
                self._log.record((question,))
            except Exception:
                pass

    # -- measurement -------------------------------------------------------

    def _fact_count(self) -> int:
        if self.learning is None:
            return 0
        try:
            return int(self.learning.facts.count())
        except Exception:
            return 0

    def _report(
        self,
        turns: list[AskingTurn],
        duration_ms: float,
        before_facts: int,
        after_facts: int,
        exhausted: bool,
    ) -> AskingReport:
        return AskingReport(
            asked=len(turns),
            answered=sum(1 for turn in turns if turn.answered),
            qualified=sum(1 for turn in turns if turn.qualified),
            student_questions=sum(1 for turn in turns if turn.source == "student"),
            bank_questions=sum(1 for turn in turns if turn.source == "bank"),
            invented_questions=sum(1 for turn in turns if turn.source == "invented"),
            relations_learned=sum(len(turn.relations_learned) for turn in turns),
            facts_gained=max(0, after_facts - before_facts),
            stopped=self._stop,
            exhausted=exhausted,
            duration_ms=duration_ms,
            teacher_name=self.teacher_name,
        )


# ------------------------------------------------------------- the held-out gate

#: One judgement: a named check the real loop must satisfy. Each is a pure
#: function of the loop, so the gate measures the *shipped* engine rather than a
#: copy of it.
@dataclass(frozen=True)
class AskingJudgement:
    key: str
    description: str


ASKING_JUDGEMENTS: tuple[AskingJudgement, ...] = (
    AskingJudgement("teacher_answers", "the answer a turn carries is the teacher's, never the learner's"),
    AskingJudgement("no_repeats", "no question is asked twice in a run"),
    AskingJudgement("stop_takes_effect", "asking the loop to stop ends it early"),
    AskingJudgement("verified_only_qualifies", "only an answer the graph corroborates becomes qualified"),
    AskingJudgement("empty_answer_is_honest", "a teacher that answers nothing is reported, not hidden"),
    AskingJudgement("exhaustion_is_honest", "a spent question source ends the run rather than looping"),
    AskingJudgement("learning_reaches_the_store", "a qualified answer reaches the learner's fact store"),
)


def _gate_ontology():
    from core.ontology import Ontology

    ontology = Ontology()
    ids = {label: ontology.add_concept_once(label).id for label in ("porosity", "permeability")}
    ontology.add_relation_once(ids["porosity"], ids["permeability"], "increases", confidence=0.6)
    return ontology


class _FakeStudent:
    """A learner whose ``answer`` would betray it if the loop ever asked it."""

    name = "learner"

    def __init__(self) -> None:
        self.proposals: list[str] = []

    def available(self) -> bool:
        return True

    def answer(self, question: str) -> str:  # pragma: no cover - must never run
        return "LEARNER MUST NEVER ANSWER"

    def propose_question(self, topic: str) -> str:
        self.proposals.append(topic)
        return f"What controls {topic}?"


class _FakeTeacher:
    """A teacher that always answers, and counts what it was asked."""

    name = "linked-teacher"

    def __init__(self, answer: str = "porosity increases permeability.") -> None:
        self._answer = answer
        self.asked: list[str] = []

    def available(self) -> bool:
        return True

    def answer(self, question: str) -> str:
        self.asked.append(question)
        return self._answer


def asking_loop_accuracy(
    judgements: tuple[AskingJudgement, ...] = ASKING_JUDGEMENTS,
) -> float:
    """The share of judgements the shipped loop satisfies.

    The gate runs the *real* :class:`AskingLoop` against a deterministic learner
    and teacher with **negative controls**, so a regression that let the learner
    answer its own question, repeated a question, ignored a stop, or trusted an
    unverified answer lowers the metric instead of passing quietly.

    The run is hermetic: the learning system's durable stores are pointed at a
    throw-away folder for the duration, so the gate measures the loop rather than
    whatever the machine happens to have learned -- and it never writes to the
    user's own data.
    """
    if not judgements:
        return 0.0
    import os
    import tempfile

    from core.paths import ENV_DATA_DIR

    checks = _GATE_CHECKS
    passed = 0
    previous = os.environ.get(ENV_DATA_DIR)
    with tempfile.TemporaryDirectory(prefix="fayd-asking-gate-") as folder:
        os.environ[ENV_DATA_DIR] = folder
        try:
            for judgement in judgements:
                check = checks.get(judgement.key)
                if check is None:
                    continue
                try:
                    if check():
                        passed += 1
                except Exception:
                    continue
        finally:
            if previous is None:
                os.environ.pop(ENV_DATA_DIR, None)
            else:
                os.environ[ENV_DATA_DIR] = previous
    return round(passed / len(judgements), 6)


def _check_teacher_answers() -> bool:
    teacher = _FakeTeacher("porosity increases permeability.")
    loop = AskingLoop(
        _FakeStudent(), teacher, ontology=_gate_ontology(), invent_with_model=False,
        questions=tuple(_bank()),
    )
    turns: list[AskingTurn] = []
    loop.run(count=3, on_turn=turns.append)
    return bool(turns) and all(turn.answer == "porosity increases permeability." for turn in turns)


def _check_no_repeats() -> bool:
    loop = AskingLoop(
        _FakeStudent(), _FakeTeacher(), ontology=_gate_ontology(),
        invent_with_model=False, questions=tuple(_bank()),
    )
    turns: list[AskingTurn] = []
    loop.run(count=6, on_turn=turns.append)
    keys = [question_key(turn.question) for turn in turns]
    return len(keys) == len(set(keys)) and len(turns) == 6


def _check_stop_takes_effect() -> bool:
    loop = AskingLoop(
        _FakeStudent(), _FakeTeacher(), ontology=_gate_ontology(),
        invent_with_model=False, questions=tuple(_bank()),
    )
    turns: list[AskingTurn] = []

    def on_turn(turn: AskingTurn) -> None:
        turns.append(turn)
        if len(turns) >= 2:
            loop.stop()

    report = loop.run(on_turn=on_turn)
    return len(turns) == 2 and report.stopped and report.asked == 2


def _check_verified_only_qualifies() -> bool:
    ontology = _gate_ontology()
    good = AskingLoop(
        _FakeStudent(), _FakeTeacher("porosity increases permeability."),
        ontology=ontology, invent_with_model=False, questions=tuple(_bank()),
    )
    good_turns: list[AskingTurn] = []
    good.run(count=1, on_turn=good_turns.append)
    bad = AskingLoop(
        _FakeStudent(), _FakeTeacher("zzz qqq wwww"),
        ontology=ontology, invent_with_model=False, questions=tuple(_bank()),
    )
    bad_turns: list[AskingTurn] = []
    bad.run(count=1, on_turn=bad_turns.append)
    return (
        bool(good_turns) and good_turns[0].qualified
        and bool(bad_turns) and not bad_turns[0].qualified
    )


def _check_empty_answer_is_honest() -> bool:
    teacher = _FakeTeacher("")
    loop = AskingLoop(
        _FakeStudent(), teacher, ontology=_gate_ontology(),
        invent_with_model=False, questions=tuple(_bank()),
    )
    turns: list[AskingTurn] = []
    report = loop.run(count=2, on_turn=turns.append)
    return (
        report.asked == 2 and report.answered == 0
        and all(turn.error for turn in turns) and not any(turn.qualified for turn in turns)
    )


def _check_exhaustion_is_honest() -> bool:
    loop = AskingLoop(
        _FakeStudent(), _FakeTeacher(), ontology=_gate_ontology(),
        invent_with_model=False, invent_questions=False, questions=tuple(_bank()[:2]),
    )
    report = loop.run()
    return report.asked == 2 and report.exhausted and not report.stopped


def _check_learning_reaches_the_store() -> bool:
    from cognition.learning_system import CognitiveLearningSystem

    learning = CognitiveLearningSystem()
    loop = AskingLoop(
        _FakeStudent(), _FakeTeacher("porosity increases permeability."),
        ontology=_gate_ontology(), learning=learning, invent_with_model=False,
        questions=tuple(_bank()),
    )
    report = loop.run(count=2)
    return report.facts_gained >= 1 and learning.facts.count() >= 1


def _bank() -> tuple[BankQuestion, ...]:
    return (
        BankQuestion(text="What is porosity?", topic="porosity", kind="bank"),
        BankQuestion(text="What is permeability?", topic="permeability", kind="bank"),
        BankQuestion(text="Why does porosity increase permeability?", topic="porosity", kind="bank"),
        BankQuestion(text="How is porosity measured?", topic="porosity", kind="bank"),
        BankQuestion(text="What controls permeability?", topic="permeability", kind="bank"),
        BankQuestion(text="What does a core sample measure?", topic="core_sample", kind="bank"),
    )


_GATE_CHECKS = {
    "teacher_answers": _check_teacher_answers,
    "no_repeats": _check_no_repeats,
    "stop_takes_effect": _check_stop_takes_effect,
    "verified_only_qualifies": _check_verified_only_qualifies,
    "empty_answer_is_honest": _check_empty_answer_is_honest,
    "exhaustion_is_honest": _check_exhaustion_is_honest,
    "learning_reaches_the_store": _check_learning_reaches_the_store,
}


__all__ = [
    "ASKING_JUDGEMENTS",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_TEACHER_NAME",
    "PAIR_TEMPLATES_AR",
    "PAIR_TEMPLATES_EN",
    "QUESTION_PROMPT",
    "SIMILARITY_LIMIT",
    "SIMILARITY_WINDOW",
    "AskingJudgement",
    "AskingLoop",
    "AskingReport",
    "AskingTurn",
    "InventedQuestionSource",
    "TurnCallback",
    "asking_loop_accuracy",
]
