"""The Mind's self-study session: ask, answer, learn, and report the benefit.

The Proactive Mind can *think* (one cycle produces ideas) and can *ask* (the
question engine turns gaps into questions), but it had no way to sit down and
**study**: ask a long list of its own questions, read the project's own answer to
each, fold what it reads into its memories, and then say honestly how much it
gained. This module is that session.

:class:`StudySession` runs the whole loop over the project's **own** objects --
the assistant's cited answers, the relation extractor, the cognitive learning
system, and the proactive mind's memory -- and reports each step through a
callback as it happens, so a UI can stream the questions and answers live.

The **benefit report** is measured, never asserted. It compares a
:class:`cognition.evaluation.CognitiveReport` taken before the session with one
taken after, and counts what the session actually changed:

* relations the graph learned (from the assistant's own answers);
* facts the learning system gained;
* new associations and episodes;
* the delta on each of the ten cognitive indicators.

Nothing here widens what the mind may believe: an answer is the grounded
assistant's cited reply (or an honest "no evidence"), a learned relation passes
through the same bounded learner every other write uses, and a question is never
treated as a fact. Dependency-free and Tk-free, so it is exercised with no model,
no display, and no network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from cognition.question_bank import BankQuestion, QuestionLog, next_questions
from cognition.storage import NAMESPACE_STUDY, _load_list, _save_list

#: The number of questions one study session asks by default -- the task's figure.
DEFAULT_SESSION_QUESTIONS = 200

#: How many questions to keep from the session in the durable record (a long
#: session should not grow the document without bound).
MAX_RECORDED_TURNS = 500


@dataclass(frozen=True)
class StudyTurn:
    """One question the mind asked and the answer it read."""

    index: int
    question: str
    answer: str
    citations: tuple[str, ...] = ()
    relations_learned: int = 0
    learned_relations: tuple[str, ...] = ()
    answered: bool = True

    def describe(self) -> str:
        mark = "" if self.answered else " (no evidence)"
        return f"Q{self.index}: {self.question}{mark}\n  A: {self.answer}"

    def to_primitive(self) -> dict[str, object]:
        return {
            "index": self.index,
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "relations_learned": self.relations_learned,
            "learned_relations": list(self.learned_relations),
            "answered": self.answered,
        }

    @classmethod
    def from_primitive(cls, data: dict) -> StudyTurn:
        return cls(
            index=int(data.get("index", 0)),
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            citations=tuple(str(c) for c in data.get("citations", ()) or ()),
            relations_learned=int(data.get("relations_learned", 0)),
            learned_relations=tuple(str(r) for r in data.get("learned_relations", ()) or ()),
            answered=bool(data.get("answered", True)),
        )


@dataclass(frozen=True)
class BenefitReport:
    """What one study session actually changed, measured before and after."""

    asked: int
    answered: int
    relations_learned: int
    facts_gained: int
    associations_gained: int
    episodes_gained: int
    indicator_delta: dict[str, float] = field(default_factory=dict)
    before_overall: float = 0.0
    after_overall: float = 0.0

    @property
    def answer_rate(self) -> float:
        return round(self.answered / self.asked, 6) if self.asked else 0.0

    @property
    def overall_delta(self) -> float:
        return round(self.after_overall - self.before_overall, 6)

    @property
    def improved(self) -> bool:
        """True when the session moved at least one durable measure forward."""
        return bool(
            self.relations_learned
            or self.facts_gained
            or self.associations_gained
            or self.episodes_gained
        )

    def describe(self) -> str:
        lines = [
            "Self-study benefit report",
            f"  questions asked: {self.asked} (answered from evidence: {self.answered}, "
            f"{self.answer_rate:.0%})",
            f"  relations learned: {self.relations_learned}",
            f"  facts gained: {self.facts_gained}",
            f"  associations gained: {self.associations_gained}",
            f"  episodes gained: {self.episodes_gained}",
            f"  cognitive indicators: {self.before_overall:.3f} → {self.after_overall:.3f} "
            f"({self.overall_delta:+.3f})",
        ]
        moved = [
            (key, delta)
            for key, delta in sorted(self.indicator_delta.items())
            if abs(delta) > 1e-9
        ]
        for key, delta in moved:
            lines.append(f"    {key}: {delta:+.3f}")
        if not moved:
            lines.append("    (no indicator moved this session)")
        if not self.improved:
            lines.append(
                "  note: the session added no new durable knowledge -- every answer "
                "was already known, which is honest, not a failure."
            )
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "asked": self.asked,
            "answered": self.answered,
            "relations_learned": self.relations_learned,
            "facts_gained": self.facts_gained,
            "associations_gained": self.associations_gained,
            "episodes_gained": self.episodes_gained,
            "indicator_delta": dict(self.indicator_delta),
            "before_overall": self.before_overall,
            "after_overall": self.after_overall,
        }

    @classmethod
    def from_primitive(cls, data: dict) -> BenefitReport:
        return cls(
            asked=int(data.get("asked", 0)),
            answered=int(data.get("answered", 0)),
            relations_learned=int(data.get("relations_learned", 0)),
            facts_gained=int(data.get("facts_gained", 0)),
            associations_gained=int(data.get("associations_gained", 0)),
            episodes_gained=int(data.get("episodes_gained", 0)),
            indicator_delta={str(k): float(v) for k, v in (data.get("indicator_delta") or {}).items()},
            before_overall=float(data.get("before_overall", 0.0)),
            after_overall=float(data.get("after_overall", 0.0)),
        )


#: The callback a streaming surface passes: it receives each :class:`StudyTurn`
#: the moment it is produced, so a UI can show the questions and answers live.
TurnCallback = Callable[[StudyTurn], None]


@dataclass
class StudyResult:
    """The whole session: every turn and its benefit report."""

    turns: tuple[StudyTurn, ...]
    report: BenefitReport

    def describe(self) -> str:
        return self.report.describe()


class StudySession:
    """Run one self-study session over the project's own knowledge.

    ``controller`` is a :class:`core.cognitive_controller.CognitiveController`
    (the same one the Proactive Mind runs on). ``assistant`` is the grounded
    assistant the questions are answered with; when it is ``None`` the project's
    default assistant is built lazily, and when it cannot be built the session
    still runs and reports every question as unanswered rather than failing.
    """

    def __init__(
        self,
        controller,
        *,
        assistant=None,
        evaluator=None,
        log: QuestionLog | None = None,
        record: bool = True,
        measure_indicators: bool = True,
    ) -> None:
        self.controller = controller
        self._assistant = assistant
        self._evaluator = evaluator
        self.log = log if log is not None else QuestionLog()
        self.record = record
        # Whether the session measures the ten cognitive indicators before and
        # after. The indicator pass is the expensive part (it runs the analogy
        # engine), so a caller that only wants the knowledge counts may turn it
        # off; the cheap counts (relations, facts, associations, episodes) are
        # always measured.
        self.measure_indicators = measure_indicators

    # -- the run ------------------------------------------------------------

    def run(
        self,
        *,
        count: int = DEFAULT_SESSION_QUESTIONS,
        on_turn: TurnCallback | None = None,
        questions: Iterable[BankQuestion] | None = None,
        baseline: object = None,
    ) -> StudyResult:
        """Ask ``count`` questions, learn from each answer, and report the gain.

        Each turn is emitted through ``on_turn`` as soon as it is produced, so a
        caller can stream the session. A question whose answer carries no evidence
        is recorded as unanswered (honest) rather than skipped, so the session
        always asks exactly ``count`` questions.

        ``baseline`` is an optional, already-measured ``CognitiveReport`` to use as
        the "before" reading (the previous session's after-state), so a later
        session need not re-measure the starting point.
        """
        if count < 1:
            raise ValueError("count must be at least 1.")
        chosen = (
            tuple(questions)
            if questions is not None
            else next_questions(self.controller.ontology, count=count, log=self.log, record=False)
        )
        assistant = self._assistant_or_default()
        evaluator = self._evaluator_or_default()
        before = self._before_report(evaluator, baseline)
        before_stats = self._learning_stats()

        turns: list[StudyTurn] = []
        for index, question in enumerate(chosen, start=1):
            turn = self._study_one(index, question, assistant)
            turns.append(turn)
            if on_turn is not None:
                on_turn(turn)

        after = self._after_report(evaluator, before)
        after_stats = self._learning_stats()
        if self.record:
            self.log.record(chosen)
        report = self._benefit(before, after, before_stats, after_stats, turns)
        self._persist(tuple(turns), report)
        return StudyResult(turns=tuple(turns), report=report)

    def _before_report(self, evaluator, baseline):
        if not self.measure_indicators:
            return None
        if baseline is not None:
            return baseline
        return evaluator.evaluate()

    def _after_report(self, evaluator, before):
        if not self.measure_indicators:
            return None
        return evaluator.evaluate()

    # -- one turn -----------------------------------------------------------

    def _study_one(self, index: int, question: BankQuestion, assistant) -> StudyTurn:
        """Ask one question, read the answer, and learn the relations it states."""
        if assistant is None:
            return StudyTurn(
                index=index,
                question=question.text,
                answer="(no assistant available to answer this question)",
                answered=False,
            )
        try:
            result = assistant.ask(question.text)
        except Exception as error:  # a failed ask is one unanswered turn, not a crash
            return StudyTurn(
                index=index,
                question=question.text,
                answer=f"(the answer could not be produced: {type(error).__name__})",
                answered=False,
            )
        text = (getattr(result, "text", "") or "").strip()
        citations = tuple(str(c) for c in getattr(result, "citations", ()) or ())
        if not text:
            return StudyTurn(index=index, question=question.text, answer="(no answer)", answered=False)
        learned = self._learn_from(text)
        self._absorb(index, question, text, citations)
        return StudyTurn(
            index=index,
            question=question.text,
            answer=text,
            citations=citations,
            relations_learned=len(learned),
            learned_relations=learned,
            answered=bool(citations),
        )

    def _absorb(self, index: int, question: BankQuestion, text: str, citations: tuple[str, ...]) -> None:
        """Fold one answered turn into the cognitive learning system.

        A fact (the answer's first complete sentence, keyed by the question) and an
        episode (what was asked and read) are stored through the learning system's
        own bounded memories, so the mind's recall can reach what it studied. Both
        are best-effort: a failure never stops the session.
        """
        try:
            from core.composition import first_sentences

            statement = first_sentences(text, limit=1) or text[:200]
        except Exception:
            statement = text[:200]
        try:
            learning = self.controller.cognitive_learning()
            learning.learn_fact(
                question.key,
                f"{question.topic}: {statement}",
                confidence=0.5,
                verified=False,
            )
            from cognition.records import Experience

            learning.record_experience(
                Experience(
                    expectation=f"an answer to: {question.text}",
                    outcome=statement,
                    lesson="studied from the project's own sources",
                    success=True,
                )
            )
        except Exception:
            return

    def _learn_from(self, text: str) -> tuple[str, ...]:
        """Read typed relations from an answer and fold them into the graph.

        Goes through the controller's own ``learn_extracted_relations`` (the same
        bounded learner every write uses), so a relation the answer states joins
        the graph only when both endpoints are known concepts and never overrides
        a seeded one. Returns the readable triples that were added.
        """
        try:
            learning = self.controller.learn_extracted_relations(text)
        except Exception:
            return ()
        return tuple(
            f"{source} {relation} {target}" for source, relation, target in learning.added
        )

    # -- measurement --------------------------------------------------------

    def _assistant_or_default(self):
        if self._assistant is not None:
            return self._assistant
        try:
            from core.assistant import build_default_assistant

            self._assistant = build_default_assistant()
        except Exception:
            self._assistant = None
        return self._assistant

    def _evaluator_or_default(self):
        if self._evaluator is not None:
            return self._evaluator
        from cognition.evaluation import CognitiveEvaluator

        self._evaluator = CognitiveEvaluator(self.controller.ontology)
        return self._evaluator

    def _learning_stats(self):
        """The durable counts that move when the mind studies something.

        Read from the learning system's own bounded *stores* (the fact store and
        the experience store) and its association counter, not from an in-RAM
        memory that may already be at its cap -- so the benefit report measures
        what the session really added, not what a capped list happened to hold.
        """
        try:
            learning = self.controller.cognitive_learning()
            stats = learning.stats()
            return (
                int(learning.facts.count()),
                int(stats.associations),
                int(learning.experiences.count()),
            )
        except Exception:
            return None

    @staticmethod
    def _benefit(before, after, before_stats, after_stats, turns) -> BenefitReport:
        relations = sum(turn.relations_learned for turn in turns)
        answered = sum(1 for turn in turns if turn.answered)
        indicator_delta: dict[str, float] = {}
        before_overall = 0.0
        after_overall = 0.0
        if before is not None and after is not None:
            indicator_delta = {
                key: round(after.scores.get(key, 0.0) - before.scores.get(key, 0.0), 6)
                for key in after.scores
            }
            before_overall = before.overall
            after_overall = after.overall
        facts = _delta(before_stats, after_stats, 0)
        associations = _delta(before_stats, after_stats, 1)
        episodes = _delta(before_stats, after_stats, 2)
        return BenefitReport(
            asked=len(turns),
            answered=answered,
            relations_learned=relations,
            facts_gained=facts,
            associations_gained=associations,
            episodes_gained=episodes,
            indicator_delta=indicator_delta,
            before_overall=before_overall,
            after_overall=after_overall,
        )

    def _persist(self, turns: tuple[StudyTurn, ...], report: BenefitReport) -> None:
        """Append the session to the durable record, best-effort and bounded."""
        try:
            existing = _load_list(NAMESPACE_STUDY)
            existing.append(
                {
                    "report": report.to_primitive(),
                    "turns": [t.to_primitive() for t in turns][-MAX_RECORDED_TURNS:],
                }
            )
            _save_list(NAMESPACE_STUDY, existing[-20:])
        except Exception:
            pass


def _delta(before, after, index: int) -> int:
    """The growth of one durable count between two readings, never negative."""
    if before is None or after is None:
        return 0
    return max(0, int(after[index]) - int(before[index]))


def study_report(controller, *, count: int = DEFAULT_SESSION_QUESTIONS, **kwargs) -> BenefitReport:
    """Run one session and return only its benefit report (a convenience)."""
    return StudySession(controller, **kwargs).run(count=count).report


def last_study_report() -> BenefitReport | None:
    """The most recent stored benefit report, or ``None`` when none was run."""
    try:
        records = _load_list(NAMESPACE_STUDY)
    except Exception:
        return None
    if not records:
        return None
    data = records[-1].get("report")
    if not isinstance(data, dict):
        return None
    return BenefitReport.from_primitive(data)


__all__ = [
    "DEFAULT_SESSION_QUESTIONS",
    "MAX_RECORDED_TURNS",
    "BenefitReport",
    "StudyResult",
    "StudySession",
    "StudyTurn",
    "last_study_report",
    "study_report",
]
