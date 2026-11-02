"""The learner-asks-the-teacher loop: Fayd Al-Ruh asks the linked model endlessly.

Everything runs headless with a deterministic learner and teacher: no model, no
display, and no network. The held-out gate is exercised too, so a regression in
the loop fails here. The durable learning stores are pointed at a throw-away
folder so the tests never touch the machine's own data.
"""

from __future__ import annotations

import pytest
from cognition.asking import (
    ASKING_JUDGEMENTS,
    AskingLoop,
    AskingReport,
    AskingTurn,
    InventedQuestionSource,
    asking_loop_accuracy,
)
from cognition.question_bank import BankQuestion, question_key


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch) -> None:
    """Keep every store the loop writes to inside the test's own folder."""
    from core.paths import ENV_DATA_DIR

    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))


def _ontology():
    from core.ontology import Ontology

    ontology = Ontology()
    ids = {label: ontology.add_concept_once(label).id for label in ("porosity", "permeability")}
    ontology.add_relation_once(ids["porosity"], ids["permeability"], "increases", confidence=0.6)
    return ontology


class _Student:
    """A learner that would betray itself if the loop ever let it answer."""

    name = "learner"

    def __init__(self, proposal: str = "What controls porosity?") -> None:
        self.proposal = proposal
        self.asked: list[str] = []

    def available(self) -> bool:
        return True

    def answer(self, question: str) -> str:  # pragma: no cover - must never run
        return "LEARNER MUST NEVER ANSWER"

    def propose_question(self, topic: str) -> str:
        self.asked.append(topic)
        return self.proposal


class _Teacher:
    """A teacher that always answers and counts what it was asked."""

    name = "linked-teacher"

    def __init__(self, answer: str = "porosity increases permeability.") -> None:
        self._answer = answer
        self.questions: list[str] = []

    def available(self) -> bool:
        return True

    def answer(self, question: str) -> str:
        self.questions.append(question)
        return self._answer


def _bank() -> tuple[BankQuestion, ...]:
    return (
        BankQuestion(text="What is porosity?", topic="porosity", kind="bank"),
        BankQuestion(text="What is permeability?", topic="permeability", kind="bank"),
        BankQuestion(text="Why does porosity increase permeability?", topic="porosity", kind="bank"),
        BankQuestion(text="How is porosity measured?", topic="porosity", kind="bank"),
    )


def _loop(student=None, teacher=None, **kwargs) -> AskingLoop:
    return AskingLoop(
        student or _Student(),
        teacher or _Teacher(),
        ontology=_ontology(),
        invent_with_model=False,
        questions=tuple(_bank()),
        **kwargs,
    )


# -- the answer is always the teacher's -------------------------------------


def test_the_answer_a_turn_carries_is_the_teachers() -> None:
    teacher = _Teacher("porosity increases permeability.")
    turns: list[AskingTurn] = []
    _loop(teacher=teacher).run(count=3, on_turn=turns.append)
    assert len(turns) == 3
    assert all(turn.answer == "porosity increases permeability." for turn in turns)
    assert all(turn.teacher_name == "linked-teacher" for turn in turns)


def test_the_learner_is_never_asked_to_answer() -> None:
    student = _Student()
    teacher = _Teacher("porosity increases permeability.")
    _loop(student=student, teacher=teacher).run(count=2)
    assert student.asked == []  # invent_with_model is off, so no proposals either
    assert len(teacher.questions) == 2


def test_an_invented_question_reaches_the_teacher() -> None:
    student = _Student("What controls porosity?")
    teacher = _Teacher("porosity increases permeability.")
    turns: list[AskingTurn] = []
    AskingLoop(
        student, teacher, ontology=_ontology(), questions=(), invent_with_model=True,
    ).run(count=1, on_turn=turns.append)
    assert turns[0].source == "student"
    assert turns[0].question == "What controls porosity?"
    assert teacher.questions == ["What controls porosity?"]


# -- no question repeats ----------------------------------------------------


def test_no_question_repeats_in_a_run() -> None:
    turns: list[AskingTurn] = []
    _loop().run(count=4, on_turn=turns.append)
    keys = [question_key(turn.question) for turn in turns]
    assert len(keys) == len(set(keys)) == 4


def test_a_near_repeat_proposal_is_refused() -> None:
    # The proposal is near-identical to the first bank question, so the guard
    # refuses it and the loop falls back to the bank's own next question.
    student = _Student("What is porosity?")
    teacher = _Teacher("porosity increases permeability.")
    turns: list[AskingTurn] = []
    AskingLoop(
        student, teacher, ontology=_ontology(), questions=tuple(_bank()), invent_with_model=True,
    ).run(count=2, on_turn=turns.append)
    assert turns[1].question != "What is porosity?"
    assert turns[1].source == "bank"


def test_the_bank_does_not_repeat_either() -> None:
    loop = _loop()
    turns: list[AskingTurn] = []
    loop.run(count=4, on_turn=turns.append)
    assert [turn.source for turn in turns] == ["bank"] * 4


# -- stop takes effect ------------------------------------------------------


def test_stop_ends_the_loop_early() -> None:
    loop = _loop()
    turns: list[AskingTurn] = []

    def on_turn(turn: AskingTurn) -> None:
        turns.append(turn)
        if len(turns) >= 2:
            loop.stop()

    report = loop.run(on_turn=on_turn)
    assert len(turns) == 2
    assert report.stopped is True
    assert report.asked == 2


def test_reset_clears_the_stop_flag() -> None:
    loop = _loop()
    loop.stop()
    assert loop.stopped is True
    loop.reset()
    assert loop.stopped is False


# -- verification and learning ---------------------------------------------


def test_only_a_verified_answer_qualifies() -> None:
    good: list[AskingTurn] = []
    _loop(teacher=_Teacher("porosity increases permeability.")).run(count=1, on_turn=good.append)
    assert good[0].qualified is True
    assert good[0].verified is True

    bad: list[AskingTurn] = []
    _loop(teacher=_Teacher("zzz qqq wwww")).run(count=1, on_turn=bad.append)
    assert bad[0].qualified is False


def test_a_qualified_answer_reaches_the_learners_fact_store() -> None:
    from cognition.learning_system import CognitiveLearningSystem

    learning = CognitiveLearningSystem()
    report = _loop(teacher=_Teacher("porosity increases permeability."), learning=learning).run(
        count=2
    )
    assert report.facts_gained >= 1
    assert learning.facts.count() >= 1


def test_the_answer_teaches_a_typed_relation() -> None:
    from core.ontology import Ontology
    from core.ontology_learner import OntologyLearner
    from core.relation_extraction import RelationExtractor

    ontology = Ontology()
    for label in ("alpha", "beta"):
        ontology.add_concept_once(label)
    learner = OntologyLearner(ontology)
    extractor = RelationExtractor(ontology, learner=learner)
    loop = AskingLoop(
        _Student(),
        _Teacher("alpha increases beta."),
        ontology=ontology,
        learn_relations=extractor.apply,
        questions=tuple(_bank()),
        invent_with_model=False,
    )
    report = loop.run(count=1)
    assert report.relations_learned == 1
    assert len(ontology.relations) == 1


# -- honesty ----------------------------------------------------------------


def test_an_empty_answer_is_reported_not_hidden() -> None:
    turns: list[AskingTurn] = []
    report = _loop(teacher=_Teacher("")).run(count=2, on_turn=turns.append)
    assert report.asked == 2
    assert report.answered == 0
    assert all(turn.error for turn in turns)
    assert not any(turn.qualified for turn in turns)


def test_a_broken_teacher_is_reported_not_fatal() -> None:
    class _Broken(_Teacher):
        def answer(self, question: str) -> str:
            raise RuntimeError("boom")

    turns: list[AskingTurn] = []
    report = _loop(teacher=_Broken()).run(count=1, on_turn=turns.append)
    assert report.answered == 0
    assert turns[0].answer == ""
    assert "failed" in turns[0].error


def test_a_spent_question_source_ends_the_run() -> None:
    loop = AskingLoop(
        _Student(), _Teacher(), ontology=_ontology(), invent_with_model=False,
        invent_questions=False, questions=tuple(_bank()[:2]),
    )
    report = loop.run()
    assert report.asked == 2
    assert report.exhausted is True
    assert report.stopped is False


def test_an_empty_bank_without_an_inventor_ends_immediately() -> None:
    loop = AskingLoop(
        _Student(), _Teacher(), ontology=_ontology(), invent_with_model=False,
        invent_questions=False, questions=(),
    )
    report = loop.run()
    assert report.asked == 0
    assert report.exhausted is True


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _loop().run(count=0)


# -- the report -------------------------------------------------------------


def test_the_report_measures_the_run() -> None:
    report = _loop(teacher=_Teacher("porosity increases permeability.")).run(count=3)
    assert isinstance(report, AskingReport)
    assert report.asked == 3
    assert report.answered == 3
    assert report.qualified == 3
    assert report.bank_questions == 3
    assert report.answer_rate == 1.0
    assert report.qualified_rate == 1.0
    assert report.duration_ms >= 0.0


def test_the_report_round_trips_to_a_primitive() -> None:
    report = _loop().run(count=2)
    payload = report.to_primitive()
    assert payload["asked"] == 2
    assert payload["teacher_name"] == "linked-teacher"


def test_a_turn_round_trips_to_a_primitive() -> None:
    turns: list[AskingTurn] = []
    _loop().run(count=1, on_turn=turns.append)
    payload = turns[0].to_primitive()
    assert payload["index"] == 1
    assert payload["question"]
    assert payload["answer"]


# -- the invented question source -------------------------------------------


def test_the_inventor_is_effectively_unbounded() -> None:
    from core.ontology import Ontology

    ontology = Ontology()
    for index in range(25):
        ontology.add_concept_once(f"concept_{index}")
    inventor = InventedQuestionSource(ontology)
    asked: set[str] = set()
    for _ in range(200):
        question = inventor.next(asked)
        if question is None:
            break
        asked.add(question.key)
    assert len(asked) == 200


def test_the_inventor_exhausts_at_its_exact_total() -> None:
    ontology = _ontology()
    inventor = InventedQuestionSource(ontology)
    asked: set[str] = set()
    while True:
        question = inventor.next(asked)
        if question is None:
            break
        asked.add(question.key)
    # 2 concepts -> one pair x ten templates.
    assert len(asked) == 10


def test_the_inventor_names_only_known_concepts() -> None:
    ontology = _ontology()
    inventor = InventedQuestionSource(ontology)
    question = inventor.next(set())
    assert question is not None
    assert ontology.concepts_in(question.text)


# -- the held-out gate ------------------------------------------------------


def test_the_gate_is_perfect_on_the_shipped_loop() -> None:
    assert asking_loop_accuracy() == 1.0


def test_the_gate_is_zero_with_no_judgements() -> None:
    assert asking_loop_accuracy(()) == 0.0


def test_the_gate_has_negative_controls() -> None:
    keys = {judgement.key for judgement in ASKING_JUDGEMENTS}
    assert "teacher_answers" in keys
    assert "verified_only_qualifies" in keys
    assert "empty_answer_is_honest" in keys
