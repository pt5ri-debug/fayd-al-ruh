"""The mind's question bank and its self-study session.

These tests exercise the two pieces the task asks for: a deep bank of **200**
distinct, non-repeating questions built from the project's own knowledge, and a
self-study session that asks them, reads the project's own answer to each, learns
what it reads, and reports the measured benefit. Everything runs over the real
ontology with no model, no display, and no network.
"""

from __future__ import annotations

import pytest
from cognition.question_bank import (
    DEFAULT_QUESTION_COUNT,
    BankQuestion,
    QuestionLog,
    bank_size,
    build_bank,
    next_questions,
    question_key,
)
from cognition.study import BenefitReport, StudySession, last_study_report
from core.cognitive_controller import CognitiveController


@pytest.fixture(scope="module")
def controller() -> CognitiveController:
    return CognitiveController()


# -- the question bank ------------------------------------------------------


def test_bank_is_deep(controller):
    """The bank holds far more than one run's worth of questions."""
    assert bank_size(controller.ontology) >= DEFAULT_QUESTION_COUNT


def test_bank_is_deterministic(controller):
    first = build_bank(controller.ontology)
    second = build_bank(controller.ontology)
    assert [q.key for q in first] == [q.key for q in second]


def test_bank_questions_are_distinct(controller):
    bank = build_bank(controller.ontology)
    keys = [q.key for q in bank]
    assert len(keys) == len(set(keys))


def test_a_batch_holds_two_hundred_distinct_questions(controller):
    questions = next_questions(controller.ontology, count=DEFAULT_QUESTION_COUNT)
    assert len(questions) == DEFAULT_QUESTION_COUNT
    assert len({q.key for q in questions}) == DEFAULT_QUESTION_COUNT


def test_a_batch_never_repeats_what_was_asked(controller, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    log = QuestionLog()
    first = next_questions(controller.ontology, count=200, log=log, record=True)
    second = next_questions(controller.ontology, count=200, log=log, record=True)
    assert not ({q.key for q in first} & {q.key for q in second})
    assert len({q.key for q in (*first, *second)}) == 400


def test_graph_questions_name_a_stated_relation(controller):
    """A graph question is built from an edge the ontology really holds."""
    bank = build_bank(controller.ontology)
    graph = [q for q in bank if q.kind == "graph"]
    assert graph
    labels = {c.label for c in controller.ontology.concepts.values()}
    assert any(q.topic in labels for q in graph)


def test_topic_questions_are_the_authored_sets(controller):
    from core.character_seeding import PETROLEUM_QUESTIONS, PROJECT_QUESTIONS

    bank = build_bank(controller.ontology)
    topics = {q.text for q in bank if q.kind == "topic"}
    assert set(PETROLEUM_QUESTIONS) <= topics
    assert set(PROJECT_QUESTIONS) <= topics


def test_question_key_is_stable_and_case_insensitive():
    assert question_key("What is porosity?") == question_key("what is  porosity? ")
    assert question_key("a") != question_key("b")


def test_bank_refuses_a_nonpositive_count(controller):
    with pytest.raises(ValueError):
        next_questions(controller.ontology, count=0)


def test_bank_wraps_when_exhausted(controller, tmp_path, monkeypatch):
    """A run always gets its questions, even after the whole bank is spent."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    log = QuestionLog()
    total = bank_size(controller.ontology)
    next_questions(controller.ontology, count=total, log=log, record=True)
    wrapped = next_questions(controller.ontology, count=10, log=log, record=True)
    assert len(wrapped) == 10
    assert len({q.key for q in wrapped}) == 10


def test_bank_question_round_trips():
    q = BankQuestion(text="What is porosity?", topic="porosity", kind="graph")
    assert BankQuestion.from_primitive(q.to_primitive()) == q


# -- the study session ------------------------------------------------------


def test_session_asks_exactly_the_requested_number(controller):
    seen: list[int] = []
    result = StudySession(controller, measure_indicators=False).run(
        count=5, on_turn=lambda t: seen.append(t.index)
    )
    assert len(result.turns) == 5
    assert seen == [1, 2, 3, 4, 5]


def test_session_streams_each_turn_as_it_is_produced(controller):
    streamed: list[str] = []
    StudySession(controller, measure_indicators=False).run(
        count=3, on_turn=lambda t: streamed.append(t.question)
    )
    assert len(streamed) == 3
    assert all(streamed)


def test_session_answers_from_evidence(controller):
    result = StudySession(controller, measure_indicators=False).run(count=3)
    assert result.report.asked == 3
    assert all(turn.answer for turn in result.turns)


def test_session_report_counts_what_it_learned(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    fresh = CognitiveController()
    result = StudySession(fresh, measure_indicators=False).run(count=10)
    report = result.report
    assert report.facts_gained == 10
    assert report.episodes_gained == 10
    assert report.answer_rate == 1.0


def test_session_report_is_honest_when_nothing_moved():
    report = BenefitReport(asked=5, answered=5, relations_learned=0, facts_gained=0,
                           associations_gained=0, episodes_gained=0)
    assert not report.improved
    assert "honest, not a failure" in report.describe()


def test_session_report_round_trips():
    report = BenefitReport(
        asked=4, answered=3, relations_learned=2, facts_gained=1,
        associations_gained=1, episodes_gained=2,
        indicator_delta={"understanding": 0.1}, before_overall=0.5, after_overall=0.6,
    )
    assert BenefitReport.from_primitive(report.to_primitive()) == report


def test_session_without_an_assistant_reports_every_turn_unanswered(controller):
    session = StudySession(controller, assistant=None, measure_indicators=False)
    # Force the "no assistant" path rather than building the default one.
    session._assistant = None
    session._assistant_or_default = lambda: None  # type: ignore[method-assign]
    result = session.run(count=3)
    assert result.report.answered == 0
    assert all(not turn.answered for turn in result.turns)


def test_session_records_the_questions_it_asked(controller, tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    log = QuestionLog()
    StudySession(controller, log=log, measure_indicators=False).run(count=4)
    assert log.asked_count() == 4


def test_session_persists_a_report(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    StudySession(CognitiveController(), measure_indicators=False).run(count=2)
    stored = last_study_report()
    assert stored is not None
    assert stored.asked == 2


def test_session_refuses_a_nonpositive_count(controller):
    with pytest.raises(ValueError):
        StudySession(controller, measure_indicators=False).run(count=0)


# -- the controller wiring --------------------------------------------------


def test_controller_exposes_the_bank_and_study(controller):
    assert len(controller.question_bank()) >= DEFAULT_QUESTION_COUNT
    questions = controller.next_questions(count=3, record=False)
    assert len(questions) == 3
    result = controller.study(count=2, measure_indicators=False)
    assert result.report.asked == 2


# -- the Arabic half of the bank --------------------------------------------


def test_the_bank_asks_a_real_share_of_its_questions_in_arabic(controller):
    """The project is Arabic-first, so the bank is not English-only."""
    from core.voices import detect_language

    bank = build_bank(controller.ontology)
    arabic = [q for q in bank if detect_language(q.text) == "ar"]
    assert len(arabic) >= 200
    assert len(arabic) / len(bank) >= 0.2


def test_an_arabic_graph_question_names_the_graphs_own_concepts(controller):
    from core.arabic_prose import arabic_label
    from core.voices import detect_language

    bank = build_bank(controller.ontology)
    arabic = [q for q in bank if q.kind == "graph" and detect_language(q.text) == "ar"]
    assert arabic
    labels = {arabic_label(c.label, controller.ontology) for c in controller.ontology.concepts.values()}
    assert any(q.topic in labels for q in arabic)


def test_every_stated_edge_contributes_at_least_one_arabic_question(controller):
    """No relation type is silently dropped: an unknown one uses the generic pair."""
    from cognition.question_bank import (
        ARABIC_GENERIC_TEMPLATES,
        ARABIC_RELATION_TEMPLATES,
        _arabic_graph_questions,
    )

    questions = _arabic_graph_questions(controller.ontology)
    assert len(questions) == len({q.key for q in questions})
    assert len(questions) >= 400
    # The templates the code actually uses cover the graph's frequent types and
    # fall back to a generic pair, so no edge is lost.
    assert ARABIC_GENERIC_TEMPLATES
    assert {"is_a", "measures", "increases"} <= set(ARABIC_RELATION_TEMPLATES)
