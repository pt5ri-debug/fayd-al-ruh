"""Tests for the decision bridge (``core.answer_decision``).

The bridge is what finally lets a *decisional* question ("what should I do to
increase permeability?") reach the project's real decision engine and come back
with the ranked recommendation the knowledge graph already states -- Gap 7 of
``docs/REASONING_ASSESSMENT.md``, the decision half. These tests pin the honesty
rules: a recommendation is only built for a question that asks *what to do*, it
offers only edges whose relation matches the direction the question asked for, it
is ranked by ``DecisionEngine.decide``, and a definitional, unrelated, or
unanswerable question yields **no** recommendation.
"""

from __future__ import annotations

import pytest
from ai.chat import format_turn, turn_payload
from core.answer_decision import (
    ACTIONABLE_RELATIONS,
    DECISION_JUDGEMENTS,
    DIRECTION_RELATIONS,
    DecisionJudgement,
    answer_decision,
    decision_recommendation_accuracy,
    direction_of,
    is_decisional,
)
from core.assistant import GroundedAssistant
from core.cognitive_controller import CognitiveController
from core.command_router import NaturalLanguageRouter
from core.documents import build_document_store
from core.jarvis import JarvisMode
from core.rag import RagRetriever
from core.reasoning_report import SOURCE_DECISION, STEP_ORDER, reasoning_report
from core.tools import build_sandbox


def _ontology():
    return CognitiveController().ontology


def _assistant() -> GroundedAssistant:
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=_ontology()))


def _jarvis(tmp_path) -> JarvisMode:
    return JarvisMode(_assistant(), router=NaturalLanguageRouter(build_sandbox(tmp_path)))


# --- the decisional cue ----------------------------------------------------


def test_what_should_i_is_decisional():
    assert is_decisional("what should I do to increase permeability?")


def test_recommend_is_decisional():
    assert is_decisional("what do you recommend for water cut?")


def test_arabic_advice_cue_is_decisional():
    assert is_decisional("بماذا تنصح لزيادة النفاذية؟")


def test_definitional_question_is_not_decisional():
    assert not is_decisional("what is permeability?")


def test_plain_statement_is_not_decisional():
    assert not is_decisional("permeability measures flow.")


# --- the direction ---------------------------------------------------------


def test_increase_direction_is_read():
    assert direction_of("what should I do to increase permeability?") == "increase"


def test_reduce_direction_is_read():
    assert direction_of("what should I do to lower water cut?") == "reduce"


def test_no_direction_is_empty():
    assert direction_of("what should I do?") == ""


def test_direction_is_deterministic_when_two_are_named():
    # "increase" is scanned first, so the same question always yields the same one.
    assert direction_of("should I increase or reduce permeability?") == "increase"


# --- the recommendation ----------------------------------------------------


def test_increase_permeability_is_recommended():
    decision = answer_decision("what should I do to increase permeability?")
    assert decision is not None
    assert decision.selected == "acid_stimulation"
    assert decision.direction == "increase"


def test_options_are_ranked_strongest_first():
    decision = answer_decision("what should I do to increase permeability?")
    assert decision is not None
    confidences = [confidence for _option, _relation, confidence in decision.options]
    assert confidences == sorted(confidences, reverse=True)


def test_only_options_in_the_direction_are_offered():
    decision = answer_decision("what should I do to increase permeability?")
    assert decision is not None
    relations = {relation for _option, relation, _confidence in decision.options}
    assert relations <= DIRECTION_RELATIONS["increase"]
    assert "reduces" not in relations


def test_maintain_direction_uses_its_own_relations():
    decision = answer_decision("what should I do to maintain reservoir pressure?")
    assert decision is not None
    relations = {relation for _option, relation, _confidence in decision.options}
    assert relations <= DIRECTION_RELATIONS["maintain"]


def test_a_direction_the_graph_cannot_serve_yields_none():
    # The graph states no edge that *reduces* permeability, so an honest bridge
    # returns nothing rather than offering an edge that runs the other way.
    assert answer_decision("what should I do to reduce permeability?") is None


def test_a_definitional_question_yields_none():
    assert answer_decision("what is permeability?") is None


def test_an_unrelated_question_yields_none():
    assert answer_decision("what should I do about the weather in Paris?") is None


def test_an_empty_question_yields_none():
    assert answer_decision("") is None


def test_a_bad_limit_is_refused():
    with pytest.raises(ValueError):
        answer_decision("what should I do to increase permeability?", limit=0)


def test_the_recommendation_describes_its_options():
    decision = answer_decision("what should I do to increase permeability?")
    assert decision is not None
    described = decision.describe()
    assert "decision engine" in described
    assert "acid_stimulation" in described


def test_actionable_relations_cover_every_direction():
    for relations in DIRECTION_RELATIONS.values():
        assert relations <= ACTIONABLE_RELATIONS


# --- the reasoning trace carries the recommendation ------------------------


def test_decision_step_sits_after_the_plan():
    assert STEP_ORDER.index(SOURCE_DECISION) == STEP_ORDER.index("The procedure the corpus states (ordered by the planner)") + 1


def test_reasoning_report_includes_a_decision_step():
    decision = answer_decision("what should I do to increase permeability?")
    report = reasoning_report("what should I do to increase permeability?", decision=decision)
    step = next(step for step in report.steps if step.label == SOURCE_DECISION)
    assert step.derived
    assert step.lines
    assert "acid_stimulation" in step.lines[0]


def test_reasoning_report_decision_step_is_empty_without_one():
    report = reasoning_report("what is permeability?")
    step = next(step for step in report.steps if step.label == SOURCE_DECISION)
    assert step.lines == ()


# --- wired into the answer -------------------------------------------------


def test_answer_carries_the_recommendation():
    answer = _assistant().ask("what should I do to increase permeability?")
    assert answer.decision is not None
    assert answer.decision.selected == "acid_stimulation"


def test_answer_carries_no_recommendation_for_a_definitional_question():
    answer = _assistant().ask("what is permeability?")
    assert answer.decision is None


def test_jarvis_turn_carries_the_recommendation(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "what should I do to increase permeability?", speak=False
    )
    assert turn.decision
    assert "acid_stimulation" in turn.decision[0]


def test_format_turn_renders_the_decision_block(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "what should I do to increase permeability?", speak=False
    )
    rendered = format_turn(turn)
    assert "ranked by the decision engine" in rendered


def test_turn_payload_carries_the_recommendation(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "what should I do to increase permeability?", speak=False
    )
    payload = turn_payload(turn)
    assert payload["decision"]
    assert "acid_stimulation" in payload["decision"][0]


# --- the held-out gate -----------------------------------------------------


def test_decision_accuracy_is_perfect_on_the_catalogue():
    assert decision_recommendation_accuracy() == 1.0


def test_decision_accuracy_is_zero_for_an_empty_catalogue():
    assert decision_recommendation_accuracy(()) == 0.0


def test_decision_judgement_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        DecisionJudgement("bad", "guess")


def test_decision_judgement_rejects_an_empty_name():
    with pytest.raises(ValueError):
        DecisionJudgement("", "recommend")


def test_catalogue_has_negative_controls():
    kinds = {judgement.kind for judgement in DECISION_JUDGEMENTS}
    assert "no-decision" in kinds


def test_an_over_firing_bridge_fails_a_negative_control():
    # A bridge that answered every decisional question would fail the
    # "no recommendation" controls, so the catalogue is a real test.
    over_firing = (
        DecisionJudgement("would over-fire", "recommend", question="what is permeability?", expected=("x",)),
    )
    assert decision_recommendation_accuracy(over_firing) == 0.0
