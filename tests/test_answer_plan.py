"""Tests for the plan bridge (``core.answer_plan``).

The bridge is what finally lets a *procedural* question ("how do I measure
porosity on a core sample?") reach the project's real planner and come back with
the ordered procedure the corpus already states -- Gap 7 of
``docs/REASONING_ASSESSMENT.md``. These tests pin the honesty rules: a plan is
only built for a question that asks *how*, it is the corpus's own wording ordered
by ``Planner.order_steps``, a bad dependency is refused rather than reordered,
and a definitional or unrelated question yields **no** plan.
"""

from __future__ import annotations

import pytest
from ai.chat import format_turn, turn_payload
from core.answer_plan import (
    PLAN_JUDGEMENTS,
    AnswerPlan,
    PlanJudgement,
    answer_plan,
    is_procedural,
    match_entry,
    order_procedure,
    plan_accuracy,
)
from core.assistant import GroundedAssistant
from core.cognitive_controller import CognitiveController
from core.command_router import NaturalLanguageRouter
from core.documents import build_document_store
from core.jarvis import JarvisMode
from core.rag import RagRetriever
from core.reasoning_report import SOURCE_PLAN, STEP_ORDER, reasoning_report
from core.tools import build_sandbox


def _ontology():
    return CognitiveController().ontology


def _assistant() -> GroundedAssistant:
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=_ontology()))


def _jarvis(tmp_path) -> JarvisMode:
    return JarvisMode(_assistant(), router=NaturalLanguageRouter(build_sandbox(tmp_path)))


# --- the procedural cue ----------------------------------------------------


def test_how_do_i_is_procedural():
    assert is_procedural("how do I measure porosity on a core sample?")


def test_how_to_is_procedural():
    assert is_procedural("how to test a well barrier")


def test_arabic_steps_cue_is_procedural():
    assert is_procedural("ما هي خطوات قياس المسامية؟")


def test_definitional_question_is_not_procedural():
    assert not is_procedural("what is porosity?")


def test_plain_statement_is_not_procedural():
    assert not is_procedural("porosity is the fraction of void space")


# --- ordering through the real planner -------------------------------------


def test_linear_procedure_keeps_its_order():
    ordered = order_procedure(
        (("A", ()), ("B", ("A",)), ("C", ("B",))), goal_description="g"
    )
    assert ordered == ("A", "B", "C")


def test_out_of_order_steps_are_ordered():
    ordered = order_procedure(
        (("C", ("B",)), ("A", ()), ("B", ("A",))), goal_description="g"
    )
    assert ordered == ("A", "B", "C")


def test_plain_step_texts_chain_on_the_previous_step():
    ordered = order_procedure(("first", "second", "third"), goal_description="g")
    assert ordered == ("first", "second", "third")


def test_circular_dependency_is_refused():
    with pytest.raises(ValueError):
        order_procedure((("A", ("B",)), ("B", ("A",))), goal_description="g")


def test_unknown_dependency_is_refused():
    with pytest.raises(ValueError):
        order_procedure((("A", ("ghost",)),), goal_description="g")


# --- matching a question to a corpus entry ---------------------------------


def test_match_entry_finds_the_core_analysis_entry():
    entry = match_entry("how do I measure porosity and permeability on a core sample?")
    assert entry is not None
    assert entry.key == "reservoir_rock_properties"


def test_match_entry_is_none_for_an_unrelated_question():
    assert match_entry("what is the capital of France?") is None


def test_match_entry_is_none_for_an_empty_question():
    assert match_entry("   ") is None


# --- the bridge itself -----------------------------------------------------


def test_answer_plan_returns_the_entrys_own_steps():
    plan = answer_plan("how do I identify the drive mechanism?")
    assert isinstance(plan, AnswerPlan)
    assert plan.source == "reservoir_drive_mechanisms"
    assert plan.steps == (
        "Identify the drive from pressure and production trends.",
        "Estimate the recovery factor for that drive.",
    )


def test_answer_plan_returns_none_for_a_definitional_question():
    assert answer_plan("what is porosity?") is None


def test_answer_plan_returns_none_for_an_unrelated_question():
    assert answer_plan("what is the capital of France?") is None


def test_answer_plan_describes_its_steps_numbered():
    plan = answer_plan("how do I measure porosity on a core sample?")
    assert plan is not None
    described = plan.describe()
    assert "1." in described and "2." in described
    assert plan.topic in described


# --- the reasoning trace carries the plan ----------------------------------


def test_plan_step_sits_between_stored_and_inferred():
    assert STEP_ORDER.index(SOURCE_PLAN) == STEP_ORDER.index("Stored in the knowledge graph") + 1


def test_reasoning_report_includes_a_plan_step():
    plan = answer_plan("how do I identify the drive mechanism?")
    report = reasoning_report("how do I identify the drive mechanism?", plan=plan)
    step = next(step for step in report.steps if step.label == SOURCE_PLAN)
    assert step.derived
    assert step.lines
    assert step.lines[0].startswith("1.")


def test_reasoning_report_plan_step_is_empty_without_a_plan():
    report = reasoning_report("what is porosity?")
    step = next(step for step in report.steps if step.label == SOURCE_PLAN)
    assert step.lines == ()


# --- wired into the answer -------------------------------------------------


def test_answer_carries_the_plan_for_a_procedural_question():
    answer = _assistant().ask("how do I measure porosity and permeability on a core sample?")
    assert answer.plan is not None
    assert answer.plan.source == "reservoir_rock_properties"


def test_answer_carries_no_plan_for_a_definitional_question():
    answer = _assistant().ask("what is porosity?")
    assert answer.plan is None


def test_jarvis_turn_carries_the_plan(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "how do I measure porosity on a core sample?", speak=False
    )
    assert turn.plan
    assert turn.plan[0].startswith("1.")


def test_format_turn_renders_the_plan_block(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "how do I measure porosity on a core sample?", speak=False
    )
    rendered = format_turn(turn)
    assert "ordered by the planner" in rendered


def test_turn_payload_carries_the_plan(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "how do I measure porosity on a core sample?", speak=False
    )
    payload = turn_payload(turn)
    assert payload["plan"]
    assert payload["plan"][0].startswith("1.")


# --- the held-out gate -----------------------------------------------------


def test_plan_accuracy_is_perfect_on_the_catalogue():
    assert plan_accuracy() == 1.0


def test_plan_accuracy_is_zero_for_an_empty_catalogue():
    assert plan_accuracy(()) == 0.0


def test_plan_judgement_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        PlanJudgement("bad", "guess")


def test_plan_judgement_rejects_an_empty_name():
    with pytest.raises(ValueError):
        PlanJudgement("", "order")


def test_catalogue_has_negative_controls():
    kinds = {judgement.kind for judgement in PLAN_JUDGEMENTS}
    assert "no-plan" in kinds
    assert "reject" in kinds
