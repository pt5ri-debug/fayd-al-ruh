"""Tests for the belief bridge (``core.answer_belief``).

The bridge is what finally lets an *evaluative* question ("does acid stimulation
increase permeability?") reach the project's real belief layer and come back with
the evidence-weighed verdict the knowledge graph already supports -- the belief
half of Gap 7 of ``docs/REASONING_ASSESSMENT.md``. These tests pin the honesty
rules: a belief is only built for a question that asks *whether a claim holds*, it
rests only on the graph's own edges between the concepts the question names, its
confidence and status are the ones ``BeliefRevisionEngine`` produced, and a
definitional question, a plain statement, or an evaluative question the graph
cannot speak to yields **no** belief.
"""

from __future__ import annotations

import pytest
from ai.chat import format_turn, turn_payload
from core.answer_belief import (
    BELIEF_JUDGEMENTS,
    GRAPH_SOURCE,
    BeliefJudgement,
    answer_belief,
    belief_accuracy,
    is_evaluative,
)
from core.assistant import GroundedAssistant
from core.belief import Belief
from core.belief_revision import BeliefRevisionEngine
from core.cognitive_controller import CognitiveController
from core.command_router import NaturalLanguageRouter
from core.documents import build_document_store
from core.evidence import Evidence
from core.jarvis import JarvisMode
from core.rag import RagRetriever
from core.reasoning_report import SOURCE_BELIEF, STEP_ORDER, reasoning_report
from core.tools import build_sandbox


def _ontology():
    return CognitiveController().ontology


def _assistant() -> GroundedAssistant:
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=_ontology()))


def _jarvis(tmp_path) -> JarvisMode:
    return JarvisMode(_assistant(), router=NaturalLanguageRouter(build_sandbox(tmp_path)))


# --- the evaluative cue ----------------------------------------------------


def test_a_does_question_is_evaluative():
    assert is_evaluative("does acid stimulation increase permeability?")


def test_an_is_question_is_evaluative():
    assert is_evaluative("is permeability high?")


def test_a_verdict_phrase_is_evaluative():
    assert is_evaluative("is it true that acid stimulation increases permeability?")


def test_arabic_verdict_cue_is_evaluative():
    assert is_evaluative("هل يزيد الحمض النفاذية؟")


def test_a_definitional_question_is_not_evaluative():
    assert not is_evaluative("what is permeability?")


def test_a_statement_is_not_evaluative():
    assert not is_evaluative("permeability measures flow.")


def test_a_question_merely_containing_is_is_not_evaluative():
    # The cue is anchored to the opening, so an interior "is" never fires.
    assert not is_evaluative("what is the knowledge graph?")


def test_an_empty_question_is_not_evaluative():
    assert not is_evaluative("   ")


# --- the belief itself -----------------------------------------------------


def test_a_supported_claim_reaches_a_verdict():
    belief = answer_belief("does acid stimulation increase permeability?")
    assert belief is not None
    assert belief.status in ("provisional", "accepted")
    assert 0.0 <= belief.confidence <= 1.0


def test_the_belief_rests_on_the_graphs_own_edges():
    belief = answer_belief("does acid stimulation increase permeability?")
    assert belief is not None
    assert any(
        line.startswith("acid_stimulation —increases→ permeability")
        for line in belief.supporting
    )


def test_the_belief_confidence_is_the_engines_own():
    # The same edges run through the real engine must give the same confidence.
    belief = answer_belief("does acid stimulation increase permeability?")
    assert belief is not None
    engine = BeliefRevisionEngine()
    probe = Belief(statement="probe", confidence=0.5)
    expected = engine.revise(
        probe,
        [
            Evidence(
                "acid_stimulation —increases→ permeability",
                GRAPH_SOURCE,
                supports=True,
                strength=0.7,
                data={"relation_type": "increases"},
            ),
            Evidence(
                "acid_stimulation —restores→ permeability",
                GRAPH_SOURCE,
                supports=True,
                strength=0.7,
                data={"relation_type": "restores"},
            ),
        ],
    )
    assert belief.confidence == pytest.approx(expected.revised.confidence)


def test_the_verdict_describes_its_evidence():
    belief = answer_belief("does acid stimulation increase permeability?")
    assert belief is not None
    described = belief.describe()
    assert "confidence" in described
    assert "acid_stimulation —increases→ permeability" in described


def test_a_definitional_question_yields_none():
    assert answer_belief("what is permeability?") is None


def test_a_statement_yields_none():
    assert answer_belief("permeability measures flow.") is None


def test_a_question_naming_one_concept_yields_none():
    assert answer_belief("is permeability high?") is None


def test_a_question_the_graph_cannot_speak_to_yields_none():
    assert answer_belief("does porosity increase water cut?") is None


def test_an_empty_question_yields_none():
    assert answer_belief("") is None


def test_a_bad_prior_is_refused():
    with pytest.raises(ValueError):
        answer_belief("does acid stimulation increase permeability?", prior=1.5)


def test_the_bridge_does_not_mutate_the_graph():
    ontology = _ontology()
    before = len(ontology.relations)
    answer_belief("does acid stimulation increase permeability?", ontology=ontology)
    assert len(ontology.relations) == before


# --- the trace step --------------------------------------------------------


def test_belief_step_sits_after_the_decision():
    assert STEP_ORDER.index(SOURCE_BELIEF) > STEP_ORDER.index(
        "The options the graph states (ranked by the decision engine)"
    )


def test_reasoning_report_includes_a_belief_step():
    belief = answer_belief("does acid stimulation increase permeability?")
    report = reasoning_report("q", belief=belief)
    step = next(step for step in report.steps if step.label == SOURCE_BELIEF)
    assert step.lines
    assert step.derived


def test_reasoning_report_belief_step_is_empty_without_one():
    report = reasoning_report("q")
    step = next(step for step in report.steps if step.label == SOURCE_BELIEF)
    assert step.lines == ()


# --- the answer path -------------------------------------------------------


def test_answer_carries_the_belief():
    answer = _assistant().ask("does acid stimulation increase permeability?")
    assert answer.belief is not None
    assert answer.belief.statement.startswith("does acid stimulation")


def test_answer_carries_no_belief_for_a_definitional_question():
    answer = _assistant().ask("what is permeability?")
    assert answer.belief is None


# --- the Jarvis / chat surfaces -------------------------------------------


def test_jarvis_turn_carries_the_belief(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "does acid stimulation increase permeability?", speak=False
    )
    assert turn.belief
    assert any("confidence" in line for line in turn.belief)


def test_format_turn_renders_the_belief_block(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "does acid stimulation increase permeability?", speak=False
    )
    rendered = format_turn(turn)
    assert "The evidence-weighed belief the graph supports:" in rendered


def test_turn_payload_carries_the_belief(tmp_path):
    turn = _jarvis(tmp_path).handle(
        "does acid stimulation increase permeability?", speak=False
    )
    payload = turn_payload(turn)
    assert payload["belief"]
    assert any("confidence" in line for line in payload["belief"])


# --- the held-out gate -----------------------------------------------------


def test_belief_accuracy_is_perfect_on_the_catalogue():
    assert belief_accuracy() == 1.0


def test_belief_accuracy_is_zero_for_an_empty_catalogue():
    assert belief_accuracy(()) == 0.0


def test_belief_judgement_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        BeliefJudgement("bad", "guess")


def test_belief_judgement_rejects_an_empty_name():
    with pytest.raises(ValueError):
        BeliefJudgement("", "no-belief")


def test_catalogue_has_negative_controls():
    kinds = {judgement.kind for judgement in BELIEF_JUDGEMENTS}
    assert "no-belief" in kinds


def test_an_over_firing_bridge_fails_a_negative_control():
    # A bridge that answered every evaluative question would fail the
    # "no belief" controls, so the catalogue is a real test.
    over_firing = (
        BeliefJudgement(
            "would over-fire",
            "status",
            question="what is permeability?",
            expected=("provisional",),
        ),
    )
    assert belief_accuracy(over_firing) == 0.0
