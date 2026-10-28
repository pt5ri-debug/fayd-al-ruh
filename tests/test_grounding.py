"""Tests for the stronger groundedness gate (``core.grounding``)."""

from __future__ import annotations

import pytest
from core.assistant import GroundedAssistant
from core.cognitive_controller import CognitiveController
from core.grounding import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_SENTENCE_SUPPORT,
    GROUNDING_JUDGEMENTS,
    GroundingCheck,
    GroundingJudgement,
    _causal_edges,
    _causal_reversals,
    _endpoint_spans,
    _find_marker,
    _label_key,
    _sentences,
    check_grounding,
    grounding_accuracy,
)
from core.model_interface import ModelResponse
from core.ontology import Ontology
from core.rag import RagRetriever, RetrievedPassage


def _passage(text: str, title: str = "Evidence") -> RetrievedPassage:
    return RetrievedPassage(
        doc_id="d1", title=title, text=text, source="src", score=0.9, signal="lexical"
    )


def _ontology() -> Ontology:
    return CognitiveController().ontology


# --- the check object ------------------------------------------------------


def test_grounding_check_rejects_out_of_range_coverage():
    with pytest.raises(ValueError):
        GroundingCheck(True, 1.5)


def test_grounding_check_describe_names_a_pass():
    assert "grounded" in check_grounding(
        "Permeability is a property of the formation.",
        [_passage("Permeability is a property of the formation.")],
    ).describe()


def test_grounding_check_describe_names_a_failure():
    check = check_grounding("Prices fell.", [_passage("The rig drills the hole.")])
    assert "ungrounded" in check.describe()
    assert check.reasons


# --- term coverage ---------------------------------------------------------


def test_grounded_answer_passes():
    check = check_grounding(
        "Permeability is a property of the formation.",
        [_passage("Permeability is a property of the formation that measures flow.")],
    )
    assert check.grounded is True
    assert check.coverage >= DEFAULT_MIN_COVERAGE


def test_one_shared_term_is_not_enough_for_a_long_answer():
    # The original gate accepted this; the stronger one must not.
    text = (
        "The drilling rig is expensive, the market is volatile, and the weather "
        "delays every operation, yet management approves the budget anyway."
    )
    check = check_grounding(text, [_passage("The drilling rig drills the hole.")])
    assert check.grounded is False
    assert check.coverage < DEFAULT_MIN_COVERAGE


def test_no_shared_term_fails():
    check = check_grounding("Prices fell sharply.", [_passage("The rig drills the hole.")])
    assert check.grounded is False


def test_no_evidence_fails_closed():
    check = check_grounding("Permeability is a property.", [])
    assert check.grounded is False
    assert "no evidence" in check.reasons[0]


def test_answer_with_no_content_terms_fails():
    check = check_grounding("the and of", [_passage("Permeability measures flow.")])
    assert check.grounded is False


# --- per-sentence support --------------------------------------------------


def test_supported_sentences_pass():
    text = "Reservoir pressure drives production. The well produces oil and gas."
    check = check_grounding(
        text, [_passage("Reservoir pressure drives production of oil and gas from the well.")]
    )
    assert check.grounded is True
    assert check.unsupported == ()


def test_majority_unsupported_sentences_fail():
    text = (
        "The drilling rig drills the hole. Porosity is a measure of void space. "
        "The market fell sharply last quarter."
    )
    check = check_grounding(text, [_passage("The drilling rig drills the hole into the formation.")])
    assert check.grounded is False
    assert len(check.unsupported) == 2
    assert any("sentences share no term" in reason for reason in check.reasons)


def test_one_unsupported_sentence_of_two_is_tolerated():
    # Half is the threshold: a single unsupported clause in two does not fail.
    text = "The drilling rig drills the hole. Prices fell sharply last quarter."
    check = check_grounding(text, [_passage("The drilling rig drills the hole into the formation.")])
    assert 1 - len(check.unsupported) / 2 >= DEFAULT_MIN_SENTENCE_SUPPORT


# --- causal direction ------------------------------------------------------


def test_label_key_turns_identifier_into_phrase():
    assert _label_key("reservoir_pressure") == "reservoir pressure"


def test_causal_edges_hold_only_causal_pairs():
    edges, endpoints = _causal_edges(_ontology())
    assert ("reservoir pressure", "production") in edges
    # An association is never an endpoint of a causal edge.
    assert "measures" not in endpoints


def test_causal_edges_none_is_empty():
    assert _causal_edges(None) == (set(), set())


def test_forward_mechanism_is_not_reported():
    assert _causal_reversals("Reservoir pressure drives production.", _ontology()) == ()


def test_reversed_mechanism_is_reported():
    reversals = _causal_reversals("Production drives reservoir pressure.", _ontology())
    assert len(reversals) == 1
    assert "reservoir pressure → production" in reversals[0]


def test_reverse_marker_is_read_backward():
    # "X is caused by Y" is the forward edge Y -> X, so it is not a reversal.
    assert _causal_reversals("Production is caused by reservoir pressure.", _ontology()) == ()
    # And the reverse of that sentence *is* a reversal.
    assert _causal_reversals("Reservoir pressure is caused by production.", _ontology())


def test_association_is_not_read_as_a_reversal():
    assert _causal_reversals("Permeability measures flow.", _ontology()) == ()


def test_causal_reversals_none_ontology_is_empty():
    assert _causal_reversals("Production drives reservoir pressure.", None) == ()


def test_reversal_fails_the_gate():
    check = check_grounding(
        "Production drives reservoir pressure.",
        [_passage("Reservoir pressure drives production in the field.")],
        ontology=_ontology(),
    )
    assert check.grounded is False
    assert check.reversals
    assert any("reversed" in reason for reason in check.reasons)


def test_forward_mechanism_passes_the_gate():
    check = check_grounding(
        "Reservoir pressure drives production.",
        [_passage("Reservoir pressure drives production in the field.")],
        ontology=_ontology(),
    )
    assert check.grounded is True


# --- helpers ---------------------------------------------------------------


def test_sentences_splits_and_drops_empties():
    assert _sentences("One. Two!  \n Three? ") == ["One.", "Two!", "Three?"]


def test_find_marker_matches_on_word_boundaries():
    # "controls" must not fire inside "controllers".
    assert _find_marker("the controllers run", ("controls",)) is None
    assert _find_marker("it controls flow", ("controls",)) is not None


def test_endpoint_spans_drops_nested_matches():
    spans = _endpoint_spans("reservoir pressure", ["reservoir pressure", "pressure"])
    labels = [label for _, _, label in spans]
    assert labels == ["reservoir pressure"]


def test_endpoint_spans_is_position_ordered():
    spans = _endpoint_spans("production follows reservoir pressure", ["production", "reservoir pressure"])
    assert spans[0][2] == "production"


# --- the held-out gate -----------------------------------------------------


def test_gate_catalogue_has_negative_controls():
    assert any(not j.expected for j in GROUNDING_JUDGEMENTS)
    assert any(j.expected for j in GROUNDING_JUDGEMENTS)
    assert any(j.ontology for j in GROUNDING_JUDGEMENTS)


def test_gate_is_perfect_on_the_catalogue():
    assert grounding_accuracy(ontology=_ontology()) == 1.0


def test_gate_empty_catalogue_is_zero():
    assert grounding_accuracy(ontology=_ontology(), judgements=()) == 0.0


def test_gate_catches_a_manufactured_pass():
    # A catalogue whose expectation the gate must *fail* proves it is measured.
    judgement = GroundingJudgement(
        "false positive control",
        "The rig is expensive and the market is volatile.",
        "The rig drills the hole.",
        expected=True,  # deliberately wrong: the gate must reject it
    )
    assert grounding_accuracy(ontology=_ontology(), judgements=(judgement,)) == 0.0


# --- assistant wiring ------------------------------------------------------


def _retriever() -> RagRetriever:
    from core.documents import build_document_store

    return RagRetriever(build_document_store(), ontology=_ontology())


def test_assistant_rejects_a_reversed_mechanism():
    class _Reversing:
        def generate(self, request):
            return ModelResponse(
                text="Production drives reservoir pressure.",
                confidence=0.9,
                model_name="bad",
            )

    answer = GroundedAssistant(_retriever(), model=_Reversing()).ask("reservoir pressure")
    assert answer.used_model is False


def test_assistant_keeps_a_grounded_model_answer():
    class _Grounded:
        def generate(self, request):
            return ModelResponse(
                text="Permeability is a property of the formation.",
                confidence=0.9,
                model_name="stub",
            )

    answer = GroundedAssistant(_retriever(), model=_Grounded()).ask("permeability")
    assert answer.used_model is True


def test_assistant_rejects_a_paragraph_riding_one_word():
    class _Padding:
        def generate(self, request):
            return ModelResponse(
                text=(
                    "The drilling rig is expensive, the market is volatile, and the "
                    "weather delays every operation, yet management approves anyway."
                ),
                confidence=0.9,
                model_name="bad",
            )

    answer = GroundedAssistant(_retriever(), model=_Padding()).ask("drilling rig")
    assert answer.used_model is False
