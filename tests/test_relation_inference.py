"""Tests for surfacing the graph's *inferred* relations as evidence."""

from __future__ import annotations

import pytest
from core.ontology import Ontology
from core.relation_inference import (
    INFERENCE_JUDGEMENTS,
    InferenceJudgement,
    InferredRelation,
    describe_inferred_relations,
    inference_accuracy,
    inferred_relations,
)


def _ontology() -> Ontology:
    from core.domain_knowledge import seed_ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_petroleum_corpus(ontology)
    return ontology


def test_inferred_relations_are_returned_for_a_known_question():
    inferences = inferred_relations(_ontology(), "What does a well penetrate?")
    assert inferences
    assert all(isinstance(item, InferredRelation) for item in inferences)
    assert all(item.source and item.target for item in inferences)


def test_inferred_relations_carry_their_steps():
    # A derivation must say which stored relations it rests on, so a reader can
    # audit the inference rather than take it on faith.
    inferences = inferred_relations(_ontology(), "What does a well penetrate?", limit=10)
    assert inferences
    assert all(item.steps for item in inferences)
    assert any("penetrates" in item.steps[0] for item in inferences)


def test_inferred_relations_are_empty_for_an_unknown_question():
    assert inferred_relations(_ontology(), "zzz-not-a-concept") == ()
    assert inferred_relations(None, "porosity") == ()
    assert inferred_relations(_ontology(), "   ") == ()


def test_inferred_relations_are_deterministic():
    ontology = _ontology()
    first = inferred_relations(ontology, "formation", limit=10)
    second = inferred_relations(ontology, "formation", limit=10)
    assert first == second


def test_inferred_relations_respect_the_limit():
    assert len(inferred_relations(_ontology(), "formation", limit=1)) <= 1
    with pytest.raises(ValueError):
        inferred_relations(_ontology(), "formation", limit=0)


def test_inferred_relations_respect_the_confidence_floor():
    strong = inferred_relations(_ontology(), "formation", limit=10, min_confidence=0.5)
    assert all(item.confidence >= 0.5 for item in strong)
    with pytest.raises(ValueError):
        inferred_relations(_ontology(), "formation", min_confidence=2.0)


def test_inferred_relations_never_relabel_an_explicit_relation():
    # "porosity increases permeability" is stored; it must not appear as an
    # inference, or a reader could not tell a written claim from a derived one.
    inferences = inferred_relations(_ontology(), "porosity permeability", limit=20)
    assert not any(
        item.source.lower() == "porosity"
        and item.target.lower() == "permeability"
        and item.relation_type == "increases"
        for item in inferences
    )


def test_describe_inferred_relations_labels_the_block():
    inferences = inferred_relations(_ontology(), "What does a well penetrate?")
    block = describe_inferred_relations(inferences)
    assert "Inferred" in block and "not stored" in block
    assert "—penetrates→" in block


def test_describe_inferred_relations_is_empty_when_there_is_nothing():
    assert describe_inferred_relations(()) == ""


def test_explain_includes_the_chain_when_present():
    item = InferredRelation("a", "increases", "b", 0.5, ("a —x→ c",))
    assert "from a —x→ c" in item.explain()
    bare = InferredRelation("a", "increases", "b", 0.5, ())
    assert bare.explain() == bare.describe()


def test_inference_judgement_rejects_bad_input():
    with pytest.raises(ValueError):
        InferenceJudgement("", "b", "increases", True)
    with pytest.raises(ValueError):
        InferenceJudgement("a", "b", "", True)


def test_inference_accuracy_is_perfect_on_the_seeded_graph():
    assert inference_accuracy(_ontology()) == 1.0


def test_inference_accuracy_is_zero_for_an_empty_catalogue():
    assert inference_accuracy(_ontology(), judgements=()) == 0.0


def test_inference_accuracy_counts_a_missed_derivation_as_wrong():
    wrong = (InferenceJudgement("porosity", "permeability", "has_property", True),)
    assert inference_accuracy(_ontology(), wrong) == 0.0


def test_inference_judgements_carry_negative_controls():
    assert any(not judgement.expected for judgement in INFERENCE_JUDGEMENTS)


def test_assistant_answer_exposes_inferred_relations():
    from core.assistant import GroundedAssistant
    from core.documents import build_document_store
    from core.rag import RagRetriever

    ontology = _ontology()
    retriever = RagRetriever(build_document_store(), ontology=ontology)
    assistant = GroundedAssistant(retriever)
    answer = assistant.ask("What does a well penetrate?")
    assert answer.inferred
    assert all(isinstance(item, InferredRelation) for item in answer.inferred)


def test_jarvis_turn_carries_inferred_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    turn = mode.handle("What does a well penetrate?", speak=False)
    assert turn.routed == "answer"
    assert turn.inferred
    assert any("—penetrates→" in line for line in turn.inferred)


def test_format_turn_renders_the_inference_block():
    from ai.chat import format_turn
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(
        heard="q",
        response="an answer",
        woke=False,
        routed="answer",
        inferred=("well —penetrates→ formation (from well —penetrates→ reservoir reservoir —contained_in→ formation)",),
    )
    text = format_turn(turn)
    assert "Inferred from the knowledge graph (derived, not stored):" in text
    assert "well —penetrates→ formation" in text


def test_turn_payload_carries_inferred():
    from ai.chat import turn_payload
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(heard="q", response="a", woke=False, routed="answer", inferred=("x —y→ z",))
    assert turn_payload(turn)["inferred"] == ["x —y→ z"]
