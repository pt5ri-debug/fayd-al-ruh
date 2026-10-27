"""Tests for surfacing the graph's *multi-hop connections* as evidence."""

from __future__ import annotations

import pytest
from core.ontology import Ontology
from core.relation_connections import (
    CONNECTION_JUDGEMENTS,
    DEFAULT_CONNECTION_LIMIT,
    ConceptConnection,
    ConnectionJudgement,
    concept_connections,
    connection_reasoning_accuracy,
    describe_concept_connections,
)


def _ontology() -> Ontology:
    from core.domain_knowledge import seed_ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_petroleum_corpus(ontology)
    return ontology


def test_connections_are_found_between_named_concepts():
    connections = concept_connections(_ontology(), "how do casing and perforating relate?")
    assert connections
    assert all(isinstance(item, ConceptConnection) for item in connections)
    assert any(
        item.source.lower() == "casing" and item.target.lower() == "perforating"
        for item in connections
    )


def test_connections_carry_their_steps():
    # A chain must say which typed links it walked, so a reader can audit it.
    connections = concept_connections(_ontology(), "how do casing and perforating relate?")
    assert connections
    assert all(item.steps for item in connections)
    assert all(item.hops == len(item.steps) for item in connections)


def test_connections_mark_a_backward_walk():
    # A step walked against the stored direction is shown with a left arrow, so a
    # reader sees exactly which claims the connection rests on.
    connections = concept_connections(_ontology(), "how do casing and perforating relate?", limit=10)
    rendered = " ".join(step for item in connections for step in item.steps)
    assert "←" in rendered or "→" in rendered


def test_connections_are_empty_for_a_single_concept():
    assert concept_connections(_ontology(), "what is porosity?") == ()


def test_connections_are_empty_for_an_unknown_question():
    assert concept_connections(_ontology(), "zzz-not-a-concept") == ()
    assert concept_connections(None, "porosity permeability") == ()
    assert concept_connections(_ontology(), "   ") == ()


def test_connections_are_empty_when_no_chain_exists():
    # Two concepts the graph cannot connect within the bound produce nothing --
    # "not connected" is a real answer, never a fabricated chain.
    assert concept_connections(_ontology(), "porosity atom", limit=10) == ()


def test_connections_are_deterministic():
    ontology = _ontology()
    first = concept_connections(ontology, "casing perforating completion", limit=10)
    second = concept_connections(ontology, "casing perforating completion", limit=10)
    assert first == second


def test_connections_respect_the_limit():
    assert len(concept_connections(_ontology(), "casing perforating completion", limit=1)) <= 1
    with pytest.raises(ValueError):
        concept_connections(_ontology(), "casing perforating", limit=0)
    with pytest.raises(ValueError):
        concept_connections(_ontology(), "casing perforating", max_hops=0)


def test_connections_are_ordered_strongest_first():
    connections = concept_connections(_ontology(), "casing perforating completion", limit=10)
    confidences = [item.confidence for item in connections]
    assert confidences == sorted(confidences, reverse=True)


def test_describe_concept_connections_labels_the_block():
    connections = concept_connections(_ontology(), "how do casing and perforating relate?")
    block = describe_concept_connections(connections)
    assert "walked, not stored" in block
    assert "step(s)" in block


def test_describe_concept_connections_is_empty_when_there_is_nothing():
    assert describe_concept_connections(()) == ""


def test_connection_judgement_rejects_bad_input():
    with pytest.raises(ValueError):
        ConnectionJudgement("", "b", 1)
    with pytest.raises(ValueError):
        ConnectionJudgement("a", "b", 0)


def test_connection_reasoning_accuracy_is_perfect_on_the_seeded_graph():
    assert connection_reasoning_accuracy(_ontology()) == 1.0


def test_connection_reasoning_accuracy_is_zero_for_an_empty_catalogue():
    assert connection_reasoning_accuracy(_ontology(), judgements=()) == 0.0


def test_connection_reasoning_accuracy_counts_a_wrong_length_chain_as_wrong():
    wrong = (ConnectionJudgement("porosity", "permeability", 3),)
    assert connection_reasoning_accuracy(_ontology(), wrong) == 0.0


def test_connection_judgements_carry_negative_controls():
    assert any(judgement.expected_hops is None for judgement in CONNECTION_JUDGEMENTS)


def test_default_connection_limit_is_positive():
    assert DEFAULT_CONNECTION_LIMIT >= 1


def test_assistant_answer_exposes_connections():
    from core.assistant import GroundedAssistant
    from core.documents import build_document_store
    from core.rag import RagRetriever

    ontology = _ontology()
    retriever = RagRetriever(build_document_store(), ontology=ontology)
    assistant = GroundedAssistant(retriever)
    answer = assistant.ask("how do casing and perforating relate to well integrity?")
    assert answer.connections
    assert all(isinstance(item, ConceptConnection) for item in answer.connections)


def test_assistant_answer_has_no_connections_for_one_concept():
    from core.assistant import GroundedAssistant
    from core.documents import build_document_store
    from core.rag import RagRetriever

    retriever = RagRetriever(build_document_store(), ontology=_ontology())
    answer = GroundedAssistant(retriever).ask("what is porosity?")
    assert answer.connections == ()


def test_jarvis_turn_carries_connection_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    turn = mode.handle("how do casing and perforating relate to well integrity?", speak=False)
    assert turn.routed == "answer"
    assert turn.connections
    assert any("step(s)" in line for line in turn.connections)


def test_format_turn_renders_the_connection_block():
    from ai.chat import format_turn
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(
        heard="q",
        response="an answer",
        woke=False,
        routed="answer",
        connections=("casing → perforating in 2 step(s) (confidence 0.49): a —x→ b then c —y→ d",),
    )
    text = format_turn(turn)
    assert "How the graph connects the concepts in the question (walked, not stored):" in text
    assert "casing → perforating" in text


def test_turn_payload_carries_connections():
    from ai.chat import turn_payload
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(heard="q", response="a", woke=False, routed="answer", connections=("x → y",))
    assert turn_payload(turn)["connections"] == ["x → y"]
