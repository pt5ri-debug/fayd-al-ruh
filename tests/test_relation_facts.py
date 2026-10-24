"""Tests for surfacing the knowledge graph's typed relations as evidence."""

from __future__ import annotations

import pytest
from core.ontology import Ontology
from core.relation_facts import (
    RelationFact,
    describe_relation_facts,
    relation_facts,
)


def _ontology() -> Ontology:
    from core.domain_knowledge import seed_ontology

    ontology = Ontology()
    seed_ontology(ontology)
    return ontology


def test_relation_facts_are_returned_for_a_known_concept():
    facts = relation_facts(_ontology(), "porosity")
    assert facts
    assert all(isinstance(fact, RelationFact) for fact in facts)
    assert all(fact.source and fact.target for fact in facts)


def test_relation_facts_reach_an_incoming_relation():
    # "density log measures porosity": asked about porosity, the incoming link
    # must be reachable, not only the outgoing ones.
    facts = relation_facts(_ontology(), "porosity", limit=50)
    assert any(fact.target.lower() == "porosity" for fact in facts)


def test_relation_facts_are_empty_for_an_unknown_question():
    assert relation_facts(_ontology(), "zzz-not-a-concept") == ()
    assert relation_facts(None, "porosity") == ()


def test_relation_facts_are_deterministic():
    ontology = _ontology()
    first = relation_facts(ontology, "permeability", limit=10)
    second = relation_facts(ontology, "permeability", limit=10)
    assert first == second


def test_relation_facts_respect_the_limit():
    facts = relation_facts(_ontology(), "porosity", limit=2)
    assert len(facts) <= 2
    with pytest.raises(ValueError):
        relation_facts(_ontology(), "porosity", limit=0)


def test_relation_facts_are_deduplicated_by_triple():
    facts = relation_facts(_ontology(), "porosity", limit=50)
    triples = [(fact.source, fact.relation_type, fact.target) for fact in facts]
    assert len(triples) == len(set(triples))


def test_relation_fact_statement_reads_underscores_as_spaces():
    fact = RelationFact("a", "depends_on", "b", 0.5)
    assert fact.statement == "a depends on b"
    assert fact.describe() == "a —depends_on→ b"


def test_describe_relation_facts_is_empty_for_no_facts():
    assert describe_relation_facts(()) == ""
    text = describe_relation_facts([RelationFact("a", "measures", "b", 0.5)])
    assert "knowledge graph" in text.lower()
    assert "a —measures→ b" in text


def test_assistant_returns_relation_facts_as_data():
    from core.assistant import GroundedAssistant
    from core.curriculum_knowledge import seed_curriculum
    from core.documents import build_document_store
    from core.rag import RagRetriever

    ontology = _ontology()
    seed_curriculum(ontology)
    retriever = RagRetriever(build_document_store(), ontology=ontology)
    answer = GroundedAssistant(retriever).ask("What is the relationship between porosity and permeability?")
    assert answer.relation_facts
    assert all(isinstance(fact, RelationFact) for fact in answer.relation_facts)
