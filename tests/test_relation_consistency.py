"""Tests for reporting contradictory relations instead of hiding them."""

from __future__ import annotations

from core.ontology import Ontology
from core.relation_consistency import (
    RelationConflict,
    describe_relation_conflicts,
    relation_conflicts,
)


def _conflicting_ontology() -> Ontology:
    from core.semantic_relation import SemanticRelation

    ontology = Ontology()
    a = ontology.add_concept_once("alpha")
    b = ontology.add_concept_once("beta")
    c = ontology.add_concept_once("gamma")
    ontology.add_relation(SemanticRelation(a.id, b.id, "increases", 0.9))
    ontology.add_relation(SemanticRelation(a.id, b.id, "reduces", 0.4))
    # An unrelated, consistent edge, so scoping has something to exclude.
    ontology.add_relation(SemanticRelation(b.id, c.id, "causes", 0.7))
    return ontology


def test_conflicting_relations_are_reported():
    conflicts = relation_conflicts(_conflicting_ontology())
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert isinstance(conflict, RelationConflict)
    assert {conflict.relation_type, conflict.opposing_type} == {"increases", "reduces"}


def test_severity_is_the_weaker_confidence():
    conflict = relation_conflicts(_conflicting_ontology())[0]
    assert conflict.severity == 0.4


def test_a_consistent_graph_reports_nothing():
    from core.domain_knowledge import seed_ontology

    ontology = Ontology()
    seed_ontology(ontology)
    assert relation_conflicts(ontology) == ()


def test_a_question_scopes_the_report_to_its_own_concepts():
    ontology = _conflicting_ontology()
    assert relation_conflicts(ontology, "alpha")
    # A question naming only an unrelated concept reports no conflict.
    assert relation_conflicts(ontology, "gamma") == ()
    assert relation_conflicts(ontology, "zzz-unknown") == ()


def test_relation_conflicts_handles_no_ontology():
    assert relation_conflicts(None) == ()
    assert relation_conflicts(None, "porosity") == ()


def test_describe_relation_conflicts_is_empty_for_none():
    assert describe_relation_conflicts(()) == ""
    text = describe_relation_conflicts(relation_conflicts(_conflicting_ontology()))
    assert "contradictory" in text.lower()
    assert "increases" in text and "reduces" in text


def test_assistant_returns_conflicts_as_data():
    from core.assistant import GroundedAssistant
    from core.documents import build_document_store
    from core.rag import RagRetriever

    ontology = _conflicting_ontology()
    retriever = RagRetriever(build_document_store(), ontology=ontology)
    answer = GroundedAssistant(retriever).ask("What does alpha do to beta?")
    assert answer.relation_conflicts
    assert isinstance(answer.relation_conflicts[0], RelationConflict)


def test_consistency_accuracy_is_perfect():
    from core.relation_consistency import consistency_accuracy

    assert consistency_accuracy() == 1.0


def test_consistency_scenarios_are_validated():
    import pytest
    from core.relation_consistency import ConsistencyScenario

    with pytest.raises(ValueError):
        ConsistencyScenario((), -1)
