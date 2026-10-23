from __future__ import annotations

import pytest
from core.concept import Concept
from core.domain_knowledge import seed_ontology
from core.ontology import Ontology
from core.ontology_reasoner import OntologyInference, OntologyReasoner, RelationRule
from core.semantic_relation import SemanticRelation


def _seeded() -> Ontology:
    ontology = Ontology()
    seed_ontology(ontology)
    return ontology


def _id(ontology: Ontology, label: str) -> str:
    return ontology.find_concepts(label)[0].id


def test_chain_rule_derives_transitive_containment():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    inference = reasoner.infer(
        ontology,
        _id(ontology, "well"),
        _id(ontology, "formation"),
        "penetrates",
    )
    assert inference is not None
    assert inference.derived is True
    assert inference.steps
    assert 0.0 < inference.confidence < 1.0


def test_shared_target_rule_derives_tool_over_property_host():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    inference = reasoner.infer(
        ontology,
        _id(ontology, "well_log"),
        _id(ontology, "reservoir"),
        "measures",
    )
    assert inference is not None
    assert inference.derived is True
    assert len(inference.steps) == 3


def test_explicit_relation_is_returned_as_not_derived():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    inference = reasoner.infer(
        ontology,
        _id(ontology, "reservoir"),
        _id(ontology, "formation"),
        "contained_in",
    )
    assert inference is not None
    assert inference.derived is False
    assert inference.steps[0].relation_type == "contained_in"


def test_derived_relations_are_not_stored_in_the_ontology():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    derived = reasoner.derive(ontology)
    assert derived
    stored = {
        (relation.source_id, relation.target_id, relation.relation_type)
        for relation in ontology.relations.values()
    }
    for inference in derived:
        key = (inference.source_id, inference.target_id, inference.relation_type)
        assert key not in stored


def test_closure_is_idempotent():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    first = reasoner.derive(ontology)
    second = reasoner.derive(ontology)
    assert first == second


def test_unknown_relation_returns_none():
    ontology = _seeded()
    reasoner = OntologyReasoner()
    assert reasoner.infer(
        ontology,
        _id(ontology, "water"),
        _id(ontology, "pressure"),
        "measures",
    ) is None


def test_rule_rejects_unknown_kind():
    with pytest.raises(ValueError):
        RelationRule("bad", "sideways", "a", "b", "c")


def test_derive_rejects_non_positive_iterations():
    with pytest.raises(ValueError):
        OntologyReasoner().derive(_seeded(), max_iterations=0)


def test_confidence_is_propagated_multiplicatively():
    ontology = Ontology()
    a = Concept("A")
    b = Concept("B")
    c = Concept("C")
    for concept in (a, b, c):
        ontology.add_concept(concept)
    ontology.add_relation(SemanticRelation(a.id, b.id, "contained_in", 0.8))
    ontology.add_relation(SemanticRelation(b.id, c.id, "contained_in", 0.5))
    inference = OntologyReasoner().infer(ontology, a.id, c.id, "contained_in")
    assert inference is not None
    assert inference.confidence == pytest.approx(0.4)


def test_inference_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        OntologyInference("a", "b", "r", 1.5, True)
