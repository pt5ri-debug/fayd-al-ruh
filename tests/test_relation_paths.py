"""Tests for bounded connection explanation between two concepts."""

from __future__ import annotations

from core.ontology import Ontology
from core.relation_paths import (
    ConnectionPath,
    RelationStep,
    describe_connection,
    find_connection,
)


def _ontology() -> Ontology:
    from core.domain_knowledge import seed_ontology

    ontology = Ontology()
    seed_ontology(ontology)
    return ontology


def _chain_ontology() -> Ontology:
    """A tiny hand-built graph, so the search is tested exactly."""
    from core.semantic_relation import SemanticRelation

    ontology = Ontology()
    a = ontology.add_concept_once("alpha")
    b = ontology.add_concept_once("beta")
    c = ontology.add_concept_once("gamma")
    ontology.add_relation(SemanticRelation(a.id, b.id, "increases", 0.8))
    ontology.add_relation(SemanticRelation(b.id, c.id, "causes", 0.5))
    return ontology


def test_direct_connection_is_one_step():
    path = find_connection(_chain_ontology(), "alpha", "gamma")
    assert isinstance(path, ConnectionPath)
    assert path.hops == 2
    assert [step.relation_type for step in path.steps] == ["increases", "causes"]


def test_connection_walks_backwards_through_an_incoming_relation():
    path = find_connection(_chain_ontology(), "gamma", "alpha")
    assert path is not None
    assert path.hops == 2
    # Walking gamma -> beta -> alpha means both steps go against the stored direction.
    assert all(step.forward is False for step in path.steps)


def test_connection_confidence_is_the_product_of_steps():
    path = find_connection(_chain_ontology(), "alpha", "gamma")
    assert path is not None
    assert path.confidence == round(0.8 * 0.5, 6)


def test_same_concept_connects_in_zero_steps():
    path = find_connection(_chain_ontology(), "alpha", "alpha")
    assert path is not None
    assert path.hops == 0
    assert "same concept" in path.describe()


def test_no_path_is_none_not_a_fabrication():
    assert find_connection(_chain_ontology(), "alpha", "zzz-unknown") is None
    assert find_connection(None, "alpha", "gamma") is None


def test_hop_bound_is_honoured():
    assert find_connection(_chain_ontology(), "alpha", "gamma", max_hops=1) is None
    assert find_connection(_chain_ontology(), "alpha", "gamma", max_hops=2) is not None
    assert find_connection(_chain_ontology(), "alpha", "gamma", max_hops=0) is None


def test_connection_is_deterministic():
    ontology = _ontology()
    first = find_connection(ontology, "porosity", "recovery_factor")
    second = find_connection(ontology, "porosity", "recovery_factor")
    assert first == second


def test_step_describe_shows_direction():
    forward = RelationStep("a", "increases", "b", 0.5, True)
    backward = RelationStep("b", "increases", "a", 0.5, False)
    assert forward.describe() == "a —increases→ b"
    assert backward.describe() == "b ←increases— a"


def test_describe_connection_names_absence_honestly():
    text = describe_connection(None, "porosity", "zzz")
    assert "No chain" in text
    assert "zzz" in text
