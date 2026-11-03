"""Knowledge validation: one verdict, explainable confidence, honest refusals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_validation import (
    DEFAULT_ACCEPT_THRESHOLD,
    FindingKind,
    KnowledgeClaim,
    KnowledgeValidator,
    ValidationFinding,
    ValidationVerdict,
    known_relation_types,
)
from core.relation import Relation


def _evidence(source: str = "SPE", supports: bool = True, strength: float = 0.9) -> Evidence:
    return Evidence(description=f"note from {source}", source=source, supports=supports, strength=strength)


def _graph_with(source: str, target: str) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_entity(Entity(EntityType.CONCEPT, source, id="S"))
    graph.add_entity(Entity(EntityType.CONCEPT, target, id="T"))
    return graph


# -- the claim ------------------------------------------------------------


def test_a_claim_needs_a_valid_confidence():
    with pytest.raises(ValueError):
        KnowledgeClaim("a", "increases", "b", confidence=1.5)


def test_the_triple_and_describe_are_stable():
    claim = KnowledgeClaim("porosity", "increases", "permeability")
    assert claim.triple == ("porosity", "increases", "permeability")
    assert claim.describe() == "porosity —increases→ permeability"


def test_a_finding_severity_is_bounded():
    with pytest.raises(ValueError):
        ValidationFinding(FindingKind.STRUCTURAL, "x", severity=2.0)


# -- the verdict ----------------------------------------------------------


def test_an_unsupported_claim_is_never_accepted():
    """THE RULE: no evidence means it can never be classified as knowledge."""
    result = KnowledgeValidator().validate(KnowledgeClaim("a", "increases", "b"))
    assert result.verdict is ValidationVerdict.NEEDS_REVIEW
    assert result.verified is False
    assert result.storable is False
    assert result.has(FindingKind.MISSING_EVIDENCE)


def test_a_well_evidenced_claim_is_accepted():
    claim = KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),), confidence=0.8)
    result = KnowledgeValidator().validate(claim)
    assert result.verdict is ValidationVerdict.ACCEPTED
    assert result.accepted is True
    assert result.storable is True
    assert result.verified is False


def test_validation_never_verifies():
    """Even an accepted claim is *not* verified -- that is a separate act."""
    claim = KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),), confidence=0.9)
    assert KnowledgeValidator().validate(claim).verified is False


def test_a_missing_endpoint_is_rejected_as_blocking():
    result = KnowledgeValidator().validate(KnowledgeClaim("", "increases", "b"))
    assert result.verdict is ValidationVerdict.REJECTED
    assert result.ok is False
    assert any(f.blocking for f in result.findings)


def test_a_self_relation_is_rejected():
    result = KnowledgeValidator().validate(KnowledgeClaim("a", "increases", "a", evidence=(_evidence(),)))
    assert result.verdict is ValidationVerdict.REJECTED


def test_an_unknown_relation_type_is_a_warning_not_a_block():
    """A text may teach a relation the seeds lack; it is reported, not refused."""
    validator = KnowledgeValidator(known_relations={"increases"})
    result = validator.validate(KnowledgeClaim("a", "novel_rel", "b", evidence=(_evidence(),)))
    assert result.verdict is not ValidationVerdict.REJECTED
    assert result.has(FindingKind.SEMANTIC)


def test_an_unknown_concept_is_a_warning():
    class Ontology:
        def find_concepts(self, label):
            return [] if label == "ghost" else [object()]

    validator = KnowledgeValidator(ontology=Ontology())
    result = validator.validate(
        KnowledgeClaim("ghost", "increases", "b", evidence=(_evidence(),), source_label="ghost", target_label="b")
    )
    assert result.has(FindingKind.SEMANTIC)
    assert result.verdict is not ValidationVerdict.REJECTED


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        KnowledgeValidator(accept_threshold=0.2, review_threshold=0.8)


# -- relationship checks --------------------------------------------------


def test_an_identical_relation_is_reported_as_a_duplicate():
    graph = _graph_with("a", "b")
    graph.add_relation(Relation(source="S", target="T", relation_type="increases"))
    result = KnowledgeValidator().validate(
        KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),)), graph=graph
    )
    assert result.has(FindingKind.DUPLICATE)


def test_an_opposing_relation_on_the_same_pair_is_a_contradiction():
    graph = _graph_with("a", "b")
    graph.add_relation(Relation(source="S", target="T", relation_type="reduces", confidence=0.7))
    result = KnowledgeValidator().validate(
        KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),), confidence=0.8), graph=graph
    )
    assert result.has(FindingKind.CONTRADICTION)
    assert result.verdict is ValidationVerdict.NEEDS_REVIEW
    # A conflict is storable (both claims preserved), but flagged.
    assert result.storable is True


def test_an_opposite_direction_is_not_a_contradiction():
    """``a reduces b`` does not oppose ``b increases a`` -- a different claim."""
    graph = _graph_with("a", "b")
    graph.add_relation(Relation(source="T", target="S", relation_type="reduces", confidence=0.7))
    result = KnowledgeValidator().validate(
        KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),)), graph=graph
    )
    assert not result.has(FindingKind.CONTRADICTION)


def test_an_unknown_endpoint_never_matches_a_stored_relation():
    """A claim naming an entity the graph has never seen cannot be a duplicate."""
    graph = _graph_with("a", "b")
    graph.add_relation(Relation(source="S", target="T", relation_type="increases"))
    result = KnowledgeValidator().validate(
        KnowledgeClaim("fresh", "increases", "b", evidence=(_evidence(),)), graph=graph
    )
    assert not result.has(FindingKind.DUPLICATE)


# -- evidence contradiction -----------------------------------------------


def test_evidence_pulling_both_ways_is_a_contradiction():
    claim = KnowledgeClaim(
        "a",
        "increases",
        "b",
        evidence=(_evidence("SPE", supports=True), _evidence("Blog", supports=False)),
    )
    result = KnowledgeValidator().validate(claim)
    assert result.has(FindingKind.CONTRADICTION)
    assert result.verdict is ValidationVerdict.NEEDS_REVIEW


# -- temporal -------------------------------------------------------------


def test_a_reversed_window_is_blocking():
    now = datetime.now(UTC)
    claim = KnowledgeClaim(
        "a",
        "increases",
        "b",
        evidence=(_evidence(),),
        valid_from=now,
        valid_until=now - timedelta(days=1),
    )
    result = KnowledgeValidator().validate(claim)
    assert result.verdict is ValidationVerdict.REJECTED
    assert result.has(FindingKind.TEMPORAL)


def test_an_expired_window_is_reported_not_blocking():
    past = datetime.now(UTC) - timedelta(days=1)
    claim = KnowledgeClaim("a", "increases", "b", evidence=(_evidence(),), valid_until=past)
    result = KnowledgeValidator().validate(claim)
    assert result.has(FindingKind.TEMPORAL)
    assert result.verdict is not ValidationVerdict.REJECTED


# -- confidence -----------------------------------------------------------


def test_the_result_carries_an_explainable_confidence():
    claim = KnowledgeClaim("a", "increases", "b", evidence=(_evidence(), _evidence("OnePetro")), confidence=0.8)
    result = KnowledgeValidator().validate(claim)
    assert 0.0 <= result.confidence <= 1.0
    assert "verdict" in result.describe()
    payload = result.to_primitive()
    assert payload["verdict"] == result.verdict.value


def test_a_low_confidence_well_formed_claim_is_provisional():
    validator = KnowledgeValidator(accept_threshold=0.9, review_threshold=0.1)
    claim = KnowledgeClaim("a", "increases", "b", evidence=(_evidence(strength=0.3),), confidence=0.3)
    result = validator.validate(claim)
    assert result.verdict is ValidationVerdict.PROVISIONAL


# -- helper ---------------------------------------------------------------


def test_known_relation_types_reads_an_ontology():
    class Relation:
        def __init__(self, relation_type):
            self.relation_type = relation_type

    class Ontology:
        relations: ClassVar = {"k": Relation("increases"), "j": Relation("reduces")}

    assert known_relation_types(Ontology()) == {"increases", "reduces"}


def test_known_relation_types_is_empty_when_the_ontology_cannot_answer():
    assert known_relation_types(object()) == set()


def test_the_accept_threshold_default_is_reasonable():
    assert 0.0 < DEFAULT_ACCEPT_THRESHOLD < 1.0
