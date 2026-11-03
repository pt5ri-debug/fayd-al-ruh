"""Entity, relation, and evidence managers: lifecycle, versioning, preservation."""

from __future__ import annotations

import pytest
from core.entity import Entity, EntityType
from core.entity_manager import EntityManager, normalize_name
from core.evidence import Evidence
from core.evidence_manager import (
    AcquisitionMethod,
    EvidenceKind,
    EvidenceManager,
)
from core.knowledge_graph import KnowledgeGraph
from core.relation import Relation
from core.relation_manager import RelationManager

# -- entity manager -------------------------------------------------------


def test_create_is_idempotent_by_normalised_name():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    first = manager.create("Porosity")
    second = manager.create("porosity")
    assert first.id == second.id
    assert len(graph.entities) == 1


def test_create_records_the_first_version():
    manager = EntityManager(KnowledgeGraph())
    entity = manager.create("porosity", confidence=0.6, properties={"unit": "fraction"})
    versions = manager.versions(entity.id)
    assert len(versions) == 1
    assert versions[0].reason == "created"
    assert versions[0].snapshot["name"] == "porosity"


def test_update_records_a_new_version():
    manager = EntityManager(KnowledgeGraph())
    entity = manager.create("porosity")
    manager.update(entity.id, confidence=0.9, properties={"measured": True}, reason="tuned")
    versions = manager.versions(entity.id)
    assert len(versions) == 2
    assert manager.latest_version(entity.id).reason == "tuned"
    assert entity.confidence == pytest.approx(0.9)
    assert entity.properties["measured"] is True


def test_update_of_an_unknown_entity_raises():
    manager = EntityManager(KnowledgeGraph())
    with pytest.raises(KeyError):
        manager.update("ghost", confidence=0.5)


def test_find_by_name_is_case_insensitive_and_can_filter_by_type():
    manager = EntityManager(KnowledgeGraph())
    entity = manager.create("porosity", entity_type=EntityType.CONCEPT)
    assert manager.find_by_name("POROSITY").id == entity.id
    assert manager.find_by_name("porosity", entity_type=EntityType.PERSON) is None


def test_versions_are_bounded():
    manager = EntityManager(KnowledgeGraph(), version_limit=3)
    entity = manager.create("porosity")
    for _ in range(5):
        manager.update(entity.id, confidence=0.5)
    assert len(manager.versions(entity.id)) == 3


def test_detect_duplicates_finds_same_name_and_type():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    manager.create("porosity")
    graph.add_entity(Entity(EntityType.CONCEPT, "porosity", id="dup"))
    candidates = manager.detect_duplicates()
    assert len(candidates) == 1
    assert "same name and type" in candidates[0].reason


def test_a_same_name_different_type_pair_is_not_an_exact_duplicate():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    manager.create("porosity", entity_type=EntityType.CONCEPT)
    graph.add_entity(Entity(EntityType.OBJECT, "porosity", id="other"))
    assert manager.detect_duplicates(threshold=1.0) == ()
    assert len(manager.detect_duplicates(threshold=0.5)) == 1


def test_merge_repoints_relations_to_the_survivor():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    survivor = manager.create("porosity")
    graph.add_entity(Entity(EntityType.CONCEPT, "permeability", id="T"))
    duplicate = Entity(EntityType.CONCEPT, "porosity", id="dup")
    graph.add_entity(duplicate)
    graph.add_relation(Relation(source="dup", target="T", relation_type="increases"))
    result = manager.merge(survivor.id, "dup")
    assert result.relations_moved == 1
    relation = next(iter(graph.relations.values()))
    assert relation.source == survivor.id
    assert "dup" not in graph.entities


def test_merge_folds_properties_without_overwriting():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    survivor = manager.create("porosity", properties={"unit": "fraction"})
    graph.add_entity(Entity(EntityType.CONCEPT, "porosity", id="dup", properties={"unit": "percent", "extra": 1}))
    manager.merge(survivor.id, "dup")
    assert survivor.properties["unit"] == "fraction"  # the survivor's value wins
    assert survivor.properties["extra"] == 1


def test_merge_preserves_the_duplicate_history_under_the_survivor():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    survivor = manager.create("porosity")
    graph.add_entity(Entity(EntityType.CONCEPT, "porosity", id="dup"))
    manager.merge(survivor.id, "dup")
    reasons = [version.reason for version in manager.versions(survivor.id)]
    assert any("merged dup" in reason for reason in reasons)


def test_merge_of_an_entity_into_itself_is_refused():
    manager = EntityManager(KnowledgeGraph())
    entity = manager.create("porosity")
    with pytest.raises(ValueError):
        manager.merge(entity.id, entity.id)


def test_merge_of_an_unknown_entity_raises():
    manager = EntityManager(KnowledgeGraph())
    entity = manager.create("porosity")
    with pytest.raises(KeyError):
        manager.merge(entity.id, "ghost")


def test_merge_duplicates_merges_only_exact_pairs():
    graph = KnowledgeGraph()
    manager = EntityManager(graph)
    manager.create("porosity")
    graph.add_entity(Entity(EntityType.CONCEPT, "porosity", id="dup"))
    results = manager.merge_duplicates()
    assert len(results) == 1
    assert len(graph.entities) == 1


def test_normalize_name_folds_case_and_spacing():
    assert normalize_name("  Porosity ") == normalize_name("porosity")


# -- relation manager -----------------------------------------------------


def test_relation_create_requires_both_endpoints():
    graph = KnowledgeGraph()
    manager = RelationManager(graph)
    graph.add_entity(Entity(EntityType.CONCEPT, "a", id="A"))
    with pytest.raises(KeyError):
        manager.create("A", "missing", "increases")


def test_relation_create_reinforces_an_identical_triple():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    first = manager.create(a.id, b.id, "increases", confidence=0.5)
    second = manager.create(a.id, b.id, "increases", confidence=0.9)
    assert second.created is False
    assert second.reinforced is True
    assert first.relation.id == second.relation.id
    assert len(graph.relations) == 1


def test_relation_create_preserves_an_opposing_relation():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    manager.create(a.id, b.id, "increases", confidence=0.7)
    result = manager.create(a.id, b.id, "reduces", confidence=0.6)
    assert result.conflict is True
    assert len(graph.relations) == 2


def test_attach_evidence_requires_known_relation_and_evidence():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    result = manager.create(a.id, b.id, "increases")
    with pytest.raises(KeyError):
        manager.attach_evidence(result.relation.id, "ghost-evidence")
    item = Evidence("note", "SPE", True, 0.9)
    graph.add_evidence(item)
    manager.attach_evidence(result.relation.id, item.id)
    assert item.id in result.relation.evidence_ids


def test_deactivate_preserves_the_relation():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    result = manager.create(a.id, b.id, "increases")
    manager.deactivate(result.relation.id, reason="superseded")
    assert result.relation.active is False
    assert graph.get_relation(result.relation.id) is not None
    manager.reactivate(result.relation.id)
    assert result.relation.active is True


def test_find_conflicts_reports_opposing_pairs():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    manager.create(a.id, b.id, "increases", confidence=0.7)
    manager.create(a.id, b.id, "reduces", confidence=0.6)
    conflicts = manager.find_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].severity == pytest.approx(0.6)


def test_find_conflicts_ignores_a_deactivated_relation_by_default():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph)
    first = manager.create(a.id, b.id, "increases", confidence=0.7)
    manager.create(a.id, b.id, "reduces", confidence=0.6)
    manager.deactivate(first.relation.id)
    assert manager.find_conflicts() == ()
    assert len(manager.find_conflicts(active_only=False)) == 1


def test_history_records_each_action_and_is_bounded():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    manager = RelationManager(graph, history_limit=2)
    result = manager.create(a.id, b.id, "increases", confidence=0.5)
    manager.create(a.id, b.id, "increases", confidence=0.6)
    manager.create(a.id, b.id, "increases", confidence=0.7)
    assert len(manager.history(result.relation.id)) == 2


def test_duplicates_reports_identical_triples():
    graph = KnowledgeGraph()
    entity_manager = EntityManager(graph)
    a = entity_manager.create("a")
    b = entity_manager.create("b")
    graph.add_relation(Relation(source=a.id, target=b.id, relation_type="increases", id="R1"))
    graph.add_relation(Relation(source=a.id, target=b.id, relation_type="increases", id="R2"))
    assert RelationManager(graph).duplicates() == (("R1", "R2"),)


# -- evidence manager -----------------------------------------------------


def test_record_stores_a_classified_item():
    graph = KnowledgeGraph()
    manager = EvidenceManager(graph, reliability={"SPE": 0.95})
    item = manager.record("measured", "SPE", strength=0.9, method=AcquisitionMethod.INGEST, reference="doc-1")
    record = manager.describe(item.id)
    assert record is not None
    assert record.kind is EvidenceKind.DIRECT
    assert record.method is AcquisitionMethod.INGEST
    assert record.reliability == pytest.approx(0.95)
    assert record.reference == "doc-1"


def test_record_inference_is_classified_as_inferred():
    manager = EvidenceManager(KnowledgeGraph())
    item = manager.record_inference("derived", reference="walk:3")
    record = manager.describe(item.id)
    assert record.kind is EvidenceKind.INFERRED
    assert record.method is AcquisitionMethod.INFERENCE
    assert record.is_direct is False


def test_an_uncalibrated_source_gets_the_default_reliability():
    manager = EvidenceManager(KnowledgeGraph())
    item = manager.record("note", "unknown-source")
    assert manager.describe(item.id).reliability == pytest.approx(0.5)


def test_a_legacy_item_without_a_kind_reads_as_direct():
    graph = KnowledgeGraph()
    legacy = Evidence("old note", "paper", True, 0.7)
    graph.add_evidence(legacy)
    record = EvidenceManager(graph).as_record(legacy)
    assert record.kind is EvidenceKind.DIRECT
    assert record.method is AcquisitionMethod.MANUAL


def test_by_kind_splits_direct_from_inferred():
    manager = EvidenceManager(KnowledgeGraph())
    manager.record("a", "SPE")
    manager.record_inference("b")
    assert manager.direct_count() == 1
    assert manager.inferred_count() == 1
    assert len(manager.by_kind(EvidenceKind.INFERRED)) == 1


def test_weight_is_strength_times_reliability():
    graph = KnowledgeGraph()
    manager = EvidenceManager(graph, reliability={"SPE": 0.8})
    item = manager.record("x", "SPE", strength=0.5)
    assert manager.weight(item.id) == pytest.approx(0.4)


def test_weight_of_an_unknown_item_raises():
    manager = EvidenceManager(KnowledgeGraph())
    with pytest.raises(KeyError):
        manager.weight("ghost")


def test_an_out_of_range_reliability_is_refused():
    with pytest.raises(ValueError):
        EvidenceManager(KnowledgeGraph(), reliability={"SPE": 1.5})


def test_for_claim_reads_attached_evidence():
    graph = KnowledgeGraph()
    manager = EvidenceManager(graph)
    item = manager.record("x", "SPE")
    graph.add_entity(Entity(EntityType.CONCEPT, "a", id="A"))
    graph.add_entity(Entity(EntityType.CONCEPT, "b", id="B"))
    relation = Relation(source="A", target="B", relation_type="increases", evidence_ids=[item.id])
    graph.add_relation(relation)
    records = manager.for_claim(relation.id)
    assert len(records) == 1
    assert "SPE" in records[0].describe()
