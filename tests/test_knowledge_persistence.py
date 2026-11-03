"""Knowledge persistence: atomic, faithful, and tolerant of a corrupt record."""

from __future__ import annotations

import pytest
from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_persistence import (
    META_KEY,
    NAMESPACE_ENTITIES,
    NAMESPACE_META,
    KnowledgePersistence,
    LoadReport,
)
from core.relation import Relation
from storage.json_store import JsonStore
from storage.sqlite_store import SqliteStore


@pytest.fixture(params=["sqlite", "json"])
def store(request, tmp_path):
    if request.param == "sqlite":
        return SqliteStore(tmp_path / "kg.db")
    return JsonStore(tmp_path / "json")


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_entity(Entity(EntityType.CONCEPT, "porosity", id="E-por", confidence=0.7))
    graph.add_entity(Entity(EntityType.CONCEPT, "permeability", id="E-per"))
    item = Evidence("measured", "SPE", True, 0.9, data={"source_reference": "doc-1"})
    graph.add_evidence(item)
    graph.add_relation(
        Relation(source="E-por", target="E-per", relation_type="increases", confidence=0.8, evidence_ids=[item.id])
    )
    return graph


# -- round trip -----------------------------------------------------------


def test_a_save_reports_what_it_wrote(store):
    report = KnowledgePersistence(store).save(_graph())
    assert report.entities == 2
    assert report.relations == 1
    assert report.evidence == 1
    assert "saved" in report.describe()


def test_a_round_trip_preserves_the_graph(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    loaded, report = persistence.load()
    assert report.ok is True
    assert len(loaded.entities) == 2
    assert len(loaded.relations) == 1
    assert len(loaded.evidence) == 1
    relation = next(iter(loaded.relations.values()))
    assert relation.relation_type == "increases"
    assert relation.confidence == pytest.approx(0.8)
    assert relation.evidence_ids


def test_a_round_trip_preserves_entity_identity_and_confidence(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    loaded, _ = persistence.load()
    entity = loaded.get_entity("E-por")
    assert entity is not None
    assert entity.name == "porosity"
    assert entity.confidence == pytest.approx(0.7)


def test_a_round_trip_preserves_evidence_metadata(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    loaded, _ = persistence.load()
    item = next(iter(loaded.evidence.values()))
    assert item.data["source_reference"] == "doc-1"


def test_a_round_trip_preserves_an_inactive_relation(store):
    graph = _graph()
    relation = next(iter(graph.relations.values()))
    relation.deactivate()
    persistence = KnowledgePersistence(store)
    persistence.save(graph)
    loaded, _ = persistence.load()
    assert next(iter(loaded.relations.values())).active is False


def test_versions_are_stored_and_read_back(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph(), versions={"E-por": [{"version": 1}]})
    assert persistence.versions() == {"E-por": [{"version": 1}]}


# -- empty and missing ----------------------------------------------------


def test_loading_an_empty_store_is_not_an_error(store):
    loaded, report = KnowledgePersistence(store).load()
    assert report.ok is False
    assert len(loaded.entities) == 0
    assert "no stored graph" in report.describe()


def test_exists_is_false_before_a_save_and_true_after(store):
    persistence = KnowledgePersistence(store)
    assert persistence.exists() is False
    persistence.save(_graph())
    assert persistence.exists() is True


# -- atomicity ------------------------------------------------------------


def test_the_meta_marker_is_written_last(store):
    """The meta document is the completion marker: its presence means a whole save."""
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    meta = store.load(NAMESPACE_META, META_KEY)
    assert isinstance(meta, dict)
    assert meta["entities"] == 2


def test_a_store_without_the_marker_is_not_a_graph(store):
    # Entity records present, but no completion marker -> no graph.
    store.save(NAMESPACE_ENTITIES, "orphan", {"id": "orphan"})
    loaded, report = KnowledgePersistence(store).load()
    assert report.ok is False
    assert len(loaded.entities) == 0


# -- corruption tolerance -------------------------------------------------


def test_a_corrupt_record_is_skipped_and_counted(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    store.save(NAMESPACE_ENTITIES, "broken", {"not": "an entity"})
    loaded, report = persistence.load()
    assert report.skipped == 1
    assert len(loaded.entities) == 2
    assert "skipped" in report.describe()


def test_a_relation_with_a_missing_entity_is_skipped(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    # Overwrite one entity with a broken record; the relation that pointed at it
    # cannot be re-added, so it is skipped -- the rest of the graph still loads.
    store.save(NAMESPACE_ENTITIES, "E-por", {"broken": True})
    _loaded, report = persistence.load()
    assert report.skipped >= 1
    assert report.entities == 1


def test_load_into_an_existing_graph_adds_to_it(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    target = KnowledgeGraph()
    target.add_entity(Entity(EntityType.CONCEPT, "extra", id="E-extra"))
    loaded, _ = persistence.load(target)
    assert loaded is target
    assert len(loaded.entities) == 3


# -- clear ----------------------------------------------------------------


def test_clear_removes_every_record(store):
    persistence = KnowledgePersistence(store)
    persistence.save(_graph())
    persistence.clear()
    assert persistence.exists() is False
    loaded, report = persistence.load()
    assert len(loaded.entities) == 0
    assert report.ok is False


def test_a_load_report_describes_an_empty_result():
    report = LoadReport(path="nowhere")
    assert "no stored graph" in report.describe()
