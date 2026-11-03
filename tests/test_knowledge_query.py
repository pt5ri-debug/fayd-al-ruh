"""The query engine reads the graph: entities, paths, evidence, contradictions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from core.entity import Entity, EntityType
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_query import (
    DEFAULT_MAX_HOPS,
    KnowledgeQueryEngine,
)
from core.relation import Relation


def _graph() -> tuple[KnowledgeGraph, dict[str, str]]:
    graph = KnowledgeGraph()
    ids: dict[str, str] = {}
    for name in ("porosity", "permeability", "flow_rate"):
        entity = Entity(EntityType.CONCEPT, name, id=f"E-{name}")
        graph.add_entity(entity)
        ids[name] = entity.id
    graph.add_relation(Relation(source=ids["porosity"], target=ids["permeability"], relation_type="increases", confidence=0.8))
    graph.add_relation(Relation(source=ids["permeability"], target=ids["flow_rate"], relation_type="controls", confidence=0.7))
    return graph, ids


# -- entity ---------------------------------------------------------------


def test_find_entity_matches_a_name_case_insensitively():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert engine.find_entity("Porosity").id == ids["porosity"]
    assert engine.find_entity("nope") is None
    assert engine.find_entity("  ") is None


def test_entities_filter_by_type_and_name():
    graph, _ = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert {e.name for e in engine.entities(name_contains="per")} == {"permeability"}
    assert len(engine.entities(entity_type="Concept")) == 3
    assert engine.entities(entity_type="Person") == ()


def test_summarize_counts_relations_and_evidence():
    graph, ids = _graph()
    item = Evidence("measured", "SPE", True, 0.9)
    graph.add_evidence(item)
    graph.add_relation(Relation(source=ids["porosity"], target=ids["flow_rate"], relation_type="affects", evidence_ids=[item.id]))
    engine = KnowledgeQueryEngine(graph)
    summary = engine.summarize(ids["porosity"])
    assert summary.relation_count == 2
    assert summary.evidence_count == 1
    assert "porosity" in summary.describe()
    assert engine.summarize("nope") is None


# -- relationships --------------------------------------------------------


def test_relations_of_an_entity_covers_both_endpoints():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.relations_of(ids["permeability"])) == 2
    assert len(engine.relations_of(ids["porosity"])) == 1


def test_between_returns_direct_relations():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.between(ids["porosity"], ids["permeability"])) == 1
    assert engine.between(ids["permeability"], ids["porosity"]) == ()


def test_neighbors_walks_the_requested_depth():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    one = {entity.name for entity in engine.neighbors(ids["porosity"], depth=1)}
    two = {entity.name for entity in engine.neighbors(ids["porosity"], depth=2)}
    assert one == {"permeability"}
    assert two == {"permeability", "flow_rate"}
    assert engine.neighbors(ids["porosity"], depth=0) == ()


# -- traversal ------------------------------------------------------------


def test_path_finds_the_shortest_chain_and_its_product_confidence():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    path = engine.path(ids["porosity"], ids["flow_rate"], max_hops=3)
    assert path is not None
    assert path.hops == 2
    assert path.confidence == pytest.approx(0.8 * 0.7, abs=1e-6)
    assert "porosity" in path.describe()
    assert "flow_rate" in path.describe()


def test_a_zero_hop_path_is_the_same_entity():
    graph, ids = _graph()
    path = KnowledgeQueryEngine(graph).path(ids["porosity"], ids["porosity"])
    assert path is not None
    assert path.hops == 0
    assert path.confidence == 1.0


def test_an_unknown_endpoint_yields_no_path():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert engine.path(ids["porosity"], "ghost") is None
    assert engine.path(ids["porosity"], ids["flow_rate"], max_hops=1) is None
    assert engine.path(ids["porosity"], ids["flow_rate"], max_hops=0) is None


def test_a_backward_step_is_marked():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    path = engine.path(ids["flow_rate"], ids["porosity"], max_hops=3)
    assert path is not None
    assert any(not step.forward for step in path.steps)


def test_the_default_hop_bound_is_reasonable():
    assert DEFAULT_MAX_HOPS >= 2


# -- evidence -------------------------------------------------------------


def test_evidence_for_relation_and_entity():
    graph, _ = _graph()
    item = Evidence("measured", "SPE", True, 0.9)
    graph.add_evidence(item)
    relation = next(iter(graph.relations.values()))
    relation.add_evidence(item.id)
    engine = KnowledgeQueryEngine(graph)
    assert [e.id for e in engine.evidence_for_relation(relation.id)] == [item.id]
    assert item.id in {e.id for e in engine.evidence_for_entity(relation.source)}


def test_by_source_filters_and_is_case_insensitive():
    graph, _ = _graph()
    graph.add_evidence(Evidence("a", "SPE", True, 0.9))
    graph.add_evidence(Evidence("b", "blog", True, 0.4))
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.by_source("spe")) == 1


# -- context and contradictions -------------------------------------------


def test_context_returns_the_relations_around_an_entity():
    graph, ids = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.context(ids["porosity"], depth=2)) == 2


def test_contradictions_report_opposing_relations_on_the_same_pair():
    graph, ids = _graph()
    graph.add_relation(Relation(source=ids["porosity"], target=ids["permeability"], relation_type="reduces", confidence=0.6))
    engine = KnowledgeQueryEngine(graph)
    found = engine.contradictions()
    assert len(found) == 1
    assert found[0].severity == pytest.approx(0.6)
    assert "vs" in found[0].describe()


def test_contradictions_can_be_scoped_to_an_entity():
    graph, ids = _graph()
    graph.add_relation(Relation(source=ids["porosity"], target=ids["permeability"], relation_type="reduces", confidence=0.6))
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.contradictions(entity_id=ids["flow_rate"])) == 0
    assert len(engine.contradictions(entity_id=ids["porosity"])) == 1


# -- temporal -------------------------------------------------------------


def test_created_between_reads_a_window():
    graph, _ = _graph()
    now = datetime.now(UTC)
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.created_between(now - timedelta(minutes=1), now + timedelta(minutes=1))) == 2
    assert engine.created_between(now + timedelta(days=1), now + timedelta(days=2)) == ()


def test_a_reversed_window_yields_nothing():
    graph, _ = _graph()
    now = datetime.now(UTC)
    engine = KnowledgeQueryEngine(graph)
    assert engine.created_between(now + timedelta(days=1), now) == ()


def test_latest_returns_newest_first():
    graph, _ = _graph()
    engine = KnowledgeQueryEngine(graph)
    assert len(engine.latest()) == 2


def test_statistics_and_describe():
    graph, _ = _graph()
    engine = KnowledgeQueryEngine(graph)
    stats = engine.statistics()
    assert stats["entities"] == 3
    assert stats["relations"] == 2
    assert "KnowledgeQueryEngine" in engine.describe()
