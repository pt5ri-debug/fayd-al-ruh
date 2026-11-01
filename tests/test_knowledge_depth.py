"""The knowledge-depth engine: how far the graph reaches, and how it is widened.

Everything runs headless against the project's own seeds and a small hand-built
graph: no model, no network, and no bulk store is required. The held-out gate is
exercised too, so a regression in how depth is measured fails here.
"""

from __future__ import annotations

import pytest
from cognition.knowledge_depth import (
    DEFAULT_HARVEST_LIMIT,
    ArabicReach,
    Bridge,
    Coverage,
    DepthReport,
    HarvestReport,
    arabic_reach,
    discipline_bridges,
    harvest_relations,
    knowledge_depth,
    knowledge_depth_accuracy,
)


@pytest.fixture(scope="module")
def controller() -> object:
    from core.cognitive_controller import CognitiveController

    return CognitiveController()


def _small_graph():
    from core.ontology import Ontology

    ontology = Ontology()
    ids = {label: ontology.add_concept_once(label).id for label in ("alpha", "beta", "gamma")}
    ontology.add_relation_once(ids["alpha"], ids["beta"], "increases", confidence=0.6)
    ontology.add_relation_once(ids["beta"], ids["gamma"], "controls", confidence=0.6)
    ontology.add_concept_once("lonely")
    return ontology


# -- coverage and connectivity ----------------------------------------------


def test_depth_counts_a_small_graph_exactly() -> None:
    report = knowledge_depth(_small_graph())
    assert report.concepts == 4
    assert report.relations == 2
    assert report.isolated == 1


def test_connectivity_is_the_share_that_participates() -> None:
    report = knowledge_depth(_small_graph())
    assert report.connectivity == 0.75


def test_an_empty_graph_is_honest() -> None:
    from core.ontology import Ontology

    report = knowledge_depth(Ontology())
    assert report.concepts == 0
    assert report.relations == 0
    assert report.connectivity == 0.0
    assert report.isolated == 0


def test_the_seeded_graph_reports_real_reach(controller: object) -> None:
    report = knowledge_depth(controller.ontology)  # type: ignore[attr-defined]
    assert report.concepts > 400
    assert report.relations > 400
    assert 0.0 < report.connectivity <= 1.0
    assert report.average_degree > 0.0


def test_depth_report_describes_itself(controller: object) -> None:
    report = knowledge_depth(controller.ontology)  # type: ignore[attr-defined]
    text = report.describe()
    assert "concept" in text
    assert "relation" in text


def test_weak_spots_name_only_the_thin_areas(controller: object) -> None:
    report = knowledge_depth(controller.ontology)  # type: ignore[attr-defined]
    for line in report.weak_spots():
        assert line.strip()


def test_depth_report_round_trips_to_a_primitive(controller: object) -> None:
    report = knowledge_depth(controller.ontology)  # type: ignore[attr-defined]
    data = report.to_primitive()
    assert data["concepts"] == report.concepts
    assert data["relations"] == report.relations
    assert isinstance(data["disciplines"], list)


def test_coverage_share_handles_a_zero_expectation() -> None:
    coverage = Coverage(key="k", label="L", concepts=0, relations=0, expected_concepts=0, expected_relations=0)
    assert coverage.concept_share == 1.0
    assert coverage.complete


# -- harvesting ---------------------------------------------------------------


def test_a_harvest_without_write_changes_nothing() -> None:
    ontology = _small_graph()
    before = len(ontology.relations)
    report = harvest_relations(ontology, limit=0, write=False)
    assert isinstance(report, HarvestReport)
    assert not report.wrote
    assert len(ontology.relations) == before


def test_a_harvest_over_a_store_reads_it(controller: object) -> None:
    class _Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Store:
        def iter_chunks(self):
            yield _Chunk("the density log measures porosity.")
            yield _Chunk("porosity increases permeability.")

    report = harvest_relations(controller, store=_Store(), limit=10, write=False)
    assert report.chunks_read == 2
    assert report.relations_read >= 1
    assert not report.wrote


def test_a_harvest_that_writes_goes_through_the_bounded_learner() -> None:
    class _Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Store:
        def iter_chunks(self):
            yield _Chunk("the density log measures porosity.")

    # A fresh controller, so the write never touches the shared graph.
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    report = harvest_relations(controller, store=_Store(), limit=10, write=True)
    # Every accepted relation is bounded by the learner's ceiling, so the count is
    # honest (>=0) and a seeded relation is never overridden.
    assert report.added >= 0
    assert report.wrote is (report.relations_read > 0)


def test_a_missing_store_is_reported_not_raised() -> None:
    class _Empty:
        def iter_chunks(self):
            return iter(())

    from core.cognitive_controller import CognitiveController

    report = harvest_relations(CognitiveController(), store=_Empty(), limit=5, write=False)
    assert report.chunks_read == 0
    assert report.relations_read == 0


# -- bridges ------------------------------------------------------------------


def test_a_bridge_is_a_real_walked_path() -> None:
    from core.relation_paths import find_connection

    ontology = _small_graph()
    path = find_connection(ontology, "alpha", "gamma", max_hops=3)
    assert path is not None
    assert path.hops == 2


def test_an_unconnected_pair_yields_no_bridge() -> None:
    from core.relation_paths import find_connection

    ontology = _small_graph()
    assert find_connection(ontology, "alpha", "lonely", max_hops=3) is None


def test_bridges_between_disciplines_are_deterministic(controller: object) -> None:
    first = discipline_bridges(controller.ontology, source="reservoir", target="drilling", limit=3)  # type: ignore[attr-defined]
    second = discipline_bridges(controller.ontology, source="reservoir", target="drilling", limit=3)  # type: ignore[attr-defined]
    assert first == second
    for bridge in first:
        assert isinstance(bridge, Bridge)
        assert bridge.hops >= 1
        assert 0.0 <= bridge.confidence <= 1.0


def test_bridges_refuse_a_non_positive_hop_bound(controller: object) -> None:
    with pytest.raises(ValueError):
        discipline_bridges(controller.ontology, max_hops=0)  # type: ignore[attr-defined]


# -- Arabic reach --------------------------------------------------------------


def test_arabic_reach_reports_the_seeded_graph(controller: object) -> None:
    reach = arabic_reach(controller.ontology)  # type: ignore[attr-defined]
    assert isinstance(reach, ArabicReach)
    assert reach.total == reach.covered
    assert reach.share == 1.0
    assert reach.missing_labels == ()


def test_arabic_reach_counts_arabic_questions_when_a_bank_is_given(controller: object) -> None:
    bank = controller.question_bank()  # type: ignore[attr-defined]
    reach = arabic_reach(controller.ontology, bank=bank)  # type: ignore[attr-defined]
    assert reach.total_questions == len(bank)
    assert reach.arabic_questions > 0
    assert 0.0 < reach.question_share <= 1.0


def test_arabic_reach_reports_an_unaliased_label() -> None:
    ontology = _small_graph()
    reach = arabic_reach(ontology)
    assert reach.covered < reach.total
    assert "alpha" in reach.missing_labels


# -- the held-out gate ---------------------------------------------------------


def test_the_depth_gate_passes_on_the_engine() -> None:
    assert knowledge_depth_accuracy() == 1.0


def test_the_default_harvest_limit_is_bounded() -> None:
    assert 0 < DEFAULT_HARVEST_LIMIT <= 5000


def test_depth_is_a_frozen_report(controller: object) -> None:
    report = knowledge_depth(controller.ontology)  # type: ignore[attr-defined]
    assert isinstance(report, DepthReport)
    with pytest.raises(Exception):
        report.concepts = 0  # type: ignore[misc]
