from core.cognitive_controller import CognitiveController
from core.domain_knowledge import (
    discipline_coverage,
    discipline_keys,
    discipline_labels,
    disciplines,
    domain_summary,
    seed_ontology,
)
from core.model_interface import CallableModel, ModelRequest, MultimodalInput
from core.ontology import Ontology
from core.self_model import SelfModel


def test_seed_ontology_adds_petroleum_concepts_and_relations():
    ontology = Ontology()
    seed_ontology(ontology)

    assert ontology.find_concepts("reservoir")
    assert ontology.find_concepts("formation")
    reservoir = ontology.find_concepts("reservoir")[0]
    targets = {relation.relation_type for relation in ontology.outgoing(reservoir.id)}
    assert "contained_in" in targets


def test_seed_ontology_is_idempotent():
    ontology = Ontology()
    first = seed_ontology(ontology)
    second = seed_ontology(ontology)

    assert len(second) == len(first)
    assert len(ontology.concepts) == len(first)
    # 13 foundation relations plus every discipline's own relations.
    expected_relations = 13 + sum(len(discipline.relations) for discipline in disciplines())
    assert len(ontology.relations) == expected_relations


def test_controller_starts_with_petroleum_frame():
    controller = CognitiveController()

    assert controller.ontology.find_concepts("permeability")
    assert "petroleum engineering" in controller.identity
    assert controller.introspect().self_model.domain == "petroleum engineering"


def test_domain_summary_names_identity_and_focus_areas():
    summary = domain_summary(SelfModel())

    assert "petroleum engineering mind" in summary
    assert "reservoir" in summary


def test_identity_frame_is_injected_into_model_requests():
    controller = CognitiveController()
    received = []

    model = CallableModel("spy", lambda request: received.append(request) or "ok")
    controller.reason(model, ModelRequest((MultimodalInput("text", "analyze the well"),)))

    assert any("petroleum engineering" in item for item in received[0].context)


def test_seed_covers_every_petroleum_discipline():
    ontology = Ontology()
    seed_ontology(ontology)

    coverage = discipline_coverage(ontology)
    assert set(coverage) == set(discipline_keys())
    assert len(coverage) >= 13
    for key, counts in coverage.items():
        assert counts["concepts"] == counts["expected_concepts"], key
        assert counts["relations"] == counts["expected_relations"], key


def test_discipline_coverage_reports_missing_concepts_honestly():
    ontology = Ontology()

    coverage = discipline_coverage(ontology)
    assert all(counts["concepts"] == 0 for counts in coverage.values())
    assert all(counts["relations"] == 0 for counts in coverage.values())


def test_discipline_labels_are_english_and_align_with_keys():
    labels = discipline_labels()

    assert set(labels) == set(discipline_keys())
    assert "drilling" in labels
    assert labels["reservoir"] == "Reservoir Engineering"


def test_domain_summary_lists_disciplines():
    summary = domain_summary(SelfModel())

    assert "Disciplines:" in summary
    assert "Drilling Engineering" in summary

