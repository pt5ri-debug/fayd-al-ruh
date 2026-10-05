from core.cognitive_controller import CognitiveController
from core.domain_knowledge import domain_summary, seed_ontology
from core.model_interface import CallableModel, ModelRequest, MultimodalInput
from core.ontology import Ontology
from core.self_model import SelfModel


def test_seed_ontology_adds_petroleum_concepts_and_relations():
    ontology = Ontology()
    seed_ontology(ontology)

    assert ontology.find_concepts("reservoir")
    assert ontology.find_concepts("مكمن")
    reservoir = ontology.find_concepts("المكمن")[0]
    targets = {relation.relation_type for relation in ontology.outgoing(reservoir.id)}
    assert "contained_in" in targets


def test_seed_ontology_is_idempotent():
    ontology = Ontology()
    first = seed_ontology(ontology)
    second = seed_ontology(ontology)

    assert len(second) == len(first)
    assert len(ontology.concepts) == len(first)
    assert len(ontology.relations) == 13


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
