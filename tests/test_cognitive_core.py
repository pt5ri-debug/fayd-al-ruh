"""The Phase-1 services are wired into the controller and reachable from it."""

from __future__ import annotations

import pytest
from core.cognitive_controller import CognitiveController
from core.event_bus import (
    EVENT_CYCLE_FINISHED,
    EVENT_KNOWLEDGE_INTEGRATED,
    EVENT_KNOWLEDGE_REJECTED,
    EVENT_RELATION_LEARNED,
)
from core.evidence import Evidence
from core.knowledge_validation import KnowledgeClaim, ValidationVerdict


@pytest.fixture()
def controller() -> CognitiveController:
    return CognitiveController()


# -- the services exist and are the controller's own ----------------------


def test_the_controller_owns_every_phase_one_service(controller):
    for name in (
        "events",
        "resource_governor",
        "entity_manager",
        "relation_manager",
        "evidence_manager",
        "knowledge_query",
        "knowledge_validator",
        "knowledge_integrator",
    ):
        assert hasattr(controller, name), name


def test_the_managers_and_query_read_the_controller_own_graph(controller):
    assert controller.entity_manager.graph is controller.graph
    assert controller.relation_manager.graph is controller.graph
    assert controller.evidence_manager.graph is controller.graph
    assert controller.knowledge_query.graph is controller.graph


def test_the_validator_was_seeded_with_the_ontology_relation_types(controller):
    assert controller.knowledge_validator.known_relations


# -- integration through the controller -----------------------------------


def test_integrating_a_claim_writes_it_and_publishes_an_event(controller):
    record = controller.integrate_knowledge(
        KnowledgeClaim(
            "porosity",
            "increases",
            "permeability",
            evidence=(Evidence("measured", "SPE", True, 0.9),),
            confidence=0.8,
        )
    )
    assert record.stored is True
    assert controller.knowledge_query.statistics()["relations"] == 1
    names = [event.name for event in controller.events.history]
    assert EVENT_KNOWLEDGE_INTEGRATED in names
    assert EVENT_RELATION_LEARNED in names


def test_an_unsupported_claim_publishes_a_rejection_and_writes_nothing(controller):
    record = controller.integrate_knowledge(KnowledgeClaim("a", "causes", "b"))
    assert record.stored is False
    assert controller.knowledge_query.statistics()["relations"] == 0
    assert EVENT_KNOWLEDGE_REJECTED in [event.name for event in controller.events.history]


def test_validate_claim_does_not_write(controller):
    result = controller.validate_claim(
        KnowledgeClaim("porosity", "increases", "permeability", evidence=(Evidence("x", "SPE", True, 0.9),))
    )
    assert result.verdict is ValidationVerdict.ACCEPTED
    assert controller.knowledge_query.statistics()["entities"] == 0


def test_query_returns_the_read_engine(controller):
    assert controller.query() is controller.knowledge_query


def test_a_caller_can_subscribe_to_the_controller_events(controller):
    seen: list[str] = []
    controller.events.subscribe(lambda event: seen.append(event.name), prefix="knowledge")
    controller.integrate_knowledge(
        KnowledgeClaim("porosity", "increases", "permeability", evidence=(Evidence("x", "SPE", True, 0.9),))
    )
    assert seen == [EVENT_KNOWLEDGE_INTEGRATED]


# -- status ---------------------------------------------------------------


def test_cognitive_core_status_reports_every_part(controller):
    status = controller.cognitive_core_status()
    assert set(status) == {"events", "resources", "graph", "integrator"}
    assert status["resources"]["enabled"] is True
    assert status["graph"]["entities"] == 0


# -- resource deferrals become events -------------------------------------


def test_a_deferred_resource_request_publishes_an_event(controller):
    from core.resource_governor import ResourceRequest

    controller.resource_governor.request(ResourceRequest("huge", memory_bytes=10**15, model=True))
    assert "resource.deferred" in [event.name for event in controller.events.history]


# -- nothing else broke ---------------------------------------------------


def test_the_ontology_is_still_seeded(controller):
    assert len(controller.ontology.concepts) > 0
    assert controller.causal_graph is not None
    assert len(controller.causal_graph.links) > 0


def test_a_full_cycle_still_runs(controller):
    """The Phase-1 wiring must not disturb the existing cognitive cycle."""
    from core.belief import Belief
    from core.evidence_processor import EvidencePolarity
    from core.observation import Observation

    belief = Belief(statement="porosity is a rock property")
    controller.graph.add_belief(belief)
    controller.state.add_belief(belief)
    observation = Observation(content="porosity is a rock property", source="test")
    result = controller.process(observation, belief, EvidencePolarity.SUPPORTS)
    assert result is not None
    assert [event.name for event in controller.events.history].count("cycle.started") == 1
    assert EVENT_CYCLE_FINISHED in [event.name for event in controller.events.history]
