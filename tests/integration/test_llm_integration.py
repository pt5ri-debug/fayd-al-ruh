"""Linking a real LLM through model_interface while keeping guardrails intact.

The model is forced into a structured shape and can only propose. These tests
prove it cannot mutate beliefs, cannot execute tools, and that its proposals
are registered through the existing validation layers.
"""

import json

import pytest
from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.model_interface import CallableModel, ModelRequest, MultimodalInput
from core.observation import Observation
from core.provenance import ProvenanceStage
from core.structured_output import MODEL_OUTPUT_SCHEMA, StructuredOutputError

pytestmark = pytest.mark.integration


def _model(payload: dict) -> CallableModel:
    text = json.dumps(payload)
    return CallableModel("fake-llm", lambda request: text)


def _full_payload() -> dict:
    return {
        "concepts": [{"label": "rain", "aliases": ["precipitation"]}],
        "evidence": [{"description": "ground is wet", "source": "sensor", "supports": True, "strength": 0.8}],
        "hypotheses": [{"statement": "rain wets the ground", "confidence": 0.7}],
        "questions": [{"text": "does rain always wet the ground", "topic": "causality", "priority": 0.6}],
        "plans": [{
            "goal": "confirm rain effect",
            "priority": 0.9,
            "steps": [
                {"description": "observe"},
                {"description": "measure", "depends_on": ["observe"]},
            ],
        }],
    }


def test_model_is_forced_to_structured_schema():
    captured = {}

    def generate(request):
        captured["schema"] = request.response_schema
        return json.dumps(_full_payload())

    controller = CognitiveController()
    controller.analyze_with_model(CallableModel("fake", generate), ModelRequest((MultimodalInput("text", "analyze"),)))

    assert captured["schema"] is MODEL_OUTPUT_SCHEMA
    assert captured["schema"]["additionalProperties"] is False


def test_proposals_are_registered_through_existing_layers():
    controller = CognitiveController()
    analysis = controller.analyze_with_model(_model(_full_payload()), ModelRequest((MultimodalInput("text", "analyze"),)))

    assert [c.label for c in analysis.concepts] == ["rain"]
    assert controller.ontology.find_concepts("precipitation")
    assert controller.graph.get_evidence(analysis.evidence[0].id) is analysis.evidence[0]
    assert controller.state.hypotheses[analysis.hypotheses[0].id] is analysis.hypotheses[0]
    assert analysis.questions[0].text.endswith("?")
    assert [step.description for step in analysis.plans[0].steps] == ["observe", "measure"]
    assert controller.graph.validate() == []


def test_proposals_are_recorded_as_provenance():
    controller = CognitiveController()
    analysis = controller.analyze_with_model(_model(_full_payload()), ModelRequest((MultimodalInput("text", "analyze"),)))

    records = controller.provenance.for_stage(ProvenanceStage.MODEL_PROPOSAL)
    assert {record.entity_id for record in records} >= set(analysis.registered_ids)
    assert all(record.source.startswith("model:") for record in records)


def test_model_cannot_change_belief_confidence_or_status():
    controller = CognitiveController()
    belief = Belief("ground stays dry", 0.5)
    controller.graph.add_belief(belief)
    before = (belief.confidence, belief.status, list(belief.supporting_evidence))

    controller.analyze_with_model(
        _model({"evidence": [{"description": "ground is wet", "source": "sensor"}]}),
        ModelRequest((MultimodalInput("text", "x"),)),
        belief=belief,
    )

    # Only evidence attachment is allowed; confidence and status are untouched.
    assert belief.confidence == before[0]
    assert belief.status == before[1]
    assert len(belief.supporting_evidence) == 1


def test_model_cannot_emit_belief_updates():
    controller = CognitiveController()
    payload = {"beliefs": [{"statement": "rain is wet", "confidence": 1.0}], "evidence": []}

    with pytest.raises(StructuredOutputError):
        controller.analyze_with_model(_model(payload), ModelRequest((MultimodalInput("text", "x"),)))


def test_model_cannot_request_tool_execution():
    controller = CognitiveController()
    payload = {"tool_calls": [{"name": "run", "args": {}}], "evidence": []}

    with pytest.raises(StructuredOutputError):
        controller.analyze_with_model(_model(payload), ModelRequest((MultimodalInput("text", "x"),)))


def test_model_never_executes_a_plan():
    controller = CognitiveController()
    payload = {"plans": [{"goal": "do something dangerous", "steps": [{"description": "rm -rf /"}]}]}

    analysis = controller.analyze_with_model(_model(payload), ModelRequest((MultimodalInput("text", "x"),)))

    # The plan is ordered but no step is executed or marked complete.
    assert all(not step.completed for step in analysis.plans[0].steps)


def test_analysis_integrates_with_cognitive_cycle():
    controller = CognitiveController()
    belief = Belief("rain wets the ground", 0.5)
    controller.graph.add_belief(belief)

    controller.analyze_with_model(
        _model({"evidence": [{"description": "ground is wet", "source": "sensor"}]}),
        ModelRequest((MultimodalInput("text", "observe rain"),)),
        belief=belief,
    )

    cycle = controller.process(
        Observation("Rain falls on the ground.", "weather_sensor", 0.9),
        belief,
        EvidencePolarity.SUPPORTS,
    )

    assert cycle.loop_result.evidence_id is not None
    assert controller.validate_contexts() == []


def test_in_project_language_model_flows_through_the_same_validation():
    """The internal transformer proposes through LLMReasoner, not around it."""
    from ai.local_model import DEFAULT_CHECKPOINT, LocalLanguageModel

    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("no committed in-project checkpoint")

    model = LocalLanguageModel(temperature=0.7, seed=0)
    controller = CognitiveController()
    analysis = controller.analyze_with_model(
        model,
        ModelRequest((MultimodalInput("text", "analyze: water makes the soil wet"),)),
    )

    # Whatever the model proposed, it arrived through the validated layers.
    assert analysis.response.model_name.startswith("fayd-local-llm")
    assert controller.graph.validate() == []
    assert all(
        record.source.startswith("model:")
        for record in controller.provenance.for_stage(ProvenanceStage.MODEL_PROPOSAL)
    )
