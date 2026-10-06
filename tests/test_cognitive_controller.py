from core.belief import Belief
from core.cognitive_controller import (
    CognitiveController,
)
from core.evidence_processor import EvidencePolarity
from core.observation import Observation


def test_controller_processes_cognitive_cycle():

    controller = CognitiveController()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    observation = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    result = controller.process(
        observation=observation,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    assert result.report.cycle_count == 1

    assert result.report.belief_count == 1

    assert (
        result.loop_result.observation
        is observation
    )

    assert controller.history.size == 1


def test_controller_uses_shared_state():

    controller = CognitiveController()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    observation = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    controller.process(
        observation=observation,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    assert (
        controller.state.beliefs[belief.id]
        is belief
    )


def test_controller_rejects_unknown_prediction():

    controller = CognitiveController()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    try:
        controller.evaluate_prediction(
            prediction_id="PRD-UNKNOWN",
            belief=belief,
            actual_result=0.5,
        )

        assert False

    except KeyError:
        pass

def test_controller_creates_cognitive_trace():

    controller = CognitiveController()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    first = Observation(
        content="The ball moved.",
        source="sensor_a",
        confidence=0.9,
    )

    second = Observation(
        content="The ball moved.",
        source="sensor_b",
        confidence=0.8,
    )

    controller.process(
        observation=first,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    controller.process(
        observation=second,
        belief=belief,
        polarity=EvidencePolarity.SUPPORTS,
    )

    assert controller.state.trace_count == 2

    trace = list(
        controller.state.traces.values()
    )[-1]

    assert trace.observation_id == second.id

    assert trace.belief_id == belief.id

    assert len(trace.pattern_keys) == 1

    assert len(trace.hypothesis_ids) == 1

    assert len(trace.prediction_ids) == 1


def test_controller_infers_ontology_relation_with_steps():
    controller = CognitiveController()

    inference = controller.infer_ontology_relation("البئر", "الطبقة", "penetrates")

    assert inference is not None
    assert inference.derived is True
    assert inference.steps
    assert 0.0 < inference.confidence < 1.0


def test_controller_returns_none_for_unknown_ontology_relation():
    controller = CognitiveController()

    assert controller.infer_ontology_relation("الماء", "الضغط", "measures") is None
    assert controller.infer_ontology_relation("لا_يوجد", "الطبقة", "penetrates") is None


def test_controller_derived_relations_stay_out_of_the_graph():
    controller = CognitiveController()

    stored = {
        (relation.source_id, relation.target_id, relation.relation_type)
        for relation in controller.ontology.relations.values()
    }

    for inference in controller.derive_ontology_relations():
        key = (inference.source_id, inference.target_id, inference.relation_type)
        assert key not in stored
