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
