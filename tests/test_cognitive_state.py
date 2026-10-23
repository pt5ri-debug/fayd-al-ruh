from core.belief import Belief
from core.cognitive_state import CognitiveState
from core.hypothesis import Hypothesis
from core.pattern_detector import Pattern
from core.prediction import Prediction
from core.prediction_error import PredictionError


def test_empty_cognitive_state():

    state = CognitiveState()

    assert state.belief_count == 0
    assert state.pattern_count == 0
    assert state.hypothesis_count == 0
    assert state.prediction_count == 0
    assert state.error_count == 0
    assert state.cycle_count == 0


def test_add_belief():

    state = CognitiveState()

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    state.add_belief(belief)

    assert state.belief_count == 1
    assert state.beliefs[belief.id] is belief


def test_add_pattern():

    state = CognitiveState()

    pattern = Pattern(
        key="the ball moved.",
        occurrences=3,
        confidence=0.8,
        memory_ids=[
            "OBS-1",
            "OBS-2",
            "OBS-3",
        ],
    )

    state.add_pattern(pattern)

    assert state.pattern_count == 1
    assert state.patterns[
        "the ball moved."
    ] is pattern


def test_add_hypothesis():

    state = CognitiveState()

    hypothesis = Hypothesis(
        statement="The ball moves repeatedly.",
        confidence=0.7,
        source_pattern="the ball moved.",
    )

    state.add_hypothesis(hypothesis)

    assert state.hypothesis_count == 1
    assert state.hypotheses[
        hypothesis.id
    ] is hypothesis


def test_add_prediction():

    state = CognitiveState()

    prediction = Prediction(
        statement="The ball will move.",
        confidence=0.7,
        hypothesis_id="HYP-001",
    )

    state.add_prediction(prediction)

    assert state.prediction_count == 1
    assert state.predictions[
        prediction.id
    ] is prediction


def test_add_prediction_error():

    state = CognitiveState()

    error = PredictionError(
        prediction_id="PRD-001",
        expected_confidence=0.8,
        actual_result=0.3,
        error=0.5,
    )

    state.add_prediction_error(error)

    assert state.error_count == 1
    assert state.prediction_errors[0] is error


def test_cycle_count():

    state = CognitiveState()

    state.next_cycle()
    state.next_cycle()
    state.next_cycle()

    assert state.cycle_count == 3
