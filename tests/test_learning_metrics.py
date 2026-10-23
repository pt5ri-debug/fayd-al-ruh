from core.cognitive_state import CognitiveState
from core.learning_metrics import LearningMetrics
from core.prediction import Prediction
from core.prediction_error import PredictionError
from core.state_history import StateHistory


def test_empty_metrics():

    state = CognitiveState()
    history = StateHistory()

    metrics = LearningMetrics.calculate(
        state,
        history,
    )

    assert metrics.prediction_count == 0
    assert metrics.resolved_prediction_count == 0
    assert metrics.correct_prediction_count == 0
    assert metrics.prediction_accuracy == 0.0
    assert metrics.average_prediction_error == 0.0
    assert metrics.confidence_change == 0.0


def test_prediction_accuracy():

    state = CognitiveState()
    history = StateHistory()

    correct = Prediction(
        statement="Prediction A",
        confidence=0.8,
        hypothesis_id="HYP-1",
    )

    incorrect = Prediction(
        statement="Prediction B",
        confidence=0.8,
        hypothesis_id="HYP-2",
    )

    correct.resolve(True)
    incorrect.resolve(False)

    state.add_prediction(correct)
    state.add_prediction(incorrect)

    metrics = LearningMetrics.calculate(
        state,
        history,
    )

    assert metrics.prediction_count == 2
    assert metrics.resolved_prediction_count == 2
    assert metrics.correct_prediction_count == 1
    assert metrics.prediction_accuracy == 0.5


def test_average_prediction_error():

    state = CognitiveState()
    history = StateHistory()

    state.add_prediction_error(
        PredictionError(
            prediction_id="PRD-1",
            expected_confidence=0.8,
            actual_result=0.6,
            error=0.2,
        )
    )

    state.add_prediction_error(
        PredictionError(
            prediction_id="PRD-2",
            expected_confidence=0.7,
            actual_result=0.3,
            error=0.4,
        )
    )

    metrics = LearningMetrics.calculate(
        state,
        history,
    )

    assert metrics.average_prediction_error == 0.3
