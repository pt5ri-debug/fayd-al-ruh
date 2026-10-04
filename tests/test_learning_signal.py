from core.learning_signal import LearningSignal
from core.prediction_error import PredictionError


def test_prediction_error_becomes_learning_signal():
    signal = LearningSignal.from_prediction_error(PredictionError("P", .8, 1., .2))
    assert signal.reward == .8 and signal.direction == 1
