from core.hypothesis_test import HypothesisTester
from core.prediction import Prediction


def test_hypothesis_test_returns_prediction_error():
    test = HypothesisTester().test(Prediction("rain", .8, "H"), 1.)
    assert test.prediction_error.error == .2 and test.supports_hypothesis
