from __future__ import annotations

from dataclasses import dataclass

from core.prediction import Prediction
from core.prediction_error import PredictionError, PredictionErrorCalculator


@dataclass(frozen=True)
class HypothesisTest:
    prediction_error: PredictionError
    supports_hypothesis: bool
    observed_result: float


class HypothesisTester:
    """Tests a prediction against an aggregate observed result in [0, 1]."""
    def __init__(self, calculator: PredictionErrorCalculator | None = None) -> None:
        self.calculator = calculator or PredictionErrorCalculator()

    def test(self, prediction: Prediction, observed_result: float) -> HypothesisTest:
        error = self.calculator.calculate(prediction, observed_result)
        return HypothesisTest(error, observed_result >= .5, observed_result)
