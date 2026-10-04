from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from core.prediction import Prediction


@dataclass
class PredictionError:
    prediction_id: str
    expected_confidence: float
    actual_result: float
    error: float
    id: str = field(default_factory=lambda: f"ERR-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not 0.0 <= self.expected_confidence <= 1.0:
            raise ValueError(
                "Expected confidence must be between 0 and 1."
            )

        if not 0.0 <= self.actual_result <= 1.0:
            raise ValueError(
                "Actual result must be between 0 and 1."
            )

        if not 0.0 <= self.error <= 1.0:
            raise ValueError(
                "Prediction error must be between 0 and 1."
            )


class PredictionErrorCalculator:
    """
    Calculates the difference between a prediction
    and the observed outcome.
    """

    def calculate(
        self,
        prediction: Prediction,
        actual_result: float,
    ) -> PredictionError:

        if not 0.0 <= actual_result <= 1.0:
            raise ValueError(
                "Actual result must be between 0 and 1."
            )

        error = abs(
            prediction.confidence
            - actual_result
        )

        error = round(error, 6)

        return PredictionError(
            prediction_id=prediction.id,
            expected_confidence=prediction.confidence,
            actual_result=actual_result,
            error=error,
        )
