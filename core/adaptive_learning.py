from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.prediction_error import PredictionError


@dataclass
class AdaptiveLearningResult:
    belief_id: str
    old_confidence: float
    new_confidence: float
    adjustment: float
    error: float


class AdaptiveLearningEngine:
    """
    Adjusts belief confidence according to prediction error.

    Learning rate limits how quickly confidence can change.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_adjustment: float = 0.2,
    ) -> None:

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                "learning_rate must be between 0 and 1."
            )

        if not 0.0 < max_adjustment <= 1.0:
            raise ValueError(
                "max_adjustment must be between 0 and 1."
            )

        self.learning_rate = learning_rate
        self.max_adjustment = max_adjustment

    def learn(
        self,
        belief: Belief,
        prediction_error: PredictionError,
    ) -> AdaptiveLearningResult:

        old_confidence = belief.confidence

        adjustment = (
            prediction_error.error
            * self.learning_rate
        )

        adjustment = min(
            adjustment,
            self.max_adjustment,
        )

        if prediction_error.actual_result < old_confidence:
            new_confidence = (
                old_confidence - adjustment
            )
        else:
            new_confidence = (
                old_confidence + adjustment
            )

        new_confidence = max(
            0.0,
            min(
                1.0,
                new_confidence,
            ),
        )

        new_confidence = round(
            new_confidence,
            6,
        )

        belief.set_confidence(
            new_confidence
        )

        return AdaptiveLearningResult(
            belief_id=belief.id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            adjustment=adjustment,
            error=prediction_error.error,
        )