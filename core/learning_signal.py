from __future__ import annotations

from dataclasses import dataclass

from core.prediction_error import PredictionError


@dataclass(frozen=True)
class LearningSignal:
    prediction_id: str
    prediction_error: float
    reward: float
    direction: int

    @classmethod
    def from_prediction_error(cls, error: PredictionError) -> "LearningSignal":
        reward = round(1.0 - error.error, 6)
        direction = 1 if error.actual_result >= error.expected_confidence else -1
        return cls(error.prediction_id, error.error, reward, direction)
