from __future__ import annotations

from dataclasses import dataclass

from core.cognitive_state import CognitiveState
from core.state_history import StateHistory


@dataclass(frozen=True)
class LearningMetrics:
    prediction_count: int
    resolved_prediction_count: int
    correct_prediction_count: int

    prediction_accuracy: float
    average_prediction_error: float

    confidence_change: float

    @classmethod
    def calculate(
        cls,
        state: CognitiveState,
        history: StateHistory,
    ) -> LearningMetrics:

        predictions = list(
            state.predictions.values()
        )

        resolved = [
            prediction
            for prediction in predictions
            if prediction.fulfilled is not None
        ]

        correct = [
            prediction
            for prediction in resolved
            if prediction.fulfilled is True
        ]

        if resolved:

            accuracy = (
                len(correct)
                / len(resolved)
            )

        else:

            accuracy = 0.0

        errors = state.prediction_errors

        if errors:

            average_error = (
                sum(
                    error.error
                    for error in errors
                )
                / len(errors)
            )

        else:

            average_error = 0.0

        return cls(
            prediction_count=len(predictions),
            resolved_prediction_count=len(resolved),
            correct_prediction_count=len(correct),
            prediction_accuracy=round(
                accuracy,
                6,
            ),
            average_prediction_error=round(
                average_error,
                6,
            ),
            confidence_change=(
                history.confidence_change()
            ),
        )
