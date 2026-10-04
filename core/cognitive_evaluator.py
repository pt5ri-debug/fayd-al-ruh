from __future__ import annotations

from dataclasses import dataclass

from core.learning_metrics import LearningMetrics


@dataclass(frozen=True)
class CognitiveEvaluation:
    learning_score: float
    prediction_score: float
    stability_score: float
    overall_score: float


class CognitiveEvaluator:
    """
    Evaluates the quality of the cognitive system
    without modifying its state.
    """

    def evaluate(
        self,
        metrics: LearningMetrics,
    ) -> CognitiveEvaluation:

        prediction_score = (
            metrics.prediction_accuracy
        )

        learning_score = (
            1.0
            - metrics.average_prediction_error
        )

        stability_score = (
            1.0
            - abs(metrics.confidence_change)
        )

        learning_score = self._clamp(
            learning_score
        )

        prediction_score = self._clamp(
            prediction_score
        )

        stability_score = self._clamp(
            stability_score
        )

        overall_score = (
            learning_score
            + prediction_score
            + stability_score
        ) / 3.0

        return CognitiveEvaluation(
            learning_score=round(
                learning_score,
                6,
            ),
            prediction_score=round(
                prediction_score,
                6,
            ),
            stability_score=round(
                stability_score,
                6,
            ),
            overall_score=round(
                overall_score,
                6,
            ),
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )