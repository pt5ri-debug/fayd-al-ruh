from __future__ import annotations

from dataclasses import dataclass

from core.cognitive_evaluator import CognitiveEvaluation
from core.cognitive_state import CognitiveState
from core.learning_metrics import LearningMetrics


@dataclass(frozen=True)
class CognitiveReport:
    cycle_count: int

    belief_count: int
    pattern_count: int
    hypothesis_count: int
    prediction_count: int
    prediction_error_count: int

    learning_score: float
    prediction_score: float
    stability_score: float
    overall_score: float

    prediction_accuracy: float
    average_prediction_error: float

    @classmethod
    def generate(
        cls,
        state: CognitiveState,
        metrics: LearningMetrics,
        evaluation: CognitiveEvaluation,
    ) -> "CognitiveReport":

        return cls(
            cycle_count=state.cycle_count,

            belief_count=state.belief_count,
            pattern_count=state.pattern_count,
            hypothesis_count=state.hypothesis_count,
            prediction_count=state.prediction_count,
            prediction_error_count=state.error_count,

            learning_score=evaluation.learning_score,
            prediction_score=evaluation.prediction_score,
            stability_score=evaluation.stability_score,
            overall_score=evaluation.overall_score,

            prediction_accuracy=(
                metrics.prediction_accuracy
            ),
            average_prediction_error=(
                metrics.average_prediction_error
            ),
        )