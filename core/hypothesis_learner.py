from __future__ import annotations

from dataclasses import dataclass

from core.hypothesis import Hypothesis
from core.learning_signal import LearningSignal


@dataclass(frozen=True)
class HypothesisLearningResult:
    hypothesis_id: str
    previous_confidence: float
    estimated_confidence: float


class HypothesisLearner:
    """Evaluates a hypothesis from outcomes without promoting it to fact."""
    def learn(self, hypothesis: Hypothesis, signal: LearningSignal) -> HypothesisLearningResult:
        estimate = max(0.0, min(1.0, hypothesis.confidence + signal.direction * signal.prediction_error * .1))
        return HypothesisLearningResult(hypothesis.id, hypothesis.confidence, round(estimate, 6))
