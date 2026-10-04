from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.learning_signal import LearningSignal


@dataclass(frozen=True)
class OnlineLearningResult:
    target_id: str
    previous: float
    updated: float
    adjustment: float


class OnlineLearner:
    """Applies one small, bounded update per incoming prediction error."""
    def __init__(self, learning_rate: float = .1) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1.")
        self.learning_rate = learning_rate

    def update_belief(self, belief: Belief, signal: LearningSignal) -> OnlineLearningResult:
        previous = belief.confidence
        adjustment = signal.direction * signal.prediction_error * self.learning_rate
        updated = round(max(0.0, min(1.0, previous + adjustment)), 6)
        belief.set_confidence(updated)
        return OnlineLearningResult(belief.id, previous, updated, round(adjustment, 6))
