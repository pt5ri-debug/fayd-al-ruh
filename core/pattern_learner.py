from __future__ import annotations

from dataclasses import dataclass

from core.learning_signal import LearningSignal
from core.pattern_detector import Pattern


@dataclass(frozen=True)
class PatternLearningResult:
    pattern_key: str
    previous_confidence: float
    estimated_confidence: float


class PatternLearner:
    def learn(self, pattern: Pattern, signal: LearningSignal) -> PatternLearningResult:
        estimate = max(0.0, min(1.0, pattern.confidence + signal.direction * signal.prediction_error * .05))
        return PatternLearningResult(pattern.key, pattern.confidence, round(estimate, 6))
