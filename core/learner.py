from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.belief_learner import BeliefLearner
from core.hypothesis import Hypothesis
from core.hypothesis_learner import HypothesisLearner, HypothesisLearningResult
from core.learning_history import LearningEvent, LearningHistory
from core.learning_signal import LearningSignal
from core.pattern_detector import Pattern
from core.pattern_learner import PatternLearner, PatternLearningResult
from core.prediction_error import PredictionError
from core.reinforcement import ReinforcementLearner, ReinforcementUpdate


@dataclass(frozen=True)
class LearningCycle:
    signal: LearningSignal
    belief_before: float
    belief_after: float
    reinforcement: ReinforcementUpdate | None


class Learner:
    """Central learning coordinator using prediction error as its only update signal."""
    def __init__(self) -> None:
        self.beliefs = BeliefLearner()
        self.reinforcement = ReinforcementLearner()
        self.history = LearningHistory()
        self.hypotheses = HypothesisLearner()
        self.patterns = PatternLearner()

    def learn_from_prediction(self, belief: Belief, error: PredictionError, action: str | None = None) -> LearningCycle:
        signal = LearningSignal.from_prediction_error(error)
        result = self.beliefs.learn(belief, signal)
        self.history.record(LearningEvent(signal, belief.id, result.previous, result.updated))
        reinforcement = self.reinforcement.reinforce(action, signal) if action is not None else None
        return LearningCycle(signal, result.previous, result.updated, reinforcement)

    def learn_hypothesis(self, hypothesis: Hypothesis, signal: LearningSignal) -> HypothesisLearningResult:
        """Re-estimate a hypothesis confidence from one verified outcome."""
        return self.hypotheses.learn(hypothesis, signal)

    def learn_pattern(self, pattern: Pattern, signal: LearningSignal) -> PatternLearningResult:
        """Re-estimate a discovered pattern confidence from one verified outcome."""
        return self.patterns.learn(pattern, signal)
