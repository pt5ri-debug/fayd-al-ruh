from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.belief_learner import BeliefLearner
from core.learning_history import LearningEvent, LearningHistory
from core.learning_signal import LearningSignal
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

    def learn_from_prediction(self, belief: Belief, error: PredictionError, action: str | None = None) -> LearningCycle:
        signal = LearningSignal.from_prediction_error(error)
        result = self.beliefs.learn(belief, signal)
        self.history.record(LearningEvent(signal, belief.id, result.previous, result.updated))
        reinforcement = self.reinforcement.reinforce(action, signal) if action is not None else None
        return LearningCycle(signal, result.previous, result.updated, reinforcement)
