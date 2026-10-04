from __future__ import annotations

from core.belief import Belief
from core.learning_signal import LearningSignal
from core.online_learner import OnlineLearner, OnlineLearningResult


class BeliefLearner:
    def __init__(self, online: OnlineLearner | None = None) -> None:
        self.online = online or OnlineLearner()

    def learn(self, belief: Belief, signal: LearningSignal) -> OnlineLearningResult:
        return self.online.update_belief(belief, signal)
