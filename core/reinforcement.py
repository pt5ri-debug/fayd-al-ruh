from __future__ import annotations

from dataclasses import dataclass

from core.learning_signal import LearningSignal


@dataclass(frozen=True)
class ReinforcementUpdate:
    action: str
    old_value: float
    new_value: float
    reward: float


class ReinforcementLearner:
    """A bounded value learner. It chooses no actions and stores no tools."""
    def __init__(self, learning_rate: float = .1) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1.")
        self.learning_rate = learning_rate
        self.values: dict[str, float] = {}

    def reinforce(self, action: str, signal: LearningSignal) -> ReinforcementUpdate:
        if not action.strip():
            raise ValueError("Action cannot be empty.")
        old = self.values.get(action, .5)
        new = round(max(0.0, min(1.0, old + self.learning_rate * (signal.reward - old))), 6)
        self.values[action] = new
        return ReinforcementUpdate(action, old, new, signal.reward)
