from __future__ import annotations

from core.action import Action
from core.state import State
from core.state_transition import StateTransition


class ActionModel:
    """Predicts an action's outcome using the declared state transition."""
    def __init__(self, transition: StateTransition | None = None) -> None:
        self.transition = transition or StateTransition()

    def predict(self, state: State, action: Action) -> State:
        return self.transition.apply(state, action)
