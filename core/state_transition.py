from __future__ import annotations

from core.action import Action
from core.state import State


class StateTransition:
    """Pure transition function used for prediction and safe simulation."""
    def apply(self, state: State, action: Action) -> State:
        if not state.satisfies(action.preconditions):
            missing = action.preconditions - state.facts
            raise ValueError(f"Action preconditions not satisfied: {sorted(missing)}")
        return State((state.facts - action.remove_effects) | action.add_effects)
