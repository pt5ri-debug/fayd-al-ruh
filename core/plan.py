from __future__ import annotations

from dataclasses import dataclass

from core.action import Action
from core.state import State
from core.state_transition import StateTransition


@dataclass(frozen=True)
class Plan:
    actions: tuple[Action, ...]
    predicted_final_state: State

    @property
    def cost(self) -> float:
        return round(sum(action.cost for action in self.actions), 6)

    def execute_simulation(self, initial_state: State, transition: StateTransition | None = None) -> State:
        """Executes only pure in-memory transitions, never external actions."""
        engine = transition or StateTransition()
        state = initial_state
        for action in self.actions:
            state = engine.apply(state, action)
        return state
