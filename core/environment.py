from __future__ import annotations

from core.action import Action
from core.observation import Observation
from core.state import State
from core.state_transition import StateTransition


class Environment:
    """Pure symbolic environment; all actions are in-memory state transitions."""
    def __init__(self, state: State | None = None, transition: StateTransition | None = None) -> None:
        self.state = state or State()
        self.transition = transition or StateTransition()

    def observe(self, confidence: float = 1.0, result: float = 1.0) -> Observation:
        content = ", ".join(sorted(self.state.facts)) or "empty environment"
        return Observation(content, "environment", confidence, {"facts": sorted(self.state.facts), "result": result})

    def apply(self, action: Action) -> State:
        self.state = self.transition.apply(self.state, action)
        return self.state
