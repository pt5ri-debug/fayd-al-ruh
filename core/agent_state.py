from __future__ import annotations

from dataclasses import dataclass

from core.observation import Observation


@dataclass
class AgentState:
    cycle_count: int = 0
    last_observation: Observation | None = None
    last_action_id: str | None = None
    last_error: str | None = None

    def record_observation(self, observation: Observation) -> None:
        self.cycle_count += 1
        self.last_observation = observation

    def record_action(self, action_id: str) -> None:
        self.last_action_id = action_id
