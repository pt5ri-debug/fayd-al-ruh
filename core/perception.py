from __future__ import annotations

from core.environment import Environment
from core.observation import Observation


class Perception:
    def perceive(self, environment: Environment) -> Observation:
        return environment.observe()
