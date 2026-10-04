from __future__ import annotations

from dataclasses import dataclass

from core.action import Action
from core.environment import Environment
from core.state import State


@dataclass(frozen=True)
class ExecutionResult:
    action_id: str
    resulting_state: State


class ActionExecutor:
    """Executes only actions against the injected symbolic Environment."""
    def execute(self, environment: Environment, action: Action) -> ExecutionResult:
        return ExecutionResult(action.id, environment.apply(action))
