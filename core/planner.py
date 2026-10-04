from __future__ import annotations

from collections import deque

from core.action import Action
from core.action_model import ActionModel
from core.goal import Goal
from core.plan import Plan
from core.state import State


class Planner:
    """Bounded breadth-first symbolic planner with predicted outcomes."""
    def __init__(self, model: ActionModel | None = None) -> None:
        self.model = model or ActionModel()

    def create_plan(self, initial_state: State, goal: Goal, actions: list[Action], max_depth: int = 6) -> Plan | None:
        if max_depth < 0:
            raise ValueError("Maximum depth cannot be negative.")
        if initial_state.satisfies(goal.desired_facts):
            return Plan((), initial_state)
        queue = deque([(initial_state, tuple())])
        seen = {initial_state.facts}
        while queue:
            state, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for action in actions:
                if not state.satisfies(action.preconditions):
                    continue
                predicted = self.model.predict(state, action)
                if predicted.facts in seen:
                    continue
                next_path = path + (action,)
                if predicted.satisfies(goal.desired_facts):
                    return Plan(next_path, predicted)
                seen.add(predicted.facts)
                queue.append((predicted, next_path))
        return None
