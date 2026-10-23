from __future__ import annotations

from collections import deque

from core.action import Action
from core.action_model import ActionModel
from core.goal import Goal, GoalStatus, PlanStep
from core.plan import Plan
from core.state import State


class Planner:
    """Single planning engine.

    ``create_plan`` searches for a symbolic action sequence; ``order_steps``
    topologically orders an explicit, dependency-declared task plan.
    """
    def __init__(self, model: ActionModel | None = None) -> None:
        self.model = model or ActionModel()

    def create_plan(self, initial_state: State, goal: Goal, actions: list[Action], max_depth: int = 6) -> Plan | None:
        if max_depth < 0:
            raise ValueError("Maximum depth cannot be negative.")
        if initial_state.satisfies(goal.desired_facts):
            return Plan((), initial_state)
        queue: deque[tuple[State, tuple[Action, ...]]] = deque([(initial_state, ())])
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
                next_path = (*path, action)
                if predicted.satisfies(goal.desired_facts):
                    return Plan(next_path, predicted)
                seen.add(predicted.facts)
                queue.append((predicted, next_path))
        return None

    def order_steps(self, goal: Goal, steps: list[PlanStep]) -> list[PlanStep]:
        """Topologically order an explicit task plan; never executes it."""
        known = {step.description for step in steps}
        for step in steps:
            if any(dependency not in known for dependency in step.depends_on):
                raise ValueError("Each plan dependency must name another step.")
        goal.status = GoalStatus.ACTIVE
        return self._topological_order(steps)

    def next_step(self, steps: list[PlanStep]) -> PlanStep | None:
        completed = {step.description for step in steps if step.completed}
        return next(
            (step for step in steps if not step.completed and set(step.depends_on) <= completed),
            None,
        )

    @staticmethod
    def _topological_order(steps: list[PlanStep]) -> list[PlanStep]:
        result: list[PlanStep] = []
        remaining = list(steps)
        while remaining:
            available = [step for step in remaining if set(step.depends_on) <= {item.description for item in result}]
            if not available:
                raise ValueError("Plan contains a circular dependency.")
            result.extend(available)
            remaining = [step for step in remaining if step not in available]
        return result
