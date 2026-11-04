from __future__ import annotations

from collections import deque
from typing import Any

from core.action import Action
from core.action_model import ActionModel
from core.goal import Goal, GoalStatus, PlanStep
from core.plan import Plan
from core.state import State


class Planner:
    """Single planning engine.

    ``create_plan`` searches for a symbolic action sequence; ``order_steps``
    topologically orders an explicit, dependency-declared task plan.

    The planner also **persists** the ordered task plans it produces, so a plan
    survives a restart (Phase 8, section 6). Persistence is opt-in: with no
    ``data_root`` the planner keeps its plans in memory only, exactly as before.
    """

    def __init__(self, model: ActionModel | None = None, *, data_root: str | None = None) -> None:
        self.model = model or ActionModel()
        self.data_root = data_root
        #: The ordered task plans this planner has produced, newest last.
        self.plans: list[list[PlanStep]] = []
        if data_root:
            self._load_plans()

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
        """Topologically order an explicit task plan; never executes it.

        The ordered plan is recorded (and persisted when a ``data_root`` is set),
        so the caller can retrieve it later.
        """
        known = {step.description for step in steps}
        for step in steps:
            if any(dependency not in known for dependency in step.depends_on):
                raise ValueError("Each plan dependency must name another step.")
        goal.status = GoalStatus.ACTIVE
        ordered = self._topological_order(steps)
        self.record_plan(ordered)
        return ordered

    def record_plan(self, steps: list[PlanStep]) -> None:
        """Remember an ordered plan (and persist it when a data root is set)."""
        self.plans.append(list(steps))
        if len(self.plans) > 200:
            self.plans = self.plans[-200:]
        self.save_plans()

    def next_step(self, steps: list[PlanStep]) -> PlanStep | None:
        completed = {step.description for step in steps if step.completed}
        return next(
            (step for step in steps if not step.completed and set(step.depends_on) <= completed),
            None,
        )

    # -------------------------------------------------------------- persistence

    def _store(self) -> Any:
        if not self.data_root:
            return None
        try:
            from storage.json_store import JsonStore

            return JsonStore(self.data_root)
        except Exception:
            return None

    def save_plans(self) -> bool:
        """Persist the recorded plans (best-effort: never raises)."""
        store = self._store()
        if store is None:
            return False
        payload = [
            [
                {"description": s.description, "depends_on": list(s.depends_on), "completed": s.completed}
                for s in steps
            ]
            for steps in self.plans
        ]
        try:
            store.save("planning", "ordered_steps", payload)
            return True
        except Exception:
            return False

    def _load_plans(self) -> int:
        store = self._store()
        if store is None:
            return 0
        try:
            data = store.load("planning", "ordered_steps")
        except Exception:
            return 0
        if not isinstance(data, list):
            return 0
        loaded: list[list[PlanStep]] = []
        for steps in data:
            if not isinstance(steps, list):
                continue
            loaded.append(
                [
                    PlanStep(
                        description=str(step.get("description", "")),
                        depends_on=tuple(str(d) for d in step.get("depends_on", ()) or ()),
                        completed=bool(step.get("completed", False)),
                    )
                    for step in steps
                    if isinstance(step, dict)
                ]
            )
        self.plans = loaded[-200:]
        return len(self.plans)

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
