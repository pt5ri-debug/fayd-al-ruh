from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    description: str
    priority: float = 0.5
    id: str = field(default_factory=lambda: f"GOAL-{uuid4().hex[:8]}")
    status: GoalStatus = GoalStatus.PENDING

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Goal description cannot be empty.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Goal priority must be between 0 and 1.")


@dataclass
class PlanStep:
    description: str
    depends_on: tuple[str, ...] = ()
    completed: bool = False


class GoalPlanner:
    """Small transparent planner: it never executes actions by itself."""

    def plan(self, goal: Goal, steps: list[PlanStep]) -> list[PlanStep]:
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
