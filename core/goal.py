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
    """A single goal model used across planning, decisions, and task tracking.

    ``desired_facts`` is the symbolic state target consumed by the planner and
    the decision/utility/risk models. Task-level goals that do not target a
    symbolic state may leave it empty and rely on ``description``/``priority``.
    """

    description: str
    desired_facts: frozenset[str] = frozenset()
    priority: float = .5
    id: str = field(default_factory=lambda: f"GOAL-{uuid4().hex[:8]}")
    status: GoalStatus = GoalStatus.PENDING

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Goal description cannot be empty.")
        if any(not fact.strip() for fact in self.desired_facts):
            raise ValueError("Goal desired facts cannot be empty strings.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Goal priority must be between 0 and 1.")


@dataclass
class PlanStep:
    description: str
    depends_on: tuple[str, ...] = ()
    completed: bool = False
