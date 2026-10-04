from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class Goal:
    """A desired set of state facts for the symbolic planner."""
    desired_facts: frozenset[str]
    description: str
    priority: float = .5
    id: str = field(default_factory=lambda: f"PLAN-GOAL-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.description.strip() or not self.desired_facts or any(not fact.strip() for fact in self.desired_facts):
            raise ValueError("Goal needs a description and desired facts.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Goal priority must be between 0 and 1.")
