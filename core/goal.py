from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class GoalStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """True when the goal can no longer make progress on its own."""
        return self in {
            GoalStatus.COMPLETED,
            GoalStatus.PARTIAL,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        }


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

    def with_status(self, status: GoalStatus) -> Goal:
        """A copy of the goal in a new status (the record itself is not mutated)."""
        from dataclasses import replace

        return replace(self, status=status)

    def to_primitive(self) -> dict[str, Any]:
        """A JSON-safe form of the goal (for durable storage)."""
        return {
            "description": self.description,
            "desired_facts": sorted(self.desired_facts),
            "priority": self.priority,
            "id": self.id,
            "status": self.status.value,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> Goal:
        """Rebuild a goal from its primitive form (tolerant of a missing field)."""
        return cls(
            description=str(data["description"]),
            desired_facts=frozenset(str(f) for f in data.get("desired_facts", ()) or ()),
            priority=float(data.get("priority", 0.5)),
            id=str(data.get("id", "")) or f"GOAL-{uuid4().hex[:8]}",
            status=GoalStatus(str(data.get("status", GoalStatus.PENDING.value))),
        )


@dataclass
class PlanStep:
    description: str
    depends_on: tuple[str, ...] = ()
    completed: bool = False


@dataclass
class GoalStore:
    """Persist symbolic goals and their statuses through the project's JsonStore.

    The durable companion of :class:`core.planning.storage.GoalStore` (which
    persists the *structured* goals): this keeps the symbolic :class:`Goal` -- the
    one the :class:`core.planner.Planner` consumes -- so a goal and its status
    survive a restart. Best-effort: a read-only data folder makes :meth:`save`
    return ``False`` rather than raising.
    """

    data_root: str | None = None
    namespace: str = "planning"
    key: str = "symbolic_goals"
    limit: int = 500
    goals: list[Goal] = field(default_factory=list)

    def add(self, goal: Goal) -> Goal:
        self.goals = [g for g in self.goals if g.id != goal.id] + [goal]
        if len(self.goals) > self.limit:
            self.goals = self.goals[-self.limit :]
        return goal

    def get(self, goal_id: str) -> Goal | None:
        return next((g for g in self.goals if g.id == goal_id), None)

    def set_status(self, goal_id: str, status: GoalStatus) -> Goal | None:
        goal = self.get(goal_id)
        if goal is None:
            return None
        updated = goal.with_status(status)
        self.add(updated)
        return updated

    def _store(self) -> Any:
        if not self.data_root:
            return None
        try:
            from storage.json_store import JsonStore

            return JsonStore(self.data_root)
        except Exception:
            return None

    def save(self) -> bool:
        store = self._store()
        if store is None:
            return False
        try:
            store.save(self.namespace, self.key, [g.to_primitive() for g in self.goals])
            return True
        except Exception:
            return False

    def load(self) -> int:
        store = self._store()
        if store is None:
            return 0
        try:
            data = store.load(self.namespace, self.key)
        except Exception:
            return 0
        if not isinstance(data, list):
            return 0
        loaded: list[Goal] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                loaded.append(Goal.from_primitive(item))
            except Exception:
                continue
        self.goals = loaded[-self.limit :]
        return len(self.goals)


__all__ = ["Goal", "GoalStatus", "GoalStore", "PlanStep"]
