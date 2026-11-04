"""The structured goal (Phase 8, section 4).

A goal is what a user (or the system itself) is trying to achieve. This module
gives it a structured record -- the fields the task lists -- and a manager that
keeps every goal with its tasks, distinguishes a *user-requested* goal from a
*system subgoal* from a *suggested* goal, and never treats an inferred goal as
explicit user authorization.

It **extends** :mod:`core.goal` (the symbolic `Goal`/`PlanStep` the planner
consumes) rather than replacing it: a structured goal can carry the symbolic
``desired_facts`` a :class:`core.planner.Planner` needs, but it also carries the
constraints, priority, deadline, permissions, and completion/verification
criteria a real task needs. Nothing here executes anything; it is state.

Dependency-free and Tk-free.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

#: How a goal came to exist. A *user* goal is explicit authorization; a
#: *subgoal* and a *suggestion* are **not** -- the coordinator must not execute
#: either without the user's own request behind it (section 4's rule).
GOAL_USER = "user"
GOAL_SUBGOAL = "subgoal"
GOAL_SUGGESTED = "suggested"


class GoalKind(str, Enum):
    """Who a goal came from -- one closed vocabulary."""

    #: Explicitly requested by the user (the only kind that authorizes action).
    USER = GOAL_USER
    #: A subgoal the system derived from a user goal (it inherits the authority).
    SUBGOAL = GOAL_SUBGOAL
    #: A goal the system *suggests*; it authorizes nothing on its own.
    SUGGESTED = GOAL_SUGGESTED

    @property
    def authorizing(self) -> bool:
        """True only for a goal the user explicitly asked for."""
        return self is GoalKind.USER


class GoalStatus(str, Enum):
    """The lifecycle of a goal -- one closed vocabulary."""

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
        return self in {
            GoalStatus.COMPLETED,
            GoalStatus.PARTIAL,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        }


@dataclass(frozen=True)
class VerificationCriteria:
    """How a goal's completion is judged (section 4's completion/verification).

    ``success_criteria`` is what "done" means; ``artifacts`` are the files or
    records that must exist; ``expected_facts`` is the symbolic state the goal
    targets. All three are optional, and an empty criterion means "the outcome
    text must be present and non-failing" -- the honest default.
    """

    success_criteria: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    expected_facts: frozenset[str] = frozenset()
    min_successful_tasks: int = 1

    def describe(self) -> str:
        parts: list[str] = []
        if self.success_criteria:
            parts.append("criteria: " + "; ".join(self.success_criteria))
        if self.artifacts:
            parts.append("artifacts: " + ", ".join(self.artifacts))
        if self.expected_facts:
            parts.append("facts: " + ", ".join(sorted(self.expected_facts)))
        return " | ".join(parts) if parts else "(outcome text must be present)"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Goal:
    """A structured goal, with everything section 4 asks for.

    The record is frozen: a goal's life is a sequence of *new* versions (via
    :meth:`with_status`), so its history is never mutated -- the same rule the
    learning and memory layers follow.
    """

    description: str
    kind: GoalKind = GoalKind.USER
    goal_id: str = field(default_factory=lambda: _new_id("goal"))
    desired_outcome: str = ""
    constraints: tuple[str, ...] = ()
    priority: float = 0.5
    deadline: str = ""
    required_resources: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    verification: VerificationCriteria = field(default_factory=VerificationCriteria)
    status: GoalStatus = GoalStatus.PENDING
    #: The ids of the tasks that serve this goal (filled by the manager).
    related_tasks: tuple[str, ...] = ()
    #: The id of the goal this one was derived from (empty for a user goal).
    parent_goal: str = ""
    #: Whether the user must approve before this goal's plan may run.
    requires_approval: bool = False
    #: The owner (a Phase-5 personality key) the goal belongs to, when assigned.
    owner: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("A goal needs a description.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Goal priority must be in [0, 1].")
        if self.kind is GoalKind.SUBGOAL and not self.parent_goal:
            raise ValueError("A subgoal needs the goal it was derived from.")

    @property
    def authorizing(self) -> bool:
        """True only when the goal authorizes action on its own."""
        return self.kind.authorizing

    def with_status(self, status: GoalStatus, **changes: object) -> Goal:
        """A copy of the goal in a new status (history is immutable)."""
        from dataclasses import replace

        return replace(self, status=status, updated_at=_now(), **changes)  # type: ignore[arg-type]

    def with_tasks(self, task_ids: tuple[str, ...]) -> Goal:
        from dataclasses import replace

        return replace(self, related_tasks=task_ids, updated_at=_now())

    def describe(self) -> str:
        owner = f" [{self.owner}]" if self.owner else ""
        extra = f" -> {self.desired_outcome}" if self.desired_outcome else ""
        return f"{self.goal_id} ({self.kind.value}{owner}): {self.description}{extra} — {self.status.value}"

    def to_primitive(self) -> dict[str, object]:
        """A JSON-safe form of the goal (for durable storage)."""
        return {
            "description": self.description,
            "kind": self.kind.value,
            "goal_id": self.goal_id,
            "desired_outcome": self.desired_outcome,
            "constraints": list(self.constraints),
            "priority": self.priority,
            "deadline": self.deadline,
            "required_resources": list(self.required_resources),
            "required_permissions": list(self.required_permissions),
            "verification": {
                "success_criteria": list(self.verification.success_criteria),
                "artifacts": list(self.verification.artifacts),
                "expected_facts": sorted(self.verification.expected_facts),
                "min_successful_tasks": self.verification.min_successful_tasks,
            },
            "status": self.status.value,
            "related_tasks": list(self.related_tasks),
            "parent_goal": self.parent_goal,
            "requires_approval": self.requires_approval,
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> Goal:
        """Rebuild a goal from its primitive form (tolerant of a missing field)."""
        raw = data.get("verification") or {}
        verification = raw if isinstance(raw, dict) else {}
        return cls(
            description=str(data["description"]),
            kind=GoalKind(str(data.get("kind", GoalKind.USER.value))),
            goal_id=str(data["goal_id"]),
            desired_outcome=str(data.get("desired_outcome", "")),
            constraints=tuple(str(c) for c in data.get("constraints", ()) or ()),
            priority=float(data.get("priority", 0.5)),  # type: ignore[arg-type]
            deadline=str(data.get("deadline", "")),
            required_resources=tuple(str(r) for r in data.get("required_resources", ()) or ()),
            required_permissions=tuple(str(p) for p in data.get("required_permissions", ()) or ()),
            verification=VerificationCriteria(
                success_criteria=tuple(
                    str(c) for c in verification.get("success_criteria", ()) or ()
                ),
                artifacts=tuple(str(a) for a in verification.get("artifacts", ()) or ()),
                expected_facts=frozenset(
                    str(f) for f in verification.get("expected_facts", ()) or ()
                ),
                min_successful_tasks=int(verification.get("min_successful_tasks", 1)),
            ),
            status=GoalStatus(str(data.get("status", GoalStatus.PENDING.value))),
            related_tasks=tuple(str(t) for t in data.get("related_tasks", ()) or ()),
            parent_goal=str(data.get("parent_goal", "")),
            requires_approval=bool(data.get("requires_approval", False)),
            owner=str(data.get("owner", "")),
            created_at=str(data.get("created_at", "")) or _now(),
            updated_at=str(data.get("updated_at", "")) or _now(),
        )


@dataclass
class GoalManager:
    """Every goal and its tasks, kept in one place and bounded.

    The manager is the *structured* counterpart of
    :class:`core.agents.goals.GoalManager` (which owns the Phase-5 per-personality
    task tree). It records a goal's versions, links its tasks, and answers the
    questions the coordinator asks: is this goal authorized? what are its
    subgoals? which goals are open?
    """

    max_goals: int = 500
    goals: dict[str, Goal] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list, repr=False)

    def add(self, goal: Goal) -> Goal:
        """Record a goal (a duplicate id replaces nothing -- ids are unique)."""
        self.goals[goal.goal_id] = goal
        self._order.append(goal.goal_id)
        self._trim()
        return goal

    def create(
        self,
        description: str,
        *,
        kind: GoalKind = GoalKind.USER,
        desired_outcome: str = "",
        constraints: tuple[str, ...] = (),
        priority: float = 0.5,
        deadline: str = "",
        required_resources: tuple[str, ...] = (),
        required_permissions: tuple[str, ...] = (),
        verification: VerificationCriteria | None = None,
        requires_approval: bool = False,
        owner: str = "",
        parent_goal: str = "",
    ) -> Goal:
        """Build and record a goal."""
        goal = Goal(
            description=description,
            kind=kind,
            desired_outcome=desired_outcome or description,
            constraints=constraints,
            priority=priority,
            deadline=deadline,
            required_resources=required_resources,
            required_permissions=required_permissions,
            verification=verification or VerificationCriteria(),
            requires_approval=requires_approval,
            owner=owner,
            parent_goal=parent_goal,
        )
        return self.add(goal)

    def get(self, goal_id: str) -> Goal | None:
        return self.goals.get(goal_id)

    def require(self, goal_id: str) -> Goal:
        goal = self.goals.get(goal_id)
        if goal is None:
            raise KeyError(f"unknown goal {goal_id!r}")
        return goal

    def update(self, goal: Goal) -> Goal:
        """Replace a goal with a new version of itself (same id)."""
        if goal.goal_id not in self.goals:
            raise KeyError(f"unknown goal {goal.goal_id!r}")
        self.goals[goal.goal_id] = goal
        return goal

    def set_status(self, goal_id: str, status: GoalStatus) -> Goal:
        return self.update(self.require(goal_id).with_status(status))

    def link_tasks(self, goal_id: str, task_ids: tuple[str, ...]) -> Goal:
        return self.update(self.require(goal_id).with_tasks(task_ids))

    def subgoals_of(self, goal_id: str) -> tuple[Goal, ...]:
        """The goals derived from ``goal_id`` (its subgoals)."""
        return tuple(g for g in self.goals.values() if g.parent_goal == goal_id)

    def open_goals(self) -> tuple[Goal, ...]:
        """Every goal that is not terminal, newest last."""
        return tuple(
            self.goals[gid]
            for gid in self._order
            if gid in self.goals and not self.goals[gid].status.terminal
        )

    def all_goals(self) -> tuple[Goal, ...]:
        return tuple(self.goals[gid] for gid in self._order if gid in self.goals)

    def authorizes(self, goal_id: str) -> bool:
        """True only when the goal (or an ancestor) is a user goal.

        This is the structural guard section 4 demands: a *suggested* goal never
        authorizes action, and a *subgoal* authorizes only when the goal it
        descends from was a user's own request.
        """
        goal = self.goals.get(goal_id)
        seen: set[str] = set()
        while goal is not None and goal.goal_id not in seen:
            seen.add(goal.goal_id)
            if goal.kind is GoalKind.USER:
                return True
            goal = self.goals.get(goal.parent_goal) if goal.parent_goal else None
        return False

    def _trim(self) -> None:
        """Drop the oldest terminal goal past the bound (never an open one)."""
        if len(self._order) <= self.max_goals:
            return
        removable = [
            gid
            for gid in self._order
            if gid in self.goals and self.goals[gid].status.terminal
        ]
        excess = len(self._order) - self.max_goals
        for gid in removable[:excess]:
            self.goals.pop(gid, None)
            self._order.remove(gid)

    def describe(self) -> str:
        if not self._order:
            return "no goals"
        return "\n".join(self.goals[gid].describe() for gid in self._order if gid in self.goals)


__all__ = [
    "GOAL_SUBGOAL",
    "GOAL_SUGGESTED",
    "GOAL_USER",
    "Goal",
    "GoalKind",
    "GoalManager",
    "GoalStatus",
    "VerificationCriteria",
]
