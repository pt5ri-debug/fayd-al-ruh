"""Autonomous planning for Fayd Al-Ruh (Phase 8).

This package turns a **goal** into a validated, inspectable **plan** -- a
dependency graph of tasks, each bound to a tool the sandbox permits -- and hands
it to :mod:`core.execution` to run under the project's existing permission and
approval gates. It is deliberately a thin layer over what already exists and
never a second implementation:

* :mod:`core.goal` / :mod:`core.plan` / :mod:`core.planner` /
  :mod:`core.plan_evaluator` -- the symbolic planning primitives, reused;
* :mod:`core.tools` / :mod:`core.tool_agent` -- the deny-by-default sandbox and
  its tool registry, reused for both selection and execution;
* :mod:`core.agents.goals` / :mod:`core.agents.routing` /
  :mod:`core.agents.capabilities` -- the Phase-5 personality agents, reused for
  task assignment and capability ceilings.

The pieces, each with a stable API and replaceable independently:

* :mod:`core.planning.goals` -- the structured Goal and its manager.
* :mod:`core.planning.tasks` -- the structured Task and its state vocabulary.
* :mod:`core.planning.decomposition` -- the task decomposition engine.
* :mod:`core.planning.dependencies` -- the dependency graph and ordering.
* :mod:`core.planning.tools` -- the tool catalogue and selection engine.
* :mod:`core.planning.plan` -- the plan representation and validator.
* :mod:`core.planning.storage` -- durable goals and plans.
* :mod:`core.planning.coordinator` -- the planning coordinator.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

from core.planning.coordinator import PlanningCoordinator, default_planner
from core.planning.decomposition import (
    DecompositionEngine,
    DecompositionError,
    default_decomposition_engine,
)
from core.planning.dependencies import DependencyError, DependencyGraph
from core.planning.goals import (
    Goal,
    GoalKind,
    GoalManager,
    GoalStatus,
    VerificationCriteria,
)
from core.planning.plan import (
    PlanStatus,
    PlanValidation,
    TaskPlan,
    validate_plan,
)
from core.planning.storage import (
    GoalStore,
    PlanStore,
    task_from_primitive,
    task_to_primitive,
)
from core.planning.tasks import Task, TaskState
from core.planning.tools import (
    RiskCategory,
    ToolCatalogue,
    ToolDescriptor,
    ToolSelection,
    select_tool,
)

__all__ = [
    "DecompositionEngine",
    "DecompositionError",
    "DependencyError",
    "DependencyGraph",
    "Goal",
    "GoalKind",
    "GoalManager",
    "GoalStatus",
    "GoalStore",
    "PlanStatus",
    "PlanStore",
    "PlanValidation",
    "PlanningCoordinator",
    "RiskCategory",
    "Task",
    "TaskPlan",
    "TaskState",
    "ToolCatalogue",
    "ToolDescriptor",
    "ToolSelection",
    "VerificationCriteria",
    "default_decomposition_engine",
    "default_planner",
    "select_tool",
    "task_from_primitive",
    "task_to_primitive",
    "validate_plan",
]
