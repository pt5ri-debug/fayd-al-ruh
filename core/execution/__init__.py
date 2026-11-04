"""Controlled execution for Fayd Al-Ruh (Phase 8).

This package runs a validated plan from :mod:`core.planning` under the project's
**existing** sandbox and permission gates, verifies the outcome, proposes safe
recovery when something fails, and records the experience through the Phase-3
learning system. It is *controlled* autonomy: nothing runs without a validated
plan, a permitted tool, and -- for a high-risk action -- a live human approval.

The pieces, each with a stable API and replaceable independently:

* :mod:`core.execution.permissions` -- the action categories, the permission
  manager, and the approval gate (over the project's own gates).
* :mod:`core.execution.engine` -- the dependency-scheduling, cancellable,
  timed-out execution engine.
* :mod:`core.execution.monitor` -- live monitoring, stall/deadlock detection,
  and the wall-clock budget.
* :mod:`core.execution.verification` -- outcome verification against the goal's
  own criteria (a tool's success is not the goal's success).
* :mod:`core.execution.recovery` -- safe failure recovery and checkpoints.
* :mod:`core.execution.history` -- the append-only, redacted audit log.
* :mod:`core.execution.coordinator` -- the one governed entry point.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

from core.execution.coordinator import (
    ExecutionCoordinator,
    ExecutionResult,
    default_executor,
)
from core.execution.engine import (
    ExecutionEngine,
    ExecutionReport,
    TaskExecution,
)
from core.execution.history import ExecutionHistory, HistoryEntry, redact
from core.execution.monitor import ExecutionMonitor, MonitorSnapshot
from core.execution.permissions import (
    ActionCategory,
    PermissionManager,
    PermissionRecord,
)
from core.execution.recovery import (
    Checkpoint,
    RecoveryAction,
    RecoveryDecision,
    RecoveryManager,
)
from core.execution.verification import (
    Outcome,
    OutcomeVerifier,
    VerificationResult,
)

__all__ = [
    "ActionCategory",
    "Checkpoint",
    "ExecutionCoordinator",
    "ExecutionEngine",
    "ExecutionHistory",
    "ExecutionMonitor",
    "ExecutionReport",
    "ExecutionResult",
    "HistoryEntry",
    "MonitorSnapshot",
    "Outcome",
    "OutcomeVerifier",
    "PermissionManager",
    "PermissionRecord",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryManager",
    "TaskExecution",
    "VerificationResult",
    "default_executor",
    "redact",
]
