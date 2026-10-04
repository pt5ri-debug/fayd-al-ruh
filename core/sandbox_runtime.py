"""Governed runtime: the single door through which environments are run.

Every run passes three gates in order:

1. a capability/permission decision (allow-list),
2. for external tools only, a human approval decision,
3. an append-only audit record.

External tool execution is refused unless all three are satisfied; the default
policy denies external tools, so nothing leaves the symbolic/sandbox tiers
without an explicit, reviewed opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.action import Action
from core.audit_log import AuditLog
from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionDecision, PermissionPolicy
from core.sandbox_environment import SandboxEnvironment, SandboxResult
from core.state import State
from core.symbolic_simulation import SimulationResult, SymbolicSimulation


@dataclass(frozen=True)
class RunOutcome:
    allowed: bool
    capability: Capability
    target: str
    reason: str
    audit_id: str
    result: object | None = None


class GovernedRuntime:
    """Runs environments only after permission, approval, and audit checks."""

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        audit: AuditLog | None = None,
        approvals: HumanApprovalGate | None = None,
    ) -> None:
        self.policy = policy or PermissionPolicy()
        self.audit = audit or AuditLog()
        self.approvals = approvals or HumanApprovalGate()

    def _authorize(
        self,
        capability: Capability,
        target: str,
        actor: str,
        context_id: str | None,
    ) -> tuple[PermissionDecision, str]:
        decision = self.policy.decide(capability, target)
        if decision.allowed and capability is Capability.EXTERNAL_TOOL:
            approval = self.approvals.decide(target)
            if not approval.approved:
                decision = PermissionDecision(capability, target, False, approval.reason)
        record = self.audit.record_decision(
            decision, actor=actor, context_id=context_id, metadata={"target": target}
        )
        return decision, record.id

    def run_symbolic(
        self,
        simulation: SymbolicSimulation,
        actions: list[Action],
        *,
        actor: str = "agent",
        context_id: str | None = None,
    ) -> RunOutcome:
        decision, audit_id = self._authorize(
            Capability.SYMBOLIC_SIMULATION, "symbolic", actor, context_id
        )
        if not decision.allowed:
            return RunOutcome(False, decision.capability, decision.target, decision.reason, audit_id)
        result: SimulationResult = simulation.run(actions)
        return RunOutcome(True, decision.capability, decision.target, decision.reason, audit_id, result)

    def run_sandbox(
        self,
        environment: SandboxEnvironment,
        actions: list[Action],
        success: Callable[[State], bool],
        *,
        actor: str = "agent",
        context_id: str | None = None,
    ) -> RunOutcome:
        decision, audit_id = self._authorize(
            Capability.SANDBOX_TASK, environment.task.name, actor, context_id
        )
        if not decision.allowed:
            return RunOutcome(False, decision.capability, decision.target, decision.reason, audit_id)
        result: SandboxResult = environment.run(actions, success)
        return RunOutcome(True, decision.capability, decision.target, decision.reason, audit_id, result)

    def run_external(
        self,
        tool_name: str,
        call: Callable[[], object],
        *,
        actor: str = "agent",
        context_id: str | None = None,
    ) -> RunOutcome:
        """Run an external tool, but only if policy AND human approval allow it."""
        decision, audit_id = self._authorize(
            Capability.EXTERNAL_TOOL, tool_name, actor, context_id
        )
        if not decision.allowed:
            return RunOutcome(False, decision.capability, tool_name, decision.reason, audit_id)
        return RunOutcome(True, decision.capability, tool_name, decision.reason, audit_id, call())
