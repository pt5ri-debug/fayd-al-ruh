"""Progressive environment ladder for running the agent in richer settings.

The agent is promoted one tier at a time:

    SYMBOLIC  ->  SANDBOX  ->  EXTERNAL

Promotion is monotonic and guarded. A tier becomes available only when the
previous tier has passed its checks, and the EXTERNAL tier additionally needs a
permission policy, an audit log, and human approval on record. The ladder never
enables external tool execution on its own; it only reports readiness, and the
``GovernedRuntime`` still enforces every gate at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from core.audit_log import AuditLog
from core.human_approval import HumanApprovalGate
from core.permissions import PermissionPolicy
from core.sandbox_runtime import GovernedRuntime


class EnvironmentTier(IntEnum):
    """Richer environments, ordered from safest to riskiest."""

    SYMBOLIC = 1
    SANDBOX = 2
    EXTERNAL = 3


@dataclass(frozen=True)
class TierReadiness:
    tier: EnvironmentTier
    available: bool
    reason: str


@dataclass
class ProgressiveEnvironment:
    """Tracks which tiers are unlocked and whether external tools are ready."""

    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    audit: AuditLog = field(default_factory=AuditLog)
    approvals: HumanApprovalGate = field(default_factory=HumanApprovalGate)
    current_tier: EnvironmentTier = EnvironmentTier.SYMBOLIC
    _symbolic_passed: bool = False
    _sandbox_passed: bool = False

    @property
    def runtime(self) -> GovernedRuntime:
        return GovernedRuntime(self.policy, self.audit, self.approvals)

    def mark_symbolic_passed(self) -> None:
        self._symbolic_passed = True
        if self.current_tier < EnvironmentTier.SANDBOX:
            self.current_tier = EnvironmentTier.SANDBOX

    def mark_sandbox_passed(self) -> None:
        self._sandbox_passed = True

    def readiness(self, tier: EnvironmentTier) -> TierReadiness:
        if tier is EnvironmentTier.SYMBOLIC:
            return TierReadiness(tier, True, "Symbolic simulation is always available.")
        if tier is EnvironmentTier.SANDBOX:
            if self._symbolic_passed:
                return TierReadiness(tier, True, "Symbolic tier passed; sandbox tasks unlocked.")
            return TierReadiness(tier, False, "Run and pass a symbolic simulation first.")
        missing: list[str] = []
        if not self._sandbox_passed:
            missing.append("sandbox tasks not passed")
        if not self.policy.allows_external:
            missing.append("no external capability in the permission policy")
        if not self.approvals.has_approval():
            missing.append("no human approval on record")
        if missing:
            return TierReadiness(tier, False, "; ".join(missing))
        return TierReadiness(tier, True, "External tier is fully gated and ready.")

    def can_run_external(self) -> bool:
        return self.readiness(EnvironmentTier.EXTERNAL).available
