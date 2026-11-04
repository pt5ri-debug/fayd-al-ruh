"""The permission and approval manager (Phase 8, section 9).

Section 9 asks for a permission system that distinguishes action categories,
requires explicit approval for **high-risk** actions, validates permissions at
execution time, records approval decisions, and never lets an agent or an LLM
grant itself a permission. It also insists that a previous approval is not
unlimited future authorization.

This module is that system, built over the project's **existing** gates -- it
does not replace them:

* :class:`core.permissions.PermissionPolicy` decides the capability;
* :class:`core.human_approval.HumanApprovalGate` is the live human approval, and
  its approvals already carry a TTL, so one approval does not authorize forever;
* :class:`core.agents.capabilities.CapabilityGateway` is the per-personality
  ceiling, so an agent cannot reach a capability its owner does not hold.

On top of those, :class:`PermissionManager` adds the three **action categories**
and the rule that a high-risk action is refused unless a live, unexpired,
per-action approval exists. Every decision is recorded, so the audit trail shows
who asked, what they asked for, and what was decided.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from core.agents.capabilities import CapabilityGateway
from core.human_approval import HumanApprovalGate
from core.permissions import PermissionPolicy
from core.planning.tools import RiskCategory, ToolDescriptor


class ActionCategory(str, Enum):
    """The three action categories section 9 names."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_risk(cls, risk: RiskCategory) -> ActionCategory:
        return {
            RiskCategory.LOW: cls.LOW,
            RiskCategory.MEDIUM: cls.MEDIUM,
            RiskCategory.HIGH: cls.HIGH,
        }[risk]


@dataclass(frozen=True)
class PermissionRecord:
    """One permission decision, kept for the audit trail."""

    actor: str
    action: str
    category: ActionCategory
    allowed: bool
    reason: str
    approved: bool = False
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def describe(self) -> str:
        verdict = "allowed" if self.allowed else "refused"
        approval = " (approved)" if self.approved else ""
        return f"{self.actor} -> {self.action} [{self.category.value}] {verdict}{approval}: {self.reason}"


@dataclass
class PermissionManager:
    """Categorise an action and decide whether it may run, then record it.

    ``approvals`` is the project's own human-approval gate: an approval must be
    requested for the exact action and is consumed by the decision, so a previous
    approval does not authorize a later action. ``gateway`` is the Phase-5
    capability gateway; when supplied, a request from an ``actor`` is first
    checked against that actor's ceiling, so an agent cannot widen its own
    authority.
    """

    approvals: HumanApprovalGate = field(default_factory=HumanApprovalGate)
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    gateway: CapabilityGateway | None = None
    #: When true, a medium-risk action also needs approval (a stricter policy).
    require_approval_for_medium: bool = False
    records: list[PermissionRecord] = field(default_factory=list)

    def categorise(self, descriptor: ToolDescriptor) -> ActionCategory:
        """The action category of a tool, from its declared risk.

        An irreversible action is always at least ``HIGH``, whatever its nominal
        risk, because it cannot be undone -- the one escalation that never
        surprises a caller.
        """
        if not descriptor.reversible:
            return ActionCategory.HIGH
        return ActionCategory.from_risk(descriptor.risk)

    def needs_approval(self, descriptor: ToolDescriptor) -> bool:
        category = self.categorise(descriptor)
        if category is ActionCategory.HIGH:
            return True
        if category is ActionCategory.MEDIUM and self.require_approval_for_medium:
            return True
        return descriptor.requires_approval

    def check(
        self,
        descriptor: ToolDescriptor,
        *,
        actor: str = "system",
        capability: object | None = None,
    ) -> PermissionRecord:
        """Decide whether an action may run now, and record the decision.

        The order is deliberate: the personality ceiling first (an agent cannot
        even ask for what it may not hold), then the project capability policy,
        then the category rule, then the live approval. A refusal is recorded and
        returned -- never raised into a caller that only wants to know.
        """
        category = self.categorise(descriptor)
        action = descriptor.tool_id

        # 1. The personality ceiling (Phase 5) -- if an actor is named.
        if self.gateway is not None and actor not in ("", "system"):
            from core.agents.capabilities import AgentCapability

            needed = AgentCapability.RUN_TOOLS
            if descriptor.mutates:
                needed = AgentCapability.WRITE_FILES
            if not self.gateway.check(actor, needed).allowed:
                return self._record(
                    actor, action, category, False,
                    f"{actor} does not hold the {needed.value} capability.",
                )

        # 2. The project capability policy.
        decision = self.policy.decide(descriptor.capability, action)
        if not decision.allowed:
            return self._record(actor, action, category, False, decision.reason)

        # 3. Category rule: a high-risk action needs approval; a low-risk one does not.
        if category is ActionCategory.HIGH or self.needs_approval(descriptor):
            approval = self.approvals.decide(action)
            if not approval.approved:
                reason = approval.reason or "a high-risk action needs the user's approval."
                return self._record(actor, action, category, False, reason)
            return self._record(actor, action, category, True, "approved for this action", approved=True)

        # 4. Low/medium: allowed (medium is audited even when not gated).
        reason = (
            "low-risk action"
            if category is ActionCategory.LOW
            else "medium-risk action (audited)"
        )
        return self._record(actor, action, category, True, reason)

    def request_approval(
        self,
        descriptor: ToolDescriptor,
        *,
        actor: str,
        justification: str = "",
        ttl_seconds: float | None = None,
    ) -> object:
        """Ask the human-approval gate for a live approval of one action.

        This is how a caller *requests*; it never approves on its own behalf --
        only the gate's ``approve`` (a human decision) does that.
        """
        return self.approvals.request(
            descriptor.tool_id, requested_by=actor, justification=justification, ttl_seconds=ttl_seconds
        )

    def _record(
        self,
        actor: str,
        action: str,
        category: ActionCategory,
        allowed: bool,
        reason: str,
        *,
        approved: bool = False,
    ) -> PermissionRecord:
        record = PermissionRecord(
            actor=actor, action=action, category=category, allowed=allowed,
            reason=reason, approved=approved,
        )
        self.records.append(record)
        return record

    def recent(self, limit: int = 20) -> tuple[PermissionRecord, ...]:
        return tuple(self.records[-limit:])

    def describe(self) -> str:
        if not self.records:
            return "no permission decisions"
        return "\n".join(record.describe() for record in self.records[-10:])


__all__ = ["ActionCategory", "PermissionManager", "PermissionRecord"]
