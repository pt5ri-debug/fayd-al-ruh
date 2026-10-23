"""Human-approval gate required before external tools may run.

Approval is a human decision, not an agent one. The runtime only proceeds when
a matching, non-expired approval exists for the requested tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    tool_name: str
    requested_by: str
    justification: str = ""
    id: str = field(default_factory=lambda: f"APR-{uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str | None = None
    decided_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("Approval tool name cannot be empty.")
        if not self.requested_by.strip():
            raise ValueError("Approval requester cannot be empty.")

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and datetime.now(UTC) >= self.expires_at

    def approve(self, approver: str, ttl_seconds: float | None = None) -> None:
        if not approver.strip():
            raise ValueError("Approver cannot be empty.")
        if approver == self.requested_by:
            raise ValueError("The requester cannot approve their own request.")
        self.status = ApprovalStatus.APPROVED
        self.decided_by = approver
        self.decided_at = datetime.now(UTC)
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValueError("Approval TTL must be positive.")
            from datetime import timedelta

            self.expires_at = self.decided_at + timedelta(seconds=ttl_seconds)

    def reject(self, approver: str) -> None:
        self.status = ApprovalStatus.REJECTED
        self.decided_by = approver
        self.decided_at = datetime.now(UTC)


@dataclass(frozen=True)
class ApprovalDecision:
    tool_name: str
    approved: bool
    reason: str
    approval_id: str | None = None


class HumanApprovalGate:
    """Tracks approval requests and answers whether a tool may run."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    def request(self, tool_name: str, requested_by: str, justification: str = "", ttl_seconds: float | None = None) -> ApprovalRequest:
        approval = ApprovalRequest(tool_name, requested_by, justification)
        if ttl_seconds is not None:
            from datetime import timedelta

            if ttl_seconds <= 0:
                raise ValueError("Approval TTL must be positive.")
            approval.expires_at = approval.created_at + timedelta(seconds=ttl_seconds)
        self._requests[approval.id] = approval
        return approval

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._requests.get(approval_id)

    def decide(self, tool_name: str) -> ApprovalDecision:
        for approval in self._requests.values():
            if approval.tool_name != tool_name:
                continue
            if approval.status is ApprovalStatus.APPROVED:
                if approval.is_expired:
                    return ApprovalDecision(tool_name, False, "Approval has expired.", approval.id)
                return ApprovalDecision(tool_name, True, "Approved by a human reviewer.", approval.id)
            if approval.status is ApprovalStatus.REJECTED:
                return ApprovalDecision(tool_name, False, "Approval was rejected.", approval.id)
        return ApprovalDecision(tool_name, False, "No human approval on record.", None)

    def has_approval(self, tool_name: str | None = None) -> bool:
        """True when a live approval exists, for a specific tool or for any tool."""
        return any(
            approval.status is ApprovalStatus.APPROVED and not approval.is_expired
            and (tool_name is None or approval.tool_name == tool_name)
            for approval in self._requests.values()
        )
