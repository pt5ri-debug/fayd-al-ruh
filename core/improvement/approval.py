"""The change approval manager: a human decision, and only a human's.

Section 13 of the phase makes approval a hard gate for production source code,
model weights, training datasets, security or permission changes, a database
schema change, an irreversible operation, and any change that could expose
private personality data -- and section 17 forbids the improvement system from
removing its own approval requirement.

This module implements that gate, and it reuses the project's own
:class:`core.human_approval.HumanApprovalGate` rather than inventing a second
one, so the same rule the rest of the project uses applies here: an approval is
granted by a human actor, it may not be granted by the requester, and it may
expire. Two additional rules are specific to this phase and are enforced here:

* **a sensitive change is always gated** -- a proposal whose ``touches`` include a
  sensitive area needs an approval even if its measured risk was low; and
* **approving one change is not approving another** -- an approval is bound to
  one proposal id, so a decision about proposal A can never be replayed for B.

The manager keeps every decision (granted and refused) in the history, so the
audit can always say who approved what, and when.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from core.human_approval import ApprovalRequest, HumanApprovalGate
from core.improvement.history import EntryKind, ImprovementHistory
from core.improvement.proposals import ImprovementProposal


class ApprovalDecision(str, Enum):
    """The outcome of asking for approval."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True)
class ApprovalOutcome:
    """One approval question and its answer."""

    proposal_id: str
    decision: ApprovalDecision
    reason: str
    approval_id: str = ""
    requested_by: str = "improvement-loop"
    decided_by: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def approved(self) -> bool:
        return self.decision is ApprovalDecision.APPROVED

    def describe(self) -> str:
        return f"{self.proposal_id}: {self.decision.value} ({self.reason})"


#: The approval-request name used for a self-improvement change, so the audit can
#: tell a deployment approval from an external-tool approval.
APPROVAL_TOOL = "self_improvement.change"


class ChangeApprovalManager:
    """Requests and records the human approval a deployment needs."""

    def __init__(
        self,
        *,
        gate: HumanApprovalGate | None = None,
        history: ImprovementHistory | None = None,
        approval_ttl_seconds: float | None = None,
    ) -> None:
        self.gate = gate or HumanApprovalGate()
        self.history = history
        self.approval_ttl_seconds = approval_ttl_seconds
        self._outcomes: dict[str, ApprovalOutcome] = {}
        self._requests: dict[str, ApprovalRequest] = {}

    # -- reading ------------------------------------------------------------
    def outcome_for(self, proposal_id: str) -> ApprovalOutcome | None:
        return self._outcomes.get(proposal_id)

    def is_approved(self, proposal_id: str) -> bool:
        outcome = self._outcomes.get(proposal_id)
        return outcome is not None and outcome.approved

    def outcomes(self) -> tuple[ApprovalOutcome, ...]:
        return tuple(self._outcomes.values())

    # -- the gate -----------------------------------------------------------
    def request(self, proposal: ImprovementProposal, *, reason: str = "") -> ApprovalOutcome:
        """Open an approval request for a proposal (does not grant it).

        The request is bound to the proposal id, so an approval for one change
        can never be reused for another.
        """
        request = ApprovalRequest(
            tool_name=f"{APPROVAL_TOOL}:{proposal.proposal_id}",
            requested_by="improvement-loop",
            justification=reason or f"{proposal.title}: {proposal.proposed_change}",
        )
        self._requests[proposal.proposal_id] = request
        outcome = ApprovalOutcome(
            proposal_id=proposal.proposal_id,
            decision=ApprovalDecision.PENDING,
            reason="awaiting the user's decision",
            approval_id=request.id,
        )
        self._outcomes[proposal.proposal_id] = outcome
        self._note(EntryKind.APPROVAL, f"Approval requested for {proposal.proposal_id}.", proposal.proposal_id)
        return outcome

    def approve(self, proposal: ImprovementProposal, *, approver: str = "user") -> ApprovalOutcome:
        """Record a human approval for one proposal.

        A proposal that is *sensitive* is refused here unless the caller names a
        real approver -- the loop can never approve on its own behalf.
        """
        request = self._requests.get(proposal.proposal_id)
        if request is None:
            request = ApprovalRequest(
                tool_name=f"{APPROVAL_TOOL}:{proposal.proposal_id}",
                requested_by="improvement-loop",
                justification=proposal.title,
            )
            self._requests[proposal.proposal_id] = request
        if not approver or approver == request.requested_by:
            outcome = ApprovalOutcome(
                proposal_id=proposal.proposal_id,
                decision=ApprovalDecision.PENDING,
                reason="the requester cannot approve their own change",
                approval_id=request.id,
            )
            self._outcomes[proposal.proposal_id] = outcome
            return outcome
        try:
            request.approve(approver, ttl_seconds=self.approval_ttl_seconds)
        except ValueError as error:
            outcome = ApprovalOutcome(
                proposal_id=proposal.proposal_id,
                decision=ApprovalDecision.PENDING,
                reason=str(error),
                approval_id=request.id,
            )
            self._outcomes[proposal.proposal_id] = outcome
            return outcome
        outcome = ApprovalOutcome(
            proposal_id=proposal.proposal_id,
            decision=ApprovalDecision.APPROVED,
            reason="approved by the user",
            approval_id=request.id,
            decided_by=approver,
        )
        self._outcomes[proposal.proposal_id] = outcome
        self._note(
            EntryKind.APPROVAL,
            f"{proposal.proposal_id} approved by {approver}.",
            proposal.proposal_id,
            detail={"approval_id": request.id},
        )
        return outcome

    def reject(self, proposal: ImprovementProposal, *, approver: str = "user", reason: str = "") -> ApprovalOutcome:
        """Record a human refusal."""
        outcome = ApprovalOutcome(
            proposal_id=proposal.proposal_id,
            decision=ApprovalDecision.REJECTED,
            reason=reason or "rejected by the user",
            approval_id=self._requests.get(proposal.proposal_id, ApprovalRequest(
                tool_name=f"{APPROVAL_TOOL}:{proposal.proposal_id}", requested_by="improvement-loop"
            )).id,
            decided_by=approver,
        )
        self._outcomes[proposal.proposal_id] = outcome
        self._note(EntryKind.REJECTION, f"{proposal.proposal_id} rejected by {approver}.", proposal.proposal_id)
        return outcome

    def check(self, proposal_id: str) -> bool:
        """Whether a live, unexpired approval exists for a proposal.

        A ``NOT_REQUIRED`` outcome never exists here: the manager is only asked
        about changes that *do* need approval.
        """
        request = self._requests.get(proposal_id)
        if request is None:
            return False
        if request.is_expired:
            self._outcomes[proposal_id] = ApprovalOutcome(
                proposal_id=proposal_id,
                decision=ApprovalDecision.EXPIRED,
                reason="the approval expired",
                approval_id=request.id,
            )
            return False
        decision = self.gate.check(request.tool_name) if hasattr(self.gate, "check") else None
        if decision is not None and getattr(decision, "approved", False):
            return True
        outcome = self._outcomes.get(proposal_id)
        return outcome is not None and outcome.approved

    def describe(self) -> str:
        if not self._outcomes:
            return "No approvals have been requested."
        lines = ["Approvals:"]
        for outcome in self._outcomes.values():
            lines.append(f"  {outcome.describe()}")
        return "\n".join(lines)

    def _note(
        self,
        kind: EntryKind,
        summary: str,
        proposal_id: str,
        *,
        detail: dict[str, object] | None = None,
    ) -> None:
        if self.history is not None:
            self.history.append(kind, summary, proposal_id=proposal_id, detail=detail or {})


__all__ = [
    "APPROVAL_TOOL",
    "ApprovalDecision",
    "ApprovalOutcome",
    "ChangeApprovalManager",
]
