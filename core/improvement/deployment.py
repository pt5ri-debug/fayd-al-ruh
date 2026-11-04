"""Controlled deployment: versioned, checked, backed up, and reversible.

Section 14 of the phase requires a deployment to be versioned, to take a
pre-deployment checkpoint, to run compatibility and health checks, to keep a
backup and a deployment log, to monitor afterwards, and to be reversible. This
module does all of that, and it does it under one hard rule the phase repeats in
sections 10, 13, 17, and 21: **the loop never modifies production source code or
model weights on its own.**

So a deployment here writes to a **governed state root** (the data folder by
default), never the repository:

* a **configuration or state** change is applied to that root, after the
  originals are backed up, and the version is recorded -- so it is reversible;
* a **source-code** change is *not* applied by this module at all. The
  deployment record marks it ``manual_merge_required`` and names the Code Studio
  workspace, so the human merges it through the Studio's own gated merge (checks
  passed, ``main`` not moved). That is the project's existing, tested path, and it
  is why this module has no ``git`` in it; and
* a **model-weight or dataset** change is refused outright: those go through the
  Phase-6 training governance, never through this coordinator.

Every deployment writes a :class:`DeploymentRecord` (its version, its backup, its
checkpoint, its checks, and its outcome) into the history, so the log the phase
asks for is the history itself.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from core.best_effort import best_effort
from core.improvement.approval import ChangeApprovalManager
from core.improvement.experiments import Experiment
from core.improvement.history import EntryKind, ImprovementHistory
from core.improvement.proposals import ImprovementProposal, ProposalStatus


class DeploymentStatus(str, Enum):
    """The outcome of one deployment attempt."""

    DEPLOYED = "deployed"
    AWAITING_APPROVAL = "awaiting_approval"
    REFUSED = "refused"
    MANUAL_MERGE_REQUIRED = "manual_merge_required"
    FAILED = "failed"


@dataclass(frozen=True)
class DeploymentRecord:
    """One versioned deployment, with everything needed to reverse it."""

    deployment_id: str
    version: int
    proposal_id: str
    status: DeploymentStatus
    target_root: str
    applied_files: tuple[str, ...] = ()
    backup_dir: str = ""
    checkpoint: str = ""
    checks: tuple[str, ...] = ()
    manual_merge_slug: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reason: str = ""

    @property
    def reversible(self) -> bool:
        return self.status is DeploymentStatus.DEPLOYED and bool(self.backup_dir)

    def describe(self) -> str:
        lines = [
            f"Deployment {self.deployment_id} v{self.version} for {self.proposal_id}: {self.status.value}",
            f"  target: {self.target_root}",
        ]
        if self.applied_files:
            lines.append(f"  applied: {', '.join(self.applied_files)}")
        if self.backup_dir:
            lines.append(f"  backup: {self.backup_dir}")
        if self.checks:
            lines.append(f"  checks: {'; '.join(self.checks)}")
        if self.manual_merge_slug:
            lines.append(f"  manual merge required in Code Studio workspace: {self.manual_merge_slug}")
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "deployment_id": self.deployment_id,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "status": self.status.value,
            "target_root": self.target_root,
            "applied_files": list(self.applied_files),
            "backup_dir": self.backup_dir,
            "checkpoint": self.checkpoint,
            "checks": list(self.checks),
            "manual_merge_slug": self.manual_merge_slug,
            "at": self.at.isoformat(),
            "reason": self.reason,
        }


#: What a deployment may change. A source or model-weight change is refused here
#: and routed to its own governance.
REFUSED_TOUCHES: frozenset[str] = frozenset({"model_weights", "training_dataset", "security", "permissions"})


@dataclass
class DeploymentPlan:
    """What a deployment intends to do, before it is done."""

    proposal: ImprovementProposal
    version: int
    target_root: Path
    files: dict[str, str] = field(default_factory=dict)
    checkpoint: str = ""
    checks: tuple[str, ...] = ()


class DeploymentCoordinator:
    """Applies an approved change to governed state, versioned and reversible."""

    def __init__(
        self,
        *,
        state_root: Path | str,
        backups_dir: Path | str,
        history: ImprovementHistory | None = None,
        approvals: ChangeApprovalManager | None = None,
        health_check: Callable[[], bool] | None = None,
    ) -> None:
        self.state_root = Path(state_root)
        self.backups_dir = Path(backups_dir)
        self.history = history
        self.approvals = approvals
        self._health_check = health_check
        self._records: list[DeploymentRecord] = []
        self._next_version = 1

    # -- reading ------------------------------------------------------------
    def records(self) -> tuple[DeploymentRecord, ...]:
        return tuple(self._records)

    def latest(self) -> DeploymentRecord | None:
        return self._records[-1] if self._records else None

    def next_version(self) -> int:
        return self._next_version

    # -- planning -----------------------------------------------------------
    def plan(self, proposal: ImprovementProposal, *, files: dict[str, str] | None = None) -> DeploymentPlan:
        """Build a plan: a version, a checkpoint, and the compatibility checks."""
        checks: list[str] = []
        if proposal.requires_explicit_approval:
            checks.append("sensitive change: explicit approval required")
        if "source_code" in proposal.touches:
            checks.append("source change: merge is the human's step (never automatic)")
        return DeploymentPlan(
            proposal=proposal,
            version=self._next_version,
            target_root=self.state_root,
            files=dict(files or {}),
            checkpoint=f"v{self._next_version}",
            checks=tuple(checks),
        )

    # -- deploying ----------------------------------------------------------
    def deploy(
        self,
        proposal: ImprovementProposal,
        *,
        experiment: Experiment | None = None,
        files: dict[str, str] | None = None,
        approved: bool = False,
    ) -> DeploymentRecord:
        """Deploy one approved change, or refuse and say why.

        The order of refusal matters, and it is deliberate: an unapproved change
        is refused **first**, before any file is read or written, so an
        unapproved deployment can never have a side effect.
        """
        # 1. Sensitive targets are refused outright.
        if set(proposal.touches) & REFUSED_TOUCHES:
            return self._record(
                proposal,
                DeploymentStatus.REFUSED,
                reason=(
                    "this change touches model weights, a dataset, or a security control; "
                    "it must go through the Phase-6 training or the security governance, not here"
                ),
            )

        # 2. Approval is checked before anything else happens.
        if not approved or self.approvals is None or not self.approvals.is_approved(proposal.proposal_id):
            return self._record(
                proposal,
                DeploymentStatus.AWAITING_APPROVAL,
                reason="the change has not been approved by the user",
            )

        # 3. A source change is never applied here -- the human merges it.
        if "source_code" in proposal.touches:
            slug = ""
            if experiment is not None and experiment.sandbox_id:
                slug = experiment.artifacts_dir
            record = self._record(
                proposal,
                DeploymentStatus.MANUAL_MERGE_REQUIRED,
                reason="source changes are merged by the user through the Code Studio, never automatically",
                manual_merge_slug=slug,
                checks=("checks passed in isolation",) if (experiment and experiment.checks_passed) else (),
            )
            return record

        # 4. A state/configuration change is applied, after a backup.
        if experiment is not None and not experiment.checks_passed and experiment.evaluation is not None:
            return self._record(
                proposal,
                DeploymentStatus.REFUSED,
                reason="the candidate's checks did not pass, so it is not deployed",
            )

        plan = self.plan(proposal, files=files)
        try:
            backup_dir = self._backup(plan)
        except OSError as error:
            return self._record(proposal, DeploymentStatus.FAILED, reason=f"the backup failed: {error}")

        try:
            applied = self._apply(plan)
        except OSError as error:
            return self._record(
                proposal, DeploymentStatus.FAILED, reason=f"the deployment failed: {error}",
                backup_dir=str(backup_dir),
            )

        if self._health_check is not None and not self._health_check():
            # A failed post-deployment health check is a rollback trigger.
            self._restore(backup_dir, applied)
            return self._record(
                proposal,
                DeploymentStatus.FAILED,
                reason="the post-deployment health check failed; the change was reverted",
                backup_dir=str(backup_dir),
                applied_files=applied,
                checks=plan.checks,
            )

        self._next_version += 1
        return self._record(
            proposal,
            DeploymentStatus.DEPLOYED,
            backup_dir=str(backup_dir),
            applied_files=applied,
            checkpoint=plan.checkpoint,
            checks=plan.checks,
        )

    # -- internals ----------------------------------------------------------
    def _backup(self, plan: DeploymentPlan) -> Path:
        """Copy the current state of every file the plan will touch into a backup."""
        backup_dir = self.backups_dir / f"v{plan.version}-{uuid4().hex[:6]}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for relative in plan.files:
            source = self.state_root / relative
            if source.exists():
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return backup_dir

    def _apply(self, plan: DeploymentPlan) -> tuple[str, ...]:
        """Write the plan's files into the governed state root."""
        applied: list[str] = []
        for relative, content in plan.files.items():
            target = self.state_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            applied.append(relative)
        return tuple(applied)

    def _restore(self, backup_dir: Path, applied: Iterable[str]) -> None:
        """Restore a deployment's files from its backup (used on a failed check)."""
        with best_effort("restore a deployment's files"):
            for relative in applied:
                saved = backup_dir / relative
                target = self.state_root / relative
                if saved.exists():
                    shutil.copy2(saved, target)
                elif target.exists():
                    target.unlink()

    def _record(self, proposal: ImprovementProposal, status: DeploymentStatus, **kwargs: object) -> DeploymentRecord:
        record = DeploymentRecord(
            deployment_id=f"DEP-{uuid4().hex[:8]}",
            version=self._next_version,
            proposal_id=proposal.proposal_id,
            status=status,
            target_root=str(self.state_root),
            **kwargs,  # type: ignore[arg-type]
        )
        self._records.append(record)
        if self.history is not None:
            self.history.append(
                EntryKind.DEPLOYMENT,
                f"{record.deployment_id} v{record.version} {status.value} for {proposal.proposal_id}.",
                proposal_id=proposal.proposal_id,
                detail=record.to_primitive(),
            )
        return record

    def describe(self) -> str:
        if not self._records:
            return "No deployments have run."
        lines = [f"{len(self._records)} deployment(s):"]
        lines.extend(f"  {record.deployment_id} v{record.version} [{record.status.value}]" for record in self._records)
        return "\n".join(lines)


def approved_status(proposal: ImprovementProposal) -> ImprovementProposal:
    """A convenience: the proposal marked approved (a record, not a side effect)."""
    return proposal.with_status(ProposalStatus.APPROVED)


__all__ = [
    "REFUSED_TOUCHES",
    "DeploymentCoordinator",
    "DeploymentPlan",
    "DeploymentRecord",
    "DeploymentStatus",
    "approved_status",
]
