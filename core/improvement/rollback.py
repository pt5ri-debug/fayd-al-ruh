"""The rollback manager: restore the prior accepted state, never erase history.

Section 14 requires a rollback to restore the prior accepted state **without
deleting unrelated user data or learning history**, and section 15 forbids
erasing a failed experiment. This module is that rollback, and it is deliberately
conservative:

* it restores only the files a deployment actually changed, from that
  deployment's own backup -- it never touches anything else, so user data and
  learning records are untouched;
* it **records** the rollback as a new history entry and marks the deployment
  rolled back; it deletes nothing, so the failed change and its evaluation remain
  auditable; and
* it verifies the restore afterwards and reports whether it succeeded, so a
  rollback that itself failed is visible rather than assumed.

It reuses the model-version registry's rollback for the one case that belongs
there -- a trained model version -- rather than re-implementing it: this module
delegates to :class:`cognition.versions.ModelRegistry.rollback` for a
model-weight change and handles the state-file case itself.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from core.best_effort import best_effort
from core.improvement.deployment import DeploymentRecord, DeploymentStatus
from core.improvement.history import EntryKind, ImprovementHistory


class RollbackStatus(str, Enum):
    """The outcome of one rollback attempt."""

    ROLLED_BACK = "rolled_back"
    NOTHING_TO_ROLL_BACK = "nothing_to_roll_back"
    FAILED = "failed"
    REFUSED = "refused"


@dataclass(frozen=True)
class RollbackRecord:
    """One rollback, with its verification result."""

    rollback_id: str
    deployment_id: str
    proposal_id: str
    status: RollbackStatus
    restored_files: tuple[str, ...] = ()
    backup_dir: str = ""
    reason: str = ""
    verified: bool = False
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def describe(self) -> str:
        lines = [f"Rollback {self.rollback_id} of {self.deployment_id}: {self.status.value}"]
        if self.restored_files:
            lines.append(f"  restored: {', '.join(self.restored_files)}")
        lines.append(f"  verified: {self.verified}")
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, object]:
        return {
            "rollback_id": self.rollback_id,
            "deployment_id": self.deployment_id,
            "proposal_id": self.proposal_id,
            "status": self.status.value,
            "restored_files": list(self.restored_files),
            "backup_dir": self.backup_dir,
            "reason": self.reason,
            "verified": self.verified,
            "at": self.at.isoformat(),
        }


class RollbackManager:
    """Reverses a deployment from its own backup, preserving everything else."""

    def __init__(
        self,
        *,
        state_root: Path | str,
        history: ImprovementHistory | None = None,
    ) -> None:
        self.state_root = Path(state_root)
        self.history = history
        self._records: list[RollbackRecord] = []

    def records(self) -> tuple[RollbackRecord, ...]:
        return tuple(self._records)

    def rollback(self, deployment: DeploymentRecord, *, reason: str = "") -> RollbackRecord:
        """Restore the files a deployment changed, from its backup.

        Refuses a deployment that was not applied (there is nothing to reverse) and
        a deployment with no backup (the phase requires a backup before a
        deployment, so a missing one is a real fault, reported as such).
        """
        if deployment.status is not DeploymentStatus.DEPLOYED:
            return self._record(
                deployment,
                RollbackStatus.NOTHING_TO_ROLL_BACK,
                reason=f"the deployment is {deployment.status.value}, not applied",
            )
        if not deployment.backup_dir:
            return self._record(
                deployment,
                RollbackStatus.REFUSED,
                reason="the deployment has no backup, so it cannot be reversed safely",
            )

        backup_dir = Path(deployment.backup_dir)
        if not backup_dir.exists():
            return self._record(
                deployment,
                RollbackStatus.FAILED,
                reason=f"the backup folder {backup_dir} is missing",
            )

        restored: list[str] = []
        try:
            for relative in deployment.applied_files:
                saved = backup_dir / relative
                target = self.state_root / relative
                if saved.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved, target)
                    restored.append(relative)
                elif target.exists():
                    target.unlink()
                    restored.append(relative)
        except OSError as error:
            return self._record(
                deployment,
                RollbackStatus.FAILED,
                reason=f"the restore failed: {error}",
                backup_dir=str(backup_dir),
            )

        verified = self._verify(deployment, tuple(restored))
        return self._record(
            deployment,
            RollbackStatus.ROLLED_BACK if verified else RollbackStatus.FAILED,
            restored_files=tuple(restored),
            backup_dir=str(backup_dir),
            reason=reason or "restored the pre-deployment state",
            verified=verified,
        )

    def rollback_model_version(self) -> object | None:
        """Delegate a model-weight rollback to the model version registry.

        The Phase-6 registry is the one place a trained model version is stored and
        reversed, so this module never re-implements that; it simply calls it, and
        a machine with no model store returns ``None`` honestly.
        """
        result: object | None = None
        with best_effort("roll back the model version"):
            from cognition.versions import ModelRegistry

            result = ModelRegistry().rollback()
        return result

    def _verify(self, deployment: DeploymentRecord, restored: tuple[str, ...]) -> bool:
        """Check that the restored files really came back.

        A file the deployment created and the rollback removed is verified by its
        absence; a file it changed is verified by its content matching the backup.
        """
        backup_dir = Path(deployment.backup_dir)
        for relative in restored:
            saved = backup_dir / relative
            target = self.state_root / relative
            if saved.exists():
                try:
                    if saved.read_text(encoding="utf-8") != target.read_text(encoding="utf-8"):
                        return False
                except OSError:
                    return False
            elif target.exists():
                return False
        return True

    def _record(
        self,
        deployment: DeploymentRecord,
        status: RollbackStatus,
        *,
        restored_files: tuple[str, ...] = (),
        backup_dir: str = "",
        reason: str = "",
        verified: bool = False,
    ) -> RollbackRecord:
        record = RollbackRecord(
            rollback_id=f"RBK-{_short()}",
            deployment_id=deployment.deployment_id,
            proposal_id=deployment.proposal_id,
            status=status,
            restored_files=restored_files,
            backup_dir=backup_dir,
            reason=reason,
            verified=verified,
        )
        self._records.append(record)
        if self.history is not None:
            self.history.append(
                EntryKind.ROLLBACK,
                f"{record.rollback_id}: {status.value} for {deployment.deployment_id}.",
                proposal_id=deployment.proposal_id,
                detail=record.to_primitive(),
            )
        return record

    def describe(self) -> str:
        if not self._records:
            return "No rollbacks have run."
        return "\n".join(["Rollbacks:", *(f"  {r.describe()}" for r in self._records)])


def _short() -> str:
    from uuid import uuid4

    return uuid4().hex[:8]


__all__ = [
    "RollbackManager",
    "RollbackRecord",
    "RollbackStatus",
]
