"""Append-only audit log for every governed action decision.

The log is intentionally immutable: records are only ever added, never edited
or removed, so it can serve as the tamper-evident trail required before any
external tool is allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from core.permissions import Capability, PermissionDecision


class AuditOutcome(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"


@dataclass(frozen=True)
class AuditRecord:
    capability: Capability
    target: str
    outcome: AuditOutcome
    reason: str
    actor: str = "agent"
    context_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: f"AUD-{uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability": self.capability.value,
            "target": self.target,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "actor": self.actor,
            "context_id": self.context_id,
            "metadata": self.metadata,
            "recorded_at": self.recorded_at.isoformat(),
        }


class AuditLog:
    """Append-only ordered log of audit records."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> AuditRecord:
        self._records.append(record)
        return record

    def record_decision(
        self,
        decision: PermissionDecision,
        *,
        actor: str = "agent",
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        return self.append(
            AuditRecord(
                capability=decision.capability,
                target=decision.target,
                outcome=AuditOutcome.ALLOWED if decision.allowed else AuditOutcome.DENIED,
                reason=decision.reason,
                actor=actor,
                context_id=context_id,
                metadata=dict(metadata or {}),
            )
        )

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def denials(self) -> tuple[AuditRecord, ...]:
        return tuple(record for record in self._records if record.outcome is AuditOutcome.DENIED)

    def for_capability(self, capability: Capability) -> tuple[AuditRecord, ...]:
        return tuple(record for record in self._records if record.capability is capability)
