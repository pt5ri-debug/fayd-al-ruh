from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class CausalLink:
    """A directed, evidence-backed causal claim: cause -> effect."""

    cause: str
    effect: str
    confidence: float = 0.5
    evidence_ids: list[str] = field(default_factory=list)
    mechanism: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"CL-{uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active: bool = True

    def __post_init__(self) -> None:
        if not self.cause.strip() or not self.effect.strip():
            raise ValueError("Cause and effect cannot be empty.")
        if self.cause == self.effect:
            raise ValueError("A causal link cannot cause itself.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Causal confidence must be between 0 and 1.")
        if self.mechanism is not None and not self.mechanism.strip():
            raise ValueError("Mechanism cannot be blank.")

    def add_evidence(self, evidence_id: str) -> None:
        if not evidence_id.strip():
            raise ValueError("Evidence ID cannot be empty.")
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)

    def deactivate(self) -> None:
        self.active = False
