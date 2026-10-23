from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class Relation:
    """
    A typed relationship between two entities.

    Example:
        Robot --SEES--> Ball

    The relation stores its own confidence,
    evidence references, provenance, and optional belief reference.
    """

    source: str
    target: str
    relation_type: str

    confidence: float = 0.5

    belief_id: str | None = None

    evidence_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    id: str = field(
        default_factory=lambda: f"R-{uuid4().hex[:8]}"
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    active: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.source.strip():
            raise ValueError(
                "Relation source cannot be empty."
            )

        if not self.target.strip():
            raise ValueError(
                "Relation target cannot be empty."
            )

        if not self.relation_type.strip():
            raise ValueError(
                "Relation type cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Relation confidence must be between 0 and 1."
            )

        if self.source == self.target:
            raise ValueError(
                "Source and target cannot be identical."
            )

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def set_confidence(self, confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Relation confidence must be between 0 and 1."
            )

        self.confidence = confidence
        self._touch()

    def attach_belief(self, belief_id: str) -> None:
        if not belief_id.strip():
            raise ValueError(
                "Belief ID cannot be empty."
            )

        self.belief_id = belief_id
        self._touch()

    def add_evidence(self, evidence_id: str) -> None:
        if not evidence_id.strip():
            raise ValueError(
                "Evidence ID cannot be empty."
            )

        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            self._touch()

    def remove_evidence(self, evidence_id: str) -> None:
        if evidence_id in self.evidence_ids:
            self.evidence_ids.remove(evidence_id)
            self._touch()

    def deactivate(self) -> None:
        self.active = False
        self._touch()

    def activate(self) -> None:
        self.active = True
        self._touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "belief_id": self.belief_id,
            "evidence_ids": self.evidence_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active": self.active,
        }
