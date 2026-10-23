from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from core.evidence import Evidence


class BeliefStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    PROVISIONAL = "provisional"
    ACCEPTED = "accepted"
    SUSPENDED = "suspended"
    REJECTED = "rejected"


@dataclass
class Belief:
    """
    Represents a changeable cognitive belief.

    A belief is not treated as absolute truth.
    It has confidence, supporting evidence,
    contradicting evidence, and a lifecycle status.

    A belief may optionally reference the hypothesis
    from which it originated.
    """

    statement: str

    confidence: float = 0.5

    status: BeliefStatus = BeliefStatus.HYPOTHESIS

    source_hypothesis_id: str | None = None

    id: str = field(
        default_factory=lambda: f"BL-{uuid4().hex[:8]}"
    )

    supporting_evidence: list[str] = field(
        default_factory=list
    )

    contradicting_evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.statement.strip():
            raise ValueError(
                "Belief statement cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Belief confidence must be between 0 and 1."
            )

        if (
            self.source_hypothesis_id is not None
            and not self.source_hypothesis_id.strip()
        ):
            raise ValueError(
                "Source hypothesis ID cannot be empty."
            )

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def add_evidence(self, evidence: Evidence) -> None:
        """
        Attach evidence to the belief.
        """

        if evidence.supports:
            if evidence.id not in self.supporting_evidence:
                self.supporting_evidence.append(
                    evidence.id
                )
        else:
            if evidence.id not in self.contradicting_evidence:
                self.contradicting_evidence.append(
                    evidence.id
                )

        self._touch()

    def set_confidence(self, confidence: float) -> None:
        """
        Update confidence while keeping it within [0, 1].
        """

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Belief confidence must be between 0 and 1."
            )

        self.confidence = confidence
        self._touch()

    def set_status(self, status: BeliefStatus) -> None:
        """
        Change the lifecycle state of the belief.
        """

        self.status = status
        self._touch()

    def evidence_count(self) -> int:
        return (
            len(self.supporting_evidence)
            + len(self.contradicting_evidence)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "status": self.status.value,
            "source_hypothesis_id": self.source_hypothesis_id,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
