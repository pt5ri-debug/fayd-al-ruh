from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass
class Prediction:
    statement: str
    confidence: float
    hypothesis_id: str

    id: str = field(
        default_factory=lambda: f"PRD-{uuid4().hex[:8]}"
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    fulfilled: bool | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError(
                "Prediction statement cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Prediction confidence must be between 0 and 1."
            )

        if not self.hypothesis_id.strip():
            raise ValueError(
                "Hypothesis ID cannot be empty."
            )

    def resolve(self, fulfilled: bool) -> None:
        self.fulfilled = fulfilled
