from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Hypothesis:
    statement: str
    confidence: float
    source_pattern: str
    id: str = field(
        default_factory=lambda: f"HYP-{uuid4().hex[:8]}"
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError(
                "Hypothesis statement cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Hypothesis confidence must be between 0 and 1."
            )

        if not self.source_pattern.strip():
            raise ValueError(
                "Source pattern cannot be empty."
            )