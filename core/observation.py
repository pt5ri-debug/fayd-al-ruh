from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Observation:
    """
    Represents an observation received from the environment.

    An observation is raw input.
    It is not automatically considered truth.
    """

    content: str

    source: str

    confidence: float = 1.0

    data: dict[str, Any] = field(
        default_factory=dict
    )

    id: str = field(
        default_factory=lambda: f"OBS-{uuid4().hex[:8]}"
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.content.strip():
            raise ValueError(
                "Observation content cannot be empty."
            )

        if not self.source.strip():
            raise ValueError(
                "Observation source cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Observation confidence must be between 0 and 1."
            )

    def to_evidence(self):
        from core.evidence import Evidence

        return Evidence(
            description=self.content,
            source=self.source,
            supports=True,
            strength=self.confidence,
            data=self.data.copy(),
        )