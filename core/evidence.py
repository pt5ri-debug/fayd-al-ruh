from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Evidence:
    """
    Represents evidence supporting or contradicting a belief/relation.
    """

    description: str

    source: str

    supports: bool = True

    strength: float = 1.0

    data: dict[str, Any] = field(default_factory=dict)

    id: str = field(
        default_factory=lambda: f"EV-{uuid4().hex[:8]}"
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.description.strip():
            raise ValueError("Evidence description cannot be empty.")

        if not self.source.strip():
            raise ValueError("Evidence source cannot be empty.")

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                "Evidence strength must be between 0 and 1."
            )