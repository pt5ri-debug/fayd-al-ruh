from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True)
class TemporalEvent:
    description: str
    start: datetime
    end: datetime | None = None
    value: float | None = None
    id: str = field(default_factory=lambda: f"TE-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Temporal event description cannot be empty.")
        if self.end is not None and self.end < self.start:
            raise ValueError("Event end cannot precede its start.")

    @property
    def finish(self) -> datetime:
        return self.end or self.start

    @property
    def duration_seconds(self) -> float:
        return (self.finish - self.start).total_seconds()
