from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class TemporalPattern:
    description: str
    occurrences: int
    average_interval: timedelta | None
    average_duration: timedelta
    direction: str
    confidence: float
