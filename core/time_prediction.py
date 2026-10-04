from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TimePrediction:
    description: str
    predicted_time: datetime
    confidence: float
    basis_occurrences: int
