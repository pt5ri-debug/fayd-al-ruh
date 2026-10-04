from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.learning_signal import LearningSignal


@dataclass(frozen=True)
class LearningEvent:
    signal: LearningSignal
    target_id: str
    before: float
    after: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LearningHistory:
    def __init__(self) -> None:
        self.events: list[LearningEvent] = []

    def record(self, event: LearningEvent) -> None:
        self.events.append(event)

    @property
    def average_error(self) -> float:
        if not self.events:
            return 0.0
        return round(sum(event.signal.prediction_error for event in self.events) / len(self.events), 6)
