from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.uncertainty import Uncertainty


@dataclass(frozen=True)
class ConfidenceRecord:
    belief_id: str
    statement: str
    uncertainty: Uncertainty


class ConfidenceMonitor:
    def inspect(self, beliefs: list[Belief], threshold: float = .8) -> list[ConfidenceRecord]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        return [ConfidenceRecord(item.id, item.statement, Uncertainty.from_confidence(item.confidence)) for item in beliefs if item.confidence < threshold]
