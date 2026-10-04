from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Uncertainty:
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("Confidence and uncertainty must be between 0 and 1.")
        if round(self.confidence + self.uncertainty, 6) != 1.0:
            raise ValueError("Confidence and uncertainty must sum to 1.")

    @classmethod
    def from_confidence(cls, confidence: float) -> "Uncertainty":
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        return cls(round(confidence, 6), round(1.0 - confidence, 6))
