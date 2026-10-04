from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InformationGap:
    """An explicit description of knowledge that is needed but not established."""
    topic: str
    known_confidence: float
    required_confidence: float = 0.8
    importance: float = 0.5

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("Information-gap topic cannot be empty.")
        for value in (self.known_confidence, self.required_confidence, self.importance):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Gap values must be between 0 and 1.")

    @property
    def resolved(self) -> bool:
        return self.known_confidence >= self.required_confidence

    @property
    def priority(self) -> float:
        """Missing confidence weighted by the practical importance of the topic."""
        return round(max(0.0, self.required_confidence - self.known_confidence) * self.importance, 6)
