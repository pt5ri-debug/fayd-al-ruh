from __future__ import annotations

from collections.abc import Iterable

from core.belief import Belief
from core.information_gap import InformationGap


class Curiosity:
    """Discovers high-value uncertainty without claiming the answer is known."""
    def discover(self, beliefs: Iterable[Belief], required_confidence: float = 0.8) -> list[InformationGap]:
        if not 0.0 <= required_confidence <= 1.0:
            raise ValueError("Required confidence must be between 0 and 1.")
        gaps = [
            InformationGap(
                topic=belief.statement,
                known_confidence=belief.confidence,
                required_confidence=required_confidence,
                importance=max(0.1, 1.0 - belief.confidence),
            )
            for belief in beliefs
            if belief.confidence < required_confidence
        ]
        return sorted(gaps, key=lambda gap: (-gap.priority, gap.topic))
