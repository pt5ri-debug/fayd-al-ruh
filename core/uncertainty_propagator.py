from __future__ import annotations

from collections.abc import Iterable

from core.uncertainty import Uncertainty


class UncertaintyPropagator:
    """Propagates confidence across a dependent chain conservatively."""
    def propagate(self, confidences: Iterable[float]) -> Uncertainty:
        values = list(confidences)
        if not values or any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Provide one or more confidences between 0 and 1.")
        confidence = 1.0
        for value in values:
            confidence *= value
        return Uncertainty.from_confidence(confidence)
