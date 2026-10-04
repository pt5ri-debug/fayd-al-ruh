from __future__ import annotations

from core.evidence import Evidence


class EvidenceWeighting:
    """Weights evidence by its strength and the calibrated reliability of its source."""
    def __init__(self, source_reliability: dict[str, float] | None = None) -> None:
        self.source_reliability = source_reliability or {}
        if any(not 0.0 <= value <= 1.0 for value in self.source_reliability.values()):
            raise ValueError("Source reliability must be between 0 and 1.")

    def weight(self, evidence: Evidence) -> float:
        reliability = evidence.data.get("source_reliability", self.source_reliability.get(evidence.source, .5))
        if not isinstance(reliability, (int, float)) or not 0.0 <= reliability <= 1.0:
            raise ValueError("Evidence source reliability must be between 0 and 1.")
        return round(evidence.strength * reliability, 6)
