from __future__ import annotations

from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting
from core.uncertainty import Uncertainty


class ConfidenceModel:
    """Evidence-weighted posterior model with a finite prior strength."""
    def __init__(self, prior_strength: float = 1.0, weighting: EvidenceWeighting | None = None) -> None:
        if prior_strength <= 0:
            raise ValueError("Prior strength must be positive.")
        self.prior_strength = prior_strength
        self.weighting = weighting or EvidenceWeighting()

    def update(self, prior_confidence: float, evidence_items: list[Evidence]) -> Uncertainty:
        if not 0.0 <= prior_confidence <= 1.0:
            raise ValueError("Prior confidence must be between 0 and 1.")
        support = sum(self.weighting.weight(item) for item in evidence_items if item.supports)
        contradict = sum(self.weighting.weight(item) for item in evidence_items if not item.supports)
        denominator = self.prior_strength + support + contradict
        confidence = (self.prior_strength * prior_confidence + support) / denominator
        return Uncertainty.from_confidence(confidence)
