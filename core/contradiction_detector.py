from __future__ import annotations

from core.contradiction import Contradiction
from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting


class ContradictionDetector:
    """Finds evidence that pulls the same belief in opposing directions."""
    def __init__(self, weighting: EvidenceWeighting | None = None) -> None:
        self.weighting = weighting or EvidenceWeighting()

    def detect(self, evidence_items: list[Evidence]) -> list[Contradiction]:
        supporting = [item for item in evidence_items if item.supports]
        contradicting = [item for item in evidence_items if not item.supports]
        return [
            Contradiction(support.id, oppose.id, round(min(self.weighting.weight(support), self.weighting.weight(oppose)), 6))
            for support in supporting
            for oppose in contradicting
        ]
