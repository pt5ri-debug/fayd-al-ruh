from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief, BeliefStatus
from core.confidence_model import ConfidenceModel
from core.contradiction_detector import ContradictionDetector
from core.evidence import Evidence
from core.uncertainty import Uncertainty


@dataclass(frozen=True)
class BeliefRevision:
    belief_id: str
    previous: Uncertainty
    revised: Uncertainty
    contradiction_count: int


class BeliefRevisionEngine:
    """Revises a belief from all known evidence, rather than treating it as binary."""
    def __init__(self, model: ConfidenceModel | None = None, detector: ContradictionDetector | None = None) -> None:
        self.model = model or ConfidenceModel()
        self.detector = detector or ContradictionDetector(self.model.weighting)

    def revise(self, belief: Belief, evidence_items: list[Evidence]) -> BeliefRevision:
        previous = Uncertainty.from_confidence(belief.confidence)
        revised = self.model.update(belief.confidence, evidence_items)
        contradictions = self.detector.detect(evidence_items)
        belief.set_confidence(revised.confidence)
        if contradictions:
            belief.set_status(BeliefStatus.SUSPENDED)
        elif revised.confidence >= .8:
            belief.set_status(BeliefStatus.ACCEPTED)
        elif revised.confidence < .2:
            belief.set_status(BeliefStatus.REJECTED)
        else:
            belief.set_status(BeliefStatus.PROVISIONAL)
        return BeliefRevision(belief.id, previous, revised, len(contradictions))
