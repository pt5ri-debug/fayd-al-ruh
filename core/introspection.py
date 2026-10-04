from __future__ import annotations

from dataclasses import dataclass

from core.evidence import Evidence


@dataclass(frozen=True)
class BeliefExplanation:
    belief_id: str
    statement: str
    confidence: float
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]


class Introspection:
    """Produces compact explainable belief provenance without exposing hidden reasoning."""
    def explain_belief(self, belief, evidence: list[Evidence]) -> BeliefExplanation:
        known = {item.id for item in evidence}
        return BeliefExplanation(
            belief.id,
            belief.statement,
            belief.confidence,
            tuple(item for item in belief.supporting_evidence if item in known),
            tuple(item for item in belief.contradicting_evidence if item in known),
        )
