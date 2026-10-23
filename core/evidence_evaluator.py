from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.evidence import Evidence


@dataclass
class EvaluationResult:
    old_confidence: float
    new_confidence: float
    supporting_strength: float
    contradicting_strength: float


class EvidenceEvaluator:
    """
    Evaluates evidence and updates belief confidence.

    This is deliberately kept separate from Belief so that
    the confidence model can evolve independently.
    """

    def evaluate(
        self,
        belief: Belief,
        evidence_items: list[Evidence],
    ) -> EvaluationResult:

        supporting_strength = sum(
            evidence.strength
            for evidence in evidence_items
            if evidence.supports
        )

        contradicting_strength = sum(
            evidence.strength
            for evidence in evidence_items
            if not evidence.supports
        )

        total_strength = (
            supporting_strength
            + contradicting_strength
        )

        old_confidence = belief.confidence

        if total_strength == 0:
            return EvaluationResult(
                old_confidence=old_confidence,
                new_confidence=old_confidence,
                supporting_strength=0.0,
                contradicting_strength=0.0,
            )

        evidence_balance = (
            supporting_strength
            / total_strength
        )

        new_confidence = (
            old_confidence * 0.5
            + evidence_balance * 0.5
        )

        new_confidence = max(
            0.0,
            min(1.0, new_confidence),
        )

        return EvaluationResult(
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            supporting_strength=supporting_strength,
            contradicting_strength=contradicting_strength,
        )

    def apply(
        self,
        belief: Belief,
        evidence_items: list[Evidence],
    ) -> EvaluationResult:

        result = self.evaluate(
            belief,
            evidence_items,
        )

        belief.set_confidence(
            result.new_confidence
        )

        return result
