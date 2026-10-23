from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.evidence import Evidence
from core.evidence_evaluator import (
    EvaluationResult,
    EvidenceEvaluator,
)


@dataclass
class LearningResult:
    belief_id: str
    previous_confidence: float
    new_confidence: float
    evidence_count: int
    learned: bool


class LearningEngine:
    """
    First-generation learning engine.

    It uses accumulated evidence to update a belief.
    """

    def __init__(
        self,
        evaluator: EvidenceEvaluator | None = None,
    ) -> None:

        self.evaluator = (
            evaluator
            if evaluator is not None
            else EvidenceEvaluator()
        )

    def learn(
        self,
        belief: Belief,
        evidence_items: list[Evidence],
    ) -> LearningResult:

        previous_confidence = belief.confidence

        if not evidence_items:

            return LearningResult(
                belief_id=belief.id,
                previous_confidence=previous_confidence,
                new_confidence=previous_confidence,
                evidence_count=0,
                learned=False,
            )

        result: EvaluationResult = (
            self.evaluator.apply(
                belief,
                evidence_items,
            )
        )

        learned = (
            result.new_confidence
            != result.old_confidence
        )

        return LearningResult(
            belief_id=belief.id,
            previous_confidence=previous_confidence,
            new_confidence=result.new_confidence,
            evidence_count=len(evidence_items),
            learned=learned,
        )
