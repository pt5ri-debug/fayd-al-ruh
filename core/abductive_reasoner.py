from __future__ import annotations

from core.inference_result import InferenceResult
from core.inference_rule import InferenceRule


class AbductiveReasoner:
    """Proposes possible explanations for an observed conclusion, ranked by confidence."""
    def explain(self, observation: str, rules: list[InferenceRule]) -> list[InferenceResult]:
        results = [
            InferenceResult(
                conclusion=f"Possible explanation for '{observation}': " + " and ".join(rule.premises),
                confidence=rule.confidence,
                method="abductive",
                premises=(observation,),
                rule_id=rule.id,
            )
            for rule in rules
            if rule.conclusion == observation
        ]
        return sorted(results, key=lambda result: -result.confidence)
