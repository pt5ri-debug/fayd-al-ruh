from __future__ import annotations

from core.inference_result import InferenceResult
from core.inference_rule import InferenceRule


class DeductiveReasoner:
    """Forward-chaining deduction. Conclusions are produced only by satisfied rules."""
    def infer(self, facts: set[str] | tuple[str, ...], rules: list[InferenceRule]) -> list[InferenceResult]:
        known = set(facts)
        results: list[InferenceResult] = []
        changed = True
        while changed:
            changed = False
            for rule in rules:
                if rule.conclusion in known or not set(rule.premises) <= known:
                    continue
                known.add(rule.conclusion)
                results.append(InferenceResult(rule.conclusion, rule.confidence, "deductive", rule.premises, rule.id))
                changed = True
        return results
