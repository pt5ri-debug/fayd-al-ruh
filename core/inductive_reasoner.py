from __future__ import annotations

from collections import Counter

from core.inference_result import InferenceResult


class InductiveReasoner:
    """Forms tentative generalizations from repeated observations."""
    def infer(self, observations: list[str], minimum_occurrences: int = 2) -> list[InferenceResult]:
        if minimum_occurrences < 2:
            raise ValueError("Induction requires at least two occurrences.")
        if not observations:
            return []
        counts = Counter(observation.strip() for observation in observations if observation.strip())
        total = len(observations)
        return [
            InferenceResult(
                conclusion=f"'{statement}' is a recurring pattern.",
                confidence=round(count / total, 6),
                method="inductive",
                premises=tuple([statement] * count),
            )
            for statement, count in counts.items()
            if count >= minimum_occurrences
        ]
