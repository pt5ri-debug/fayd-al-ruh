from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Counterfactual:
    cause: str
    effect: str
    factual_confidence: float
    counterfactual_confidence: float

    @property
    def changed(self) -> bool:
        return self.factual_confidence != self.counterfactual_confidence
