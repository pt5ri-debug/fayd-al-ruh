from __future__ import annotations

from dataclasses import dataclass

from core.causal_effect import CausalEffect


@dataclass(frozen=True)
class CausalExperiment:
    cause: str
    effect: str
    treated_outcomes: tuple[float, ...]
    control_outcomes: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.treated_outcomes or not self.control_outcomes:
            raise ValueError("An experiment needs treated and control outcomes.")
        if any(not 0.0 <= value <= 1.0 for value in self.treated_outcomes + self.control_outcomes):
            raise ValueError("Experiment outcomes must be between 0 and 1.")

    def estimate_effect(self) -> CausalEffect:
        treated = sum(self.treated_outcomes) / len(self.treated_outcomes)
        control = sum(self.control_outcomes) / len(self.control_outcomes)
        return CausalEffect(self.cause, self.effect, round(treated, 6), round(control, 6), round(treated - control, 6))
