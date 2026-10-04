from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalEffect:
    cause: str
    effect: str
    treated_outcome: float
    control_outcome: float
    effect_size: float

    def __post_init__(self) -> None:
        if not self.cause.strip() or not self.effect.strip():
            raise ValueError("Cause and effect cannot be empty.")
        if any(not 0.0 <= value <= 1.0 for value in (self.treated_outcome, self.control_outcome)):
            raise ValueError("Outcome probabilities must be between 0 and 1.")
        if round(self.effect_size, 6) != round(self.treated_outcome - self.control_outcome, 6):
            raise ValueError("Effect size must equal treated outcome minus control outcome.")
