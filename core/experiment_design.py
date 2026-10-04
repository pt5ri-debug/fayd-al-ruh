from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentDesign:
    name: str
    variable: str
    intervention: str
    trials: int = 1
    control_description: str = "baseline"

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.name, self.variable, self.intervention, self.control_description)):
            raise ValueError("Experiment design fields cannot be empty.")
        if self.trials < 1:
            raise ValueError("An experiment needs at least one trial.")
