from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Utility:
    value: float
    benefit: float
    cost: float

    def __post_init__(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in (self.value, self.benefit, self.cost)):
            raise ValueError("Utility values must be between 0 and 1.")
