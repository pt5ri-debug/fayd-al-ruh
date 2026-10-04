from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief


@dataclass(frozen=True)
class BeliefConflict:
    first: Belief
    second: Belief
    severity: float

    def __post_init__(self) -> None:
        if self.first.id == self.second.id:
            raise ValueError("A belief cannot conflict with itself.")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("Conflict severity must be between 0 and 1.")

    @property
    def belief_ids(self) -> tuple[str, str]:
        return self.first.id, self.second.id
