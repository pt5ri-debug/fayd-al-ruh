from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Procedure:
    name: str
    steps: tuple[str, ...]
    confidence: float = .5

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.steps or any(not step.strip() for step in self.steps):
            raise ValueError("Procedure needs a name and non-empty steps.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Procedure confidence must be between 0 and 1.")


class ProceduralMemory:
    """Stores declarative procedure descriptions; it never executes them."""
    def __init__(self) -> None:
        self.procedures: dict[str, Procedure] = {}

    def learn(self, procedure: Procedure) -> None:
        self.procedures[procedure.name] = procedure

    def retrieve(self, name: str) -> Procedure | None:
        return self.procedures.get(name)
