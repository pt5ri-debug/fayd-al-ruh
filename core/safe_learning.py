from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SandboxResult:
    name: str
    reward: float
    observation: str


class LearningSandbox:
    """Runs registered pure simulations only; no shell, network, or file access."""

    def __init__(self) -> None:
        self._experiments: dict[str, Callable[[dict[str, float]], SandboxResult]] = {}

    def register(self, name: str, experiment: Callable[[dict[str, float]], SandboxResult]) -> None:
        if not name.strip():
            raise ValueError("Experiment name cannot be empty.")
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists.")
        self._experiments[name] = experiment

    def run(self, name: str, state: dict[str, float]) -> SandboxResult:
        if name not in self._experiments:
            raise KeyError(f"Unknown experiment: {name}")
        if any(not isinstance(value, (int, float)) for value in state.values()):
            raise ValueError("Sandbox state values must be numeric.")
        result = self._experiments[name](state.copy())
        if not -1.0 <= result.reward <= 1.0:
            raise ValueError("Experiment reward must be between -1 and 1.")
        return result
