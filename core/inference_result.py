from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    conclusion: str
    confidence: float
    method: str
    premises: tuple[str, ...]
    rule_id: str | None = None

    def __post_init__(self) -> None:
        if not self.conclusion.strip():
            raise ValueError("Inference conclusion cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Inference confidence must be between 0 and 1.")
