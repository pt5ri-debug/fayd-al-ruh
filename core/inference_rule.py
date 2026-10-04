from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class InferenceRule:
    """A transparent rule: all premises imply one conclusion."""
    premises: tuple[str, ...]
    conclusion: str
    confidence: float = 1.0
    name: str = "rule"
    id: str = field(default_factory=lambda: f"RULE-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.premises or any(not premise.strip() for premise in self.premises):
            raise ValueError("An inference rule needs non-empty premises.")
        if not self.conclusion.strip():
            raise ValueError("Rule conclusion cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Rule confidence must be between 0 and 1.")
