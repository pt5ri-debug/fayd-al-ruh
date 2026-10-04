from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class Question:
    text: str
    topic: str
    priority: float
    id: str = field(default_factory=lambda: f"Q-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.text.rstrip().endswith("?"):
            raise ValueError("Question text must be non-empty and end with '?'.")
        if not self.topic.strip():
            raise ValueError("Question topic cannot be empty.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("Question priority must be between 0 and 1.")
