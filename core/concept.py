from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class Concept:
    label: str
    aliases: frozenset[str] = field(default_factory=frozenset)
    id: str = field(default_factory=lambda: f"CON-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Concept label cannot be empty.")

    @property
    def terms(self) -> frozenset[str]:
        return frozenset({self.label.casefold(), *(alias.casefold() for alias in self.aliases)})
