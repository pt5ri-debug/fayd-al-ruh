from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class Action:
    """A declarative action; it has no external side effects."""
    name: str
    preconditions: frozenset[str] = field(default_factory=frozenset)
    add_effects: frozenset[str] = field(default_factory=frozenset)
    remove_effects: frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0
    id: str = field(default_factory=lambda: f"ACT-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.name.strip() or self.cost < 0:
            raise ValueError("Action needs a name and non-negative cost.")
        if self.add_effects & self.remove_effects:
            raise ValueError("An action cannot add and remove the same fact.")
