from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class State:
    facts: frozenset[str] = field(default_factory=frozenset)

    def satisfies(self, required: frozenset[str]) -> bool:
        return required <= self.facts
