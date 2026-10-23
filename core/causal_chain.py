from __future__ import annotations

from dataclasses import dataclass

from core.causal_link import CausalLink


@dataclass(frozen=True)
class CausalChain:
    """An ordered explanation composed of compatible causal links."""

    links: tuple[CausalLink, ...]

    def __post_init__(self) -> None:
        if not self.links:
            raise ValueError("A causal chain needs at least one link.")
        if any(left.effect != right.cause for left, right in zip(self.links, self.links[1:], strict=False)):
            raise ValueError("Causal chain links must connect in order.")

    @property
    def cause(self) -> str:
        return self.links[0].cause

    @property
    def effect(self) -> str:
        return self.links[-1].effect

    @property
    def confidence(self) -> float:
        value = 1.0
        for link in self.links:
            value *= link.confidence
        return round(value, 6)

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.cause, *tuple(link.effect for link in self.links))
