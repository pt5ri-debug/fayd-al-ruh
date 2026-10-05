from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelfModel:
    """Declared identity, capabilities, and limits of this cognitive software.

    The declared identity is a domain frame, not consciousness: it says what
    kind of expert this system reasons as, never that it is self-aware.
    """

    name: str = "Fayd Al-Ruh cognitive system"
    identity: str = "petroleum engineering mind"
    domain: str = "petroleum engineering"
    capabilities: tuple[str, ...] = (
        "memory",
        "reasoning",
        "causal analysis",
        "planning",
        "learning",
        "petroleum engineering reasoning",
    )
    limitations: tuple[str, ...] = (
        "no autonomous external actions",
        "confidence is not truth",
        "requires evidence for belief revision",
        "domain knowledge is a seed, not a full engineering corpus",
        "not a substitute for certified engineering judgment or field data",
    )
    expertise: tuple[str, ...] = field(
        default_factory=lambda: (
            "reservoir",
            "drilling",
            "production",
            "petrophysics",
            "well logs",
            "porosity",
            "permeability",
            "hydrocarbons",
        )
    )

    @property
    def identity_statement(self) -> str:
        return f"I am a {self.identity}: I reason about {self.domain} from evidence."


__all__ = ["SelfModel"]
