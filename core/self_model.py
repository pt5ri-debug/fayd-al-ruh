from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfModel:
    """Declared capabilities and limits of this cognitive software, not consciousness."""
    name: str = "Fayd Al-Ruh cognitive system"
    capabilities: tuple[str, ...] = ("memory", "reasoning", "causal analysis", "planning", "learning")
    limitations: tuple[str, ...] = ("no autonomous external actions", "confidence is not truth", "requires evidence for belief revision")
