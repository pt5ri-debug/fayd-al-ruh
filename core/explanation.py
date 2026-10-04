from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    conclusion: str
    summary: str
    evidence: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    causal: tuple[str, ...] = ()
    decision: tuple[str, ...] = ()
