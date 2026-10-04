from __future__ import annotations

from dataclasses import dataclass

from core.decision import Decision


@dataclass(frozen=True)
class DecisionTrace:
    candidates: tuple[Decision, ...]
    selected: Decision | None
    reason: str
