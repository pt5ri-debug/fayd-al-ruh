from __future__ import annotations

from dataclasses import dataclass

from core.action import Action
from core.risk import Risk
from core.utility import Utility


@dataclass(frozen=True)
class Decision:
    action: Action
    utility: Utility
    risk: Risk
    score: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("Decision score must be between -1 and 1.")
