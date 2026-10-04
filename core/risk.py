from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Risk:
    value: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Risk must be between 0 and 1.")
