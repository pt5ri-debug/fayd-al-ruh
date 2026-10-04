from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyScore:
    name: str
    value: float
    attempts: int


class StrategyLearning:
    """Learns which named reasoning strategy has worked best in a context."""
    def __init__(self) -> None:
        self._scores: dict[str, StrategyScore] = {}

    def record(self, name: str, outcome: float) -> StrategyScore:
        if not name.strip() or not 0.0 <= outcome <= 1.0:
            raise ValueError("Strategy name and outcome must be valid.")
        previous = self._scores.get(name, StrategyScore(name, .5, 0))
        attempts = previous.attempts + 1
        value = round(previous.value + (outcome - previous.value) / attempts, 6)
        score = StrategyScore(name, value, attempts)
        self._scores[name] = score
        return score

    def select(self, candidates: list[str]) -> StrategyScore | None:
        known = [self._scores[name] for name in candidates if name in self._scores]
        return max(known, key=lambda score: (score.value, score.attempts, score.name)) if known else None
