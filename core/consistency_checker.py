from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable

from core.belief import Belief
from core.belief_conflict import BeliefConflict


@dataclass(frozen=True)
class ConsistencyReport:
    consistent: bool
    conflicts: tuple[BeliefConflict, ...]


class ConsistencyChecker:
    """Detects explicit statement/negation conflicts without assuming correlations are causal."""
    def check(self, beliefs: Iterable[Belief]) -> ConsistencyReport:
        items = list(beliefs)
        conflicts: list[BeliefConflict] = []
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                if self._opposes(first.statement, second.statement):
                    conflicts.append(BeliefConflict(first, second, round(min(first.confidence, second.confidence), 6)))
        return ConsistencyReport(not conflicts, tuple(conflicts))

    @staticmethod
    def _normalise(statement: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", statement.casefold())).strip()

    @classmethod
    def _opposes(cls, first: str, second: str) -> bool:
        left, right = cls._normalise(first), cls._normalise(second)
        prefixes = ("not ", "no ", "لا ", "ليس ")
        return any((left.startswith(prefix) and left[len(prefix):] == right) or (right.startswith(prefix) and right[len(prefix):] == left) for prefix in prefixes)
