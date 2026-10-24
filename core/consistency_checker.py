from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from core.belief import Belief
from core.belief_conflict import BeliefConflict
from core.relation_polarity import opposes


@dataclass(frozen=True)
class ConsistencyReport:
    consistent: bool
    conflicts: tuple[BeliefConflict, ...]


class ConsistencyChecker:
    """Detects explicit statement/negation conflicts without assuming correlations are causal.

    Two beliefs conflict when one is the explicit negation of the other (the
    original statement-level rule) *or* when they attach the two poles of the
    same relation axis to the same ordered pair of concepts -- "waterflood
    increases recovery" against "waterflood reduces recovery". The second rule
    reads the relation type out of the belief's ``metadata`` (``relation_type``,
    plus ``source_id``/``target_id``), so it is *direction-aware*: it never calls
    "A increases B" and "B reduces A" contradictory, which a free-text check on
    the verbs alone cannot promise. A belief with no relation metadata is judged
    by the statement rule alone.
    """

    def check(self, beliefs: Iterable[Belief]) -> ConsistencyReport:
        items = list(beliefs)
        conflicts: list[BeliefConflict] = []
        for index, first in enumerate(items):
            for second in items[index + 1 :]:
                if self._opposes(first.statement, second.statement) or self._relations_oppose(first, second):
                    conflicts.append(BeliefConflict(first, second, round(min(first.confidence, second.confidence), 6)))
        return ConsistencyReport(not conflicts, tuple(conflicts))

    @staticmethod
    def _relation_axis(belief: Belief) -> tuple[str, str, str] | None:
        """The ``(source_id, target_id, relation_type)`` a belief asserts, if any."""
        metadata = belief.metadata or {}
        relation_type = str(metadata.get("relation_type", "")).strip()
        if not relation_type:
            return None
        source_id = str(metadata.get("source_id", "")).strip()
        target_id = str(metadata.get("target_id", "")).strip()
        if not source_id or not target_id:
            return None
        return source_id, target_id, relation_type

    @classmethod
    def _relations_oppose(cls, first: Belief, second: Belief) -> bool:
        """Whether two beliefs assert opposing relations on the same ordered pair.

        The pair must match in *direction* (``A increases B`` does not conflict
        with ``B reduces A``) and the two relation types must be explicit
        antonyms (``increases``/``reduces``), so two unrelated typed relations are
        never called contradictory.
        """
        left, right = cls._relation_axis(first), cls._relation_axis(second)
        if left is None or right is None:
            return False
        left_source, left_target, left_type = left
        right_source, right_target, right_type = right
        if (left_source, left_target) != (right_source, right_target):
            return False
        return opposes(left_type, right_type)

    @staticmethod
    def _normalise(statement: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", statement.casefold())).strip()

    @classmethod
    def _opposes(cls, first: str, second: str) -> bool:
        left, right = cls._normalise(first), cls._normalise(second)
        prefixes = ("not ", "no ", "لا ", "ليس ")
        return any((left.startswith(prefix) and left[len(prefix):] == right) or (right.startswith(prefix) and right[len(prefix):] == left) for prefix in prefixes)
