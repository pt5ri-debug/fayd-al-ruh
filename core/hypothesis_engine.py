from __future__ import annotations

from core.hypothesis import Hypothesis
from core.pattern_detector import Pattern


class HypothesisEngine:
    """
    Converts discovered patterns into testable hypotheses.

    A hypothesis is not treated as established knowledge.
    """

    def create(
        self,
        pattern: Pattern,
    ) -> Hypothesis:

        statement = (
            f"Pattern '{pattern.key}' "
            f"may represent a recurring event."
        )

        confidence = min(
            pattern.confidence,
            0.8,
        )

        return Hypothesis(
            statement=statement,
            confidence=confidence,
            source_pattern=pattern.key,
        )