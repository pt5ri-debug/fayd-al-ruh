from __future__ import annotations

import itertools
from collections import Counter

from core.causal_link import CausalLink


class CausalDiscovery:
    """Proposes directional candidates from repeated ordered observations.

    Candidates are marked as discovered and still require experimental or
    evidential validation before being treated as causal knowledge.
    """
    def discover(self, ordered_events: list[str], minimum_occurrences: int = 2) -> list[CausalLink]:
        if minimum_occurrences < 2:
            raise ValueError("Causal discovery needs at least two occurrences.")
        transitions = Counter(itertools.pairwise(ordered_events))
        total = max(1, sum(transitions.values()))
        return [
            CausalLink(cause, effect, round(count / total, 6), metadata={"status": "candidate", "occurrences": count})
            for (cause, effect), count in transitions.items()
            if cause != effect and count >= minimum_occurrences
        ]
