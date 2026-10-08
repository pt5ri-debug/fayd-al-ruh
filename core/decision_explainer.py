from __future__ import annotations

from core.composition import decision_sentences
from core.decision_trace import DecisionTrace


class DecisionExplainer:
    def explain(self, trace: DecisionTrace) -> tuple[str, ...]:
        candidates = tuple(
            (item.action.name, item.utility.value, item.risk.value, item.score)
            for item in trace.candidates
        )
        selected = trace.selected.action.name if trace.selected is not None else None
        return decision_sentences(selected, trace.reason, candidates)
