from __future__ import annotations

from core.decision_trace import DecisionTrace


class DecisionExplainer:
    def explain(self, trace: DecisionTrace) -> tuple[str, ...]:
        candidates = tuple(
            f"{item.action.name}: utility {item.utility.value:.2f}, risk {item.risk.value:.2f}, score {item.score:.2f}"
            for item in trace.candidates
        )
        selected = (f"Selected: {trace.selected.action.name}. {trace.reason}",) if trace.selected is not None else (trace.reason,)
        return selected + candidates
