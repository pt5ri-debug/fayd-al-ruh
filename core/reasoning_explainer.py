from __future__ import annotations

from core.reasoning_trace import ReasoningTrace


class ReasoningExplainer:
    def explain(self, trace: ReasoningTrace) -> tuple[str, ...]:
        return tuple(
            f"{result.method}: {', '.join(result.premises)} → {result.conclusion} (confidence {result.confidence:.2f})"
            for result in trace.results
        )
