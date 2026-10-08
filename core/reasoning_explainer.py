from __future__ import annotations

from core.composition import reasoning_step_sentence
from core.reasoning_trace import ReasoningTrace


class ReasoningExplainer:
    def explain(self, trace: ReasoningTrace) -> tuple[str, ...]:
        return tuple(
            reasoning_step_sentence(
                result.method,
                result.premises,
                result.conclusion,
                result.confidence,
            )
            for result in trace.results
        )
