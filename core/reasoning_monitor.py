from __future__ import annotations

from dataclasses import dataclass

from core.reasoning_trace import ReasoningTrace


@dataclass(frozen=True)
class ReasoningSummary:
    conclusion_count: int
    methods: tuple[str, ...]
    conclusions: tuple[str, ...]


class ReasoningMonitor:
    def inspect(self, trace: ReasoningTrace) -> ReasoningSummary:
        return ReasoningSummary(len(trace.results), tuple(dict.fromkeys(result.method for result in trace.results)), trace.conclusions)
