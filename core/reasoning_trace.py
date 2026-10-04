from __future__ import annotations

from dataclasses import dataclass, field

from core.inference_result import InferenceResult


@dataclass
class ReasoningTrace:
    """Auditable chronological trace; it contains no hidden chain of thought."""
    results: list[InferenceResult] = field(default_factory=list)

    def add(self, result: InferenceResult) -> None:
        self.results.append(result)

    @property
    def conclusions(self) -> tuple[str, ...]:
        return tuple(result.conclusion for result in self.results)
