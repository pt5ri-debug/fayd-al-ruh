from __future__ import annotations

from dataclasses import dataclass

from core.cognitive_state import CognitiveState
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory


@dataclass(frozen=True)
class QualityAssessment:
    logical_consistency: float
    memory_retrieval: float
    calibration: float
    overall: float


class EvaluationSuite:
    """Deterministic checks for integrity, retrieval, and confidence quality."""

    def assess(self, graph: KnowledgeGraph, memory: CognitiveMemory, state: CognitiveState) -> QualityAssessment:
        consistency = 1.0 if not graph.validate() else 0.0
        observations = memory.find_by_type("observation")
        memory_score = 1.0 if not observations else min(1.0, sum(item.access_count > 0 for item in observations) / len(observations))
        errors = state.prediction_errors
        calibration = 1.0 if not errors else max(0.0, 1.0 - sum(error.error ** 2 for error in errors) / len(errors))
        overall = (consistency + memory_score + calibration) / 3
        return QualityAssessment(*(round(value, 6) for value in (consistency, memory_score, calibration, overall)))
