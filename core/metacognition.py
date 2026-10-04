from __future__ import annotations

from dataclasses import dataclass

from core.cognitive_state import CognitiveState
from core.confidence_monitor import ConfidenceMonitor, ConfidenceRecord
from core.error_monitor import ErrorMonitor, ErrorSummary
from core.knowledge_state import KnowledgeState
from core.self_model import SelfModel


@dataclass(frozen=True)
class MetacognitiveReport:
    self_model: SelfModel
    knowledge: KnowledgeState
    low_confidence_beliefs: tuple[ConfidenceRecord, ...]
    errors: ErrorSummary


class Metacognition:
    """Read-only assessment of what the system knows, lacks, and gets wrong."""
    def __init__(self, confidence: ConfidenceMonitor | None = None, errors: ErrorMonitor | None = None) -> None:
        self.confidence_monitor = confidence or ConfidenceMonitor()
        self.error_monitor = errors or ErrorMonitor()
        self.self_model = SelfModel()

    def assess(self, state: CognitiveState) -> MetacognitiveReport:
        beliefs = list(state.beliefs.values())
        return MetacognitiveReport(self.self_model, KnowledgeState.capture(state), tuple(self.confidence_monitor.inspect(beliefs)), self.error_monitor.inspect(state.prediction_errors))
