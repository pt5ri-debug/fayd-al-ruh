from __future__ import annotations

from core.belief import Belief
from core.cognitive_controller import CognitiveController, CognitiveCycleResult
from core.evidence_processor import EvidencePolarity
from core.observation import Observation


class Cognition:
    """Adapter from agent perception to the existing cognitive controller."""
    def process(self, controller: CognitiveController, observation: Observation, belief: Belief, polarity: EvidencePolarity) -> CognitiveCycleResult:
        return controller.process(observation, belief, polarity)
