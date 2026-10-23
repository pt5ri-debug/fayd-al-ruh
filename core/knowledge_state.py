from __future__ import annotations

from dataclasses import dataclass

from core.belief import BeliefStatus
from core.cognitive_state import CognitiveState


@dataclass(frozen=True)
class KnowledgeState:
    belief_count: int
    accepted_count: int
    provisional_count: int
    suspended_count: int
    rejected_count: int
    unknown_count: int

    @classmethod
    def capture(cls, state: CognitiveState, required_confidence: float = .8) -> KnowledgeState:
        beliefs = list(state.beliefs.values())
        return cls(
            belief_count=len(beliefs),
            accepted_count=sum(item.status is BeliefStatus.ACCEPTED for item in beliefs),
            provisional_count=sum(item.status is BeliefStatus.PROVISIONAL for item in beliefs),
            suspended_count=sum(item.status is BeliefStatus.SUSPENDED for item in beliefs),
            rejected_count=sum(item.status is BeliefStatus.REJECTED for item in beliefs),
            unknown_count=sum(item.confidence < required_confidence for item in beliefs),
        )
