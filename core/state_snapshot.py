from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.cognitive_state import CognitiveState


@dataclass(frozen=True)
class StateSnapshot:
    """
    Immutable snapshot of the cognitive state.
    """

    timestamp: datetime

    cycle_count: int
    belief_count: int
    pattern_count: int
    hypothesis_count: int
    prediction_count: int
    error_count: int

    average_belief_confidence: float

    @classmethod
    def capture(
        cls,
        state: CognitiveState,
    ) -> "StateSnapshot":

        beliefs = list(
            state.beliefs.values()
        )

        if beliefs:

            average_confidence = (
                sum(
                    belief.confidence
                    for belief in beliefs
                )
                / len(beliefs)
            )

        else:

            average_confidence = 0.0

        average_confidence = round(
            average_confidence,
            6,
        )

        return cls(
            timestamp=datetime.now(timezone.utc),
            cycle_count=state.cycle_count,
            belief_count=state.belief_count,
            pattern_count=state.pattern_count,
            hypothesis_count=state.hypothesis_count,
            prediction_count=state.prediction_count,
            error_count=state.error_count,
            average_belief_confidence=average_confidence,
        )