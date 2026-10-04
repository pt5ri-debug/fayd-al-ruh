from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class CognitiveTrace:
    """
    Records the causal chain of a cognitive process.
    """

    observation_id: str

    id: str = field(
        default_factory=lambda: f"TRACE-{uuid4().hex[:8]}"
    )

    pattern_keys: list[str] = field(
        default_factory=list
    )

    hypothesis_ids: list[str] = field(
        default_factory=list
    )

    prediction_ids: list[str] = field(
        default_factory=list
    )

    prediction_error_ids: list[str] = field(
        default_factory=list
    )

    belief_id: str | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def add_pattern(
        self,
        pattern_key: str,
    ) -> None:

        if pattern_key not in self.pattern_keys:
            self.pattern_keys.append(pattern_key)

    def add_hypothesis(
        self,
        hypothesis_id: str,
    ) -> None:

        if hypothesis_id not in self.hypothesis_ids:
            self.hypothesis_ids.append(hypothesis_id)

    def add_prediction(
        self,
        prediction_id: str,
    ) -> None:

        if prediction_id not in self.prediction_ids:
            self.prediction_ids.append(prediction_id)

    def add_prediction_error(
        self,
        error_id: str,
    ) -> None:

        if error_id not in self.prediction_error_ids:
            self.prediction_error_ids.append(error_id)

    def set_belief(
        self,
        belief_id: str,
    ) -> None:

        self.belief_id = belief_id