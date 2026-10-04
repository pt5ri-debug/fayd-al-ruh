from __future__ import annotations

from dataclasses import dataclass, field

from core.belief import Belief
from core.hypothesis import Hypothesis
from core.pattern_detector import Pattern
from core.prediction import Prediction
from core.prediction_error import PredictionError
from core.cognitive_trace import CognitiveTrace


@dataclass
class CognitiveState:
    """
    Central snapshot of the system's current cognitive state.
    """

    beliefs: dict[str, Belief] = field(
        default_factory=dict
    )

    patterns: dict[str, Pattern] = field(
        default_factory=dict
    )

    hypotheses: dict[str, Hypothesis] = field(
        default_factory=dict
    )

    predictions: dict[str, Prediction] = field(
        default_factory=dict
    )

    prediction_errors: list[PredictionError] = field(
        default_factory=list
    )

    traces: dict[str, CognitiveTrace] = field(
        default_factory=dict
    )

    cycle_count: int = 0

    def add_belief(
        self,
        belief: Belief,
    ) -> None:

        self.beliefs[belief.id] = belief

    def add_pattern(
        self,
        pattern: Pattern,
    ) -> None:

        self.patterns[pattern.key] = pattern

    def add_hypothesis(
        self,
        hypothesis: Hypothesis,
    ) -> None:

        self.hypotheses[hypothesis.id] = hypothesis

    def add_prediction(
        self,
        prediction: Prediction,
    ) -> None:

        self.predictions[prediction.id] = prediction

    def add_prediction_error(
        self,
        error: PredictionError,
    ) -> None:

        self.prediction_errors.append(error)

    def next_cycle(self) -> None:
        self.cycle_count += 1

    def add_trace(
        self,
        trace: CognitiveTrace,
    ) -> None:
        self.traces[trace.id] = trace
    

    @property
    def belief_count(self) -> int:
        return len(self.beliefs)

    @property
    def pattern_count(self) -> int:
        return len(self.patterns)

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)

    @property
    def prediction_count(self) -> int:
        return len(self.predictions)

    @property
    def error_count(self) -> int:
        return len(self.prediction_errors)

    @property
    def trace_count(self) -> int:
        return len(self.traces)