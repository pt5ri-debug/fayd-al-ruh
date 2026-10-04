from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from core.experiment_design import ExperimentDesign
from core.hypothesis import Hypothesis
from core.prediction import Prediction


@dataclass(frozen=True)
class Experiment:
    hypothesis: Hypothesis
    prediction: Prediction
    design: ExperimentDesign
    id: str = field(default_factory=lambda: f"EXP-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if self.prediction.hypothesis_id != self.hypothesis.id:
            raise ValueError("Experiment prediction must belong to its hypothesis.")
