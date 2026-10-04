from __future__ import annotations

from dataclasses import dataclass

from core.hypothesis_test import HypothesisTest
from core.observation import Observation


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    observations: tuple[Observation, ...]
    test: HypothesisTest

    @property
    def prediction_error(self):
        return self.test.prediction_error
