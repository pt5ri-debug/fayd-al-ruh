from __future__ import annotations

from collections.abc import Callable

from core.experiment_design import ExperimentDesign
from core.observation import Observation


class ObservationCollector:
    """Collects observations from an injected source; it owns no sensors or tools."""
    def collect(self, design: ExperimentDesign, source: Callable[[ExperimentDesign, int], Observation]) -> list[Observation]:
        observations = [source(design, trial) for trial in range(design.trials)]
        if any(not isinstance(observation, Observation) for observation in observations):
            raise TypeError("Experiment source must return Observation instances.")
        return observations
