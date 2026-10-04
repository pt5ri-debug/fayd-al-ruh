from __future__ import annotations

from collections.abc import Callable

from core.experiment import Experiment
from core.experiment_result import ExperimentResult
from core.hypothesis_test import HypothesisTester
from core.observation import Observation
from core.observation_collector import ObservationCollector


class ExperimentRunner:
    """Runs a supplied pure observation source and produces a testable result."""
    def __init__(self, collector: ObservationCollector | None = None, tester: HypothesisTester | None = None) -> None:
        self.collector = collector or ObservationCollector()
        self.tester = tester or HypothesisTester()

    def run(self, experiment: Experiment, source: Callable[[object, int], Observation]) -> ExperimentResult:
        observations = self.collector.collect(experiment.design, source)
        values = [observation.data.get("result", observation.confidence) for observation in observations]
        if any(not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Observation result values must be between 0 and 1.")
        observed_result = sum(values) / len(values)
        test = self.tester.test(experiment.prediction, observed_result)
        experiment.prediction.resolve(test.supports_hypothesis)
        return ExperimentResult(experiment.id, tuple(observations), test)
