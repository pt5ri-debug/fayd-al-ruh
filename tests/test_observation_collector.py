from core.experiment_design import ExperimentDesign
from core.observation import Observation
from core.observation_collector import ObservationCollector


def test_collector_requests_each_trial_from_injected_source():
    observations = ObservationCollector().collect(ExperimentDesign("test", "x", "do", 2), lambda design, trial: Observation(str(trial), "test"))
    assert len(observations) == 2
