from core.experiment import Experiment
from core.experiment_design import ExperimentDesign
from core.experiment_runner import ExperimentRunner
from core.hypothesis import Hypothesis
from core.observation import Observation
from core.prediction import Prediction


def test_runner_completes_hypothesis_to_prediction_error_cycle():
    hypothesis = Hypothesis("rain", .8, "pattern")
    experiment = Experiment(hypothesis, Prediction("rain", .8, hypothesis.id), ExperimentDesign("test", "x", "do", 2))
    result = ExperimentRunner().run(experiment, lambda design, trial: Observation("result", "sim", data={"result": 1.}))
    assert result.prediction_error.error == .2
