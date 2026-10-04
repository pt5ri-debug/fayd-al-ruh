from core.experiment import Experiment
from core.experiment_design import ExperimentDesign
from core.hypothesis import Hypothesis
from core.prediction import Prediction


def test_experiment_connects_hypothesis_and_prediction():
    hypothesis = Hypothesis("rain", .7, "pattern")
    experiment = Experiment(hypothesis, Prediction("rain", .7, hypothesis.id), ExperimentDesign("test", "x", "do x"))
    assert experiment.prediction.hypothesis_id == hypothesis.id
