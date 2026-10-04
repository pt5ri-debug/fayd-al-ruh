from core.experiment_design import ExperimentDesign


def test_design_requires_positive_trial_count():
    assert ExperimentDesign("rain test", "water", "add water", 2).trials == 2
