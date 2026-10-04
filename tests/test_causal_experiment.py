from core.causal_experiment import CausalExperiment


def test_experiment_estimates_treated_minus_control_effect():
    effect = CausalExperiment("rain", "wet", (.9, .8), (.2, .3)).estimate_effect()
    assert effect.effect_size == .6
