from core.uncertainty_propagator import UncertaintyPropagator


def test_propagator_combines_dependent_confidences():
    value = UncertaintyPropagator().propagate([.9, .8])
    assert value.confidence == .72 and value.uncertainty == .28
