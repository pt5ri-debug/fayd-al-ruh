from core.uncertainty import Uncertainty


def test_uncertainty_is_complement_of_confidence():
    value = Uncertainty.from_confidence(.73)
    assert value.confidence == .73 and value.uncertainty == .27
