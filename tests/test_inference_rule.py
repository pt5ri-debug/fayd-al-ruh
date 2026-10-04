import pytest
from core.inference_rule import InferenceRule


def test_rule_requires_premises_and_conclusion():
    rule = InferenceRule(("rain",), "wet_ground", 0.9)
    assert rule.conclusion == "wet_ground"
    with pytest.raises(ValueError):
        InferenceRule((), "wet_ground")
