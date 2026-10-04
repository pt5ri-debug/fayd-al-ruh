from core.abductive_reasoner import AbductiveReasoner
from core.inference_rule import InferenceRule


def test_abduction_returns_ranked_possible_explanations():
    rules = [InferenceRule(("rain",), "wet", .8), InferenceRule(("sprinkler",), "wet", .6)]
    results = AbductiveReasoner().explain("wet", rules)
    assert results[0].method == "abductive"
    assert "rain" in results[0].conclusion
