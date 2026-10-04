from core.deductive_reasoner import DeductiveReasoner
from core.inference_rule import InferenceRule


def test_deduction_chains_only_in_forward_direction():
    rules = [InferenceRule(("rain",), "wet"), InferenceRule(("wet",), "slippery")]
    results = DeductiveReasoner().infer({"rain"}, rules)
    assert [result.conclusion for result in results] == ["wet", "slippery"]
