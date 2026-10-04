from core.inference import Inference
from core.inference_rule import InferenceRule
from core.reasoning_engine import ReasoningEngine


def test_reasoning_contract():
    assert ReasoningEngine().deduce(Inference(("rain",)), [InferenceRule(("rain",), "wet")]).conclusions == ("wet",)
