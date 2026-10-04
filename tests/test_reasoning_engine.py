from core.inference import Inference
from core.inference_rule import InferenceRule
from core.reasoning_engine import ReasoningEngine


def test_engine_coordinates_deduction():
    trace = ReasoningEngine().deduce(Inference(("rain",)), [InferenceRule(("rain",), "wet")])
    assert trace.conclusions == ("wet",)
