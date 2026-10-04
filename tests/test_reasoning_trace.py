from core.inference_result import InferenceResult
from core.reasoning_trace import ReasoningTrace


def test_trace_preserves_visible_conclusions():
    trace = ReasoningTrace()
    trace.add(InferenceResult("wet", .9, "deductive", ("rain",)))
    assert trace.conclusions == ("wet",)
