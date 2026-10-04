from core.inference_result import InferenceResult
from core.reasoning_monitor import ReasoningMonitor
from core.reasoning_trace import ReasoningTrace


def test_reasoning_monitor_summarises_visible_trace():
    trace = ReasoningTrace([InferenceResult("wet", .9, "deductive", ("rain",))])
    assert ReasoningMonitor().inspect(trace).methods == ("deductive",)
