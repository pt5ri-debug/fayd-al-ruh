from core.inference_result import InferenceResult
from core.reasoning_explainer import ReasoningExplainer
from core.reasoning_trace import ReasoningTrace


def test_reasoning_explainer_shows_premise_to_conclusion():
    text = ReasoningExplainer().explain(ReasoningTrace([InferenceResult("wet", .9, "deductive", ("rain",))]))[0]
    assert "rain → wet" in text
