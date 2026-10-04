from __future__ import annotations

from core.abductive_reasoner import AbductiveReasoner
from core.deductive_reasoner import DeductiveReasoner
from core.inductive_reasoner import InductiveReasoner
from core.inference import Inference
from core.inference_rule import InferenceRule
from core.reasoning_monitor import ReasoningMonitor, ReasoningSummary
from core.reasoning_trace import ReasoningTrace
from core.uncertainty import Uncertainty
from core.uncertainty_propagator import UncertaintyPropagator


class ReasoningEngine:
    """Coordinates explicit deductive, inductive, and abductive reasoning."""
    def __init__(self) -> None:
        self.deductive = DeductiveReasoner()
        self.inductive = InductiveReasoner()
        self.abductive = AbductiveReasoner()
        self.monitor = ReasoningMonitor()
        self.uncertainty = UncertaintyPropagator()

    def deduce(self, inference: Inference, rules: list[InferenceRule]) -> ReasoningTrace:
        trace = ReasoningTrace()
        for result in self.deductive.infer(inference.facts, rules):
            trace.add(result)
        return trace

    def induce(self, observations: list[str]) -> ReasoningTrace:
        trace = ReasoningTrace()
        for result in self.inductive.infer(observations):
            trace.add(result)
        return trace

    def abduce(self, observation: str, rules: list[InferenceRule]) -> ReasoningTrace:
        trace = ReasoningTrace()
        for result in self.abductive.explain(observation, rules):
            trace.add(result)
        return trace

    def summarize(self, trace: ReasoningTrace) -> ReasoningSummary:
        """Summarize a completed reasoning trace without exposing hidden steps."""
        return self.monitor.inspect(trace)

    def propagate_uncertainty(self, confidences: list[float]) -> Uncertainty:
        """Combine chained confidences conservatively for a derived conclusion."""
        return self.uncertainty.propagate(confidences)
