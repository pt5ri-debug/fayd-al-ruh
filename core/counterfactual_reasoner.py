from __future__ import annotations

from core.causal_graph import CausalGraph
from core.causal_intervention import CausalIntervention
from core.causal_reasoner import CausalReasoner
from core.counterfactual import Counterfactual


class CounterfactualReasoner:
    """Answers 'what if X had not happened?' from the explicit causal graph."""
    def what_if_not(self, graph: CausalGraph, cause: str, effect: str) -> Counterfactual:
        factual = CausalReasoner().explain(graph, cause, effect)
        factual_confidence = factual.confidence if factual is not None else 0.0
        intervention = CausalIntervention().predict_without(graph, cause, effect)
        return Counterfactual(cause, effect, factual_confidence, intervention.predicted_confidence)
