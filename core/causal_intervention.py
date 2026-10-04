from __future__ import annotations

from dataclasses import dataclass

from core.causal_effect import CausalEffect
from core.causal_experiment import CausalExperiment
from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.causal_reasoner import CausalReasoner


@dataclass(frozen=True)
class InterventionResult:
    blocked_cause: str
    effect: str
    predicted_confidence: float
    explanation: str


class CausalIntervention:
    """Runs a do-style intervention in a copied causal graph; no real-world action occurs."""
    def predict_without(self, graph: CausalGraph, blocked_cause: str, effect: str) -> InterventionResult:
        simulated = CausalGraph()
        for link in graph.links.values():
            if link.cause != blocked_cause:
                simulated.add_link(CausalLink(link.cause, link.effect, link.confidence, list(link.evidence_ids), link.mechanism, dict(link.metadata), link.id, link.created_at, link.active))
        best = max((CausalReasoner().explain(simulated, source, effect) for source in {link.cause for link in simulated.links.values()}), key=lambda chain: chain.confidence if chain is not None else -1, default=None)
        confidence = best.confidence if best is not None else 0.0
        return InterventionResult(blocked_cause, effect, confidence, "Effect estimated after removing outgoing influence of the intervention cause.")

    def estimate_effect(self, experiment: CausalExperiment) -> CausalEffect:
        """Estimate a causal effect from a controlled experiment's outcomes."""
        return experiment.estimate_effect()
