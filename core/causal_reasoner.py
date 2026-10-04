from __future__ import annotations

from core.causal_chain import CausalChain
from core.causal_effect import CausalEffect
from core.causal_explainer import CausalExplainer
from core.causal_graph import CausalGraph
from core.counterfactual import Counterfactual


class CausalReasoner:
    """Finds directed causal explanations; it never infers reverse causation."""

    def __init__(self) -> None:
        self.explainer = CausalExplainer()

    def explain(self, graph: CausalGraph, cause: str, effect: str, max_depth: int = 4) -> CausalChain | None:
        if max_depth < 1:
            raise ValueError("Maximum depth must be at least 1.")
        candidates: list[CausalChain] = []

        def visit(node: str, links: tuple, visited: set[str]) -> None:
            if len(links) >= max_depth:
                return
            for link in graph.effects_of(node):
                if link.effect in visited:
                    continue
                next_links = links + (link,)
                if link.effect == effect:
                    candidates.append(CausalChain(next_links))
                else:
                    visit(link.effect, next_links, visited | {link.effect})

        visit(cause, (), {cause})
        if not candidates:
            return None
        return max(candidates, key=lambda chain: (chain.confidence, -len(chain.links)))

    def effects(self, graph: CausalGraph, cause: str) -> list[str]:
        return [link.effect for link in graph.effects_of(cause)]

    def explain_chain(self, chain: CausalChain) -> str:
        """Render a human-readable explanation of a discovered causal chain."""
        return self.explainer.explain_chain(chain)

    def explain_effect(self, effect: CausalEffect) -> str:
        """Render a human-readable explanation of a measured causal effect."""
        return self.explainer.explain_effect(effect)

    def explain_counterfactual(self, result: Counterfactual) -> str:
        """Render a human-readable explanation of a counterfactual simulation."""
        return self.explainer.explain_counterfactual(result)
