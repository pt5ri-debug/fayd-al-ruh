from __future__ import annotations

from core.causal_chain import CausalChain
from core.causal_effect import CausalEffect
from core.counterfactual import Counterfactual


class CausalExplainer:
    def explain_chain(self, chain: CausalChain) -> str:
        mechanisms = [link.mechanism for link in chain.links if link.mechanism]
        explanation = f"{chain.cause} may lead to {chain.effect} through {' → '.join(chain.nodes)} (confidence {chain.confidence:.2f})."
        return explanation + (f" Mechanism: {'; '.join(mechanisms)}." if mechanisms else "")

    def explain_effect(self, effect: CausalEffect) -> str:
        return f"Intervening on {effect.cause} changed {effect.effect} by {effect.effect_size:+.2f} compared with control."

    def explain_counterfactual(self, result: Counterfactual) -> str:
        return f"With {result.cause}, {result.effect} had confidence {result.factual_confidence:.2f}; without it, the simulated confidence is {result.counterfactual_confidence:.2f}."
