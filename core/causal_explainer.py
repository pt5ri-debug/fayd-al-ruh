from __future__ import annotations

from core.causal_chain import CausalChain
from core.causal_effect import CausalEffect
from core.composition import (
    causal_chain_sentence,
    causal_effect_sentence,
    counterfactual_sentence,
)
from core.counterfactual import Counterfactual


class CausalExplainer:
    def explain_chain(self, chain: CausalChain) -> str:
        mechanisms = [link.mechanism for link in chain.links if link.mechanism]
        return causal_chain_sentence(
            chain.cause,
            chain.effect,
            chain.nodes,
            chain.confidence,
            mechanisms,
        )

    def explain_effect(self, effect: CausalEffect) -> str:
        return causal_effect_sentence(effect.cause, effect.effect, effect.effect_size)

    def explain_counterfactual(self, result: Counterfactual) -> str:
        return counterfactual_sentence(
            result.cause,
            result.effect,
            result.factual_confidence,
            result.counterfactual_confidence,
        )
