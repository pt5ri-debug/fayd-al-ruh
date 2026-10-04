from core.causal_effect import CausalEffect
from core.causal_explainer import CausalExplainer


def test_explainer_uses_interventional_effect_size():
    assert "+0.60" in CausalExplainer().explain_effect(CausalEffect("rain", "wet", .8, .2, .6))
