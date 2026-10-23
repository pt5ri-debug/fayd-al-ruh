from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.counterfactual_reasoner import CounterfactualReasoner


def test_causal_counterfactual_contract():
    graph = CausalGraph()
    graph.add_link(CausalLink("rain", "wet", .9))
    assert CounterfactualReasoner().what_if_not(graph, "rain", "wet").changed
