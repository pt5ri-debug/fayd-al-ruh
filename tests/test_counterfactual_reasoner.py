from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.counterfactual_reasoner import CounterfactualReasoner


def test_counterfactual_compares_factual_and_without_cause():
    graph = CausalGraph()
    graph.add_link(CausalLink("rain", "wet", .9))
    result = CounterfactualReasoner().what_if_not(graph, "rain", "wet")
    assert result.factual_confidence == .9 and result.counterfactual_confidence == 0.0
