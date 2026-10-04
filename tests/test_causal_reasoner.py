from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.causal_reasoner import CausalReasoner


def test_reasoner_finds_best_directed_explanation():
    graph = CausalGraph()
    graph.add_link(CausalLink("rain", "wet_ground", 0.9))
    graph.add_link(CausalLink("wet_ground", "slippery", 0.8))
    chain = CausalReasoner().explain(graph, "rain", "slippery")
    assert chain is not None
    assert chain.nodes == ("rain", "wet_ground", "slippery")
    assert chain.confidence == 0.72
    assert CausalReasoner().explain(graph, "slippery", "rain") is None
