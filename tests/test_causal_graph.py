from core.causal_graph import CausalGraph
from core.causal_link import CausalLink


def test_causal_graph_queries_preserve_direction():
    graph = CausalGraph()
    link = CausalLink("rain", "wet_ground")
    graph.add_link(link)
    assert graph.effects_of("rain") == [link]
    assert graph.causes_of("wet_ground") == [link]
    assert graph.effects_of("wet_ground") == []
