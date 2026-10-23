from core.causal_graph import CausalGraph
from core.causal_intervention import CausalIntervention
from core.causal_link import CausalLink


def test_intervention_removes_direct_influence():
    graph = CausalGraph()
    graph.add_link(CausalLink("rain", "wet", .9))
    assert CausalIntervention().predict_without(graph, "rain", "wet").predicted_confidence == 0.0
