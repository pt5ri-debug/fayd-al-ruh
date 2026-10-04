from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.causal_validator import CausalValidator


def test_validator_reports_missing_evidence():
    graph = CausalGraph()
    graph.add_link(CausalLink("A", "B", evidence_ids=["EV-missing"]))
    result = CausalValidator().validate_graph(graph, evidence_ids={"EV-known"})
    assert result.valid is False
    assert "EV-missing" in result.errors[0]
