from core.causal_discovery import CausalDiscovery


def test_discovery_proposes_repeated_ordered_transition_as_candidate():
    links = CausalDiscovery().discover(["rain", "wet", "rain", "wet"], 2)
    assert links[0].cause == "rain" and links[0].effect == "wet"
    assert links[0].metadata["status"] == "candidate"
