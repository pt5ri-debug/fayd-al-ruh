import pytest
from core.causal_link import CausalLink


def test_causal_link_has_direction_confidence_and_evidence():
    link = CausalLink("rain", "wet_ground", 0.9, ["EV-1"], "water falls")
    assert (link.cause, link.effect, link.confidence) == ("rain", "wet_ground", 0.9)
    link.add_evidence("EV-1")
    assert link.evidence_ids == ["EV-1"]


def test_self_causation_is_rejected():
    with pytest.raises(ValueError):
        CausalLink("rain", "rain")
