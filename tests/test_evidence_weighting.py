from core.evidence import Evidence
from core.evidence_weighting import EvidenceWeighting


def test_weighting_includes_source_reliability():
    weighting = EvidenceWeighting({"sensor": .8})
    assert weighting.weight(Evidence("rain", "sensor", strength=.9)) == .72
