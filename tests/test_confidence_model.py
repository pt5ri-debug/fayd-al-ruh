from core.confidence_model import ConfidenceModel
from core.evidence import Evidence


def test_supporting_evidence_increases_confidence_and_reduces_uncertainty():
    result = ConfidenceModel().update(.5, [Evidence("rain", "sensor", strength=1., data={"source_reliability": 1.})])
    assert result.confidence == .75 and result.uncertainty == .25
