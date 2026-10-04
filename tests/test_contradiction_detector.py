from core.contradiction_detector import ContradictionDetector
from core.evidence import Evidence


def test_detector_finds_opposing_evidence():
    evidence = [Evidence("rain", "a", True), Evidence("no rain", "b", False)]
    assert len(ContradictionDetector().detect(evidence)) == 1
