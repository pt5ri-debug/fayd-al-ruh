from core.evidence import Evidence
from core.evidence_trace import EvidenceTrace


def test_evidence_trace_separates_support_and_contradiction():
    trace = EvidenceTrace.from_evidence([Evidence("yes", "a", True), Evidence("no", "b", False)])
    assert len(trace.supporting_ids) == 1 and len(trace.contradicting_ids) == 1
