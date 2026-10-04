from core.evidence import Evidence
from core.explanation_engine import ExplanationEngine


def test_engine_composes_evidence_based_summary():
    explanation = ExplanationEngine().explain("wet ground", [Evidence("rain", "sensor")])
    assert "1 supporting" in explanation.summary and explanation.evidence
