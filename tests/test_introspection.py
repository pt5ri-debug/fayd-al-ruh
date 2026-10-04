from core.belief import Belief
from core.evidence import Evidence
from core.introspection import Introspection


def test_introspection_explains_belief_from_evidence_ids():
    evidence = Evidence("rain", "sensor"); belief = Belief("rain"); belief.add_evidence(evidence)
    assert Introspection().explain_belief(belief, [evidence]).supporting_evidence_ids == (evidence.id,)
