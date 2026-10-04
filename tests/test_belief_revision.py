from core.belief import Belief, BeliefStatus
from core.belief_revision import BeliefRevisionEngine
from core.evidence import Evidence


def test_revision_suspends_belief_when_evidence_conflicts():
    belief = Belief("rain", .5)
    result = BeliefRevisionEngine().revise(belief, [Evidence("yes", "a", True), Evidence("no", "b", False)])
    assert result.contradiction_count == 1 and belief.status is BeliefStatus.SUSPENDED
