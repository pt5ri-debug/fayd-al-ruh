from core.belief import Belief, BeliefStatus
from core.belief_conflict import BeliefConflict
from core.conflict_resolver import ConflictResolver


def test_resolver_suspends_weaker_conflicting_belief():
    strong, weak = Belief("rain", .9), Belief("not rain", .5)
    result = ConflictResolver().resolve(BeliefConflict(strong, weak, .5))
    assert result.resolved_belief_id == strong.id
    assert weak.status is BeliefStatus.SUSPENDED
