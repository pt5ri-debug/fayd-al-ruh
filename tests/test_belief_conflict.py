import pytest
from core.belief import Belief
from core.belief_conflict import BeliefConflict


def test_conflict_requires_two_beliefs():
    belief = Belief("rain")
    with pytest.raises(ValueError):
        BeliefConflict(belief, belief, .5)
