from core.belief import Belief
from core.curiosity import Curiosity


def test_curiosity_discovers_low_confidence_beliefs():
    gaps = Curiosity().discover([Belief("uncertain", .3), Belief("known", .9)])
    assert [gap.topic for gap in gaps] == ["uncertain"]
