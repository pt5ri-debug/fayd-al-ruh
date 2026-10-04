from core.temporal_relation import TemporalRelation


def test_relation_has_before_after_and_overlap():
    assert TemporalRelation.BEFORE.value == "before"
