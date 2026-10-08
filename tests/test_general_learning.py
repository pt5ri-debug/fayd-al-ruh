"""Tests for general, knowledge-grounded learning (link prediction)."""

from ml.general_learning import (
    RelationLearningReport,
    relation_examples,
    train_relation_outcome,
)


def test_relation_examples_are_distinct_and_nonempty():
    relations = relation_examples()
    assert relations
    keys = {(r.source, r.relation, r.target) for r in relations}
    assert len(keys) == len(relations)


def test_link_prediction_generalises_beyond_random():
    _model, report = train_relation_outcome()
    # Hit@1 ranks the true target against three decoys, so random is 0.25.
    assert report.hit_at_1 > 0.25
    assert report.generalisation_gain > 0.0
    assert report.mean_rank < 2.5


def test_report_serialises_and_is_stable_across_runs():
    _model_a, first = train_relation_outcome()
    _model_b, second = train_relation_outcome()
    assert isinstance(first, RelationLearningReport)
    assert first.hit_at_1 == second.hit_at_1
    data = first.as_dict()
    assert data["relations"] == first.relations
    assert 0.0 <= data["hit_at_1"] <= 1.0
