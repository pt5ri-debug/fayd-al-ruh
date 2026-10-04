from core.self_improvement import SelfImprovement


def test_improvement_plan_requires_human_review():
    assert SelfImprovement().propose([]).requires_human_review
