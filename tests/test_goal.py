from core.goal import Goal


def test_planning_goal_has_desired_facts():
    assert Goal(frozenset({"dry"}), "make ground dry").desired_facts == frozenset({"dry"})
