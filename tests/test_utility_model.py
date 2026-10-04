from core.action import Action
from core.goal import Goal
from core.utility_model import UtilityModel


def test_utility_rewards_goal_progress():
    value = UtilityModel().evaluate(Action("grow", add_effects=frozenset({"plant"})), Goal(frozenset({"plant"}), "grow"))
    assert value.benefit == 1.0
