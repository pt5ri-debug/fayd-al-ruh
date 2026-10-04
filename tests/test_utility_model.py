from core.action import Action
from core.goal import Goal
from core.utility_model import UtilityModel


def test_utility_rewards_goal_progress():
    value = UtilityModel().evaluate(Action("grow", add_effects=frozenset({"plant"})), Goal("grow", frozenset({"plant"})))
    assert value.benefit == 1.0
