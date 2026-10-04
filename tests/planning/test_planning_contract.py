from core.action import Action
from core.goal import Goal
from core.planner import Planner
from core.state import State


def test_planning_contract_finds_goal_action():
    assert Planner().create_plan(State(), Goal(frozenset({"done"}), "finish"), [Action("finish", add_effects=frozenset({"done"}))]) is not None
