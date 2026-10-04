from core.action import Action
from core.goal import Goal
from core.planner import Planner
from core.state import State


def test_planner_finds_multi_step_plan():
    actions = [Action("collect", add_effects=frozenset({"water"})), Action("grow", frozenset({"water"}), frozenset({"plant"}))]
    plan = Planner().create_plan(State(), Goal("grow plant", frozenset({"plant"})), actions)
    assert plan is not None and [action.name for action in plan.actions] == ["collect", "grow"]
