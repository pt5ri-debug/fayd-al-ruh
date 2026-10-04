from core.action import Action
from core.decision_engine import DecisionEngine
from core.goal import Goal
from core.state import State


def test_engine_selects_feasible_high_utility_low_risk_action():
    goal = Goal("grow", frozenset({"plant"}))
    actions = [Action("grow", add_effects=frozenset({"plant"})), Action("blocked", frozenset({"water"}), frozenset({"plant"}))]
    trace = DecisionEngine().decide(State(), goal, actions)
    assert trace.selected is not None and trace.selected.action.name == "grow"
