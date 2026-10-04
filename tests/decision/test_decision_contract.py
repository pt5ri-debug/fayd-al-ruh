from core.action import Action
from core.decision_engine import DecisionEngine
from core.goal import Goal
from core.state import State


def test_decision_contract_selects_action():
    assert DecisionEngine().decide(State(), Goal("finish", frozenset({"done"})), [Action("finish", add_effects=frozenset({"done"}))]).selected is not None
