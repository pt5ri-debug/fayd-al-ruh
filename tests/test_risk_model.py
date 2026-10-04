from core.action import Action
from core.goal import Goal
from core.risk_model import RiskModel
from core.state import State


def test_risk_detects_missing_preconditions():
    risk = RiskModel().evaluate(State(), Action("grow", frozenset({"water"})), Goal(frozenset({"plant"}), "grow"))
    assert risk.value > 0 and risk.reasons
