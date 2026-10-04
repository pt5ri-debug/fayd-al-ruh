from core.action import Action
from core.decision import Decision
from core.risk import Risk
from core.utility import Utility


def test_decision_binds_action_utility_and_risk():
    assert Decision(Action("observe"), Utility(.5, .5, .1), Risk(.1, ()), .45).action.name == "observe"
