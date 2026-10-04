from core.action import Action
from core.decision import Decision
from core.decision_explainer import DecisionExplainer
from core.decision_trace import DecisionTrace
from core.risk import Risk
from core.utility import Utility


def test_decision_explainer_shows_selected_action():
    decision = Decision(Action("observe"), Utility(.8, .8, .1), Risk(.1, ()), .75)
    assert "Selected: observe" in DecisionExplainer().explain(DecisionTrace((decision,), decision, "best"))[0]
