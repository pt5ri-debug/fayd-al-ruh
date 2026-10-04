from core.goal import Goal
from core.plan import Plan
from core.plan_evaluator import PlanEvaluator
from core.state import State


def test_evaluator_reports_prediction_error():
    goal = Goal("dry", frozenset({"dry"}))
    result = PlanEvaluator().evaluate(Plan((), State(frozenset({"dry"}))), goal, State(frozenset({"wet"})))
    assert result.goal_achieved is False and result.prediction_error == 1.0
