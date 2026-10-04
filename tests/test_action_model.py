from core.action import Action
from core.action_model import ActionModel
from core.state import State


def test_action_model_predicts_transition():
    assert "dry" in ActionModel().predict(State(), Action("dry", add_effects=frozenset({"dry"}))).facts
