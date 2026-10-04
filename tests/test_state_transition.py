from core.action import Action
from core.state import State
from core.state_transition import StateTransition


def test_transition_applies_declared_effects():
    state = State(frozenset({"wet"}))
    action = Action("dry", frozenset({"wet"}), frozenset({"dry"}), frozenset({"wet"}))
    assert StateTransition().apply(state, action).facts == frozenset({"dry"})
