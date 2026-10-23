from core.action import Action
from core.environment import Environment
from core.state import State


def test_environment_applies_pure_symbolic_action():
    environment = Environment(State())
    environment.apply(Action("see", add_effects=frozenset({"seen"})))
    assert "seen" in environment.state.facts
