from core.action import Action
from core.action_executor import ActionExecutor
from core.environment import Environment


def test_executor_returns_resulting_symbolic_state():
    result = ActionExecutor().execute(Environment(), Action("see", add_effects=frozenset({"seen"})))
    assert "seen" in result.resulting_state.facts
