from core.action import Action
from core.plan import Plan
from core.state import State


def test_plan_simulation_is_pure_state_update():
    action = Action("observe", add_effects=frozenset({"seen"}))
    plan = Plan((action,), State(frozenset({"seen"})))
    assert plan.execute_simulation(State()).facts == frozenset({"seen"})
