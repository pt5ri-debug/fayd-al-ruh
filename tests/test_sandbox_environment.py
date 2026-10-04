import pytest

from core.action import Action
from core.sandbox_environment import SandboxEnvironment
from core.sandbox_tasks import TASKS, task_by_name
from core.symbolic_simulation import SymbolicSimulation


def _success(state) -> bool:
    return "wet_soil" in state.facts


def test_task_catalog_has_unique_names():
    names = [task.name for task in TASKS]
    assert len(names) == len(set(names))
    assert task_by_name("wet-garden").allowed_action_names == frozenset({"water", "wait"})


def test_sandbox_solves_explicit_task():
    environment = SandboxEnvironment(task_by_name("wet-garden"))
    result = environment.run(
        [Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))], _success
    )

    assert result.success
    assert result.terminated == "success"
    assert result.reward == 1.0
    assert result.violations == ()


def test_sandbox_rejects_action_outside_allow_list():
    environment = SandboxEnvironment(task_by_name("wet-garden"))
    result = environment.run(
        [Action("detonate", add_effects=frozenset({"wet_soil"}))], _success
    )

    assert not result.success
    assert result.accepted_steps == 0
    assert "not allowed" in result.violations[0]


def test_sandbox_enforces_step_budget():
    environment = SandboxEnvironment(task_by_name("wet-garden"))
    actions = [Action("wait", frozenset({"dry_soil"}), frozenset()) for _ in range(10)]
    result = environment.run(actions, _success)

    assert result.terminated == "step_limit"
    assert len(result.steps) == task_by_name("wet-garden").max_steps


def test_sandbox_exports_replayable_trace():
    environment = SandboxEnvironment(task_by_name("wet-garden"))
    environment.run([Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))], _success)

    trace = environment.to_trace()
    replayed = SymbolicSimulation.replay(trace)

    assert replayed.final_state.facts == environment.state.facts


_NEW_TASK_SOLUTIONS = {
    "plant-seed": (
        "sprout",
        [
            Action("till", frozenset({"hard_soil"}), frozenset({"tilled_soil"}),
                   frozenset({"hard_soil"})),
            Action("plant", frozenset({"tilled_soil", "seed_available"}),
                   frozenset({"seed_planted"}), frozenset({"seed_available"})),
            Action("water", frozenset({"seed_planted"}), frozenset({"seed_watered"})),
            Action("wait", frozenset({"seed_watered"}), frozenset({"sprout"}),
                   frozenset({"seed_watered"})),
        ],
    ),
    "boil-water": (
        "tea_ready",
        [
            Action("fill", frozenset({"empty_pot", "water_tap"}), frozenset({"pot_filled"}),
                   frozenset({"empty_pot"})),
            Action("heat", frozenset({"pot_filled", "stove_on"}),
                   frozenset({"water_boiling"}), frozenset({"pot_filled"})),
            Action("steep", frozenset({"water_boiling", "tea_available"}),
                   frozenset({"tea_ready"})),
        ],
    ),
    "recharge-lamp": (
        "lamp_on",
        [
            Action("open", frozenset({"panel_closed"}), frozenset({"panel_open"}),
                   frozenset({"panel_closed"})),
            Action("swap", frozenset({"panel_open", "spare_cell"}),
                   frozenset({"cell_replaced"}), frozenset({"spare_cell"})),
            Action("switch", frozenset({"cell_replaced"}), frozenset({"lamp_on"}),
                   frozenset({"cell_replaced"})),
        ],
    ),
}


@pytest.mark.parametrize("task_name", sorted(_NEW_TASK_SOLUTIONS))
def test_new_tasks_have_a_valid_solution_path(task_name):
    goal_fact, plan = _NEW_TASK_SOLUTIONS[task_name]
    task = task_by_name(task_name)
    environment = SandboxEnvironment(task)

    result = environment.run(plan, lambda state: goal_fact in state.facts)

    assert result.success
    assert result.terminated == "success"
    assert result.violations == ()
    assert goal_fact in result.final_state.facts
    assert len(result.steps) <= task.max_steps


@pytest.mark.parametrize("task_name", sorted(_NEW_TASK_SOLUTIONS))
def test_new_tasks_reject_actions_outside_allow_list(task_name):
    task = task_by_name(task_name)
    environment = SandboxEnvironment(task)
    intruder = Action("detonate", add_effects=frozenset({"sprout", "tea_ready", "lamp_on"}))

    result = environment.run([intruder], lambda state: False)

    assert not result.success
    assert result.accepted_steps == 0
    assert "not allowed" in result.violations[0]


def test_task_rejects_duplicate_action_names():
    from core.sandbox_environment import SandboxTask
    from core.symbolic_simulation import ActionSpec

    with pytest.raises(ValueError):
        SandboxTask("bad", "duplicate", (ActionSpec("a"), ActionSpec("a")))
