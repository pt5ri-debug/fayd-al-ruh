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


def test_task_rejects_duplicate_action_names():
    from core.sandbox_environment import SandboxTask
    from core.symbolic_simulation import ActionSpec

    with pytest.raises(ValueError):
        SandboxTask("bad", "duplicate", (ActionSpec("a"), ActionSpec("a")))
