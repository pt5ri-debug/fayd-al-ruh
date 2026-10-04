from core.action import Action
from core.symbolic_simulation import SymbolicSimulation


def _water() -> Action:
    return Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))


def test_simulation_records_deterministic_trace():
    simulation = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=4)
    result = simulation.run([_water()])

    assert result.final_state.facts == frozenset({"dry_soil", "wet_soil"})
    assert result.trace.step_count == 1
    assert result.trace.terminated == "completed"
    assert result.trace.steps[0].action.name == "water"


def test_simulation_is_replayable_from_serialized_trace():
    simulation = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=4)
    result = simulation.run([_water()])

    payload = result.trace.to_json()
    restored = type(result.trace).from_json(payload)
    replayed = SymbolicSimulation.replay(restored)

    assert replayed.final_state.facts == result.final_state.facts
    assert restored == result.trace


def test_simulation_replay_matches_across_runs():
    actions = [_water()]
    first = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=4).run(actions)
    second = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=4).run(actions)

    assert first.trace.to_json() == second.trace.to_json()


def test_simulation_enforces_step_limit():
    simulation = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=1)
    result = simulation.run([_water(), _water()])

    assert result.trace.terminated == "step_limit"
    assert result.trace.step_count == 1


def test_simulation_records_precondition_violation():
    simulation = SymbolicSimulation(frozenset(), max_steps=4)
    result = simulation.run([_water()])

    assert result.trace.step_count == 0
    assert result.violations


def test_simulation_rejects_step_after_termination():
    simulation = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=1)
    simulation.run([_water(), _water()])

    try:
        simulation.apply(_water())
    except RuntimeError as error:
        assert "terminated" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")
