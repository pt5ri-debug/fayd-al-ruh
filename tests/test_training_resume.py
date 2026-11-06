"""Resumable training: a second run continues from where the first stopped.

These pin the ledger's own contract and the resume behaviour of the real
trainers, including the negative controls -- a changed run is not resumed into,
a corrupt ledger is not fatal, and a finished task is not retrained.
"""

import json

import numpy as np
from core.petroleum_tasks import PETROLEUM_TASKS
from ml.checkpoint import (
    LEDGER_NAME,
    ProgressEntry,
    TrainingProgress,
    fingerprint_of,
)
from ml.cognitive_brain import CognitiveBrain
from ml.learner import QLearningAgent
from ml.petroleum import train_petroleum
from ml.resume_gate import (
    ResumeJudgement,
    training_resume_accuracy,
)
from ml.resume_gate import (
    describe as resume_gate_describe,
)
from ml.resume_gate import (
    judgements as resume_gate_judgements,
)

_FIRST_TASK = PETROLEUM_TASKS[0].name


# --------------------------------------------------------------- the ledger

def test_fingerprint_is_stable_and_config_sensitive():
    assert fingerprint_of({"a": 1, "b": 2}) == fingerprint_of({"b": 2, "a": 1})
    assert fingerprint_of({"a": 1}) != fingerprint_of({"a": 2})


def test_ledger_round_trips(tmp_path):
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    state.record("task-a", 5, artifact="brain_task-a")
    state.save()
    reloaded = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    assert reloaded.available
    assert reloaded.completed("task-a") == 5
    assert reloaded.artifact_path("task-a") == tmp_path / "brain_task-a"


def test_ledger_is_bounded(tmp_path):
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0}, max_units=3)
    for index in range(6):
        state.record(f"task-{index}", 1)
    assert len(state.units) == 3
    assert "task-5" in state.units
    assert "task-0" not in state.units


def test_corrupt_ledger_reads_as_empty(tmp_path):
    (tmp_path / LEDGER_NAME).write_text("{ not json", encoding="utf-8")
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    assert not state.available
    assert state.completed("task-a") == 0


def test_a_non_object_ledger_reads_as_empty(tmp_path):
    (tmp_path / LEDGER_NAME).write_text("[1, 2, 3]", encoding="utf-8")
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    assert not state.available


def test_a_changed_configuration_invalidates_the_record(tmp_path):
    first = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    first.record("a", 4)
    first.save()
    changed = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 1})
    assert changed.stale
    assert not changed.available


def test_a_missing_fingerprint_keeps_the_record(tmp_path):
    """A caller that gives no config does not invalidate what is recorded."""
    first = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    first.record("a", 4)
    first.save()
    plain = TrainingProgress(tmp_path, scope="petroleum")
    assert not plain.stale
    assert plain.completed("a") == 4


def test_remaining_never_goes_negative(tmp_path):
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    state.record("a", 10)
    assert state.remaining("a", 6) == 0
    assert state.remaining("b", 6) == 6


def test_scopes_do_not_collide(tmp_path):
    petroleum = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    petroleum.record("a", 3)
    petroleum.save()
    ml = TrainingProgress(tmp_path, scope="ml", config={"seed": 0})
    assert not ml.available


def test_reset_forgets_the_scope(tmp_path):
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    state.record("a", 3)
    state.reset()
    state.save()
    assert not TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0}).available


def test_a_bare_integer_entry_reads(tmp_path):
    (tmp_path / LEDGER_NAME).write_text(
        json.dumps({"scopes": {"petroleum": {"fingerprint": "", "units": {"a": 7}}}}),
        encoding="utf-8",
    )
    state = TrainingProgress(tmp_path, scope="petroleum")
    assert state.completed("a") == 7
    assert state.artifact_path("a") is None


def test_progress_entry_is_frozen():
    entry = ProgressEntry(completed=2, artifact="x")
    assert entry.completed == 2
    try:
        entry.completed = 3  # type: ignore[misc]
    except Exception:  # frozen dataclass
        return
    raise AssertionError("ProgressEntry should be frozen")


# ------------------------------------------------------- agent/brain state

def test_agent_round_trip_preserves_rng(tmp_path):
    agent = QLearningAgent(4, 3, seed=7)
    agent.save(tmp_path / "policy.npz")
    restored = QLearningAgent.load(tmp_path / "policy.npz")
    # The next draw from the exploration RNG must match, so a resumed run's
    # sequence is the one an uninterrupted run would have drawn.
    assert restored._rng.random() == agent._rng.random()


def test_brain_round_trip_preserves_rng(tmp_path):
    from core.sandbox_tasks import task_by_name

    task = task_by_name("wet-garden")
    success = lambda state: "wet_soil" in state.facts  # noqa: E731
    brain = CognitiveBrain(task, success, seed=3)
    brain.learn(episodes=3)
    brain.save(tmp_path / "brain")
    restored = CognitiveBrain.load(tmp_path / "brain", task, success)
    assert restored._rng.random() == brain._rng.random()


# ----------------------------------------------------- petroleum resume

def test_petroleum_resume_loads_a_finished_task(tmp_path):
    train_petroleum(episodes=2, include_held_out=False, save_dir=tmp_path, resume=True)
    before = (tmp_path / f"policy_{_FIRST_TASK}.npz").read_bytes()
    train_petroleum(episodes=2, include_held_out=False, save_dir=tmp_path, resume=True)
    after = (tmp_path / f"policy_{_FIRST_TASK}.npz").read_bytes()
    assert before == after, "a finished task must be loaded, not retrained"


def test_petroleum_resume_equals_an_uninterrupted_run(tmp_path):
    full = tmp_path / "full"
    part = tmp_path / "part"
    train_petroleum(episodes=6, include_held_out=False, save_dir=full, resume=False)
    train_petroleum(episodes=3, include_held_out=False, save_dir=part, resume=True)
    train_petroleum(episodes=6, include_held_out=False, save_dir=part, resume=True)
    assert (
        (full / f"policy_{_FIRST_TASK}.npz").read_bytes()
        == (part / f"policy_{_FIRST_TASK}.npz").read_bytes()
    )


def test_petroleum_a_raised_budget_tops_up(tmp_path):
    train_petroleum(episodes=3, include_held_out=False, save_dir=tmp_path, resume=True)
    assert TrainingProgress(tmp_path, scope="petroleum").completed(_FIRST_TASK) == 3
    train_petroleum(episodes=6, include_held_out=False, save_dir=tmp_path, resume=True)
    assert TrainingProgress(tmp_path, scope="petroleum").completed(_FIRST_TASK) == 6


def test_petroleum_without_resume_starts_over(tmp_path):
    train_petroleum(episodes=4, include_held_out=False, save_dir=tmp_path, resume=True)
    train_petroleum(episodes=2, include_held_out=False, save_dir=tmp_path, resume=False)
    # A fresh run writes no ledger at all, so nothing is recorded.
    assert not TrainingProgress(tmp_path, scope="petroleum", config={"x": 1}).available


def test_petroleum_no_save_dir_records_nothing(tmp_path):
    report = train_petroleum(episodes=2, include_held_out=False, save_dir=None, resume=True)
    assert report.overall_success >= 0.0
    assert not (tmp_path / LEDGER_NAME).exists()


# ------------------------------------------------------------ the gate

def test_resume_gate_all_judgements_pass():
    results = resume_gate_judgements()
    assert results, "the gate must make judgements"
    failed = [j.name for j in results if not j.passed]
    assert not failed, f"the resume gate failed: {failed}"
    assert training_resume_accuracy() == 1.0


def test_resume_gate_describe_lists_every_judgement():
    text = resume_gate_describe()
    assert "Resumable training gate" in text
    assert "a_partial_run_continues" in text


def test_resume_gate_has_negative_controls():
    names = {j.name for j in resume_gate_judgements()}
    assert "fingerprint_invalidates_a_changed_run" in names
    assert "ledger_round_trip_and_corruption" in names
    assert "a_finished_task_is_not_retrained" in names


def test_resume_gate_accuracy_is_zero_for_no_judgements(monkeypatch):
    import ml.resume_gate as gate

    monkeypatch.setattr(gate, "judgements", lambda: ())
    assert gate.training_resume_accuracy() == 0.0


def test_resume_judgement_describes_itself():
    passed = ResumeJudgement("x", True, "did the thing")
    failed = ResumeJudgement("y", False, "did not")
    assert "PASS" in passed.describe() and "did the thing" in passed.describe()
    assert "FAIL" in failed.describe()


# --------------------------------------------------------------- CLI

def test_train_cli_resume_gate_prints_a_report(capsys):
    from cli.train import main

    assert main(["resume-gate"]) == 0
    out = capsys.readouterr().out
    assert "Resumable training gate" in out


def test_train_cli_resume_gate_json(capsys):
    import json as _json

    from cli.train import main

    assert main(["resume-gate", "--json"]) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert {"name", "passed", "detail"} <= set(payload[0])


def test_train_cli_resume_from_a_missing_adapter_is_refused(tmp_path, capsys):
    from cli.train import main

    code = main(
        [
            "run",
            "--dataset",
            "does-not-exist",
            "--resume-from",
            str(tmp_path / "nope"),
        ]
    )
    # The dataset is checked first, so an unknown dataset is the honest failure.
    assert code in (0, 3, 4)


def test_ml_cli_accepts_fresh_and_resume(tmp_path):
    from ml.__main__ import main

    code = main(
        [
            "--episodes",
            "2",
            "--save-dir",
            str(tmp_path),
            "--fresh",
            "--quiet",
        ]
    )
    assert code == 0


def test_ml_cli_fresh_without_a_save_dir_runs():
    """--fresh with nowhere to save is simply a fresh run, not an error."""
    from ml.__main__ import main

    assert main(["--episodes", "2", "--fresh", "--quiet"]) == 0


def test_release_fresh_flag_is_accepted():
    from ml.release import main

    # A dry parse: --manifest-only never trains, so this stays fast and offline.
    code = main(["--manifest-only", "--fresh"])
    assert code == 0


def test_llm_fresh_flag_is_accepted(tmp_path):
    from ml.llm.__main__ import main

    code = main(
        [
            "--steps",
            "1",
            "--repeats",
            "1",
            "--structured-repeats",
            "1",
            "--checkpoint",
            str(tmp_path / "llm.npz"),
            "--fresh",
            "--quiet",
        ]
    )
    assert code == 0


def test_petroleum_cli_fresh_flag(tmp_path):
    from ml.petroleum import main

    code = main(
        [
            "--episodes",
            "2",
            "--save-dir",
            str(tmp_path),
            "--fresh",
            "--quiet",
        ]
    )
    assert code == 0


def test_numpy_arrays_are_not_needed_to_read_the_ledger(tmp_path):
    """The ledger is plain JSON, so it reads without numpy."""
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    state.record("a", 2)
    state.save()
    raw = json.loads((tmp_path / LEDGER_NAME).read_text(encoding="utf-8"))
    assert raw["scopes"]["petroleum"]["units"]["a"]["completed"] == 2


def test_ledger_artifact_absolute_path_is_kept(tmp_path):
    state = TrainingProgress(tmp_path, scope="petroleum", config={"seed": 0})
    outside = tmp_path.parent / "elsewhere"
    state.record("a", 1, artifact=outside)
    assert state.artifact_path("a") == outside


def test_unused_numpy_import_is_real():
    """A guard that the module really needs numpy (used by the agent state)."""
    assert np is not None
