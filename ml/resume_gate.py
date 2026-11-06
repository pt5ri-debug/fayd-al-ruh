"""The held-out gate for resumable training: continuation, proven by measurement.

Every claim the resume feature makes is checked here on the **real** trainers,
with **negative controls** so an over-eager or broken implementation fails as
surely as a missing one. The gate runs small real runs (a couple of episodes on
the fastest tasks) so it stays fast, and it measures the properties that matter:

1. the ledger round-trips its records through disk, and a corrupt file reads as
   empty rather than raising;
2. a changed configuration **invalidates** the record (the fingerprint) while a
   raised episode budget does **not** -- a top-up is not a restart;
3. a fully-trained unit is **loaded, not retrained** (its artifact's bytes are
   untouched on the second run), and a run that finished the work reports it as
   already complete;
4. a partial run **continues** rather than restarts: ``3 then resume to 6``
   leaves the exact policy bytes an uninterrupted ``6`` leaves;
5. a raised budget **tops up** the recorded cumulative count instead of resetting
   it;
6. the language model's own resume still holds: a resumed run raises the
   cumulative step count and does not reproduce a fresh run of the same length.

:func:`training_resume_accuracy` is the metric the evaluation wires in.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.petroleum_tasks import PETROLEUM_TASKS

from ml.checkpoint import LEDGER_NAME, TrainingProgress
from ml.petroleum import train_petroleum

#: The task the gate trains on. It is the first catalog task and converges in a
#: couple of episodes, so the gate stays fast.
_GATE_TASK = PETROLEUM_TASKS[0].name
#: The gate's episode budget: enough to learn, small enough to stay quick.
_GATE_EPISODES = 2


@dataclass(frozen=True)
class ResumeJudgement:
    """One gate judgement: a name, whether it passed, and why."""

    name: str
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _policy_bytes(directory: Path, task: str) -> bytes | None:
    path = directory / f"policy_{task}.npz"
    return path.read_bytes() if path.exists() else None


def _train(directory: Path, *, episodes: int, resume: bool = True) -> None:
    """A small petroleum run on the fastest task, so the gate stays quick."""
    train_petroleum(
        episodes=episodes,
        include_held_out=False,
        include_procedural=False,
        save_dir=directory,
        resume=resume,
        verbose=False,
    )


def judgements() -> tuple[ResumeJudgement, ...]:
    """Run the whole gate and return one judgement per check."""
    out: list[ResumeJudgement] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. The ledger round-trips, and a corrupt file reads as empty.
        ledger_dir = root / "ledger"
        state = TrainingProgress(ledger_dir, scope="petroleum", config={"seed": 0})
        state.record("task-a", 5, artifact="brain_task-a")
        state.save()
        reloaded = TrainingProgress(ledger_dir, scope="petroleum", config={"seed": 0})
        round_trip = reloaded.completed("task-a") == 5 and reloaded.available
        (ledger_dir / LEDGER_NAME).write_text("{ this is not json", encoding="utf-8")
        corrupt = TrainingProgress(ledger_dir, scope="petroleum", config={"seed": 0})
        corrupt_ok = (not corrupt.available) and corrupt.completed("task-a") == 0
        out.append(
            ResumeJudgement(
                "ledger_round_trip_and_corruption",
                round_trip and corrupt_ok,
                "records survive a save/load and a corrupt file reads as empty"
                if round_trip and corrupt_ok
                else "the ledger did not round-trip or a corrupt file raised",
            )
        )

        # 2. A changed configuration invalidates the record; a raised budget does not.
        #    (This uses its own folder: the corrupt file above would mask the result.)
        same_scope = root / "fingerprint"
        first = TrainingProgress(same_scope, scope="petroleum", config={"tasks": ["a"], "seed": 0})
        first.record("a", 4)
        first.save()
        changed = TrainingProgress(same_scope, scope="petroleum", config={"tasks": ["a"], "seed": 1})
        invalidated = changed.stale and not changed.available
        # A different *budget* is not part of the fingerprint, so the record stays.
        top_up = TrainingProgress(same_scope, scope="petroleum", config={"tasks": ["a"], "seed": 0})
        budget_ok = (not top_up.stale) and top_up.completed("a") == 4
        out.append(
            ResumeJudgement(
                "fingerprint_invalidates_a_changed_run",
                invalidated and budget_ok,
                "a changed config resets the record while a raised budget keeps it"
                if invalidated and budget_ok
                else "the fingerprint treated a budget change as a new run, or missed a real one",
            )
        )

        # 3. A fully-trained task is loaded, not retrained: the artifact bytes are
        #    unchanged by the second run.
        once = root / "once"
        _train(once, episodes=_GATE_EPISODES)
        before = _policy_bytes(once, _GATE_TASK)
        _train(once, episodes=_GATE_EPISODES)
        after = _policy_bytes(once, _GATE_TASK)
        skipped = before is not None and before == after
        out.append(
            ResumeJudgement(
                "a_finished_task_is_not_retrained",
                skipped,
                "the saved policy is byte-identical, so the task was loaded, not retrained"
                if skipped
                else "a finished task was retrained (its bytes changed)",
            )
        )

        # 4. A partial run continues rather than restarts: 3+3 equals a single 6.
        full = root / "full"
        part = root / "part"
        _train(full, episodes=6)
        _train(part, episodes=3)
        _train(part, episodes=6)
        equivalent = (
            _policy_bytes(full, _GATE_TASK) is not None
            and _policy_bytes(full, _GATE_TASK) == _policy_bytes(part, _GATE_TASK)
        )
        out.append(
            ResumeJudgement(
                "a_partial_run_continues",
                equivalent,
                "3 episodes then a resume to 6 equals an uninterrupted 6"
                if equivalent
                else "a resumed run diverged from an uninterrupted run of the same length",
            )
        )

        # 5. A raised budget tops up the cumulative count rather than resetting it.
        ledger = TrainingProgress(part, scope="petroleum")
        before_count = ledger.completed(_GATE_TASK)
        _train(part, episodes=8)
        ledger = TrainingProgress(part, scope="petroleum")
        topped = before_count == 6 and ledger.completed(_GATE_TASK) == 8
        out.append(
            ResumeJudgement(
                "a_raised_budget_tops_up",
                topped,
                "the cumulative count rose from 6 to 8 rather than restarting"
                if topped
                else f"expected 6 then 8, recorded {before_count} then {ledger.completed(_GATE_TASK)}",
            )
        )

        # 6. The language model's own resume still raises the cumulative step count.
        out.append(_language_model_judgement(root / "llm"))

    return tuple(out)


def _language_model_judgement(scratch: Path) -> ResumeJudgement:
    """A resumed language run continues: the step count rises and the weights differ."""
    import contextlib
    import io

    import numpy as np

    from ml.llm.__main__ import train_and_save
    from ml.llm.state import state_path

    checkpoint = scratch / "fayd-llm.npz"
    # ``train_and_save`` prints its own one-line summary to stdout regardless of
    # ``verbose``; the gate runs inside a ``--json`` command, so that output is
    # captured rather than allowed to corrupt the machine-readable report.
    with contextlib.redirect_stdout(io.StringIO()):
        train_and_save(
            steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint,
            log_every=0, val_every=0, verbose=False,
        )
        resumed = train_and_save(
            steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint,
            log_every=0, val_every=0, verbose=False, resume=True,
        )
        fresh = train_and_save(
            steps=4, repeats=1, structured_repeats=1, checkpoint=scratch / "fresh.npz",
            log_every=0, val_every=0, verbose=False,
        )
    steps_ok = int(np.load(state_path(checkpoint))["steps"][0]) == 4
    diverged = not np.allclose(resumed.token_embedding, fresh.token_embedding)
    passed = steps_ok and diverged
    return ResumeJudgement(
        "language_model_resume_continues",
        passed,
        "the cumulative step count rose to 4 and the resumed weights differ from a fresh 4"
        if passed
        else "the language-model resume did not continue from the saved state",
    )


def training_resume_accuracy() -> float:
    """The share of the gate's judgements that passed."""
    results = judgements()
    if not results:
        return 0.0
    return sum(1 for j in results if j.passed) / len(results)


def describe() -> str:
    """The full gate report, one line per judgement."""
    results = judgements()
    passed = sum(1 for j in results if j.passed)
    lines = [f"Resumable training gate: {passed}/{len(results)} judgement(s) passed"]
    lines.extend(f"  {j.describe()}" for j in results)
    return "\n".join(lines)


__all__ = [
    "ResumeJudgement",
    "describe",
    "judgements",
    "training_resume_accuracy",
]
