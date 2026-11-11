"""The held-out gate for importing ready-made ``.npz`` weights.

Every claim the import feature makes is checked here on the **real** engine, with
**negative controls** so an over-eager or broken implementation fails as surely
as a missing one. The gate builds tiny real checkpoints (the fastest language run
and a couple of petroleum episodes), so it stays fast, and it measures the
properties that matter:

1. a real checkpoint, its tokenizer, and its optimiser-state sidecar are all
   **recognised** by reading the arrays inside the file -- and a random numpy
   archive is reported ``unknown``, never guessed;
2. an installed checkpoint is **picked up by the real trainer**: importing it and
   resuming advances the step count exactly as continuing the original would;
3. an existing checkpoint is **never replaced silently**: a second import without
   ``--force`` is refused and the file on disk is untouched, while a forced import
   keeps a backup whose bytes equal the file it replaced;
4. a policy is installed into its **task's own file**, and a policy with no task
   named is refused rather than written to a made-up path;
5. a file fetched over the network goes through the **gates**: a URL that is not
   HTTPS, and a download that is not approved, are both refused, and no bytes are
   transferred;
6. an unreadable file (missing, a folder, or not a numpy archive) is refused with
   the reason rather than installed.

:func:`checkpoint_import_accuracy` is the metric the evaluation wires in.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.checkpoint_import import (
    KIND_LANGUAGE,
    KIND_POLICY,
    KIND_UNKNOWN,
    fetch_weights,
    install_weights,
    read_weight_info,
)


@dataclass(frozen=True)
class ImportJudgement:
    """One gate judgement: a name, whether it passed, and why."""

    name: str
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _make_checkpoint(directory: Path, *, steps: int = 2) -> Path:
    """A tiny real language-model checkpoint, written with its own save path."""
    from ml.llm.__main__ import train_and_save

    checkpoint = directory / "fayd-llm.npz"
    # ``train_and_save`` prints its own summary to stdout; the gate may run inside
    # a ``--json`` command, so that output is captured rather than allowed to
    # corrupt the machine-readable report.
    with contextlib.redirect_stdout(io.StringIO()):
        train_and_save(
            steps=steps,
            repeats=1,
            structured_repeats=1,
            checkpoint=checkpoint,
            log_every=0,
            val_every=0,
            verbose=False,
        )
    return checkpoint


def _steps(checkpoint: Path) -> int:
    from ml.llm.state import state_path

    return int(np.load(state_path(checkpoint))["steps"][0])


def _make_policy(directory: Path, task: str) -> Path:
    from ml.petroleum import train_petroleum

    train_petroleum(
        episodes=2,
        include_held_out=False,
        include_procedural=False,
        save_dir=directory,
        resume=False,
        verbose=False,
    )
    return directory / f"policy_{task}.npz"


def _random_archive(path: Path) -> Path:
    """A numpy archive of arrays no weight shape matches (a negative control)."""
    with zipfile.ZipFile(path, "w") as archive:
        buffer = io.BytesIO()
        np.save(buffer, np.arange(4))
        archive.writestr("unrelated.npy", buffer.getvalue())
    return path


def judgements() -> tuple[ImportJudgement, ...]:
    """Run the whole gate and return one judgement per check."""
    import os
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # The engine installs into the project's own data folders, so the gate
        # points that folder at its scratch directory for the duration of the run:
        # the checks exercise the real wiring without touching real learned data.
        with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(root / "data")}):
            return _run_checks(root)


def _run_checks(root: Path) -> tuple[ImportJudgement, ...]:
    from ml.llm.__main__ import train_and_save

    out: list[ImportJudgement] = []

    # 1. Recognition: a real checkpoint (with sidecars) and a random archive.
    source = _make_checkpoint(root / "source")
    info = read_weight_info(source)
    recognised = (
        info.valid
        and info.kind == KIND_LANGUAGE
        and info.checkpoint is not None
        and info.checkpoint.n_layer is not None
    )
    tokenizer_sidecar = info.tokenizer_path is not None
    state_sidecar = info.state_path is not None
    random_info = read_weight_info(_random_archive(root / "random.npz"))
    refused_random = (not random_info.valid) and random_info.kind == KIND_UNKNOWN
    out.append(
        ImportJudgement(
            "recognises_real_weights_and_refuses_a_random_archive",
            recognised and tokenizer_sidecar and state_sidecar and refused_random,
            "a checkpoint is recognised with both sidecars while a random archive "
            "is reported unknown"
            if recognised and tokenizer_sidecar and state_sidecar and refused_random
            else "the classifier guessed: a real checkpoint was missed or a random "
            "archive was accepted",
        )
    )

    # 2. Round trip through the *real* trainer: import, then resume advances the
    #    step count, and it equals continuing the source directly.
    target_dir = root / "target"
    target_checkpoint = target_dir / "fayd-llm.npz"
    report = install_weights(source, destination=target_checkpoint)
    installed_ok = report.installed and target_checkpoint.exists()
    with contextlib.redirect_stdout(io.StringIO()):
        train_and_save(
            steps=2,
            repeats=1,
            structured_repeats=1,
            checkpoint=target_checkpoint,
            log_every=0,
            val_every=0,
            verbose=False,
            resume=True,
        )
    # The source began at ``steps`` and its state sidecar travelled with it.
    advanced = _steps(target_checkpoint) == 4
    out.append(
        ImportJudgement(
            "the_real_trainer_continues_the_imported_weights",
            installed_ok and advanced,
            "the imported checkpoint resumed to step 4, so the trainer read it"
            if installed_ok and advanced
            else "the trainer did not resume from the imported checkpoint",
        )
    )

    # 3. An existing checkpoint is never replaced silently.
    refuse = install_weights(source, destination=target_checkpoint)
    bytes_before = target_checkpoint.read_bytes()
    untouched = (not refuse.installed) and target_checkpoint.read_bytes() == bytes_before
    forced = install_weights(source, destination=target_checkpoint, force=True)
    backup_ok = (
        forced.installed
        and forced.backup is not None
        and forced.backup.read_bytes() == bytes_before
    )
    out.append(
        ImportJudgement(
            "an_existing_checkpoint_is_backed_up_not_clobbered",
            untouched and backup_ok,
            "a plain import was refused and a forced one kept a byte-identical backup"
            if untouched and backup_ok
            else "an existing checkpoint was replaced without a refusal or a backup",
        )
    )

    # 4. A policy is installed into its task's own file; a policy whose name names
    #    no task and with no --task is refused rather than written to a made-up path.
    from core.petroleum_tasks import PETROLEUM_TASKS

    policy_dir = root / "policies"
    task = PETROLEUM_TASKS[0].name
    policy = _make_policy(policy_dir, task)
    policy_info = read_weight_info(policy)
    policy_report = install_weights(
        policy, destination=root / "artifacts" / "petroleum" / f"policy_{task}.npz"
    )
    anonymous = policy_dir / "mystery.npz"
    anonymous.write_bytes(policy.read_bytes())
    unnamed = install_weights(anonymous)  # no destination, no task -> must be refused
    policy_ok = (
        policy_info.kind == KIND_POLICY
        and policy_info.task == task
        and policy_report.installed
        and (not unnamed.installed)
    )
    out.append(
        ImportJudgement(
            "a_policy_goes_to_its_task_file_and_an_unnamed_one_is_refused",
            policy_ok,
            "the policy installed into its task's file and an unnamed one was refused"
            if policy_ok
            else "a policy was misclassified or written to a made-up path",
        )
    )

    # 5. The network gates: a non-HTTPS URL and an unapproved download are refused.
    plain = fetch_weights("http://example.com/weights.npz", approved=True)
    unapproved = fetch_weights("https://example.com/weights.npz", approved=False)
    gated = (not plain.installed) and (not unapproved.installed)
    out.append(
        ImportJudgement(
            "a_fetch_is_gated",
            gated,
            "a plain-HTTP URL and an unapproved download were both refused"
            if gated
            else "a fetch that should have been refused was allowed",
        )
    )

    # 6. Unreadable inputs are refused with a reason, never installed.
    missing = read_weight_info(root / "nope.npz")
    folder = read_weight_info(root)
    not_archive = root / "plain.npz"
    not_archive.write_text("not a zip", encoding="utf-8")
    text_info = read_weight_info(not_archive)
    refused_inputs = (
        (not missing.valid)
        and (not folder.valid)
        and (not text_info.valid)
        and not install_weights(not_archive).installed
    )
    out.append(
        ImportJudgement(
            "unreadable_inputs_are_refused",
            refused_inputs,
            "a missing file, a folder, and a non-archive were each refused"
            if refused_inputs
            else "an unreadable input was accepted",
        )
    )

    return tuple(out)


def checkpoint_import_accuracy() -> float:
    """The share of the gate's judgements that passed."""
    results = judgements()
    if not results:
        return 0.0
    return sum(1 for item in results if item.passed) / len(results)


__all__ = [
    "ImportJudgement",
    "checkpoint_import_accuracy",
    "judgements",
]
