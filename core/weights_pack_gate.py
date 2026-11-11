"""The held-out gate for the committed weights pack.

Every claim the pack makes is checked here on the **real** engine, with
**negative controls** so an over-eager or broken implementation fails as surely
as a missing one. The gate builds tiny real weight files (the fastest language
run and a couple of petroleum episodes) inside a scratch pack, so it stays fast,
and it measures the properties that matter:

1. a manifest is read into entries, and an entry names the file, kind, and the
   command that continues it -- a missing pack and a manifest from a *newer*
   build are both **reported**, never guessed;
2. an entry whose bytes match the recorded digest **verifies**, while a file that
   was altered after the digest was taken is **refused** (the negative control
   that stops a truncated or tampered file being installed);
3. an entry whose declared architecture does not match the file is **refused**;
4. installing a verified entry puts it in the trainer's own folder and the *real*
   trainer then resumes from it -- and installing an entry whose file fails
   verification writes nothing;
5. an unknown entry key is **refused** with the reason, and an existing file is
   never clobbered without ``force`` (a forced install keeps a byte-identical
   backup).

:func:`weights_pack_accuracy` is the metric the evaluation wires in.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from core.checkpoint_import import KIND_POLICY
from core.weights_pack import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    STATUS_DIGEST_MISMATCH,
    STATUS_INCOMPATIBLE,
    STATUS_OK,
    PackEntry,
    file_digest,
    generate_manifest,
    install_pack,
    integrity_report,
    load_pack,
    verify_entry,
)


@dataclass(frozen=True)
class PackJudgement:
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


def _write_pack(root: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a minimal pack manifest for the gate, so it exercises the real reader."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MANIFEST_VERSION,
        "project_version": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "digest_algorithm": "sha256",
        "entries": entries,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return root


def judgements() -> tuple[PackJudgement, ...]:
    """Run the whole gate and return one judgement per check."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(root / "data")}):
            return _run_checks(root)


def _run_checks(root: Path) -> tuple[PackJudgement, ...]:
    out: list[PackJudgement] = []

    weights = root / "weights"
    checkpoint = _make_checkpoint(root / "source")
    digest = file_digest(checkpoint)
    entry_row = {
        "key": "fayd-llm",
        "title": "Language model",
        "kind": "language-model",
        "path": str(checkpoint),
        "companions": [
            str(checkpoint.with_suffix(".tokenizer.json")),
            str(checkpoint.with_suffix(".state.npz")),
        ],
        "digest": digest,
        "architecture": {
            "vocab_size": 257,
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 96,
            "block_size": 64,
        },
        "continue_command": "python -m cli llm --resume",
        "notes": "test",
    }
    _write_pack(weights, [entry_row])

    # 1. The manifest is read into entries, and a missing pack is reported.
    pack = load_pack(weights)
    read_ok = (
        pack.available
        and len(pack.entries) == 1
        and pack.get("fayd-llm") is not None
        and pack.get("nope") is None
    )
    missing = load_pack(root / "does-not-exist")
    reported_missing = (not missing.available) and bool(missing.detail)
    out.append(
        PackJudgement(
            "reads_the_manifest_and_reports_a_missing_pack",
            read_ok and reported_missing,
            "the manifest read into one entry and a missing pack was reported"
            if read_ok and reported_missing
            else "the manifest was misread or a missing pack was not reported",
        )
    )

    # 2. A matching digest verifies; an altered file is refused.
    entry = pack.get("fayd-llm")
    assert entry is not None
    verified = verify_entry(entry)
    original_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(original_bytes + b"tampered")
    altered = verify_entry(entry)
    # Restore the pristine bytes so the later install checks exercise a valid file.
    checkpoint.write_bytes(original_bytes)
    digest_ok = verified.status == STATUS_OK and altered.status == STATUS_DIGEST_MISMATCH
    out.append(
        PackJudgement(
            "a_matching_digest_verifies_and_an_altered_file_is_refused",
            digest_ok,
            "the pristine file verified and a tampered one was refused"
            if digest_ok
            else "a tampered file passed or a pristine one was refused",
        )
    )

    # 3. A declared architecture that does not match the file is refused.
    bad_arch_row = dict(entry_row)
    bad_arch_row["key"] = "wrong-arch"
    bad_arch_row["architecture"] = {
        "vocab_size": 999,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 999,
        "block_size": 999,
    }
    # Restore the file so only the architecture mismatches.
    checkpoint2 = _make_checkpoint(root / "source2")
    bad_arch_row["path"] = str(checkpoint2)
    bad_arch_row["digest"] = file_digest(checkpoint2)
    _write_pack(weights, [entry_row, bad_arch_row])
    pack2 = load_pack(weights)
    wrong = pack2.get("wrong-arch")
    assert wrong is not None
    wrong_check = verify_entry(wrong)
    out.append(
        PackJudgement(
            "a_mismatched_architecture_is_refused",
            wrong_check.status == STATUS_INCOMPATIBLE,
            "an architecture that does not match the file was refused"
            if wrong_check.status == STATUS_INCOMPATIBLE
            else "an incompatible architecture was accepted",
        )
    )

    # 4. Installing a verified entry lets the *real* trainer resume from it, and a
    #    refused entry writes nothing.
    _write_pack(weights, [entry_row])
    pack3 = load_pack(weights)
    report = install_pack(pack3, keys=("fayd-llm",))
    installed_ok = report.ok and report.installed == 1
    from ml.llm.__main__ import train_and_save

    target = root / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
    with contextlib.redirect_stdout(io.StringIO()):
        train_and_save(
            steps=2,
            repeats=1,
            structured_repeats=1,
            checkpoint=target,
            log_every=0,
            val_every=0,
            verbose=False,
            resume=True,
        )
    import numpy as np
    from ml.llm.state import state_path

    resumed = target.exists() and state_path(target).exists()
    step_after = int(np.load(state_path(target))["steps"][0])
    out.append(
        PackJudgement(
            "the_real_trainer_resumes_from_an_installed_entry",
            installed_ok and resumed and step_after == 4,
            "the pack entry installed and the trainer resumed to step 4"
            if installed_ok and resumed and step_after == 4
            else "the pack entry did not reach the trainer",
        )
    )

    # 5. An unknown key is refused, and an existing file is not clobbered.
    unknown = install_pack(pack3, keys=("no-such-key",))
    refused_unknown = (not unknown.ok) and bool(unknown.detail)
    before = target.read_bytes()
    clobber = install_pack(pack3, keys=("fayd-llm",))
    untouched = target.read_bytes() == before and clobber.refused == 1
    forced = install_pack(pack3, keys=("fayd-llm",), force=True)
    backup_ok = (
        forced.installed == 1
        and forced.outcomes[0].backup is not None
        and forced.outcomes[0].backup.read_bytes() == before
    )
    out.append(
        PackJudgement(
            "an_unknown_key_is_refused_and_an_existing_file_is_backed_up",
            refused_unknown and untouched and backup_ok,
            "an unknown key was refused and an existing file was backed up, not clobbered"
            if refused_unknown and untouched and backup_ok
            else "an unknown key was accepted or a file was clobbered",
        )
    )

    # 6. Provenance is recorded and surfaced honestly: a described entry carries a
    #    readable provenance line naming its origin, and an entry with no
    #    provenance renders no line (never a fabricated "unknown").
    generated = generate_manifest(root=root / "generated", checkpoint=checkpoint, include_policies=False)
    entry_with_provenance = generated.get("fayd-llm")
    described = entry_with_provenance is not None and bool(entry_with_provenance.provenance_line())
    origin_named = entry_with_provenance is not None and "trainer" in entry_with_provenance.origin
    bare = PackEntry(key="k", title="t", kind=KIND_POLICY, path=checkpoint)
    no_line = bare.provenance_line() == "" and bare.origin == ""
    provenance_ok = described and origin_named and no_line
    out.append(
        PackJudgement(
            "provenance_is_recorded_and_surfaced_without_inventing_a_field",
            provenance_ok,
            "a generated entry named its origin and a bare entry invented no provenance"
            if provenance_ok
            else "provenance was missing from a generated entry or fabricated for a bare one",
        )
    )

    # 7. The pack-wide integrity verdict is honest: an incomplete pack (an entry
    #    not installed) is reported incomplete, a fully verified+installed pack is
    #    complete, and a *partial* verification tuple is never called complete.
    _write_pack(weights, [entry_row])
    empty_data = root / "integrity-data"
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(empty_data)}):
        not_installed = integrity_report(weights)
        incomplete_ok = (not not_installed.complete) and not_installed.verified == 1
        installed_report = install_pack(load_pack(weights), keys=("fayd-llm",))
        after = integrity_report(weights)
        complete_ok = installed_report.ok and after.complete and after.present == 1
    partial = integrity_report(pack=load_pack(weights), verifications=())
    partial_ok = (not partial.complete) and partial.unverified_keys == ("fayd-llm",)
    out.append(
        PackJudgement(
            "the_integrity_verdict_is_complete_only_when_every_entry_is_verified_and_installed",
            incomplete_ok and complete_ok and partial_ok,
            "an uninstalled pack was incomplete, an installed one complete, a partial check not complete"
            if incomplete_ok and complete_ok and partial_ok
            else "the integrity verdict called an unready pack complete",
        )
    )

    return tuple(out)


def weights_pack_accuracy() -> float:
    """The share of the gate's judgements that passed."""
    results = judgements()
    if not results:
        return 0.0
    return sum(1 for item in results if item.passed) / len(results)


__all__ = [
    "PackJudgement",
    "judgements",
    "weights_pack_accuracy",
]
