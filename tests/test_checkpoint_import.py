"""Importing ready-made ``.npz`` weights so the project's trainers continue them.

These pin the import engine's own contract -- recognition, installation, backups,
the fetch gate, and the listing -- and the wiring that makes the feature reachable:
the CLI (`python -m cli checkpoints`, `python -m cli models import-weights`), the
trainer's own flag (`python -m ml.llm --import-checkpoint`), and the Model Manager.
The negative controls matter as much as the happy paths: a random archive must not
be guessed to be a checkpoint, an existing file must not be clobbered, and an
unapproved fetch must transfer nothing.
"""

from __future__ import annotations

import contextlib
import io
import os
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from core.checkpoint_import import (
    BACKUP_SUFFIX,
    KIND_LANGUAGE,
    KIND_MODEL,
    KIND_POLICY,
    KIND_STATE,
    KIND_UNKNOWN,
    KINDS,
    CheckpointImportError,
    ImportReport,
    InstalledWeights,
    WeightInfo,
    checkpoint_import_dir,
    classify_members,
    destination_for,
    fetch_weights,
    install_weights,
    list_installed,
    npy_members,
    read_weight_info,
    resume_plan,
)
from core.checkpoint_import_gate import (
    checkpoint_import_accuracy,
)
from core.checkpoint_import_gate import (
    judgements as import_gate_judgements,
)
from core.petroleum_tasks import PETROLEUM_TASKS

_FIRST_TASK = PETROLEUM_TASKS[0].name


# ------------------------------------------------------------------ helpers


def _train_checkpoint(directory: Path, *, steps: int = 2) -> Path:
    """A tiny real language-model checkpoint (with its sidecars)."""
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


def _train_policy(directory: Path, task: str) -> Path:
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


def _write_random_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        buffer = io.BytesIO()
        np.save(buffer, np.arange(3))
        archive.writestr("something.npy", buffer.getvalue())
    return path


# ------------------------------------------------------------------- reading


def test_npy_members_reads_names_from_the_archive(tmp_path):
    archive = tmp_path / "weights.npz"
    np.savez(archive, config=np.arange(3), p0=np.zeros(2))
    members = npy_members(archive)
    assert set(members) == {"config", "p0"}


def test_npy_members_is_empty_for_a_non_archive(tmp_path):
    path = tmp_path / "plain.npz"
    path.write_text("not a zip", encoding="utf-8")
    assert npy_members(path) == ()
    assert npy_members(tmp_path / "missing.npz") == ()


def test_classify_members_reads_each_kind():
    assert classify_members(("config", "p0", "W")) == KIND_LANGUAGE
    assert classify_members(("steps", "t", "m0")) == KIND_STATE
    assert classify_members(("W", "epsilon", "alpha")) == KIND_POLICY
    assert classify_members(("kind", "W1")) == KIND_MODEL
    assert classify_members(("foo", "bar")) == KIND_UNKNOWN


def test_read_weight_info_recognises_a_real_checkpoint(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    info = read_weight_info(source)
    assert info.valid
    assert info.kind == KIND_LANGUAGE
    assert info.resumable
    assert info.checkpoint is not None
    assert info.checkpoint.n_layer is not None
    # The sidecars the trainer wrote travel with it.
    assert info.tokenizer_path is not None
    assert info.state_path is not None
    assert "language-model" in info.describe()


def test_read_weight_info_refuses_a_random_archive(tmp_path):
    info = read_weight_info(_write_random_archive(tmp_path / "random.npz"))
    assert not info.valid
    assert info.kind == KIND_UNKNOWN
    assert not info.resumable


def test_read_weight_info_refuses_missing_folder_and_non_archive(tmp_path):
    missing = read_weight_info(tmp_path / "nope.npz")
    assert not missing.valid and "no file" in missing.detail.lower()
    folder = read_weight_info(tmp_path)
    assert not folder.valid and "folder" in folder.detail.lower()
    text = tmp_path / "plain.npz"
    text.write_text("not a zip", encoding="utf-8")
    info = read_weight_info(text)
    assert not info.valid and "not a numpy" in info.detail.lower()


def test_weight_info_round_trips_through_primitive(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    info = read_weight_info(source)
    primitive = info.to_primitive()
    assert primitive["kind"] == KIND_LANGUAGE
    assert primitive["valid"] is True
    assert primitive["resumable"] is True
    assert isinstance(primitive["checkpoint"], dict)


def test_a_policy_names_its_task(tmp_path):
    policy = _train_policy(tmp_path, _FIRST_TASK)
    info = read_weight_info(policy)
    assert info.kind == KIND_POLICY
    assert info.task == _FIRST_TASK


# ------------------------------------------------------------- destinations


def test_destination_for_points_at_the_trainers_own_folders(tmp_path):
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        from core.paths import artifacts_dir, checkpoints_dir

        assert destination_for(KIND_LANGUAGE) == checkpoints_dir() / "fayd-llm.npz"
        assert destination_for(KIND_STATE) == checkpoints_dir() / "fayd-llm.state.npz"
        assert (
            destination_for(KIND_POLICY, task="drill-well")
            == artifacts_dir() / "petroleum" / "policy_drill-well.npz"
        )
        assert destination_for(KIND_MODEL, name="m.npz") == artifacts_dir() / "m.npz"


def test_destination_for_refuses_a_policy_with_no_task():
    with pytest.raises(CheckpointImportError):
        destination_for(KIND_POLICY)


def test_destination_for_refuses_an_unknown_kind():
    with pytest.raises(CheckpointImportError):
        destination_for("nonsense")


def test_resume_plan_names_the_trainer_per_kind():
    assert resume_plan(KIND_LANGUAGE).command == "python -m cli training-center lm-train --resume"
    assert resume_plan(KIND_STATE).command == "python -m cli training-center lm-train --resume"
    assert resume_plan(KIND_POLICY, task="x").command == "python -m cli petroleum --resume"
    assert resume_plan(KIND_MODEL).command == "python -m cli ml"
    # An unknown kind names no command rather than inventing one.
    assert resume_plan(KIND_UNKNOWN).command == ""


# --------------------------------------------------------------- installing


def test_install_weights_copies_into_the_trainers_folder(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    target = tmp_path / "target" / "fayd-llm.npz"
    report = install_weights(source, destination=target)
    assert report.installed
    assert target.exists()
    assert target.read_bytes() == source.read_bytes()
    # The tokenizer and state sidecars travel with the checkpoint.
    assert target.with_suffix(".tokenizer.json").exists()
    assert target.with_suffix(".state.npz").exists()
    assert report.plan is not None and report.plan.command.endswith("--resume")


def test_install_weights_refuses_to_clobber_without_force(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    target = tmp_path / "target" / "fayd-llm.npz"
    install_weights(source, destination=target)
    before = target.read_bytes()
    second = install_weights(source, destination=target)
    assert not second.installed
    assert "--force" in second.detail
    assert target.read_bytes() == before


def test_install_weights_force_keeps_a_backup(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    target = tmp_path / "target" / "fayd-llm.npz"
    install_weights(source, destination=target)
    before = target.read_bytes()
    forced = install_weights(source, destination=target, force=True)
    assert forced.installed
    assert forced.backup is not None
    assert forced.backup.read_bytes() == before
    assert BACKUP_SUFFIX in forced.backup.name


def test_install_weights_refuses_an_unreadable_file(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_text("nope", encoding="utf-8")
    report = install_weights(bad, destination=tmp_path / "target.npz")
    assert not report.installed
    assert not report.info.valid


def test_install_weights_reports_a_file_already_in_place(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    report = install_weights(source, destination=source)
    assert not report.installed
    assert "already in place" in report.detail.lower()


def test_import_report_round_trips_through_primitive(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    report = install_weights(source, destination=tmp_path / "target" / "fayd-llm.npz")
    primitive = report.to_primitive()
    assert primitive["installed"] is True
    assert primitive["plan"]["command"].endswith("--resume")


# ---------------------------------------------------------------- fetching


def test_fetch_weights_refuses_a_plain_http_url():
    report = fetch_weights("http://example.com/w.npz", approved=True)
    assert not report.installed
    assert "https" in report.detail.lower()


def test_fetch_weights_refuses_without_approval():
    report = fetch_weights("https://example.com/w.npz", approved=False)
    assert not report.installed


def test_fetch_weights_refuses_a_non_npz_name():
    report = fetch_weights("https://example.com/w.bin", approved=True)
    assert not report.installed
    assert "npz" in report.detail.lower()


def test_checkpoint_import_dir_is_git_ignored_data(tmp_path):
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert checkpoint_import_dir() == tmp_path / "data" / "checkpoint_import"


# ----------------------------------------------------------------- listing


def test_list_installed_reads_the_project_folders(tmp_path):
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert list_installed() == ()
        source = _train_checkpoint(tmp_path / "run")
        install_weights(source)  # no destination -> the project's own folder
        installed = list_installed()
        kinds = {item.kind for item in installed}
        assert KIND_LANGUAGE in kinds
        assert KIND_STATE in kinds
        for item in installed:
            assert isinstance(item, InstalledWeights)
            assert item.describe()


# ------------------------------------------------------------- end to end


def test_the_real_trainer_continues_imported_weights(tmp_path):
    """Import a checkpoint, then resume the real trainer: the steps advance."""
    from ml.llm.__main__ import train_and_save
    from ml.llm.state import state_path

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        source = _train_checkpoint(tmp_path / "run")
        report = install_weights(source)
        assert report.installed
        assert report.destination is not None
        with contextlib.redirect_stdout(io.StringIO()):
            train_and_save(
                steps=2,
                repeats=1,
                structured_repeats=1,
                checkpoint=report.destination,
                log_every=0,
                val_every=0,
                verbose=False,
                resume=True,
            )
        assert int(np.load(state_path(report.destination))["steps"][0]) == 4


def test_ml_llm_import_checkpoint_flag(tmp_path):
    """`python -m ml.llm --import-checkpoint F` installs and resumes in one step."""
    from core.paths import checkpoints_dir
    from ml.llm.__main__ import main as llm_main
    from ml.llm.state import state_path

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        source = _train_checkpoint(tmp_path / "run")
        checkpoint = checkpoints_dir() / "fayd-llm.npz"
        code = llm_main(
            [
                "--steps", "2",
                "--repeats", "1",
                "--structured-repeats", "1",
                "--checkpoint", str(checkpoint),
                "--import-checkpoint", str(source),
                "--quiet",
            ]
        )
        assert code == 0
        assert checkpoint.exists()
        assert int(np.load(state_path(checkpoint))["steps"][0]) == 4


def test_ml_llm_import_checkpoint_refuses_a_bad_file(tmp_path):
    from ml.llm.__main__ import main as llm_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        bad = tmp_path / "bad.npz"
        bad.write_text("nope", encoding="utf-8")
        code = llm_main(
            [
                "--checkpoint", str(tmp_path / "data" / "fayd-llm.npz"),
                "--import-checkpoint", str(bad),
                "--steps", "1",
                "--quiet",
            ]
        )
        assert code == 3


# --------------------------------------------------------------- the CLI


def test_cli_inspect_and_plan(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    source = _train_checkpoint(tmp_path / "run")
    assert cli_main(["inspect", str(source)]) == 0
    assert "language-model" in capsys.readouterr().out
    assert cli_main(["plan", str(source)]) == 0
    assert "--resume" in capsys.readouterr().out


def test_cli_inspect_refuses_a_bad_file(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    bad = tmp_path / "bad.npz"
    bad.write_text("nope", encoding="utf-8")
    assert cli_main(["inspect", str(bad)]) == 3
    assert "not a numpy" in capsys.readouterr().out.lower()


def test_cli_status_empty_then_installed(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["status"]) == 3
        capsys.readouterr()
        source = _train_checkpoint(tmp_path / "run")
        assert cli_main(["install", str(source)]) == 0
        capsys.readouterr()
        assert cli_main(["status"]) == 0
        assert "language-model" in capsys.readouterr().out


def test_cli_install_refuses_without_force(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        source = _train_checkpoint(tmp_path / "run")
        assert cli_main(["install", str(source)]) == 0
        capsys.readouterr()
        assert cli_main(["install", str(source)]) == 4
        assert "--force" in capsys.readouterr().out


def test_cli_fetch_needs_approval(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    assert cli_main(["fetch", "https://example.com/w.npz"]) == 4
    assert "approve" in capsys.readouterr().err.lower()


def test_cli_kinds_and_list(tmp_path, capsys):
    from cli.checkpoints import main as cli_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["kinds"]) == 0
        out = capsys.readouterr().out
        for kind in KINDS:
            assert kind in out
        assert cli_main(["list"]) == 0


def test_cli_json_output(tmp_path, capsys):
    import json

    from cli.checkpoints import main as cli_main

    source = _train_checkpoint(tmp_path / "run")
    assert cli_main(["inspect", str(source), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == KIND_LANGUAGE


# -------------------------------------------------------- model manager


def test_model_manager_delegates_the_import(tmp_path):
    from core.model_manager import ModelManager

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        manager = ModelManager(tmp_path)
        source = _train_checkpoint(tmp_path / "run")
        info = manager.importable_weights(source)
        assert info.kind == KIND_LANGUAGE
        report = manager.import_weights(source)
        assert report.installed
        assert any(item.kind == KIND_LANGUAGE for item in manager.installed_weights())


def test_models_cli_import_weights(tmp_path, capsys):
    from cli.models import main as models_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        source = _train_checkpoint(tmp_path / "run")
        assert models_main(["import-weights", str(source)]) == 0
        assert "imported" in capsys.readouterr().out
        assert models_main(["installed-weights"]) == 0
        assert "language-model" in capsys.readouterr().out
        assert models_main(["weight-kinds"]) == 0


def test_models_cli_import_weights_missing_file(tmp_path, capsys):
    from cli.models import main as models_main

    assert models_main(["import-weights", str(tmp_path / "nope.npz")]) == 3
    assert "No file" in capsys.readouterr().err


# --------------------------------------------------- weights pack (Phase 2)


def _write_pack(tmp_path):
    """Write a tiny real pack manifest over a real weight file."""
    import json

    from core.checkpoint_import import KIND_LANGUAGE
    from core.weights_pack import MANIFEST_NAME, file_digest

    source = _train_checkpoint(tmp_path / "run")
    root = tmp_path / "weights"
    root.mkdir()
    digest = file_digest(source)
    manifest = {
        "version": 1,
        "project_version": "test",
        "entries": [
            {
                "key": "fayd-llm",
                "kind": KIND_LANGUAGE,
                "title": "Fayd Al-Ruh language model",
                "path": str(source),
                "digest": digest,
                "continue_command": "python -m cli training-center lm-train --resume",
            }
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_model_manager_reads_the_weights_pack(tmp_path):
    from core.model_manager import ModelManager

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        root = _write_pack(tmp_path)
        manager = ModelManager(tmp_path)
        pack = manager.weights_pack(root)
        assert pack.available
        assert pack.get("fayd-llm") is not None
        assert manager.continue_command("fayd-llm", root) == "python -m cli training-center lm-train --resume"
        status = manager.weights_pack_status(root)
        assert status.ready == 1
        verifications = manager.verify_weights_pack(root)
        assert verifications and verifications[0].ok
        assert manager.verify_weight_entry("fayd-llm", root).ok


def test_model_manager_refuses_an_unknown_pack_entry(tmp_path):
    from core.model_manager import ModelManager
    from core.weights_pack import WeightsPackError

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        root = _write_pack(tmp_path)
        manager = ModelManager(tmp_path)
        try:
            manager.verify_weight_entry("nope", root)
        except WeightsPackError:
            pass
        else:  # pragma: no cover - the refusal is the point
            raise AssertionError("an unknown key must be refused")


def test_models_cli_reads_the_shipped_weights_pack(capsys):
    """The CLI's pack commands read the *shipped* pack read-only."""
    from cli.models import main as models_main

    # The repo ships `weights/manifest.json`, so this exercises the real path.
    assert models_main(["weights-pack", "--list"]) == 0
    out = capsys.readouterr().out
    assert "fayd-llm" in out
    assert models_main(["verify-weights", "fayd-llm"]) == 0
    assert "OK" in capsys.readouterr().out
    assert models_main(["continue-weights", "fayd-llm"]) == 0
    assert "training-center lm-train --resume" in capsys.readouterr().out
    assert models_main(["continue-weights", "nope"]) == 3


def test_models_cli_pack_commands_exist():
    from cli.models import _build_parser

    parser = _build_parser()
    for command in (
        "weights-pack",
        "verify-weights",
        "install-pack",
        "continue-weights",
    ):
        argv = [command, "fayd-llm"] if command == "continue-weights" else [command]
        assert parser.parse_args(argv).command == command


# ------------------------------------------------------------------- gate


def test_the_gate_passes_on_the_real_engine():
    assert checkpoint_import_accuracy() == 1.0


def test_the_gate_has_negative_controls():
    names = {judgement.name for judgement in import_gate_judgements()}
    assert "recognises_real_weights_and_refuses_a_random_archive" in names
    assert "an_existing_checkpoint_is_backed_up_not_clobbered" in names
    assert "a_fetch_is_gated" in names
    assert "unreadable_inputs_are_refused" in names


def test_the_gate_does_not_touch_real_data(tmp_path):
    """The gate isolates FAYD_DATA_DIR, so it never writes into a real data folder."""
    marker = tmp_path / "real" / "ml" / "llm" / "checkpoints"
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "real")}):
        assert checkpoint_import_accuracy() == 1.0
    assert not marker.exists()


# ------------------------------------------------------------- shape guards


def test_weight_info_defaults_are_honest():
    info = WeightInfo(path=Path("x.npz"))
    assert info.kind == KIND_UNKNOWN
    assert not info.valid
    assert not info.resumable


def test_import_report_describes_each_state(tmp_path):
    source = _train_checkpoint(tmp_path / "run")
    report = install_weights(source, destination=tmp_path / "t" / "fayd-llm.npz")
    assert isinstance(report, ImportReport)
    text = report.describe()
    assert "imported" in text and "continue with" in text
