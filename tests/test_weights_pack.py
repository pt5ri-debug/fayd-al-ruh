"""The committed weights pack: what the project ships, and how it is continued.

These pin the pack engine's own contract -- reading a manifest, verifying an entry
against its digest, kind, and architecture, installing a verified entry into the
trainer's folder, and refusing a file the pack cannot vouch for -- plus the wiring
that makes the feature reachable: the CLI (``python -m cli weights``) and the
shipped pack the repository carries. The negative controls matter as much as the
happy paths: an altered file, a mismatched architecture, an unknown key, and a
newer manifest must each be refused rather than accepted.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from unittest import mock

from core.paths import weights_pack_dir
from core.weights_pack import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    STATUS_DIGEST_MISMATCH,
    STATUS_INCOMPATIBLE,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_WRONG_KIND,
    PackEntry,
    WeightsPackError,
    file_digest,
    generate_manifest,
    install_pack,
    integrity_report,
    load_pack,
    pack_status,
    verify_entry,
    verify_pack,
)
from core.weights_pack_gate import judgements as pack_gate_judgements
from core.weights_pack_gate import weights_pack_accuracy

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


def _language_row(checkpoint: Path, *, key: str = "fayd-llm") -> dict[str, object]:
    return {
        "key": key,
        "title": "Language model",
        "kind": "language-model",
        "path": str(checkpoint),
        "companions": [
            str(checkpoint.with_suffix(".tokenizer.json")),
            str(checkpoint.with_suffix(".state.npz")),
        ],
        "digest": file_digest(checkpoint),
        "architecture": {
            "vocab_size": 257,
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 96,
            "block_size": 64,
        },
        "continue_command": "python -m cli llm --resume",
        "notes": "test entry",
    }


def _write_pack(root: Path, entries: list[dict[str, object]], *, version: int = MANIFEST_VERSION) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "project_version": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "digest_algorithm": "sha256",
        "entries": entries,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return root


# ------------------------------------------------------------------ reading


def test_load_pack_reads_the_manifest(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    assert pack.available
    assert pack.version == MANIFEST_VERSION
    assert pack.keys() == ("fayd-llm",)
    assert pack.get("fayd-llm") is not None
    assert pack.get("FAYD-LLM") is not None  # case-insensitive lookup
    assert pack.get("nope") is None


def test_load_pack_reports_a_missing_pack(tmp_path):
    pack = load_pack(tmp_path / "absent")
    assert not pack.available
    assert "No weights pack" in pack.detail


def test_load_pack_refuses_a_newer_manifest(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)], version=999)
    pack = load_pack(root)
    assert not pack.available
    assert "version 999" in pack.detail


def test_load_pack_refuses_an_unknown_kind(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    row = _language_row(checkpoint)
    row["kind"] = "not-a-kind"
    root = _write_pack(tmp_path / "weights", [row])
    pack = load_pack(root)
    assert not pack.available
    assert "not-a-kind" in pack.detail


def test_load_pack_tolerates_a_corrupt_manifest(tmp_path):
    root = tmp_path / "weights"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    pack = load_pack(root)
    assert not pack.available
    assert "Could not read" in pack.detail


def test_entry_round_trips_through_primitive(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    primitive = entry.to_primitive()
    assert primitive["key"] == "fayd-llm"
    assert primitive["kind"] == "language-model"
    assert primitive["resumable"] is True
    assert primitive["continue_command"] == "python -m cli llm --resume"


# --------------------------------------------------------------- verification


def test_verify_a_pristine_entry(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    result = verify_entry(entry)
    assert result.ok and result.status == STATUS_OK
    assert result.size_bytes > 0


def test_verify_refuses_a_tampered_file(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    result = verify_entry(entry)
    assert result.status == STATUS_DIGEST_MISMATCH
    assert not result.ok


def test_verify_refuses_a_missing_file(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    checkpoint.unlink()
    assert verify_entry(entry).status == STATUS_MISSING


def test_verify_refuses_a_mismatched_architecture(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    row = _language_row(checkpoint)
    row["architecture"] = {"vocab_size": 999, "n_layer": 12, "n_head": 12, "n_embd": 999, "block_size": 999}
    root = _write_pack(tmp_path / "weights", [row])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    result = verify_entry(entry)
    assert result.status == STATUS_INCOMPATIBLE
    assert "architecture" in result.detail.lower() or "vocab_size" in result.detail


def test_verify_refuses_a_wrong_kind(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    row = _language_row(checkpoint)
    row["kind"] = "policy"  # a language checkpoint is not a policy
    root = _write_pack(tmp_path / "weights", [row])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    assert verify_entry(entry).status == STATUS_WRONG_KIND


def test_verify_refuses_a_missing_sidecar(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    entry = load_pack(root).get("fayd-llm")
    assert entry is not None
    checkpoint.with_suffix(".tokenizer.json").unlink()
    result = verify_entry(entry)
    assert result.status == STATUS_INCOMPATIBLE
    assert "sidecar" in result.detail.lower()


def test_verify_pack_covers_every_entry(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    policy = _train_policy(tmp_path / "policies", "drill-well")
    root = _write_pack(
        tmp_path / "weights",
        [
            _language_row(checkpoint),
            {
                "key": "policy-drill-well",
                "title": "Policy",
                "kind": "policy",
                "path": str(policy),
                "digest": file_digest(policy),
                "task": "drill-well",
                "continue_command": "python -m cli petroleum --resume",
            },
        ],
    )
    results = verify_pack(load_pack(root))
    assert len(results) == 2
    assert all(item.ok for item in results)


# ------------------------------------------------------------------ install


def test_install_entry_puts_it_in_the_trainers_folder(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        report = install_pack(pack, keys=("fayd-llm",))
        assert report.ok and report.installed == 1
        target = tmp_path / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
        assert target.exists()
        assert target.read_bytes() == checkpoint.read_bytes()


def test_install_refuses_a_tampered_entry_and_writes_nothing(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        report = install_pack(pack, keys=("fayd-llm",))
        assert report.refused == 1
        assert not (tmp_path / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz").exists()


def test_install_refuses_an_unknown_key(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        report = install_pack(pack, keys=("no-such-key",))
        assert not report.ok
        assert "no-such-key" in report.detail


def test_install_does_not_clobber_without_force(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert install_pack(pack, keys=("fayd-llm",)).installed == 1
        second = install_pack(pack, keys=("fayd-llm",))
        assert second.refused == 1
        assert "--force" in second.outcomes[0].detail


def test_install_force_keeps_a_backup(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        install_pack(pack, keys=("fayd-llm",))
        target = tmp_path / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
        before = target.read_bytes()
        forced = install_pack(pack, keys=("fayd-llm",), force=True)
        assert forced.installed == 1
        backup = forced.outcomes[0].backup
        assert backup is not None and backup.read_bytes() == before


def test_install_a_policy_goes_to_its_task_file(tmp_path):
    policy = _train_policy(tmp_path / "policies", "drill-well")
    root = _write_pack(
        tmp_path / "weights",
        [
            {
                "key": "policy-drill-well",
                "title": "Policy",
                "kind": "policy",
                "path": str(policy),
                "digest": file_digest(policy),
                "task": "drill-well",
                "continue_command": "python -m cli petroleum --resume",
            }
        ],
    )
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        report = install_pack(pack, keys=("policy-drill-well",))
        assert report.ok
        target = tmp_path / "data" / "ml" / "artifacts" / "petroleum" / "policy_drill-well.npz"
        assert target.exists()


def test_install_pack_with_no_keys_installs_everything(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    pack = load_pack(root)
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        report = install_pack(pack)
        assert report.installed == len(pack.entries)


# ------------------------------------------------------------------- status


def test_pack_status_reports_ready_and_present(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        status = pack_status(root)
        assert status.ready == 1 and status.present == 0
        install_pack(load_pack(root), keys=("fayd-llm",))
        status = pack_status(root)
        assert status.ready == 1 and status.present == 1


# -------------------------------------------------------------- generation


def test_generate_manifest_reads_the_weights_on_disk(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    policy_dir = tmp_path / "policies"
    _train_policy(policy_dir, "drill-well")
    root = tmp_path / "weights"
    pack = generate_manifest(
        root=root,
        checkpoint=checkpoint,
        policy_dir=policy_dir,
    )
    assert pack.root == root
    assert (root / MANIFEST_NAME).exists()
    keys = set(pack.keys())
    assert "fayd-llm" in keys
    assert "policy-drill-well" in keys
    # A regenerated pack verifies against the files it indexed.
    assert all(item.ok for item in verify_pack(load_pack(root)))


def test_generate_manifest_records_a_real_digest(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = tmp_path / "weights"
    pack = generate_manifest(root=root, checkpoint=checkpoint, include_policies=False)
    entry = pack.get("fayd-llm")
    assert entry is not None
    assert entry.digest == file_digest(checkpoint)
    assert entry.architecture.get("n_layer") == 3


# ------------------------------------------------------------------- shipped


def test_the_repository_ships_a_verifiable_pack():
    """The pack the repository carries must be present and verify cleanly."""
    pack = load_pack(weights_pack_dir())
    assert pack.available, pack.detail
    assert pack.get("fayd-llm") is not None
    verifications = verify_pack(pack)
    assert verifications
    assert all(item.ok for item in verifications), [
        item.describe() for item in verifications if not item.ok
    ]


def test_the_shipped_language_entry_is_resumable():
    pack = load_pack(weights_pack_dir())
    entry = pack.get("fayd-llm")
    assert entry is not None
    assert entry.kind == "language-model"
    assert entry.resumable
    assert entry.continue_command == "python -m cli llm --resume"


# ---------------------------------------------------------------------- CLI


def test_cli_status_and_list(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["status", "--root", str(root)]) == 0
        assert "verify cleanly" in capsys.readouterr().out
        assert cli_main(["list", "--root", str(root)]) == 0
        assert "fayd-llm" in capsys.readouterr().out


def test_cli_verify_and_plan(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    assert cli_main(["verify", "fayd-llm", "--root", str(root)]) == 0
    assert "[OK]" in capsys.readouterr().out
    assert cli_main(["plan", "fayd-llm", "--root", str(root)]) == 0
    assert "--resume" in capsys.readouterr().out


def test_cli_verify_refuses_a_tampered_file(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    assert cli_main(["verify", "fayd-llm", "--root", str(root)]) == 4
    assert "REFUSED" in capsys.readouterr().out


def test_cli_install_and_unknown_key(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["install", "fayd-llm", "--root", str(root)]) == 0
        assert "installed" in capsys.readouterr().out
        assert cli_main(["install", "nope", "--root", str(root)]) == 4


def test_cli_json_output(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    assert cli_main(["status", "--root", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pack"]["available"] is True
    assert payload["ready"] == 1


def test_cli_generate(tmp_path, capsys):
    from cli.weights import main as cli_main

    # Generating against the checkout packs the committed weights it finds.
    root = tmp_path / "weights"
    assert cli_main(["generate", "--root", str(root)]) == 0
    assert (root / MANIFEST_NAME).exists()
    assert cli_main(["list", "--root", str(root)]) == 0
    assert "fayd-llm" in capsys.readouterr().out


def test_cli_status_reports_a_missing_pack(tmp_path, capsys):
    from cli.weights import main as cli_main

    assert cli_main(["status", "--root", str(tmp_path / "absent")]) == 3


# -------------------------------------------------------- continue (Phase 3)


def test_cli_continue_installs_then_names_the_trainer(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["continue", "fayd-llm", "--root", str(root)]) == 0
        out = capsys.readouterr().out
        assert "installed" in out
        assert "--resume" in out
        destination = tmp_path / "data" / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"
        assert destination.exists()


def test_cli_continue_json_names_the_command(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["continue", "fayd-llm", "--root", str(root), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["key"] == "fayd-llm"
        assert payload["command"] == "python -m cli llm --resume"
        assert payload["ran"] is False


def test_cli_continue_refuses_an_unknown_key(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    assert cli_main(["continue", "nope", "--root", str(root)]) == 3


def test_cli_continue_refuses_a_tampered_entry(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["continue", "fayd-llm", "--root", str(root)]) == 4


def test_select_keys_filters_by_kind(tmp_path):
    from core.weights_pack import select_keys

    checkpoint = _train_checkpoint(tmp_path / "run")
    policy = _train_policy(tmp_path / "artifacts", "complete-well")
    root = _write_pack(
        tmp_path / "weights",
        [
            _language_row(checkpoint),
            {
                "key": "policy-complete-well",
                "title": "Policy: complete well",
                "kind": "policy",
                "path": str(policy),
                "digest": file_digest(policy),
                "task": "complete-well",
                "continue_command": "python -m cli petroleum --resume",
            },
        ],
    )
    pack = load_pack(root)
    assert select_keys(pack, "language-model") == ("fayd-llm",)
    assert select_keys(pack, "policy") == ("policy-complete-well",)
    assert select_keys(pack, "model") == ()


def test_llm_trainer_refuses_an_unknown_pack_entry_before_training(capsys):
    """--install-weights installs a *named* entry, refusing an unknown key."""
    from ml.llm.__main__ import main as llm_main

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(Path.cwd() / "nope")}):
        assert llm_main(["--install-weights", "definitely-not-a-key"]) == 3
    assert "No weights-pack entry" in capsys.readouterr().out


def test_petroleum_trainer_installs_the_policy_entries(tmp_path, capsys):
    """--install-weights installs every policy into --save-dir, then trains."""
    from ml import petroleum

    checkpoint = _train_checkpoint(tmp_path / "run")
    policy = _train_policy(tmp_path / "artifacts", "complete-well")
    _write_pack(
        tmp_path / "weights",
        [
            _language_row(checkpoint),
            {
                "key": "policy-complete-well",
                "title": "Policy: complete well",
                "kind": "policy",
                "path": str(policy),
                "digest": file_digest(policy),
                "task": "complete-well",
                "continue_command": "python -m cli petroleum --resume",
            },
        ],
    )

    class _Report:
        def as_dict(self):
            return {"stub": True}

    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}), mock.patch(
        "core.paths.weights_pack_dir", return_value=tmp_path / "weights"
    ), mock.patch.object(petroleum, "train_petroleum", return_value=_Report()):
        assert petroleum.main(["--install-weights", "--json"]) == 0
        installed = (
            tmp_path / "data" / "ml" / "artifacts" / "petroleum" / "policy_complete-well.npz"
        )
        assert installed.exists()


def test_the_base_trainer_exposes_the_install_weights_flag():
    import contextlib
    import io

    from ml.__main__ import main as ml_main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            ml_main(["--help"])
        except SystemExit:
            pass
    assert "--install-weights" in buffer.getvalue()


# ------------------------------------------------------ provenance (Phase 4)


def test_generated_entries_record_provenance(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    pack = generate_manifest(
        root=tmp_path / "weights", checkpoint=checkpoint, include_policies=False
    )
    entry = pack.get("fayd-llm")
    assert entry is not None
    assert entry.origin.startswith("the project's own trainer")
    assert entry.provenance["digest_algorithm"] == "sha256"
    assert "MIT" in entry.provenance["license"]
    line = entry.provenance_line()
    assert "producer:" in line and "license:" in line
    # The generated manifest on disk keeps the provenance.
    on_disk = load_pack(tmp_path / "weights").get("fayd-llm")
    assert on_disk is not None and on_disk.provenance["producer"] == entry.provenance["producer"]


def test_provenance_line_leaves_absent_fields_out():
    bare = PackEntry(key="k", title="t", kind="policy", path=Path("/x"))
    assert bare.provenance_line() == ""
    assert bare.origin == ""
    partial = PackEntry(
        key="k", title="t", kind="policy", path=Path("/x"), provenance={"seed": 7}
    )
    assert partial.provenance_line() == "seed: 7"
    assert partial.origin == ""


def test_cli_show_prints_the_provenance(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    assert cli_main(["show", "fayd-llm", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "language-model" in out
    assert cli_main(["show", "nope", "--root", str(root)]) == 3


# -------------------------------------------------------- integrity (Phase 5)


def test_integrity_is_incomplete_until_installed(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        before = integrity_report(root)
        assert before.available
        assert before.total == 1 and before.verified == 1
        assert not before.complete
        install_pack(load_pack(root), keys=("fayd-llm",))
        after = integrity_report(root)
        assert after.complete and after.present == 1


def test_integrity_reports_a_tampered_entry(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
        report = integrity_report(root)
        assert not report.complete
        assert [item.entry.key for item in report.failed] == ["fayd-llm"]


def test_integrity_never_calls_a_partial_check_complete(tmp_path):
    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        install_pack(load_pack(root), keys=("fayd-llm",))
        partial = integrity_report(pack=load_pack(root), verifications=())
        assert not partial.complete
        assert partial.unverified_keys == ("fayd-llm",)


def test_integrity_reports_a_missing_pack(tmp_path):
    report = integrity_report(tmp_path / "absent")
    assert not report.available
    assert not report.complete
    assert report.describe()


def test_cli_integrity_reports_and_exits(tmp_path, capsys):
    from cli.weights import main as cli_main

    checkpoint = _train_checkpoint(tmp_path / "run")
    root = _write_pack(tmp_path / "weights", [_language_row(checkpoint)])
    with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(tmp_path / "data")}):
        assert cli_main(["integrity", "--root", str(root)]) == 4
        assert "incomplete" in capsys.readouterr().out
        install_pack(load_pack(root), keys=("fayd-llm",))
        assert cli_main(["integrity", "--root", str(root), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["complete"] is True and payload["total"] == 1


# --------------------------------------------------------------------- menu


def test_menu_and_dispatch_wire_the_pack():
    from cli.__main__ import _COMMANDS
    from cli.menu import MENU, run_weights

    assert _COMMANDS["weights"] is run_weights
    assert _COMMANDS["weights-pack"] is run_weights
    assert _COMMANDS["checkpoints"].__name__ == "run_checkpoints"
    labels = [label for label, _action, _args in MENU]
    assert any("Weights pack" in label for label in labels)


# --------------------------------------------------------------------- gate


def test_the_gate_passes_on_the_real_engine():
    assert weights_pack_accuracy() == 1.0


def test_the_gate_has_negative_controls():
    names = {item.name for item in pack_gate_judgements()}
    assert "a_matching_digest_verifies_and_an_altered_file_is_refused" in names
    assert "a_mismatched_architecture_is_refused" in names
    assert "an_unknown_key_is_refused_and_an_existing_file_is_backed_up" in names


def test_weights_pack_error_is_a_runtime_error():
    assert issubclass(WeightsPackError, RuntimeError)


def test_pack_entry_defaults_are_honest():
    entry = PackEntry(key="k", title="t", kind="language-model", path=Path("/x"))
    assert entry.digest == ""
    assert entry.architecture == {}
    assert entry.resumable is True
    primitive = entry.to_primitive()
    assert primitive["provenance"] == {}


def test_pack_status_of_a_missing_pack_is_reported(tmp_path):
    status = pack_status(tmp_path / "absent")
    assert not status.pack.available
    assert status.ready == 0 and status.present == 0
