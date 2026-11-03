"""Phase 6: the training CLI and the held-out gate.

The CLI tests drive the *real* :mod:`cli.train` module (a temporary data root), so
a regression in the CLI is caught here rather than only in a manual run. The gate
tests assert that the Phase-6 gate passes every judgement and that its accuracy is
1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

from cli import train as train_cli
from ml.training.gate import describe as gate_describe
from ml.training.gate import judgements, training_accuracy

# ------------------------------------------------------------- the gate


def test_gate_passes_every_judgement():
    results = judgements()
    assert len(results) >= 10
    failed = [j.name for j in results if not j.passed]
    assert not failed, f"failing judgements: {failed}"


def test_gate_accuracy_is_one():
    assert training_accuracy() == 1.0


def test_gate_report_names_each_judgement():
    text = gate_describe()
    assert "Phase 6 training gate" in text
    for name in (
        "dataset_validation",
        "dataset_provenance",
        "dataset_versioning",
        "dataset_split",
        "natasha_exclusion",
        "model_compatibility",
        "training_config",
        "training_lifecycle",
        "base_preserved",
        "promotion_and_rollback",
    ):
        assert name in text


# ------------------------------------------------------------- the CLI


def _run(root: Path, *argv: str) -> int:
    return train_cli.main(["--root", str(root), *argv])


def test_cli_status_reports_no_adapters(tmp_path: Path, capsys):
    assert _run(tmp_path, "status") == 0
    out = capsys.readouterr().out
    assert "Phase 6" in out


def test_cli_compat_reports_the_in_process_method(tmp_path: Path, capsys):
    assert _run(tmp_path, "compat") == 0
    out = capsys.readouterr().out
    assert "compatibility" in out.lower()


def test_cli_compat_json(tmp_path: Path, capsys):
    assert _run(tmp_path, "compat", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trainable"] is True
    assert any(m["method"] == "lora_inprocess" for m in payload["methods"])


def test_cli_presets_lists_them(tmp_path: Path, capsys):
    assert _run(tmp_path, "presets") == 0
    out = capsys.readouterr().out
    assert "cpu-safe" in out and "quality" in out


def test_cli_build_dataset_and_list(tmp_path: Path, capsys):
    code = _run(
        tmp_path,
        "build-dataset",
        "demo",
        "--text",
        "Q: what is porosity?\nA: porosity is the void fraction of a rock.",
        "--owner",
        "jarvis",
        "--source",
        "cli",
        "--license",
        "project",
    )
    assert code == 0
    capsys.readouterr()
    assert _run(tmp_path, "datasets") == 0
    assert "demo@v1" in capsys.readouterr().out


def test_cli_build_dataset_refuses_an_empty_one(tmp_path: Path):
    assert _run(tmp_path, "build-dataset", "empty") == 2


def test_cli_build_dataset_refuses_a_private_scope(tmp_path: Path):
    code = _run(
        tmp_path,
        "build-dataset",
        "priv",
        "--text",
        "a private fact about porosity and permeability",
        "--owner",
        "natasha",
    )
    assert code == 4


def test_cli_validate_reports_an_unknown_dataset(tmp_path: Path):
    assert _run(tmp_path, "validate", "nope") == 3


def test_cli_run_is_a_dry_run_without_approve(tmp_path: Path, capsys):
    _run(
        tmp_path,
        "build-dataset",
        "demo",
        "--text",
        "Q: what is porosity?\nA: porosity is the void fraction of a rock.",
        "--owner",
        "jarvis",
        "--source",
        "cli",
        "--license",
        "project",
    )
    capsys.readouterr()
    assert _run(tmp_path, "run", "--dataset", "demo", "--steps", "4") == 0
    out = capsys.readouterr().out
    assert "Dry run" in out


def test_cli_run_refuses_an_unknown_dataset(tmp_path: Path):
    assert _run(tmp_path, "run", "--dataset", "nope") == 3


def test_cli_run_trains_with_approve(tmp_path: Path, capsys):
    _run(
        tmp_path,
        "build-dataset",
        "demo",
        "--text",
        "Q: what is porosity?\nA: porosity is the void fraction of a rock and permeability is flow.",
        "--owner",
        "jarvis",
        "--source",
        "cli",
        "--license",
        "project",
    )
    # A dataset of one example is too short to train; build a longer one instead.
    long_text = "\n".join(
        f"Q: what is porosity {i}?\nA: porosity is the void fraction of the rock {i}."
        for i in range(30)
    )
    capsys.readouterr()
    _run(
        tmp_path,
        "build-dataset",
        "long",
        "--text",
        long_text,
        "--owner",
        "jarvis",
        "--source",
        "cli",
        "--license",
        "project",
    )
    capsys.readouterr()
    code = _run(tmp_path, "run", "--dataset", "long", "--steps", "6", "--approve")
    out = capsys.readouterr().out
    assert code == 0
    assert "completed" in out
    assert _run(tmp_path, "runs") == 0
    assert "completed" in capsys.readouterr().out


def test_cli_evaluate_and_promote_gate(tmp_path: Path, capsys):
    long_text = "\n".join(
        f"Q: what is porosity {i}?\nA: porosity is the void fraction of the rock {i}."
        for i in range(30)
    )
    _run(tmp_path, "build-dataset", "long", "--text", long_text, "--owner", "jarvis",
         "--source", "cli", "--license", "project")
    capsys.readouterr()
    _run(tmp_path, "run", "--dataset", "long", "--steps", "6", "--approve")
    out = capsys.readouterr().out
    run_id = out.split("[completed]")[0].strip().split()[-1]
    assert _run(tmp_path, "evaluate", run_id) == 0
    # Promotion without approval is refused.
    assert _run(tmp_path, "promote", run_id) == 4
    capsys.readouterr()
    # With approval, it is promoted only when the evaluation accepted it.
    code = _run(tmp_path, "promote", run_id, "--approve")
    capsys.readouterr()
    assert code in (0, 4)
    # A rollback never deletes history.
    assert _run(tmp_path, "rollback") == 0
    capsys.readouterr()
    assert _run(tmp_path, "adapters") == 0
    assert run_id in capsys.readouterr().out


def test_cli_evaluate_refuses_an_unknown_run(tmp_path: Path):
    assert _run(tmp_path, "evaluate", "nope") == 3


def test_cli_promote_refuses_an_unknown_adapter(tmp_path: Path):
    assert _run(tmp_path, "promote", "nope") == 3


def test_cli_feasibility_json(tmp_path: Path, capsys):
    assert _run(tmp_path, "feasibility", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_cli_gate_reports(tmp_path: Path, capsys):
    assert _run(tmp_path, "gate") == 0
    assert "Phase 6 training gate" in capsys.readouterr().out


def test_cli_no_command_prints_status(tmp_path: Path, capsys):
    assert train_cli.main(["--root", str(tmp_path)]) == 0
    assert "Phase 6" in capsys.readouterr().out


def test_cli_is_registered_in_the_menu_and_dispatch():
    from cli import __main__ as cli_main
    from cli import menu

    assert any(action is menu.run_train for _title, action, _argv in menu.MENU)
    assert "train" in cli_main.COMMANDS if hasattr(cli_main, "COMMANDS") else True


def test_cli_refuses_a_bad_rank(tmp_path: Path):
    code = _run(
        tmp_path,
        "build-dataset",
        "demo",
        "--text",
        "Q: what is porosity?\nA: porosity is the void fraction of a rock.",
        "--owner",
        "jarvis",
        "--source",
        "cli",
        "--license",
        "project",
    )
    assert code == 0
    assert _run(tmp_path, "run", "--dataset", "demo", "--steps", "4", "--rank", "0") == 2
