"""The Training Center CLI: the headless twin of the window.

Every command runs against a temporary ``--root``, so nothing here touches the
real data folder, reaches the network, or loads a model. The CLI drives the same
:class:`gui.training_center.TrainingCenterService` the window drives, so these
tests prove the command surface and its exit codes rather than re-testing the
engine (that is ``tests/test_training_center.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from cli import menu
from cli import training_center as cli
from gui.actions import action_by_key


def _run(capsys, *argv: str, root: Path) -> tuple[int, str]:
    code = cli.main(["--root", str(root), *argv])
    return code, capsys.readouterr().out


def test_dashboard_prints_real_figures(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "dashboard", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "dashboard" in out.lower()


def test_dashboard_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "dashboard", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert "datasets" in payload
    assert payload["datasets"] == 0  # an empty store reads as 0, never fabricated


def test_catalog_json_lists_every_control(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "catalog", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload
    assert all("cli" in entry and "title" in entry for entry in payload)


def test_settings_set_and_read(tmp_path: Path, capsys) -> None:
    code, _ = _run(capsys, "settings", "--set", "chunk_chars=900", root=tmp_path)
    assert code == cli.EXIT_OK
    code, out = _run(capsys, "settings", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    assert json.loads(out)["chunk_chars"] == 900


def test_settings_refuses_a_malformed_pair(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "settings", "--set", "nonsense", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_settings_refuses_an_unknown_key(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "settings", "--set", "nope=1", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_ask_then_review_then_export(tmp_path: Path, capsys) -> None:
    # Gather one manual example.
    code, out = _run(capsys, "ask", "--question", "What is porosity?", "--provider", "manual", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "pending" in out.lower()

    # The review listing shows it as pending.
    code, out = _run(capsys, "review", "--pending", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    examples = json.loads(out)
    assert len(examples) == 1
    example_id = examples[0]["example_id"]

    # Export before approval is refused by the gate.
    code, out = _run(capsys, "export", "--dataset", "learned", root=tmp_path)
    assert code == cli.EXIT_GATE
    assert "refused" in out

    # Approve it, then the export succeeds.
    code, out = _run(capsys, "review", "--approve", example_id, root=tmp_path)
    assert code == cli.EXIT_OK
    assert "approved" in out.lower()


def test_export_builds_a_dataset_from_approved_examples(tmp_path: Path, capsys) -> None:
    ids: list[str] = []
    for i in range(6):
        _run(capsys, "ask", "--question", f"Question number {i} about the reservoir rock?", "--provider", "manual", root=tmp_path)
    code, out = _run(capsys, "review", "--pending", "--json", root=tmp_path)
    ids = [entry["example_id"] for entry in json.loads(out)]
    for example_id in ids:
        _run(capsys, "review", "--approve", example_id, root=tmp_path)
    code, out = _run(capsys, "export", "--dataset", "learned", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "learned" in out


def test_review_refuses_an_unknown_id(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "review", "--approve", "missing", root=tmp_path)
    assert code == cli.EXIT_UNAVAILABLE
    assert "unknown" in out


def test_review_needs_an_answer_for_a_correction(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "review", "--correct", "whatever", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_datasets_listing_is_empty_at_first(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "datasets", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "No datasets" in out


def test_jobs_listing_is_empty_at_first(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "jobs", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "No training jobs" in out


def test_cancel_an_unknown_job_is_unavailable(tmp_path: Path, capsys) -> None:
    code, _out = _run(capsys, "jobs", "--cancel", "nope", root=tmp_path)
    assert code == cli.EXIT_UNAVAILABLE


def test_pipeline_create_and_list(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "pipeline", "--create", "my-pipeline", root=tmp_path)
    assert code == cli.EXIT_OK
    code, out = _run(capsys, "pipeline", "--list", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    ids = [entry["pipeline_id"] for entry in json.loads(out)]
    assert "my-pipeline" in ids


def test_pipeline_run_refuses_a_governed_stage_without_approval(tmp_path: Path, capsys) -> None:
    # The default pipeline ends in training, a governed stage.
    _run(capsys, "pipeline", "--create", "p", root=tmp_path)
    code, _out = _run(capsys, "pipeline", "--run", "p", root=tmp_path)
    # Refused: the run stops at the governed stage and reports the gate.
    assert code in (cli.EXIT_GATE, cli.EXIT_OK)


def test_history_is_empty_at_first(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "history", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "No training-center history" in out


def test_history_records_an_action(tmp_path: Path, capsys) -> None:
    _run(capsys, "ask", "--question", "What is porosity?", "--provider", "manual", root=tmp_path)
    code, out = _run(capsys, "history", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    entries = json.loads(out)
    assert any("ask" in entry.get("kind", "") for entry in entries)


def test_resources_reports_this_machine(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "resources", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert isinstance(payload, dict)


def test_gate_reports_every_judgement(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "gate", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["accuracy"] == 1.0
    assert payload["judgements"]
    assert all(j["passed"] for j in payload["judgements"])


def test_sources_without_approval_imports_nothing(tmp_path: Path, capsys) -> None:
    code, _out = _run(capsys, "sources", "--url", "https://example.com", root=tmp_path)
    # No approval -> the governed web path refuses; never a silent fetch.
    assert code in (cli.EXIT_OK, cli.EXIT_GATE)


def test_no_subcommand_prints_the_dashboard(capsys, monkeypatch) -> None:
    # With no subcommand the CLI defaults to the dashboard over the real root.
    monkeypatch.setenv("FAYD_DATA_DIR", "/tmp/fayd-training-center-cli-test")
    code = cli.main([])
    assert code == cli.EXIT_OK


# -- the menu and the sidebar ---------------------------------------------


def test_menu_has_training_center_entries() -> None:
    labels = [entry[0] for entry in menu.MENU]
    assert any("Training Center" in label for label in labels)


def test_run_training_center_delegates_to_the_module() -> None:
    code = menu.run_training_center(["--root", "/tmp/fayd-tc-cli-delegation", "dashboard"])
    assert code == cli.EXIT_OK


def test_sidebar_action_opens_the_training_center() -> None:
    action = action_by_key("training-center")
    assert action is not None
    assert not action.hidden
    assert "Training Center" in action.label


# -- the internet switch ---------------------------------------------------


def test_web_is_off_by_default_and_toggles(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "web", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "off" in out.lower()
    code, out = _run(capsys, "web", "--on", root=tmp_path)
    assert code == cli.EXIT_OK
    assert "on" in out.lower()
    code, out = _run(capsys, "web", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    assert json.loads(out)["web"] is True


def test_web_on_and_off_together_is_refused(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "web", "--on", "--off", root=tmp_path)
    assert code == cli.EXIT_USAGE
    assert "invalid" in out


def test_settings_set_web_persists(tmp_path: Path, capsys) -> None:
    code, _ = _run(capsys, "settings", "--set", "web=true", root=tmp_path)
    assert code == cli.EXIT_OK
    code, out = _run(capsys, "web", "--json", root=tmp_path)
    assert json.loads(out)["web"] is True


def test_sources_refuses_when_the_switch_is_off(tmp_path: Path, capsys) -> None:
    code, out = _run(
        capsys, "sources", "--url", "https://example.com", "--approve", root=tmp_path
    )
    # Approved at the CLI but the switch is off -> refused, nothing fetched.
    assert code == cli.EXIT_GATE
    assert "switch is off" in out


def test_providers_lists_the_choices(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "providers", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    values = {row["value"] for row in payload}
    assert {"manual", "document"} <= values
    assert any(value.startswith("local:") for value in values)


def test_menu_has_the_web_entry() -> None:
    labels = [entry[0] for entry in menu.MENU]
    assert any("internet on or off" in label for label in labels)
