"""The Model Manager CLI, driven through its own ``main``.

The CLI is the headless twin of the window, so these tests assert the two things
that matter: it drives the *same* manager the GUI does (nothing reimplemented),
and it reports honestly -- a missing model is a stated reason and a real exit
code, never a traceback. A ``FAYD_DATA_DIR``-style isolation is used so a test run
never writes into the developer's own data folder.
"""

from __future__ import annotations

import json
import struct

import pytest
from cli import models as models_cli


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Keep every registry write inside the test's own folder."""
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path / "data"))
    yield


def _gguf_key(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw


def _gguf_uint32(value: int) -> bytes:
    return struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, *, name="CLI Model", architecture="llama", context=4096):
    entries = [
        ("general.name", _gguf_string(name)),
        ("general.architecture", _gguf_string(architecture)),
        ("general.file_type", _gguf_uint32(15)),
        (f"{architecture}.context_length", _gguf_uint32(context)),
    ]
    body = b"".join(_gguf_key(key) + payload for key, payload in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, len(entries)) + body)
    return path


def run(capsys, *argv):
    code = models_cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------- list


def test_list_with_an_empty_registry_still_lists_the_catalog(capsys):
    code, out, _ = run(capsys, "list")
    assert code == 0
    # The shipped catalog is what the program knows about on a fresh machine.
    assert "not-downloaded" in out or "ready" in out


def test_list_registered_only_is_empty_on_a_fresh_machine(capsys):
    code, out, _ = run(capsys, "list", "--registered-only")
    assert code == 0
    assert "No models" in out


def test_list_json_is_machine_readable(capsys):
    code, out, _ = run(capsys, "list", "--json")
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload
    for row in payload:
        assert {"model_id", "display_name", "state", "ready", "runtime"} <= set(row)


def test_registering_then_listing_shows_the_model(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    code, out, _ = run(capsys, "register", str(target), "--name", "Mine")
    assert code == 0
    assert "Registered Mine" in out

    code, out, _ = run(capsys, "list", "--registered-only")
    assert code == 0
    assert "Mine" in out
    assert "ready" in out


def test_registering_a_missing_file_fails_with_a_reason(tmp_path, capsys):
    code, _, err = run(capsys, "register", str(tmp_path / "nope.gguf"))
    assert code == 3
    assert "error:" in err


def test_registering_a_non_model_file_fails(tmp_path, capsys):
    target = tmp_path / "notes.txt"
    target.write_text("just words", encoding="utf-8")
    code, _, err = run(capsys, "register", str(target))
    assert code == 3
    assert "error:" in err


# ------------------------------------------------------------------- status


def test_status_reports_the_subsystem(capsys):
    code, out, _ = run(capsys, "status")
    assert code == 0
    assert "Model Manager" in out


def test_status_json_carries_the_counts(capsys):
    code, out, _ = run(capsys, "status", "--json")
    assert code == 0
    payload = json.loads(out)
    assert {"registered", "catalog", "ready", "runtimes"} <= set(payload)
    assert isinstance(payload["runtimes"], str)


# ----------------------------------------------------------------- runtimes


def test_runtimes_reports_each_runtime(capsys):
    code, out, _ = run(capsys, "runtimes")
    assert code == 0
    assert "llama.cpp" in out


def test_runtimes_json_names_the_capabilities(capsys):
    code, out, _ = run(capsys, "runtimes", "--json")
    assert code == 0
    payload = json.loads(out)
    names = {row["runtime"] for row in payload}
    assert "llama.cpp" in names
    assert "fayd-inprocess" in names
    for row in payload:
        assert "gpu_offload" in row


# ---------------------------------------------------------------- discovery


def test_discover_finds_a_file_in_an_explicit_root(tmp_path, capsys):
    write_gguf(tmp_path / "scan" / "found.gguf")
    code, out, _ = run(capsys, "discover", "--root", str(tmp_path / "scan"))
    assert code == 0
    assert "found.gguf" in out


def test_discover_json_reports_the_roots_and_files(tmp_path, capsys):
    write_gguf(tmp_path / "scan" / "found.gguf")
    code, out, _ = run(capsys, "discover", "--root", str(tmp_path / "scan"), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["files"]
    assert payload["files"][0]["format"] == "gguf"


def test_discover_on_an_empty_folder_says_so(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, _ = run(capsys, "discover", "--root", str(empty))
    assert code == 0
    assert "No model files found" in out


# -------------------------------------------------------------- show/inspect


def test_show_prints_the_model(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf", name="Shown Model")
    run(capsys, "register", str(target))
    code, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "show", model_id)
    assert code == 0
    assert "Shown Model" in out
    assert model_id in out


def test_show_json_is_machine_readable(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target), "--name", "Readable")
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "show", model_id, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["display_name"] == "Readable"
    assert payload["ready"] is True


def test_show_an_unknown_model_exits_three(capsys):
    code, _, err = run(capsys, "show", "not-a-model")
    assert code == 3
    assert "error:" in err


def test_inspect_re_reads_the_file(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf", name="Inspected")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "inspect", model_id)
    assert code == 0
    assert "Inspected" in out
    assert "usable:     True" in out


def test_inspect_json_reports_integrity(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "inspect", model_id, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["usable"] is True
    assert payload["format"] == "gguf"


def test_inspect_of_a_vanished_file_exits_three(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]
    target.unlink()

    code, _, _ = run(capsys, "inspect", model_id)
    assert code == 3


# -------------------------------------------------------------------- check


def test_check_reports_the_memory_estimate_is_an_estimate(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "check", model_id)
    # llama.cpp is not installed in CI, so the honest answer is "not here".
    assert code in (0, 3)
    # The estimate is named as an estimate, never as a measurement.
    assert "estimate" in out
    assert "GPU offload" in out


def test_check_of_a_vanished_file_exits_three(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]
    target.unlink()

    code, out, _ = run(capsys, "check", model_id)
    assert code == 3
    assert "cannot run" in out


# -------------------------------------------------------------------- load


def test_load_without_the_runtime_fails_honestly(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "load", model_id)
    assert code == 3
    assert "could not load" in out or "not installed" in out


def test_unload_with_nothing_loaded_is_success(capsys):
    code, out, _ = run(capsys, "unload")
    assert code == 0
    assert out.strip()


# ------------------------------------------------------------------ settings


def test_settings_show_the_defaults(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "settings", model_id)
    assert code == 0
    assert "temperature" in out


def test_settings_set_persists_a_value(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "settings", model_id, "--set", "temperature=0.2")
    assert code == 0
    assert "Saved" in out

    code, out, _ = run(capsys, "settings", model_id, "--json")
    assert code == 0
    assert json.loads(out)["temperature"] == pytest.approx(0.2)


def test_settings_set_rejects_an_out_of_range_value(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, _, err = run(capsys, "settings", model_id, "--set", "temperature=99")
    assert code == 2
    assert "error:" in err


def test_settings_set_needs_key_equals_value(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, _, err = run(capsys, "settings", model_id, "--set", "temperature")
    assert code == 2
    assert "key=value" in err


def test_settings_reset_returns_to_the_defaults(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]
    run(capsys, "settings", model_id, "--set", "temperature=0.2")

    code, out, _ = run(capsys, "settings", model_id, "--reset")
    assert code == 0
    assert "Reset" in out
    _, out, _ = run(capsys, "settings", model_id, "--json")
    assert json.loads(out)["temperature"] == pytest.approx(0.7)


# ------------------------------------------------------------------- update


def test_update_changes_a_display_name(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target), "--name", "Before")
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "update", model_id, "--name", "After")
    assert code == 0
    assert "After" in out
    _, out, _ = run(capsys, "show", model_id)
    assert "After" in out


def test_update_with_nothing_to_change_exits_two(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, _, err = run(capsys, "update", model_id)
    assert code == 2
    assert "nothing to update" in err


def test_update_with_a_blank_name_is_refused(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, _, err = run(capsys, "update", model_id, "--name", "   ")
    assert code == 3
    assert "error:" in err


# ------------------------------------------------------------------- remove


def test_remove_without_confirmation_exits_four(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, _, err = run(capsys, "remove", model_id)
    assert code == 4
    assert "refused" in err
    # The record is still there.
    _, out, _ = run(capsys, "list", "--registered-only", "--json")
    assert json.loads(out)


def test_remove_with_confirmation_forgets_the_record_and_keeps_the_file(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    _, listed, _ = run(capsys, "list", "--registered-only", "--json")
    model_id = json.loads(listed)[0]["model_id"]

    code, out, _ = run(capsys, "remove", model_id, "--yes")
    assert code == 0
    assert "not touched" in out
    assert target.is_file()
    _, out, _ = run(capsys, "list", "--registered-only", "--json")
    assert json.loads(out) == []


def test_remove_of_an_unknown_model_exits_three(capsys):
    code, _, err = run(capsys, "remove", "nope", "--yes")
    assert code == 3
    assert "error:" in err


# ---------------------------------------------------------------------- log


def test_log_records_a_registration(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target), "--name", "Logged")

    code, out, _ = run(capsys, "log")
    assert code == 0
    assert "Logged" in out


def test_log_json_is_machine_readable(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target))
    code, out, _ = run(capsys, "log", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload
    assert {"at", "operation", "ok", "message"} <= set(payload[0])


def test_log_is_empty_on_a_fresh_machine(capsys):
    code, out, _ = run(capsys, "log")
    assert code == 0
    assert "No model operations" in out


def test_log_limit_is_honoured(tmp_path, capsys):
    for index in range(5):
        write_gguf(tmp_path / f"m{index}.gguf")
        run(capsys, "register", str(tmp_path / f"m{index}.gguf"), "--name", f"M{index}")
    code, out, _ = run(capsys, "log", "--limit", "2", "--json")
    assert code == 0
    assert len(json.loads(out)) == 2


# ---------------------------------------------------------------- robustness


def test_the_cli_never_prints_a_traceback_on_a_bad_command(capsys):
    with pytest.raises(SystemExit) as caught:
        models_cli.main(["not-a-command"])
    assert caught.value.code == 2


def test_a_register_of_a_directory_is_refused(tmp_path, capsys):
    folder = tmp_path / "a-folder"
    folder.mkdir()
    code, _, err = run(capsys, "register", str(folder))
    assert code == 3
    assert "error:" in err


def test_import_custom_with_nothing_to_import_says_so(capsys):
    code, out, _ = run(capsys, "import-custom")
    assert code == 0
    assert "No hand-added models" in out


def test_the_cli_is_registered_in_the_dispatcher():
    from cli import __main__ as cli_main

    assert "models" in cli_main._COMMANDS
    assert "model-manager" in cli_main._COMMANDS


def test_the_menu_offers_the_model_manager():
    from cli.menu import MENU, run_models

    labels = [label for label, _action, _args in MENU]
    assert any("Model Manager" in label for label in labels)
    entries = [entry for entry in MENU if entry[1] is run_models]
    assert entries


def test_select_lists_the_known_tasks(capsys):
    code = models_cli.main(["select"])
    out = capsys.readouterr().out
    assert code == 0
    assert "petroleum" in out


def test_select_of_an_unknown_task_exits_two(capsys):
    code = models_cli.main(["select", "no-such-task"])
    assert code == 2
    assert "Unknown task" in capsys.readouterr().err


def test_select_reports_when_nothing_can_run_the_task(capsys):
    code = models_cli.main(["select", "petroleum"])
    assert code == 3
    assert "No model can run" in capsys.readouterr().out


def test_select_json_is_machine_readable(capsys):
    code = models_cli.main(["select", "petroleum", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"] == "petroleum"
    assert payload["models"] == []


# ------------------------------------------------------- search + dashboard


def test_search_matches_a_registered_model(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target), "--name", "Findable")
    code, out, _ = run(capsys, "search", "findable")
    assert code == 0
    assert "Findable" in out


def test_search_with_no_match_exits_three(capsys):
    code, out, _ = run(capsys, "search", "no-such-model-name")
    assert code == 3
    assert "No models matched" in out


def test_search_json_is_machine_readable(tmp_path, capsys):
    target = write_gguf(tmp_path / "m.gguf")
    run(capsys, "register", str(target), "--name", "Findable")
    code, out, _ = run(capsys, "search", "findable", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["query"] == "findable"
    assert payload["models"]


def test_search_filters_by_state(capsys):
    code, out, _ = run(capsys, "search", "--state", "ready")
    # A fresh machine has nothing ready, so this is the honest empty answer.
    assert code == 3
    assert "No models matched" in out


def test_active_reports_nothing_loaded_on_a_fresh_machine(capsys):
    code, out, _ = run(capsys, "active")
    assert code == 0
    assert "No model is loaded" in out


def test_active_json_is_machine_readable(capsys):
    code, out, _ = run(capsys, "active", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["models"] == []


def test_diagnostics_reports_the_machine_and_the_runtimes(capsys):
    code, out, _ = run(capsys, "diagnostics")
    assert code == 0
    assert "This machine" in out
    assert "Runtimes" in out
    assert "llama.cpp" in out


def test_diagnostics_json_is_machine_readable(capsys):
    code, out, _ = run(capsys, "diagnostics", "--json")
    assert code == 0
    payload = json.loads(out)
    assert "resources" in payload
    assert "runtimes" in payload
    assert "active" in payload
    assert isinstance(payload["status"]["registered"], int)


def test_the_menu_offers_active_and_diagnostics():
    from cli.menu import MENU

    labels = [label for label, _action, _args in MENU]
    assert any("loaded right now" in label for label in labels)
    assert any("recent errors" in label for label in labels)
