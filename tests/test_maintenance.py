"""The Maintenance service and CLI: backup/restore, diagnostics, shortcuts.

The service is exercised headless (no display, no model): a temporary data root
holds a small store to back up and restore, and the diagnostics and the shortcut
map are read from the modules that own them.
"""

from __future__ import annotations

from pathlib import Path

from gui.maintenance import (
    STORE_KNOWLEDGE,
    STORE_LEARNING,
    STORE_MEMORY,
    STORE_TITLES,
    MaintenanceService,
)


def _service(tmp_path: Path) -> MaintenanceService:
    return MaintenanceService(tmp_path)


# ------------------------------------------------------------------ statuses


def test_store_statuses_name_every_store() -> None:
    service = MaintenanceService()
    stores = {status.store for status in service.store_statuses()}
    assert stores == {STORE_MEMORY, STORE_LEARNING, STORE_KNOWLEDGE}


def test_an_absent_store_is_reported_absent_not_faked(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for status in service.store_statuses():
        assert status.present is False
        assert "absent" in status.describe()


def test_a_present_store_is_reported_present(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "shard-000.jsonl").write_text("{}\n", encoding="utf-8")
    service = _service(tmp_path)
    status = next(s for s in service.store_statuses() if s.store == STORE_KNOWLEDGE)
    assert status.present is True
    assert "present" in status.describe()


# ------------------------------------------------------------------- backup


def test_backup_copies_a_store_and_reports_a_real_count(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "shard-000.jsonl").write_text("a\n", encoding="utf-8")
    (knowledge / "shard-001.jsonl").write_text("b\n", encoding="utf-8")
    service = _service(tmp_path)
    result = service.backup(STORE_KNOWLEDGE)
    assert result.ok is True
    assert result.records == 2
    assert Path(result.path).is_dir()
    assert (Path(result.path) / "shard-000.jsonl").read_text(encoding="utf-8") == "a\n"


def test_backup_of_an_absent_store_fails_honestly(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.backup(STORE_MEMORY)
    assert result.ok is False
    assert "nothing to back up" in result.message


def test_backup_of_an_unknown_store_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.backup("nonsense")
    assert result.ok is False
    assert "unknown store" in result.message


def test_backup_honours_an_explicit_destination(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "shard-000.jsonl").write_text("a\n", encoding="utf-8")
    target = tmp_path / "my-backup"
    service = _service(tmp_path)
    result = service.backup(STORE_KNOWLEDGE, destination=target)
    assert result.ok is True
    assert Path(result.path) == target
    assert target.is_dir()


# ------------------------------------------------------------------ restore


def test_restore_replaces_the_store_and_keeps_a_safety_copy(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "shard-000.jsonl").write_text("original\n", encoding="utf-8")
    service = _service(tmp_path)
    backup = service.backup(STORE_KNOWLEDGE)
    assert backup.ok is True
    # Change the store, then restore the backup over it.
    (knowledge / "shard-000.jsonl").write_text("changed\n", encoding="utf-8")
    result = service.restore(STORE_KNOWLEDGE, backup.path)
    assert result.ok is True
    assert (knowledge / "shard-000.jsonl").read_text(encoding="utf-8") == "original\n"
    safety = knowledge.with_name(knowledge.name + ".pre-restore")
    assert safety.is_dir()
    assert (safety / "shard-000.jsonl").read_text(encoding="utf-8") == "changed\n"


def test_restore_of_a_missing_backup_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.restore(STORE_KNOWLEDGE, tmp_path / "does-not-exist")
    assert result.ok is False
    assert "does not exist" in result.message


def test_restore_of_an_empty_backup_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty-backup"
    empty.mkdir()
    service = _service(tmp_path)
    result = service.restore(STORE_KNOWLEDGE, empty)
    assert result.ok is False
    assert "empty" in result.message


def test_restore_of_an_unknown_store_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    result = service.restore("nonsense", tmp_path)
    assert result.ok is False
    assert "unknown store" in result.message


# -------------------------------------------------------------- diagnostics


def test_diagnostics_read_the_machine_and_the_stores(tmp_path: Path) -> None:
    service = _service(tmp_path)
    lines = service.diagnostics()
    labels = {line.label for line in lines}
    # The machine's own figures are read (a platform always reports the system).
    assert "system" in labels or "Models" in labels
    # Each store's state is included.
    assert STORE_TITLES[STORE_KNOWLEDGE] in labels


def test_recent_log_is_empty_or_a_tuple(tmp_path: Path) -> None:
    service = _service(tmp_path)
    entries = service.recent_log()
    assert isinstance(entries, tuple)


# --------------------------------------------------------------- shortcuts


def test_shortcuts_are_reported_with_a_scope() -> None:
    service = MaintenanceService()
    shortcuts = service.shortcuts()
    assert shortcuts
    assert all(shortcut.keys and shortcut.action and shortcut.scope for shortcut in shortcuts)


def test_shortcuts_match_the_real_bindings() -> None:
    """Every listed key is one the project really binds somewhere.

    The list is documentation; this keeps it honest by checking each binding's
    keys against the source that installs them.
    """
    app_source = Path("gui/app.py").read_text(encoding="utf-8").lower()
    clipboard_source = Path("gui/clipboard.py").read_text(encoding="utf-8").lower()
    composer_source = Path("gui/chat_composer.py").read_text(encoding="utf-8").lower()
    controls_source = Path("gui/window_controls.py").read_text(encoding="utf-8").lower()
    combined = app_source + clipboard_source + composer_source
    for shortcut in MaintenanceService().shortcuts():
        if shortcut.keys.startswith("Ctrl+"):
            letter = shortcut.keys[len("Ctrl+") :].lower()
            forms = (f"<control-{letter}>", f"<control-key-{letter}>")
            assert any(form in combined for form in forms), (
                f"{shortcut.keys} is listed but not bound"
            )
        if shortcut.keys == "Enter":
            assert "<return>" in composer_source
        if shortcut.keys == "Shift+Enter":
            assert "<shift-return>" in composer_source
        if shortcut.keys == "Escape":
            assert "<escape>" in controls_source


# ------------------------------------------------------------------- CLI


def test_cli_status_prints_the_page(capsys) -> None:
    from cli.maintenance import main

    code = main(["status", "--data-root", "/tmp/fayd-cli-test"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Maintenance" in out
    assert "Stores" in out


def test_cli_stores_json(capsys) -> None:
    import json

    from cli.maintenance import main

    code = main(["stores", "--json", "--data-root", "/tmp/fayd-cli-test"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert {entry["store"] for entry in payload} == {STORE_MEMORY, STORE_LEARNING, STORE_KNOWLEDGE}


def test_cli_backup_and_restore_round_trip(tmp_path: Path, capsys) -> None:
    from cli.maintenance import main

    knowledge = tmp_path / "knowledge_base"
    knowledge.mkdir(parents=True)
    (knowledge / "shard-000.jsonl").write_text("original\n", encoding="utf-8")
    code = main(["backup", STORE_KNOWLEDGE, "--to", str(tmp_path / "bk"), "--data-root", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "bk" / "shard-000.jsonl").exists()
    (knowledge / "shard-000.jsonl").write_text("changed\n", encoding="utf-8")
    code = main(
        ["restore", STORE_KNOWLEDGE, "--from", str(tmp_path / "bk"), "--approve", "--data-root", str(tmp_path)]
    )
    assert code == 0
    assert (knowledge / "shard-000.jsonl").read_text(encoding="utf-8") == "original\n"
    capsys.readouterr()


def test_cli_restore_without_approval_is_refused(tmp_path: Path, capsys) -> None:
    from cli.maintenance import main

    code = main(["restore", STORE_KNOWLEDGE, "--from", str(tmp_path), "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 4
    assert "approve" in out


def test_cli_backup_unknown_store_is_an_argument_error(capsys) -> None:
    # argparse rejects an unknown choice with SystemExit(2).
    import pytest
    from cli.maintenance import main

    with pytest.raises(SystemExit) as excinfo:
        main(["backup", "nonsense"])
    assert excinfo.value.code == 2


def test_cli_shortcuts_json(capsys) -> None:
    import json

    from cli.maintenance import main

    code = main(["shortcuts", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(entry["keys"] == "Ctrl+N" for entry in payload)


def test_maintenance_is_reachable_from_the_cli_dispatch() -> None:
    from cli.__main__ import _COMMANDS

    assert "maintenance" in _COMMANDS
    assert "backup" in _COMMANDS


def test_maintenance_menu_entries_exist() -> None:
    from cli.menu import run_maintenance

    assert run_maintenance(["shortcuts"]) == 0


# ----------------------------------------------------------- memory & knowledge


def test_memory_lines_for_an_unknown_owner_are_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.memory_lines("") == ()
    # A private character's memory is refused by the isolation policy; the view
    # reports it as empty rather than raising into the window.
    assert service.memory_lines("natasha") == ()


def test_memory_summary_is_readable(tmp_path: Path) -> None:
    service = _service(tmp_path)
    summary = service.memory_summary("fayd_al_ruh")
    assert isinstance(summary, str) and summary


def test_memory_lines_read_a_real_fact(tmp_path: Path) -> None:
    from core.character_learning import CharacterLearning

    CharacterLearning("jarvis", root=tmp_path, in_memory=False).add_fact("waterflood supports pressure")
    service = _service(tmp_path)
    texts = [line.text for line in service.memory_lines("jarvis")]
    assert any("waterflood" in text for text in texts)


def test_knowledge_search_blank_query_is_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.knowledge_search("") == ()


def test_knowledge_search_absent_store_is_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.knowledge_search("porosity") == ()


def test_cli_memory_requires_an_owner(capsys) -> None:
    from cli.maintenance import main

    code = main(["memory", ""])
    assert code == 2


def test_cli_memory_json(capsys) -> None:
    import json

    from cli.maintenance import main

    code = main(["memory", "fayd_al_ruh", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["owner"] == "fayd_al_ruh"
    assert "entries" in payload


def test_cli_knowledge_requires_a_query(capsys) -> None:
    from cli.maintenance import main

    code = main(["knowledge", ""])
    assert code == 2


def test_cli_knowledge_empty_store_is_honest(capsys, tmp_path: Path) -> None:
    from cli.maintenance import main

    code = main(["knowledge", "porosity", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Nothing matched" in out
