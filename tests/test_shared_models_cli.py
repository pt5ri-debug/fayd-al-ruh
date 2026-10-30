"""The shared-models CLI: the headless twin of the window's shared-model controls.

It drives the *same* :mod:`core.shared_models` module, so the CLI and the window
can never disagree about what is installed. These tests pin the commands
(``status``, ``models``, ``list``, ``download``, ``download-all``, ``migrate``),
the exit codes, the ``--json`` output, and the fact that a download needs
``--approve``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli import shared_models as cli
from core.shared_models import CATALOG


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# -- status ---------------------------------------------------------------
def test_status_reports_an_empty_folder_and_exits_three(data_root: Path, capsys) -> None:
    assert cli.main(["status"]) == 3
    out = capsys.readouterr().out
    assert "Shared language models" in out
    assert "installed: no" in out


def test_status_reports_a_present_model(data_root: Path, capsys) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / CATALOG["qwen3.5-2b"].filename).write_bytes(b"GGUF")
    assert cli.main(["status"]) == 0
    assert CATALOG["qwen3.5-2b"].filename in capsys.readouterr().out


def test_status_json_is_machine_readable(data_root: Path, capsys) -> None:
    assert cli.main(["status", "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is False
    assert "shared_models" in payload["folder"]


# -- models / list --------------------------------------------------------
def test_models_lists_the_catalog(data_root: Path, capsys) -> None:
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    for key in CATALOG:
        assert key in out


def test_models_json_lists_every_entry(data_root: Path, capsys) -> None:
    assert cli.main(["models", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["key"] for row in rows} == set(CATALOG)


def test_list_reports_no_file_and_exits_three(data_root: Path, capsys) -> None:
    assert cli.main(["list"]) == 3
    assert "No shared model file" in capsys.readouterr().out


def test_list_shows_the_files_present(data_root: Path, capsys) -> None:
    folder = data_root / "shared_models"
    folder.mkdir(parents=True)
    (folder / "a.gguf").write_bytes(b"x")
    assert cli.main(["list"]) == 0
    assert "a.gguf" in capsys.readouterr().out


# -- download -------------------------------------------------------------
def test_download_without_approve_is_refused(data_root: Path, capsys) -> None:
    assert cli.main(["download"]) == 4
    assert "--approve" in capsys.readouterr().out


def test_download_with_approve_fetches(data_root: Path, monkeypatch, capsys) -> None:
    import sys

    class FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
            target = Path(local_dir) / filename
            target.write_bytes(b"GGUF")
            return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    assert cli.main(["download", "--approve", "--model", "qwen3.5-0.8b"]) == 0
    assert "Shared model ready" in capsys.readouterr().out


def test_download_all_without_approve_is_refused(data_root: Path, capsys) -> None:
    assert cli.main(["download-all"]) == 4
    assert "--approve" in capsys.readouterr().out


def test_download_all_fetches_every_model(data_root: Path, monkeypatch, capsys) -> None:
    import sys

    class FakeHub:
        @staticmethod
        def hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
            target = Path(local_dir) / filename
            target.write_bytes(b"GGUF")
            return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    assert cli.main(["download-all", "--approve"]) == 0
    assert "shared model(s) ready" in capsys.readouterr().out


# -- migrate --------------------------------------------------------------
def test_migrate_moves_the_legacy_model(data_root: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(data_root))
    natasha = data_root / "natasha"
    natasha.mkdir()
    legacy = natasha / CATALOG["qwen3-1.7b"].filename
    legacy.write_bytes(b"GGUF")

    assert cli.main(["migrate"]) == 0
    out = capsys.readouterr().out
    assert "Moved 1 model" in out
    assert (data_root / "shared_models" / legacy.name).is_file()


def test_migrate_json_is_machine_readable(data_root: Path, capsys) -> None:
    assert cli.main(["migrate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "moved" in payload
    assert "skipped" in payload


# -- the menu wiring ------------------------------------------------------
def test_the_menu_exposes_shared_models() -> None:
    from cli.menu import MENU, run_shared_models

    labels = [label for label, _action, _argv in MENU]
    assert any("Shared models" in label for label in labels)
    assert callable(run_shared_models)


def test_the_command_is_registered() -> None:
    from cli.__main__ import _COMMANDS

    assert "shared-models" in _COMMANDS
    assert "shared" in _COMMANDS
