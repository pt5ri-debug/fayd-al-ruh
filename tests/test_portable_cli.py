"""The portable menu: command dispatch, validation, and loop control."""

from __future__ import annotations

from collections.abc import Sequence

import cli.menu as menu
from cli.__main__ import main as cli_main


def test_run_choice_rejects_out_of_range_and_non_numeric(capsys) -> None:
    assert menu.run_choice("99") == 1
    assert menu.run_choice("abc") == 1
    out = capsys.readouterr().out
    assert "out of range" in out
    assert "Unrecognised" in out


def test_run_choice_quit_codes() -> None:
    assert menu.run_choice("q") == 0
    assert menu.run_choice("quit") == 0
    assert menu.run_choice("") == 0


def test_run_choice_dispatches_to_action(monkeypatch) -> None:
    calls: list[tuple[Sequence[str], int]] = []

    def fake_action(argv: Sequence[str]) -> int:
        calls.append((argv, 0))
        return 0

    monkeypatch.setattr(menu, "MENU", (("Fake", fake_action, ("--x",)),))
    assert menu.run_choice("1") == 0
    assert calls == [(("--x",), 0)]


def test_menu_loop_stops_on_quit(monkeypatch) -> None:
    answers = iter(["1", "q"])
    # Replace the first action so no real work runs.
    monkeypatch.setattr(menu, "MENU", (("Noop", lambda argv: 0, ()),))
    code = menu.menu_loop(reader=lambda _prompt: next(answers))
    assert code == 0


def test_cli_paths_command(capsys) -> None:
    assert cli_main(["paths"]) == 0
    out = capsys.readouterr().out
    assert "Data folder" in out
    assert "Checkpoint" in out


def test_cli_unknown_command(capsys) -> None:
    assert cli_main(["nope"]) == 1
    assert "Unknown command" in capsys.readouterr().out


def test_run_ml_adds_default_save_dir(monkeypatch, tmp_path) -> None:
    captured: dict[str, Sequence[str]] = {}

    def fake_run_module(name: str, argv: Sequence[str]) -> int:
        captured["name"] = name
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    monkeypatch.setattr(menu, "artifacts_dir", lambda: tmp_path / "artifacts")

    menu.run_ml(())
    assert captured["name"] == "ml.__main__"
    assert "--save-dir" in captured["argv"]
    assert str(tmp_path / "artifacts") in captured["argv"]

    menu.run_ml(["--episodes", "5"])
    assert captured["argv"] == ["--episodes", "5", "--save-dir", str(tmp_path / "artifacts")]


def test_run_ml_respects_explicit_save_dir(monkeypatch, tmp_path) -> None:
    captured: dict[str, Sequence[str]] = {}

    def fake_run_module(name: str, argv: Sequence[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    menu.run_ml(["--save-dir", str(tmp_path / "explicit")])
    assert captured["argv"].count("--save-dir") == 1


def test_missing_dependency_message_for_numpy() -> None:
    error = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    message = menu.missing_dependency_message(error)
    assert "numpy" in message
    assert "pip install -r requirements.txt" in message


def test_missing_dependency_message_for_other_module() -> None:
    error = ModuleNotFoundError("No module named 'x'", name="some_pkg")
    message = menu.missing_dependency_message(error)
    assert "some_pkg" in message
    assert "requirements.txt" not in message


def test_run_choice_reports_missing_dependency(monkeypatch, capsys) -> None:
    def boom(argv: Sequence[str]) -> int:
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(menu, "MENU", (("Boom", boom, ()),))
    assert menu.run_choice("1") == 1
    out = capsys.readouterr().out
    assert "Missing dependency: 'numpy'" in out
    assert "pip install -r requirements.txt" in out


def test_run_petroleum_adds_default_save_dir(monkeypatch, tmp_path) -> None:
    captured: dict[str, Sequence[str]] = {}

    def fake_run_module(name: str, argv: Sequence[str]) -> int:
        captured["name"] = name
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    monkeypatch.setattr(menu, "artifacts_dir", lambda: tmp_path / "artifacts")

    menu.run_petroleum(["--episodes", "5"])
    assert captured["name"] == "ml.petroleum"
    assert captured["argv"] == [
        "--episodes",
        "5",
        "--save-dir",
        str(tmp_path / "artifacts" / "petroleum"),
    ]


def test_run_petroleum_respects_explicit_save_dir(monkeypatch, tmp_path) -> None:
    captured: dict[str, Sequence[str]] = {}

    def fake_run_module(name: str, argv: Sequence[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    menu.run_petroleum(["--save-dir", str(tmp_path / "explicit")])
    assert captured["argv"].count("--save-dir") == 1


def test_cli_petroleum_command_dispatches(monkeypatch) -> None:
    calls: list[Sequence[str]] = []

    def fake(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    import cli.__main__ as cli_entry

    monkeypatch.setitem(cli_entry._COMMANDS, "petroleum", fake)
    assert cli_entry.main(["petroleum", "--episodes", "5"]) == 0
    assert calls == [["--episodes", "5"]]
