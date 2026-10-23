"""The Scene Studio CLI: the headless twin of the window.

These drive the real :func:`cli.scene3d.main` against a temporary data folder, so
they exercise the *same* service the window uses -- no display, no mocks. They
pin what matters: the read-only commands print what they promise, the working
scene accumulates across invocations (add a figure, then pose it, then render),
JSON output is machine-readable, the demo builds a whole scene, a bad file is an
honest non-zero exit, and the module is import-safe with no Tk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from cli import scene3d as cli


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the Studio's data folder at a temporary directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_status_prints_the_capabilities(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["status"], capsys)
    assert code == cli.EXIT_OK
    assert "Scene Studio" in output
    assert "figures" in output
    assert "software rasteriser" in output


def test_status_json_is_machine_readable(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["status", "--json"], capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(output)
    assert "adult" in payload["figures"]
    assert payload["poses"]
    assert "honesty" in payload


def test_figures_lists_the_presets(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["figures"], capsys)
    assert code == cli.EXIT_OK
    for preset in ("adult", "athletic", "slim", "child"):
        assert preset in output


def test_props_lists_the_catalog(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["props"], capsys)
    assert code == cli.EXIT_OK
    assert "ground" in output


def test_poses_lists_the_presets(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["poses"], capsys)
    assert code == cli.EXIT_OK
    assert "wave" in output


def test_materials_and_primitives_list(data_root: Path, capsys: pytest.CaptureFixture[str]):
    assert run(["materials"], capsys)[0] == cli.EXIT_OK
    assert run(["primitives"], capsys)[0] == cli.EXIT_OK


def test_help_prints_the_command_vocabulary(data_root: Path, capsys: pytest.CaptureFixture[str]):
    code, output = run(["help"], capsys)
    assert code == cli.EXIT_OK
    assert "Commands:" in output


def test_the_working_scene_accumulates_across_invocations(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    assert run(["new", "--name", "Demo"], capsys)[0] == cli.EXIT_OK
    code, output = run(["add-figure", "athletic"], capsys)
    assert code == cli.EXIT_OK
    assert "Athletic" in output

    code, output = run(["pose-preset", "wave"], capsys)
    assert code == cli.EXIT_OK
    assert "wave" in output

    code, output = run(["frame"], capsys)
    assert code == cli.EXIT_OK

    code, output = run(["render", "--width", "200", "--height", "160"], capsys)
    assert code == cli.EXIT_OK
    assert "software rasteriser" in output
    # The render is a real file in the Studio's renders folder.
    renders = list((data_root / "scene3d" / "renders").glob("*.png"))
    assert renders


def test_pose_with_no_figure_is_an_honest_failure(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    run(["new", "--name", "Empty", "--no-ground"], capsys)
    code, output = run(["pose-preset", "wave"], capsys)
    assert code == cli.EXIT_EMPTY
    assert "no figure" in output.lower()


def test_add_figure_with_an_unknown_preset_is_a_usage_error(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    code, output = run(["add-figure", "robot"], capsys)
    assert code == cli.EXIT_EMPTY
    assert "Unknown figure preset" in output


def test_command_runs_the_shared_interpreter(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    run(["new", "--name", "Cmd", "--no-ground"], capsys)
    code, output = run(["command", "add figure slim"], capsys)
    assert code == cli.EXIT_OK
    assert "Slim" in output

    code, output = run(["command", "dance wildly"], capsys)
    assert code == cli.EXIT_OK
    assert "Did not understand" in output


def test_render_json_reports_the_file(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    run(["new", "--name", "Json", "--no-ground"], capsys)
    run(["add-figure", "adult"], capsys)
    run(["frame"], capsys)
    code, output = run(["render", "--width", "200", "--height", "160", "--json"], capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(output)
    assert payload["ok"] is True
    assert Path(payload["path"]).is_file()


def test_demo_builds_and_renders_a_whole_scene(
    data_root: Path, capsys: pytest.CaptureFixture[str]
):
    code, output = run(
        ["demo", "--figure", "slim", "--pose", "walk", "--width", "200", "--height", "160"],
        capsys,
    )
    assert code == cli.EXIT_OK
    assert "slim" in output.lower()
    assert "software rasteriser" in output


def test_export_obj_writes_a_file(data_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    run(["new", "--name", "Exported", "--no-ground"], capsys)
    run(["add-figure", "adult"], capsys)
    target = tmp_path / "out.obj"
    code, output = run(["export-obj", str(target)], capsys)
    assert code == cli.EXIT_OK
    assert target.is_file()
    assert "Exported" in output


def test_save_then_list(data_root: Path, capsys: pytest.CaptureFixture[str]):
    run(["new", "--name", "Kept"], capsys)
    code, output = run(["save"], capsys)
    assert code == cli.EXIT_OK
    assert "Saved the scene" in output
    code, output = run(["list"], capsys)
    assert code == cli.EXIT_OK
    assert "Kept" in output


def test_open_a_missing_model_is_a_file_error(
    data_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, output = run(["open", str(tmp_path / "gone.obj")], capsys)
    assert code == cli.EXIT_FILE
    assert "gone.obj" in output


def test_open_an_obj_model(data_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    source = tmp_path / "tri.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    code, output = run(["open", str(source)], capsys)
    assert code == cli.EXIT_OK
    assert "tri" in output


def test_the_module_imports_without_tk():
    # cli/scene3d.py must not pull in Tk, so the CLI works headless.
    import importlib

    module = importlib.import_module("cli.scene3d")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import tkinter" not in source
    assert "tkinter" not in sys.modules or "gui.scene3d_dialog" not in sys.modules


def test_the_cli_menu_offers_the_scene_studio():
    """The Scene Studio is reachable from the CLI menu, beside the other studios."""
    from cli import menu

    labels = [title for title, _action, _argv in menu.visible_menu()]
    assert any(label.startswith("Scene Studio: build, pose, and render") for label in labels)
    assert any(label == "Scene Studio: render the built-in figure to a PNG" for label in labels)


def test_run_scene3d_dispatches_to_the_cli_module(data_root: Path, capsys):
    """``run_scene3d`` runs the headless twin, not a copy of the logic."""
    from cli.menu import run_scene3d

    code = run_scene3d(("status",))
    output = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "Scene Studio" in output


def test_the_scene3d_command_is_wired_into_the_cli_dispatch():
    """``python -m cli scene3d`` (and the ``scene`` alias) reach the Studio."""
    from cli import __main__ as cli_main

    assert "scene3d" in cli_main._COMMANDS
    assert "scene" in cli_main._COMMANDS
    assert cli_main._COMMANDS["scene3d"] is cli_main._COMMANDS["scene"]
