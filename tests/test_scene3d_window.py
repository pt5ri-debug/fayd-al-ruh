"""Window-level tests for the Scene Studio (skipped with no display).

These drive the *real* Tk window against the *real*
:class:`gui.scene3d.SceneStudioService`, so the page is exercised the way a user
drives it: add a figure from the sidebar, pose it, frame the camera, render (the
worker thread and the UI-thread queue included), and run a command from the
composer -- all with no GPU, no model, and no network.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest

from gui.scene3d import SceneStudioService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _wait_until(root, predicate, *, timeout: float = 20.0) -> bool:
    """Run the real Tk main loop until ``predicate`` holds (or time runs out)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


def _all_text_widgets(parent) -> list:
    import tkinter as tk

    found = []
    for child in parent.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_all_text_widgets(child))
    return found


def _transcript_text(dialog) -> str:
    return "\n".join(widget.get("1.0", "end") for widget in _all_text_widgets(dialog._transcript))


def _list_items(listbox) -> list[str]:
    return [listbox.get(index) for index in range(listbox.size())]


def _close_window(root, dialog) -> None:
    """Tear the window down only once its worker has finished.

    A worker thread still in flight holds the dialog (and so the Tk root); if the
    root were destroyed first, the interpreter would be freed on the worker's own
    thread, which Tk forbids (``the Tcl interpreter is leaked because it was
    deallocated in a thread other than the one it was created in``). Waiting for
    the run to drain first is what a user closing an idle window does anyway.
    """
    _wait_until(root, lambda: not dialog._busy, timeout=5.0)
    root.destroy()
    # Collect the dead root *here*, on the main thread, while no worker is alive.
    # Left to a later test's worker thread, Tk's own finalizer would run off the
    # main thread and refuse (gh-83274), leaking the interpreter.
    gc.collect()


def _figure_index(dialog) -> int:
    return next(
        index
        for index, item in enumerate(_list_items(dialog._object_list))
        if "figure" in item
    )


def _dialog(tmp_path: Path):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.scene3d_dialog import SceneStudioDialog

    service = SceneStudioService(
        scenes_dir=tmp_path / "scenes",
        renders_dir=tmp_path / "renders",
        library_dir=tmp_path / "library",
    )
    root = tk.Tk()
    root.withdraw()
    dialog = SceneStudioDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    return root, dialog, service


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_with_the_chat_window_style(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog.window.title() == "Scene Studio"
        assert dialog._transcript is not None
        assert dialog._command is not None
        assert dialog._viewport is not None
        # The same title-bar trio every window installs.
        assert dialog._window_controls is not None
        # The honesty line is in the transcript from the first moment.
        assert "software rasteriser" in _transcript_text(dialog)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_adding_a_figure_from_the_sidebar_shows_it_in_the_object_list(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("athletic")
        assert _wait_until(
            root, lambda: any("Athletic" in item for item in _list_items(dialog._object_list))
        )
        assert "Athletic" in service.scene.object_names()
        # Selecting the figure fills the bone list.
        index = next(
            i for i, item in enumerate(_list_items(dialog._object_list)) if "Athletic" in item
        )
        dialog._object_list.selection_clear(0, "end")
        dialog._object_list.selection_set(index)
        dialog._on_select_object()
        assert _wait_until(root, lambda: dialog._bone_list.size() > 0)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_applying_a_pose_preset_changes_the_figure(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        assert _wait_until(root, lambda: "Adult" in service.scene.object_names())
        dialog._object_list.selection_clear(0, "end")
        dialog._object_list.selection_set(_figure_index(dialog))
        dialog._on_select_object()
        dialog._on_pose_preset("wave")
        obj = service.scene.find_object("Adult")
        assert obj.pose
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_rendering_runs_on_a_worker_and_shows_a_preview(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        dialog._on_frame()
        assert _wait_until(root, lambda: bool(dialog._viewport_source), timeout=30.0)
        assert Path(dialog._viewport_source).is_file()
        assert dialog._photo is not None
        assert (tmp_path / "renders").is_dir()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_rendering_and_saving_writes_the_chosen_file(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        dialog._on_frame()
        target = tmp_path / "chosen.png"
        dialog._render_viewport(save_path=str(target), announce=True)
        assert _wait_until(root, lambda: target.is_file(), timeout=30.0)
        assert "Render" in _transcript_text(dialog)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_command_box_runs_an_instruction(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._command_var.set("add figure slim")
        dialog._on_command()
        assert _wait_until(root, lambda: "Slim" in service.scene.object_names())
        assert "Slim" in _transcript_text(dialog)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_command_box_reports_an_unknown_instruction(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._command_var.set("teleport the figure")
        dialog._on_command()
        assert _wait_until(root, lambda: "Did not understand" in _transcript_text(dialog))
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_bone_button_rotates_the_selected_bone(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        assert _wait_until(root, lambda: "Adult" in service.scene.object_names())
        dialog._object_list.selection_clear(0, "end")
        dialog._object_list.selection_set(_figure_index(dialog))
        dialog._on_select_object()
        assert _wait_until(root, lambda: dialog._bone_list.size() > 0)
        dialog._bone_list.selection_clear(0, "end")
        dialog._bone_list.selection_set(0)
        dialog._rotate_bone("x", 15)
        obj = service.scene.find_object("Adult")
        assert obj.pose  # the selected bone was rotated
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_clearing_the_pose_from_the_window(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        dialog._object_list.selection_clear(0, "end")
        dialog._object_list.selection_set(_figure_index(dialog))
        dialog._on_select_object()
        dialog._on_pose_preset("sit")
        obj = service.scene.find_object("Adult")
        assert obj.pose
        dialog._on_clear_pose()
        assert service.scene.find_object("Adult").pose == {}
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_new_scene_resets_the_window(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        dialog._on_new_scene()
        assert service.scene.object_names() == ("Ground",)
        assert _wait_until(root, lambda: dialog._object_list.size() == 1)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_an_obj_from_the_window(tmp_path: Path) -> None:
    source = tmp_path / "tri.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._load(str(source))
        assert _wait_until(root, lambda: "tri" in service.scene.object_names())
        assert _wait_until(root, lambda: "Loaded" in _transcript_text(dialog))
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_never_touches_the_network(tmp_path: Path, monkeypatch) -> None:
    # The Studio is fully local: it must not open a socket while building,
    # adding a figure, posing, and rendering.
    import socket

    def forbidden(*args, **kwargs):  # pragma: no cover - only on failure
        raise AssertionError("the Scene Studio must not open a network connection")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._on_add_figure("adult")
        dialog._on_pose_preset("wave")
        dialog._on_frame()
        assert _wait_until(root, lambda: bool(service.scene.object_names()))
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_attach_menu_offers_the_scene_studio(tmp_path: Path) -> None:
    """The composer's ＋ menu names the Scene Studio and opens its window."""
    import tkinter as tk

    from gui.app import FaydGUI

    from tests.test_gui_design import _service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        menu = app._build_attach_menu()
        end = menu.index("end")
        assert end is not None
        labels = [
            menu.entrycget(index, "label")
            for index in range(end + 1)
            if menu.type(index) == "command"
        ]
        assert any("Scene Studio" in label for label in labels)
        # Opening it builds the real window without raising, and the window owns
        # the same service the chat window's ChatService exposes.
        app._open_scene_studio()
        root.update()
        assert app._scene_studio_dialog is not None
        assert app._scene_studio_dialog.window.title() == "Scene Studio"
        assert app._scene_studio_dialog.service is app.service.scene3d
        # Opening it a second time focuses the existing window, not a new one.
        first = app._scene_studio_dialog
        app._open_scene_studio()
        assert app._scene_studio_dialog is first
        first._close()
        root.update()
        assert app._scene_studio_dialog is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_sidebar_action_routes_to_the_window(tmp_path: Path) -> None:
    """The hidden sidebar action opens the Studio instead of running the CLI."""
    import tkinter as tk

    from gui.actions import action_by_key
    from gui.app import FaydGUI

    from tests.test_gui_design import _service

    action = action_by_key("scene-studio")
    assert action is not None
    assert action.hidden is True

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._on_action(action)
        root.update()
        assert app._scene_studio_dialog is not None
        assert app._scene_studio_dialog.window.title() == "Scene Studio"
        app._scene_studio_dialog._close()
        root.update()
    finally:
        root.destroy()
