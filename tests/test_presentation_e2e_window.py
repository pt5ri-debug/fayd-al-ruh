"""End-to-end window tests for the Presentation Studio download path.

These drive the *real* Tk window against the *real* ``PresentationService`` and
prove the file-download workflow a user performs: export through the window, and
"Save as…" to a chosen path, with the saved file reopened and checked. They skip
when no display is available, like the other window tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gui.presentation import PresentationService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class ScriptedAssistant:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def __call__(self, _question: str) -> str:
        return self.answer


PROSE = (
    "Porosity is the fraction of the bulk volume that is void space. "
    "Permeability is the ability of the rock to transmit a fluid."
)


def _ready_service(tmp_path: Path) -> tuple[PresentationService, str]:
    """A service with one project walked to approved content; return both."""
    from gui.presentation import SlideSpec

    service = PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant(PROSE),
    )
    project = service.new_project("Download deck")
    service.draft_outline(
        project.project_id, text="Reservoir\n    - Porosity\n    - Permeability"
    )
    service.approve_outline(project.project_id)
    service.add_slides(
        project.project_id,
        [
            SlideSpec(title="Porosity", content=["Void fraction"]),
            SlideSpec(title="Permeability", content=["Flow"]),
        ],
    )
    service.review_slides(project.project_id)
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    return service, project.project_id


def _wait_until(root, predicate, *, timeout: float = 10.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_exports_and_the_download_button_tracks_it(tmp_path: Path) -> None:
    """The window's export writes a file and enables the download control."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service, project_id = _ready_service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._refresh_projects()
        dialog._select_project(project_id)
        root.update()

        assert dialog._export_button.cget("state") == "normal"
        dialog._on_export()
        assert _wait_until(root, lambda: not dialog._busy)

        reports = service.export_history(project_id)
        assert reports and reports[-1].ok
        saved = Path(reports[-1].path)
        assert saved.exists() and saved.suffix == ".pptx"

        # The download control now reflects a real, downloadable file.
        status = service.status(project_id)
        assert status.download
        assert Path(status.download["path"]) == saved
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_saves_as_to_a_chosen_path(tmp_path: Path) -> None:
    """`Save as…` writes the deck to the user's chosen location."""
    import tkinter as tk

    from gui import presentation_dialog as module
    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service, project_id = _ready_service(tmp_path)
    chosen = tmp_path / "chosen" / "my-deck.pptx"

    # Stand in for the native save dialog: return the path the user would pick.
    def fake_asksaveasfilename(**_kwargs):
        chosen.parent.mkdir(parents=True, exist_ok=True)
        return str(chosen)

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._refresh_projects()
        dialog._select_project(project_id)
        root.update()

        original = module.filedialog.asksaveasfilename
        module.filedialog.asksaveasfilename = fake_asksaveasfilename
        try:
            dialog._on_export_as()
            assert _wait_until(root, lambda: not dialog._busy)
        finally:
            module.filedialog.asksaveasfilename = original

        assert chosen.exists(), "Save as… did not write the chosen file"
        assert chosen.suffix == ".pptx"
        # The export history records exactly the chosen destination.
        reports = service.export_history(project_id)
        assert any(Path(report.path) == chosen and report.ok for report in reports)
        dialog._close()
    finally:
        root.destroy()
