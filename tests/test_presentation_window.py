"""Window-level tests for the Presentation Studio dialog (skipped with no display).

These drive the *real* Tk window against the *real* ``PresentationService`` and a
scripted assistant, so the page is exercised the way a user drives it: discuss,
draft and approve an outline, draft and review the slides, approve the content,
and export. The scripted assistant keeps the run free of any model or network.
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
    """A deterministic stand-in for the project's assistant (no model needed)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def __call__(self, _question: str) -> str:
        return self.answer


OUTLINE = "Reservoir\n- Porosity\n- Permeability"

PROSE = (
    "Porosity is the fraction of void space in a rock and it controls storage. "
    "Permeability is the ability to transmit fluids and it controls flow rate. "
    "Well logs estimate both quantities in the subsurface."
)


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant(PROSE),
    )


def _wait_until(root, predicate, *, timeout: float = 10.0) -> bool:
    """Run the real Tk main loop until ``predicate`` holds (or time runs out)."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


def _all_text_widgets(parent) -> list:
    """Every ``tk.Text`` under ``parent``, so a test can search the transcript."""
    import tkinter as tk

    found = []
    for child in parent.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_all_text_widgets(child))
    return found


def _transcript_text(dialog) -> str:
    return "\n".join(
        widget.get("1.0", "end") for widget in _all_text_widgets(dialog._transcript)
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_and_offers_the_workflow(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import STEPS, TITLE, PresentationDialog

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root,
            _service(tmp_path),
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        assert dialog.window.title() == TITLE
        assert set(dialog._step_buttons) == {key for key, _label in STEPS}
        # With no project yet, the export button is disabled and the page says why.
        assert dialog._export_button.cget("state") == "disabled"
        assert "press **new presentation** to start" in _transcript_text(dialog).lower()
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_walks_the_whole_workflow_and_exports(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()

        project = service.new_project("Petrophysics")
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        # 1. Discuss the subject (the scripted assistant answers).
        dialog._question.insert("1.0", "What is porosity?")
        dialog._on_discuss()
        assert _wait_until(root, lambda: not dialog._busy)
        assert "porosity is the fraction" in _transcript_text(dialog).lower()

        # 2-4. Draft the outline from the assistant's prose, review, approve.
        service.draft_outline(project.project_id)
        dialog.refresh()
        root.update()
        dialog._on_review_outline()
        dialog._on_approve_outline()
        root.update()

        # 5-6. Draft the slides, then move them into review.
        dialog.draft_slides(None)
        assert _wait_until(root, lambda: not dialog._busy)
        dialog._on_review_slides()
        root.update()
        assert dialog._slide_list.size() >= 1

        # 7. Approve every slide, then the content.
        for slide_id in list(dialog._slide_ids):
            service.approve_slide(project.project_id, slide_id)
        dialog.refresh()
        root.update()
        dialog._on_approve_content()
        root.update()

        # 8. Export, and the gate is open.
        assert dialog._export_button.cget("state") == "normal"
        dialog._on_export()
        assert _wait_until(root, lambda: not dialog._busy)
        assert "exported" in _transcript_text(dialog).lower()

        reports = service.export_history(project.project_id)
        assert reports and reports[-1].ok
        assert Path(reports[-1].path).exists()
        assert Path(reports[-1].path).suffix == ".pptx"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_slide_editor_loads_and_saves_a_slide(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project = service.new_project("Deck")
    service.draft_outline(project.project_id, text=OUTLINE)
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id)
    service.review_slides(project.project_id)

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()
        assert dialog._slide_list.size() >= 1

        # Select the first slide: the editor loads it and becomes visible.
        dialog._slide_list.selection_set(0)
        dialog._on_slide_select()
        root.update()
        assert dialog._selected_slide
        assert dialog._slide_panel.winfo_ismapped()
        assert dialog._slide_title_var.get()

        # Edit the title and save it through the editor's own controls.
        dialog._slide_title_var.set("Edited in the window")
        dialog._on_save_slide()
        assert _wait_until(root, lambda: not dialog._busy)
        reloaded = service.get(project.project_id).slide(dialog._selected_slide)
        assert reloaded.title == "Edited in the window"

        # Approve, withdraw, and remove through the editor's own controls.
        dialog._on_approve_slide()
        root.update()
        assert service.get(project.project_id).slide(dialog._selected_slide).approved
        dialog._on_unapprove_slide()
        root.update()
        assert not service.get(project.project_id).slide(dialog._selected_slide).approved
        before = dialog._slide_list.size()
        dialog._on_remove_slide()
        root.update()
        assert dialog._slide_list.size() == before - 1
        dialog._close()
    finally:
        root.destroy()


def _column_overflows(widget) -> list[str]:
    """Every mapped descendant that requests more width than it was given."""
    problems: list[str] = []
    try:
        width, requested = widget.winfo_width(), widget.winfo_reqwidth()
    except Exception:
        return problems
    if widget.winfo_ismapped() and requested > width + 1:
        problems.append(f"{widget.winfo_class()} requests {requested}px in {width}px")
    for child in widget.winfo_children():
        problems.extend(_column_overflows(child))
    return problems


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_messages_never_overflow_the_transcript_column(tmp_path: Path) -> None:
    """A message must wrap inside the column, not request pixels it cannot have.

    A ``tk.Text`` measures ``width`` in *characters*, so the message width has to
    be converted from the column's pixels; passing the pixel count straight
    through asked for thousands of pixels and pushed a horizontal scrollbar over
    the transcript.
    """
    import tkinter as tk

    from gui.app import DARK_THEME, LIGHT_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    for theme in (DARK_THEME, LIGHT_THEME):
        root = tk.Tk()
        root.geometry("1400x900")
        try:
            dialog = PresentationDialog(
                root, service, theme=theme, font="TkDefaultFont", mono="TkFixedFont"
            )
            project = service.new_project("Layout")
            dialog._refresh_projects()
            dialog._select_project(project.project_id)
            dialog._add_message("assistant", "A long assistant answer. " * 40)
            dialog._add_message("user", "short question")
            for _ in range(20):
                root.update()
            assert not _column_overflows(dialog._transcript)
            dialog._close()
        finally:
            root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_messages_are_re_measured_when_the_column_changes(tmp_path: Path) -> None:
    """A reflow must re-measure every row, not leave it sized for the old column."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        project = service.new_project("Layout")
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        dialog._add_message("assistant", "A long assistant answer. " * 40)
        for _ in range(10):
            root.update()

        def widths() -> dict[str, int]:
            return {text: widget.cget("width") for _r, text, _row, widget, _c in dialog._row_frames}

        # A narrower column must shrink every row's character width.
        before = widths()
        dialog._available_column = lambda: 300  # type: ignore[method-assign]
        dialog._reflow()
        after = widths()
        assert before and after
        assert all(after[text] < before[text] for text in before)
        # And the re-measured rows still fit the column they were measured for.
        assert all(width > 0 for width in after.values())
        assert not _column_overflows(dialog._transcript)
        dialog._close()
    finally:
        root.destroy()
