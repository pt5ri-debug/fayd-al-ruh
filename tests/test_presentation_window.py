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


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_batch_control_drafts_a_bounded_number_of_slides(tmp_path: Path) -> None:
    """The window's batch spinbox sets the service's batch size."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        project = service.new_project("Batch")
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        assert dialog._batch_var.get() == str(service.batch_size)
        dialog._batch_var.set("2")
        dialog._on_batch_size()
        assert service.batch_size == 2
        assert "2 slide(s) per batch" in dialog._export_note.cget("text")
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_editor_edits_a_slide_section_and_subsection(tmp_path: Path) -> None:
    """The section/subsection fields are loaded and saved by the editor."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        project = service.new_project("Editor")
        service.draft_outline(project.project_id, text="Reservoir\n  - Porosity")
        service.approve_outline(project.project_id)
        service.draft_slides_from_outline(project.project_id, batch=1)
        service.review_slides(project.project_id)
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        (slide,) = service.get(project.project_id).ordered_slides()
        dialog._load_slide(slide.slide_id)
        assert dialog._slide_section_var.get() == "Reservoir"
        assert dialog._slide_subsection_var.get() == "Porosity"
        dialog._slide_subsection_var.set("Storage")
        dialog._on_save_slide()
        assert _wait_until(root, lambda: not dialog._busy)
        reloaded = service.get(project.project_id).slide(slide.slide_id)
        assert reloaded.section == "Reservoir"
        assert reloaded.subsection == "Storage"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_command_box_runs_a_slide_instruction(tmp_path: Path) -> None:
    """The command box drives the shared command parser."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        project = service.new_project("Command")
        service.draft_outline(project.project_id, text=OUTLINE)
        service.approve_outline(project.project_id)
        service.draft_slides_from_outline(project.project_id, batch=2)
        service.review_slides(project.project_id)
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        dialog._command_var.set("Approve slides 1-2")
        dialog._on_run_command()
        assert _wait_until(root, lambda: not dialog._busy)
        assert "Approved slide(s) 1, 2" in dialog._export_note.cget("text")
        assert len(service.get(project.project_id).approved_slides) == 2
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_deck_look_controls_persist_the_project_settings(tmp_path: Path) -> None:
    """The theme/size/numbers controls write through to the stored settings."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        project = service.new_project("Look")
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        # The widgets mirror the stored defaults on open.
        assert dialog._theme_choice.get() == "professional"
        assert dialog._aspect_choice.get() == "16:9"

        dialog._theme_choice.set("dark")
        dialog._on_theme_change()
        dialog._aspect_choice.set("4:3")
        dialog._on_aspect_change()
        dialog._numbers_var.set(False)
        dialog._on_numbers_change()

        saved = service.get(project.project_id).settings
        assert saved.theme == "dark"
        assert saved.aspect == "4:3"
        assert saved.show_slide_numbers is False

        # The engine note names the engine an export would really use.
        assert "Engine:" in dialog._engine_note.cget("text")
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_management_buttons_track_the_selection(tmp_path: Path) -> None:
    """Rename/Duplicate/Delete are live only while a project is selected."""
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
        # No project yet: every management button is disabled.
        assert dialog._rename_button.cget("state") == "disabled"
        assert dialog._duplicate_button.cget("state") == "disabled"
        assert dialog._delete_button.cget("state") == "disabled"

        project = service.new_project("Manage")
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()
        assert dialog._rename_button.cget("state") == "normal"
        assert dialog._duplicate_button.cget("state") == "normal"
        assert dialog._delete_button.cget("state") == "normal"

        # The dashboard row carries the id, a date, a slide count, and the state.
        row = dialog._project_list.get(0)
        assert project.project_id in row
        assert "slide(s)" in row
        assert "Discussion" in row
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_check_and_preview_actions_report_the_deck(tmp_path: Path) -> None:
    """Check the deck and Preview show the honest, in-order content."""
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

        # The check runs with no file written and reports the deck's state
        # (here it is not yet ready, because the content is not approved).
        dialog._on_validate()
        root.update()
        transcript = _transcript_text(dialog)
        assert "is not ready" in transcript
        assert project.project_id in transcript
        assert "issue:" in transcript

        # Approve every slide, then check and preview the ready deck.
        for slide_id in list(dialog._slide_ids):
            service.approve_slide(project.project_id, slide_id)
        service.approve_content(project.project_id)
        dialog.refresh()
        root.update()

        dialog._on_validate()
        root.update()
        assert "is valid" in _transcript_text(dialog)

        # The preview names every approved slide, in order, before any export.
        dialog._on_preview()
        root.update()
        preview = _transcript_text(dialog)
        assert "slide(s) will be written" in preview
        for slide in service.get(project.project_id).ordered_slides():
            assert f"{slide.number}. {slide.title}" in preview
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_export_button_validates_before_it_writes(tmp_path: Path) -> None:
    """The export path refuses an invalid deck and only opens after approval."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project = service.new_project("Gate")
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

        # Content is not approved yet, so the export button stays disabled.
        assert dialog._export_button.cget("state") == "disabled"
        # The download folder button is off until a real export exists.
        assert dialog._download_button.cget("state") == "disabled"

        for slide_id in list(dialog._slide_ids):
            service.approve_slide(project.project_id, slide_id)
        dialog.refresh()
        root.update()
        dialog._on_approve_content()
        root.update()
        assert dialog._export_button.cget("state") == "normal"

        dialog._on_export()
        assert _wait_until(root, lambda: not dialog._busy)
        # After a successful export the folder button becomes available.
        assert dialog._download_button.cget("state") == "normal"
        dialog._close()
    finally:
        root.destroy()

