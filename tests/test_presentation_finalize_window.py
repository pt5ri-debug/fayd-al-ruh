"""End-to-end window tests for the step 6 -> step 8 fix and the final editor.

These drive the *real* Tk windows against the *real* ``PresentationService`` and
prove the workflow the user performs:

* **Studio** -- review the slides, press *Approve the content*, and reach the
  exportable state, with the final editor opened for the approved deck;
* **Final editor** -- polish a slide, re-approve, and export a real .pptx.

They skip when no display is available, like the other window tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gui.presentation import PresentationService, SlideSpec

PROSE = (
    "Porosity is the fraction of the bulk volume that is void space. "
    "Permeability is the ability of the rock to transmit a fluid."
)


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class ScriptedAssistant:
    def __init__(self, answer: str = PROSE) -> None:
        self.answer = answer

    def __call__(self, _question: str) -> str:
        return self.answer


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant(),
    )


def _at_slide_review(service: PresentationService) -> str:
    """A project walked to SLIDE_REVIEW with two unapproved slides."""
    project = service.new_project("Workflow deck")
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
    return project.project_id


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
def test_the_studio_wires_its_own_model_before_a_draft(tmp_path: Path) -> None:
    """Item 2: the window actually loads the Studio's Qwen3 model to draft.

    The author is wired on the worker thread just before the draft, so opening the
    window never loads a multi-gigabyte model, and a missing model is a silent
    fallback to the assistant rather than an error.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project = service.new_project("Author wiring")
    calls: list[str] = []

    def ensure_author():
        calls.append("called")
        return object()

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root,
            service,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
            ensure_author=ensure_author,
        )
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()
        assert calls == [], "opening the window must not load the model"

        dialog.draft_outline(text="Reservoir\n    - Porosity")
        assert _wait_until(root, lambda: not dialog._busy)
        assert calls == ["called"], "the draft must wire the Studio's own model first"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_failing_author_load_falls_back_to_the_assistant(tmp_path: Path) -> None:
    """A missing extra or weights must not break drafting."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project = service.new_project("Fallback")

    def ensure_author():
        raise RuntimeError("no model installed")

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root,
            service,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
            ensure_author=ensure_author,
        )
        dialog._refresh_projects()
        dialog._select_project(project.project_id)
        root.update()

        dialog.draft_outline(text="Reservoir\n    - Porosity")
        assert _wait_until(root, lambda: not dialog._busy)
        assert service.get(project.project_id).outline is not None
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_studio_approves_the_content_and_opens_the_final_editor(tmp_path: Path) -> None:
    """The fix: step 7 completes the workflow instead of raising."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    opened: list[str] = []

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root,
            service,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
            on_open_final_editor=lambda: opened.append("final"),
        )
        dialog._refresh_projects()
        dialog._select_project(project_id)
        root.update()

        # Step 6 is available; step 7 becomes available once the deck is under review.
        assert dialog._step_buttons["review-slides"].cget("state") == "normal"
        assert dialog._step_buttons["approve-content"].cget("state") == "normal"

        dialog._on_approve_content()
        root.update()

        project = service.get(project_id)
        assert project.content_approved, "step 7 must complete, not raise"
        assert len(project.approved_slides) == 2
        assert opened == ["final"], "the final editor must be opened for the approved deck"

        # Step 8 is now enabled in the Studio as well.
        dialog._refresh_steps(project.state)
        assert dialog._step_buttons["export"].cget("state") == "normal"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_studio_reports_which_slide_is_empty(tmp_path: Path) -> None:
    """A blank slide names itself, so the user knows what to fix."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.add_slides(project_id, [SlideSpec(title="", content=[])])

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._refresh_projects()
        dialog._select_project(project_id)
        root.update()

        dialog._on_approve_content()
        root.update()

        assert not service.get(project_id).content_approved
        note = dialog._export_note.cget("text")
        assert "3" in note, f"the empty slide's number must be named, got: {note!r}"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_studio_offers_approve_all_slides(tmp_path: Path) -> None:
    """The one-click batch approval is wired to the service."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_dialog import PresentationDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)

    root = tk.Tk()
    try:
        dialog = PresentationDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._refresh_projects()
        dialog._select_project(project_id)
        root.update()

        assert not service.get(project_id).all_slides_approved
        dialog._on_approve_all_slides()
        root.update()
        assert service.get(project_id).all_slides_approved
        dialog._close()
    finally:
        root.destroy()


# ------------------------------------------------------------ the final editor


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_exports_the_approved_deck(tmp_path: Path) -> None:
    """The new window writes a real .pptx from the approved slides."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_finalize_dialog import PresentationFinalizeDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    root = tk.Tk()
    try:
        dialog = PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        assert dialog._export_button.cget("state") == "normal"
        assert dialog._slide_ids, "the approved slides must be listed"

        dialog._on_export()
        assert _wait_until(root, lambda: not dialog._busy)

        reports = service.export_history(project_id)
        assert reports and reports[-1].ok
        saved = Path(reports[-1].path)
        assert saved.exists() and saved.suffix == ".pptx"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_edits_a_slide_and_re_arms_the_gate(tmp_path: Path) -> None:
    """Editing in the final editor disables the export until the slides are re-approved."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_finalize_dialog import PresentationFinalizeDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    edits: list[str] = []

    root = tk.Tk()
    try:
        dialog = PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
            on_edit=lambda: edits.append("edited"),
        )
        root.update()

        # Load the first slide into the editor, change its body, and save.
        dialog._load_slide(dialog._slide_ids[0])
        dialog._slide_body.delete("1.0", "end")
        dialog._slide_body.insert("1.0", "A polished bullet")
        dialog._on_save_slide()
        assert _wait_until(root, lambda: not dialog._busy)

        assert service.get(project_id).state == "SLIDE_REVIEW"
        assert dialog._export_button.cget("state") == "disabled"
        assert edits == ["edited"], "the Studio behind must be told to refresh"

        # Re-approving the slides makes the deck exportable again.
        dialog._on_approve_all_slides()
        service.approve_content(project_id)
        dialog.refresh(keep_note=True)
        root.update()
        assert dialog._export_button.cget("state") == "normal"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_moves_a_slide_and_keeps_the_order(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_finalize_dialog import PresentationFinalizeDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    root = tk.Tk()
    try:
        dialog = PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        before = [slide.title for slide in service.get(project_id).ordered_slides()]

        dialog._selected_slide = dialog._slide_ids[1]
        dialog._on_move(-1)
        root.update()

        after = [slide.title for slide in service.get(project_id).ordered_slides()]
        assert after == list(reversed(before))
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_updates_the_deck_theme(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_finalize_dialog import PresentationFinalizeDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    root = tk.Tk()
    try:
        dialog = PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        dialog._theme_choice.set("dark")
        dialog._on_theme_change()
        root.update()
        assert service.get(project_id).settings.theme == "dark"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_validates_and_previews_without_crashing(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.presentation_finalize_dialog import PresentationFinalizeDialog

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    root = tk.Tk()
    try:
        dialog = PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        before = len(dialog._row_frames)
        dialog._on_validate()
        dialog._on_preview()
        root.update()
        assert len(dialog._row_frames) > before, "check and preview report in the transcript"
        dialog._close()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_final_editor_saves_as_to_a_chosen_path(tmp_path: Path) -> None:
    import tkinter as tk

    from gui import presentation_finalize_dialog as module
    from gui.app import DARK_THEME

    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    chosen = tmp_path / "chosen" / "final-deck.pptx"

    def fake_asksaveasfilename(**_kwargs):
        chosen.parent.mkdir(parents=True, exist_ok=True)
        return str(chosen)

    root = tk.Tk()
    try:
        dialog = module.PresentationFinalizeDialog(
            root,
            service,
            project_id=project_id,
            theme=DARK_THEME,
            font="TkDefaultFont",
            mono="TkFixedFont",
        )
        root.update()
        original = module.filedialog.asksaveasfilename
        module.filedialog.asksaveasfilename = fake_asksaveasfilename
        try:
            dialog._on_export_as()
            assert _wait_until(root, lambda: not dialog._busy)
        finally:
            module.filedialog.asksaveasfilename = original
        assert chosen.exists()
        dialog._close()
    finally:
        root.destroy()
