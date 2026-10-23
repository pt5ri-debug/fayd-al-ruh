"""End-to-end window tests for the Course window and its Studio trigger.

These drive the *real* Tk windows against the *real* ``PresentationService`` and
prove the workflow the user performs:

* the **Studio** offers the Course window only once the deck's own slides match a
  course, and hands the chosen course key out;
* the **Course window** shows the course's concepts with the covered ones marked,
  hands the missing concepts back to the Studio's question box, and jumps to the
  slide that covers a concept.

They skip when no display is available, like the other window tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gui.presentation import PresentationService, SlideSpec


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


pytestmark = pytest.mark.skipif(not _display_available(), reason="no display available")


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store", output_dir=tmp_path / "out", ask=lambda _q: "x"
    )


def _petrophysics_deck(service: PresentationService) -> str:
    project = service.new_project("Petrophysics deck")
    service.draft_outline(project.project_id, text="Petrophysics\n    - Porosity")
    service.approve_outline(project.project_id)
    service.add_slide(
        project.project_id,
        SlideSpec(title="Effective porosity", content=["Effective porosity of the rock"]),
    )
    service.add_slide(
        project.project_id,
        SlideSpec(title="Absolute permeability", content=["Absolute permeability in darcies"]),
    )
    service.add_slide(
        project.project_id,
        SlideSpec(title="Water saturation", content=["Water saturation of the pores"]),
    )
    service.review_slides(project.project_id)
    return project.project_id


def _theme():
    from gui.app import DARK_THEME

    return DARK_THEME


def _studio(root, service, opened: list):
    from gui.presentation_dialog import PresentationDialog

    return PresentationDialog(
        root,
        service,
        theme=_theme(),
        font="TkDefaultFont",
        mono="TkFixedFont",
        on_open_course=opened.append,
    )


def _course_window(root, service, project_id: str, course_key: str, events: list):
    from gui.presentation_course_dialog import PresentationCourseDialog

    return PresentationCourseDialog(
        root,
        service,
        project_id=project_id,
        course_key=course_key,
        theme=_theme(),
        font="TkDefaultFont",
        mono="TkFixedFont",
        on_use_concepts=lambda text: events.append(("use", text)),
        on_open_slide=lambda number: events.append(("slide", number)),
    )


def test_the_studio_offers_the_course_once_the_deck_reaches_it(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    empty = service.new_project("Empty")
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        opened: list[str] = []
        dialog = _studio(root, service, opened)

        dialog._select_project(empty.project_id)
        root.update()
        assert dialog._course_button.cget("state") == "disabled"
        dialog._on_course_button()
        assert opened == []

        dialog._select_project(project_id)
        root.update()
        assert dialog._course_button.cget("state") == "normal"
        dialog._on_course_button()
        assert opened == ["petrophysics"]
        dialog._close()
    finally:
        root.destroy()


def test_the_course_window_marks_what_the_deck_covers(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        dialog = _course_window(root, service, project_id, "petrophysics", [])
        root.update()
        entries = [dialog._concept_list.get(i) for i in range(dialog._concept_list.size())]
        assert any(entry.startswith("✓ effective porosity") for entry in entries)
        assert any(entry.startswith("· hydrocarbon saturation") for entry in entries)
        assert dialog._course_label.cget("text") == "Petrophysics"
        assert dialog._branch_label.cget("text") == "Petroleum Disciplines"
        dialog._close()
    finally:
        root.destroy()


def test_the_course_window_hands_the_missing_concepts_back(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        events: list = []
        dialog = _course_window(root, service, project_id, "petrophysics", events)
        root.update()
        assert dialog._use_button.cget("state") == "normal"
        dialog._on_use_missing()
        assert events and events[-1][0] == "use"
        text = events[-1][1]
        assert "- water saturation" not in text
        assert "- hydrocarbon saturation" in text
        dialog._close()
    finally:
        root.destroy()


def test_the_course_window_jumps_to_the_slide_that_covers_a_concept(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        events: list = []
        dialog = _course_window(root, service, project_id, "petrophysics", events)
        root.update()
        # First concept is "effective porosity", which slide 1 covers.
        dialog._concept_list.selection_set(0)
        dialog._on_concept_select()
        assert dialog._goto_button.cget("state") == "normal"
        dialog._on_goto_concept()
        assert events[-1] == ("slide", 1)
        # A missing concept has no slide, so the button is disabled.
        last = dialog._concept_list.size() - 1
        dialog._concept_list.selection_set(last)
        dialog._on_concept_select()
        assert dialog._goto_button.cget("state") == "disabled"
        dialog._close()
    finally:
        root.destroy()


def test_the_studio_puts_curriculum_concepts_in_the_question_box(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        dialog = _studio(root, service, [])
        dialog._select_project(project_id)
        root.update()
        dialog.use_course_concepts("- hydrocarbon saturation\n- formation factor")
        root.update()
        assert "hydrocarbon saturation" in dialog._question.get("1.0", "end")
        dialog._close()
    finally:
        root.destroy()


def test_the_studio_reveals_the_slide_that_covers_a_concept(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        dialog = _studio(root, service, [])
        dialog._select_project(project_id)
        root.update()
        dialog.reveal_slide_number(2)
        root.update()
        slide = service.get(project_id).slide(dialog._selected_slide)
        assert slide.number == 2
        dialog._close()
    finally:
        root.destroy()


def test_the_course_window_switches_to_another_matching_course(tmp_path: Path) -> None:
    import tkinter as tk

    service = _service(tmp_path)
    project_id = _petrophysics_deck(service)

    root = tk.Tk()
    try:
        dialog = _course_window(root, service, project_id, "petrophysics", [])
        root.update()
        matches = service.course_matches(project_id)
        assert dialog._match_list.size() == len(matches)
        # Selecting a different course re-reads it from the one service.
        dialog._match_list.selection_clear(0, "end")
        dialog._match_list.selection_set(0)
        dialog._on_match_select()
        root.update()
        assert dialog._course_label.cget("text") == matches[0].label
        dialog._close()
    finally:
        root.destroy()
