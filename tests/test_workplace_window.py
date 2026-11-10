"""Window-level tests for the Workplace (skipped with no display).

These drive the *real* Tk window against the *real*
:class:`gui.workplace.WorkplaceService`, so the page is exercised the way a user
drives it: the dashboard reads real figures, the institution tree and the ten
divisions are drawn, a division is added and deactivated from the sidebar, and the
taxonomy is listed -- all with no model and no network.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest
from gui.workplace import WorkplaceService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _wait_until(root, predicate, *, timeout: float = 20.0) -> bool:
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


def _close_window(root, dialog) -> None:
    _wait_until(root, lambda: not dialog._busy, timeout=5.0)
    root.destroy()
    gc.collect()


def _dialog(tmp_path: Path):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.workplace_dialog import WorkplaceDialog

    service = WorkplaceService(root=tmp_path)
    root = tk.Tk()
    root.withdraw()
    dialog = WorkplaceDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    return root, dialog, service


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_with_the_chat_window_style(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog.window.title() == "Workplace"
        assert dialog._sidebar is not None
        assert dialog._spinner is not None
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_dashboard_reads_real_figures(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog._dash_labels["divisions"].cget("text") == "10"
        assert dialog._dash_labels["institutions"].cget("text") == "1"
        assert "ready" in dialog._status_line.cget("text").lower()
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_tree_and_divisions_are_drawn(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        tree = dialog._tree.get("1.0", "end")
        assert "Egyptian Petroleum Research Institute" in tree
        assert "Technical Support & Technology Center" in tree
        assert dialog._division_list.size() == 10
        assert "Engineering Services" in dialog._division_list.get(0)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_taxonomy_is_listed(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        text = dialog._taxonomy_text.get("1.0", "end")
        assert "Welding Engineering" in text
        assert "WPS / PQR / WPQ" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_division_profile_is_shown(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._division_list.selection_clear(0, "end")
        dialog._division_list.selection_set(0)
        dialog._on_division_selected()
        text = dialog._profile_text.get("1.0", "end")
        assert "Scope of services" in text
        assert "Engineering Services" in dialog._profile_title.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_adding_a_profile_item_updates_the_profile(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        dialog._division_list.selection_clear(0, "end")
        dialog._division_list.selection_set(0)
        dialog._on_division_selected()
        division = dialog._selected_division()
        service.add_division_detail(
            division_id=division.record_id, kind="standard", title="API 1104"
        )
        dialog._refresh_profile()
        assert "API 1104" in dialog._profile_text.get("1.0", "end")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_language_and_direction_controls_reflect_and_change_settings(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        assert dialog._language.get() == "en"
        dialog._language.set("ar")
        dialog._on_language()
        assert service.settings().language == "ar"
        assert service.settings().rtl is True
        dialog._rtl.set(False)
        dialog._on_rtl()
        assert service.settings().rtl is False
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_deactivating_a_division_updates_the_list(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._division_list.selection_clear(0, "end")
        dialog._division_list.selection_set(0)
        dialog._on_division_selected()
        assert dialog._toggle_active_button.cget("text") == "Deactivate"
        dialog._on_toggle_active()
        assert dialog._division_list.size() == 9
        dialog._show_inactive.set(True)
        dialog._refresh_divisions()
        assert dialog._division_list.size() == 10
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_honesty_note_is_shown(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "proposed, configurable defaults" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_dashboard_reads_the_new_domain_figures(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.domain.create_project(number="P-1", title="A")
        service.domain.create_person(name="Ali")
        dialog._refresh_dashboard()
        assert dialog._dash_labels["active_projects"].cget("text") == "0"
        assert dialog._dash_labels["personnel"].cget("text") == "1"
        assert dialog._dash_labels["standards"].cget("text") == "0"
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_records_section_lists_a_project(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.domain.create_project(number="TSTC-2026-001", title="Coating inspection")
        dialog._record_entity.set("projects")
        dialog._refresh_records()
        text = dialog._records_text.get("1.0", "end")
        assert "PRJ-" in text
        assert "TSTC-2026-001" in text
        assert "Coating inspection" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_audit_trail_is_shown(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.domain.create_person(name="Ali")
        dialog._on_show_audit()
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "create" in text and "people" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_work_section_lists_a_project_and_a_person(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.domain.create_project(number="TSTC-2026-500", title="Flowlines")
        service.domain.create_person(name="Ali Hassan", role="Welding Engineer")
        dialog._refresh_work()
        assert dialog._project_list.size() == 1
        assert "TSTC-2026-500" in dialog._project_list.get(0)
        assert "Ali Hassan" in dialog._people_list.get(0)
        # The selected project's view is drawn in the sidebar.
        assert "Flowlines" in dialog._project_text.get("1.0", "end")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_competency_matrix_is_shown_in_the_transcript(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        person = service.domain.create_person(name="Mona")
        service.domain.add_qualification(person_id=person.record_id, title="CSWIP 3.1")
        dialog._on_show_matrix()
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "CSWIP 3.1" in text
        assert "Mona" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_library_section_lists_a_standard(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.add_standard(body="API", number="1104", title="Welding of Pipelines")
        dialog._refresh_library()
        assert dialog._library_list.size() == 1
        assert "API 1104" in dialog._library_list.get(0)
        assert "Official standard" in dialog._library_kinds_label.cget("text")
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_revision_comparison_is_shown(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        first = service.add_standard(
            body="API", number="1104", edition="21st", published_on="2013-01-01"
        )
        service.record_standard_revision(
            first.record_id, edition="22nd", published_on="2021-09-01"
        )
        dialog._refresh_library()
        dialog._library_list.selection_clear(0, "end")
        dialog._library_list.selection_set(0)
        dialog._on_compare_revisions()
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "22nd" in text and "edition" in text
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_document_registry_and_verification(tmp_path: Path) -> None:
    root, dialog, service = _dialog(tmp_path)
    try:
        service.register_document_text("Coating procedure text.", title="Coating procedure")
        dialog._refresh_library()
        assert dialog._document_list.size() == 1
        dialog._document_list.selection_clear(0, "end")
        dialog._document_list.selection_set(0)
        dialog._on_read_document()
        dialog._on_verify_document()
        text = "\n".join(w.get("1.0", "end") for w in _all_text_widgets(dialog._transcript))
        assert "Coating procedure text." in text
        assert "matches" in text
    finally:
        _close_window(root, dialog)
