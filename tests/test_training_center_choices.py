"""The Training Center's dropdowns: the choice dialog and the window's prompts.

The pure label/value resolution runs with no display; the window parts are
skipped unless a display (or Xvfb) is available, exactly like
``tests/test_training_center_window.py``. No model, network, or training run ever
happens here -- the window is opened with a stand-in service.
"""

from __future__ import annotations

import tkinter as tk

import pytest
from gui.choice_dialog import display_for, normalise_choices, resolve_choice

from tests.test_gui_design import _display_available
from tests.test_training_center_window import StubService, _dialog

# -- the pure label/value resolution --------------------------------------


def test_normalise_splits_pairs_and_plain_strings() -> None:
    rows, displays = normalise_choices(["a", ("b", "B label")])
    assert rows == (("a", "a"), ("b", "B label"))
    assert displays == ("a", "B label")


def test_normalise_drops_a_duplicate_label_but_keeps_both_rows() -> None:
    rows, displays = normalise_choices([("a", "Same"), ("b", "Same")])
    assert rows == (("a", "Same"), ("b", "Same"))
    # The label appears once in the dropdown, so it never shadows a value.
    assert displays == ("Same",)


def test_resolve_maps_a_label_to_its_value_and_keeps_an_unknown_name() -> None:
    rows, _ = normalise_choices([("manual", "A hand-written answer"), ("document", "The sources")])
    assert resolve_choice("The sources", rows) == "document"  # a picked label
    assert resolve_choice("document", rows) == "document"  # a typed value
    assert resolve_choice("a-new-dataset", rows) == "a-new-dataset"  # free typing


def test_display_for_shows_a_known_value_by_its_label() -> None:
    rows, _ = normalise_choices([("manual", "A hand-written answer")])
    assert display_for("manual", rows) == "A hand-written answer"
    assert display_for("unknown", rows) == "unknown"
    assert display_for("", rows) == ""


# -- the window's dropdown wiring (needs a display) -----------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_has_a_web_switch_that_reaches_the_service(monkeypatch) -> None:
    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        toggle = dialog._web_toggle
        assert toggle is not None
        assert service.web_enabled() is False
        # Clicking the switch turns the internet on and the service is told.
        toggle._on_click()
        assert service.web_enabled() is True
        assert toggle._on is True
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_setting_a_setting_uses_the_choice_dialog(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        answers = iter(["chunk_chars", "900"])
        monkeypatch.setattr(dialog_module, "ask_choice", lambda *a, **k: next(answers))
        dialog._on_set_setting()
        assert service.settings_set == [("chunk_chars", 900)]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_setting_the_web_key_refreshes_the_switch(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        answers = iter(["web", "true"])
        monkeypatch.setattr(dialog_module, "ask_choice", lambda *a, **k: next(answers))
        dialog._on_set_setting()
        assert service.web_enabled() is True
        assert dialog._web_toggle._on is True
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_cloud_action_turns_the_internet_on_when_the_user_agrees(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        monkeypatch.setattr(dialog_module.messagebox, "askyesno", lambda *a, **k: True)
        assert dialog._require_web("Asking a cloud model") is True
        assert service.web_enabled() is True
        assert dialog._web_toggle._on is True
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_cloud_action_is_refused_when_the_user_declines(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        monkeypatch.setattr(dialog_module.messagebox, "askyesno", lambda *a, **k: False)
        assert dialog._require_web("Asking a cloud model") is False
        assert service.web_enabled() is False
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_asking_a_question_uses_the_provider_dropdown(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        answers = iter(["What is porosity?", "manual"])
        monkeypatch.setattr(dialog_module.simpledialog, "askstring", lambda *a, **k: next(answers))
        monkeypatch.setattr(dialog_module, "ask_choice", lambda *a, **k: next(answers))
        dialog._on_ask()
        # The provider dropdown chose a real value the service accepts.
        assert service.asked == [("What is porosity?", "manual")]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_learning_topic_is_an_editable_dropdown(monkeypatch) -> None:
    import gui.training_center_dialog as dialog_module

    root = tk.Tk()
    try:
        service = StubService()
        dialog = _dialog(root, service)
        captured: dict[str, object] = {}

        def fake_choice(*args, **kwargs):
            captured["values"] = kwargs.get("values")
            return ""

        monkeypatch.setattr(dialog_module, "ask_choice", fake_choice)
        dialog._on_generate()
        # The topic box offered the project's own topics, and cancelling stopped.
        assert captured["values"] == service.topic_choices()
    finally:
        root.destroy()
