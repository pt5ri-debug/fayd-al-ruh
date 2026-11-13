"""The Training Center's bilingual help: the guide, the CLI, and the window.

The guide is generated from the catalogue, so these tests prove three things:
that it is genuinely bilingual and complete, that the CLI and the window show the
same text, and that the **?** control beside the window's minimize / maximize /
close buttons really opens it (and reuses one window). The window parts skip
without a display, exactly like the other window tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli import training_center as cli
from core.training_center.catalog import CAPABILITIES, SECTIONS, capabilities_for
from core.training_center.help import (
    LANGUAGES,
    help_capability_count,
    help_sections,
    help_text,
    help_title,
    resolve_language,
)

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - no libtk on this machine
    tk = None  # type: ignore[assignment]

from tests.test_gui_design import _display_available

# -- the guide text -------------------------------------------------------


def test_the_guide_is_bilingual() -> None:
    arabic = help_text("ar")
    english = help_text("en")
    assert arabic != english
    assert "مركز التدريب" in arabic
    assert "Training Center" in english


def test_the_guide_documents_every_control_in_both_languages() -> None:
    for language in LANGUAGES:
        text = help_text(language)
        for cap in CAPABILITIES:
            title = cap.title_ar if language == "ar" and cap.title_ar else cap.title
            assert title in text, f"{cap.key} missing from the {language} guide"
        for section in SECTIONS:
            if capabilities_for(section):
                assert section in help_sections()


def test_the_guide_count_matches_the_catalogue() -> None:
    assert help_capability_count() == len(CAPABILITIES)


def test_the_guide_explains_what_happens_and_the_command() -> None:
    english = help_text("en")
    # Each control's own return value and its equivalent command are present, so a
    # reader learns what pressing it does, not only its name.
    cap = CAPABILITIES[0]
    assert cap.outcome in english
    assert cap.cli in english


def test_the_guide_states_the_safety_rules() -> None:
    english = help_text("en")
    assert "unreviewed" in english.lower()
    assert "Natasha is private" in english


def test_unknown_language_falls_back_to_english() -> None:
    assert resolve_language("fr") == "en"
    assert resolve_language("") == "en"
    assert resolve_language("ar") == "ar"
    assert resolve_language("AR") == "ar"


def test_titles_are_bilingual() -> None:
    assert help_title("ar") != help_title("en")


# -- the CLI --------------------------------------------------------------


def _run(capsys, *argv: str, root: Path) -> tuple[int, str]:
    code = cli.main(["--root", str(root), *argv])
    return code, capsys.readouterr().out


def test_cli_help_prints_the_arabic_guide(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "help", root=tmp_path)
    assert code == cli.EXIT_OK
    assert help_title("ar") in out


def test_cli_help_english(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "help", "--lang", "en", root=tmp_path)
    assert code == cli.EXIT_OK
    assert help_title("en") in out


def test_cli_help_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    code, out = _run(capsys, "help", "--lang", "en", "--json", root=tmp_path)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["language"] == "en"
    assert payload["capabilities"] == len(CAPABILITIES)
    assert payload["title"] == help_title("en")
    assert payload["text"] == help_text("en")


def test_the_menu_carries_the_guide_entry() -> None:
    from cli.menu import MENU, run_training_center

    titles = [title for title, _action, _args in MENU]
    assert any("complete guide" in title for title in titles)
    entry = next(e for e in MENU if "complete guide" in e[0])
    assert entry[1] is run_training_center
    assert entry[2] == ("help",)


# -- the window -----------------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_header_carries_a_help_control() -> None:
    from tests.test_training_center_window import StubService, _dialog

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        button = dialog._window_controls.help_button
        assert button is not None
        assert str(button.cget("text")) == "?"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_pressing_help_opens_the_guide_window_and_reuses_it() -> None:
    from gui.training_center_help import TrainingCenterHelpDialog

    from tests.test_training_center_window import StubService, _dialog

    root = tk.Tk()
    try:
        dialog = _dialog(root, StubService())
        dialog.open_help()
        root.update()
        help_window = dialog._help_window
        assert isinstance(help_window, TrainingCenterHelpDialog)
        first = help_window.window
        dialog.open_help()  # a second press must reuse the open window
        root.update()
        assert dialog._help_window.window is first
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_help_window_switches_language_in_place() -> None:
    from gui.app import DARK_THEME
    from gui.training_center_help import TrainingCenterHelpDialog

    root = tk.Tk()
    try:
        window = TrainingCenterHelpDialog(
            root, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono", language="ar"
        )
        root.update()
        assert "مركز التدريب" in window._text.get("1.0", "end")
        window._toggle_language()
        root.update()
        english = window._text.get("1.0", "end")
        assert "Training Center" in english
        assert help_title("en") in window.window.title()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_help_window_draws_the_heading_markup() -> None:
    from gui.app import DARK_THEME
    from gui.training_center_help import TrainingCenterHelpDialog

    root = tk.Tk()
    try:
        window = TrainingCenterHelpDialog(
            root, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono", language="en"
        )
        root.update()
        rendered = window._text.get("1.0", "end")
        # The plain guide's underlines became Markdown headings and the bullets a
        # Markdown list, so the shared painter styles them.
        assert "# " in rendered
        assert "## " in rendered
        assert "### " in rendered
        assert "- " in rendered
        # No underline rule survives as literal text.
        assert "====" not in rendered
    finally:
        root.destroy()
