"""Word-like font and paragraph settings, and the sidebar section that applies them.

Two halves. The **headless** half proves the settings themselves: a profile
validates and clamps a hand-edited value, it round-trips through the store, an
empty field means "keep the project's own", and a profile folds onto
:class:`gui.typography.Typography` so the family, the size, the styles, and the
paragraph really reach the font tuple and the block metrics. The **window** half
proves the feature the user asked for: a *Fonts & paragraphs* entry of its own in
the sidebar (below *Themes*) offers the controls for **three** scopes, the windows
scope restyles the main window *and* every dialog already open, and the chat scope
touches only the conversation -- the two never disturb each other.

The window parts are skipped unless a display (or Xvfb) is available, exactly like
``tests/test_gui.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from gui import rich_text
from gui.typography import SCALE, for_root, resolve
from gui.typography_settings import (
    ALIGNMENT_NOTE,
    ALIGNMENTS,
    FONT_SIZES,
    LINE_SPACINGS,
    LISTS_NOTE,
    MAX_LINE_SPACING,
    MAX_SIZE_PT,
    MIN_LINE_SPACING,
    SCOPE_CHAT,
    SCOPE_CONTROLS,
    SCOPE_LABELS,
    SCOPE_LISTS,
    SCOPE_SUMMARIES,
    SCOPE_WINDOWS,
    SCOPES,
    SPACE_STEPS,
    ParagraphStyle,
    TypographyProfile,
    default_profile,
    load_profile,
    load_profiles,
    save_profile,
    save_profiles,
    scope_shows,
)


def _bubble_widgets(widget: Any) -> list[Any]:
    """Every ``Text`` under ``widget`` (the message bodies of a transcript)."""
    import tkinter as tk

    found: list[Any] = []
    for child in widget.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_bubble_widgets(child))
    return found


def _texts(widget: Any) -> list[str]:
    """The text of every message body under ``widget``, in order."""
    return [w.get("1.0", "end-1c") for w in _bubble_widgets(widget)]


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point the whole program's data folder at a fresh temp directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


# ------------------------------------------------------------------ the scopes


def test_the_three_scopes_are_named_and_described() -> None:
    """Each scope has a readable label and a one-line summary."""
    assert SCOPES == (SCOPE_WINDOWS, SCOPE_LISTS, SCOPE_CHAT)
    for scope in SCOPES:
        assert SCOPE_LABELS[scope]
        assert SCOPE_SUMMARIES[scope]


def test_the_lists_scope_offers_only_what_a_list_box_can_draw() -> None:
    """A ``Listbox`` has no paragraph controls, so the lists block shows none."""
    assert scope_shows(SCOPE_LISTS, "font")
    assert scope_shows(SCOPE_LISTS, "size")
    assert scope_shows(SCOPE_LISTS, "styles")
    assert not scope_shows(SCOPE_LISTS, "align")
    assert not scope_shows(SCOPE_LISTS, "line_spacing")
    assert not scope_shows(SCOPE_LISTS, "space")
    # The two text scopes keep every control.
    for scope in (SCOPE_WINDOWS, SCOPE_CHAT):
        for feature in ("font", "size", "styles", "align", "line_spacing", "space"):
            assert scope_shows(scope, feature), (scope, feature)
    # An unknown scope shows nothing rather than raising.
    assert not scope_shows("nonsense", "font")


def test_the_control_sets_name_the_same_features_the_panel_draws() -> None:
    """The declaration and the panel cannot drift: both use the same names."""
    assert set(SCOPE_CONTROLS) == set(SCOPES)


# ---------------------------------------------------------------- the profile


def test_a_default_profile_keeps_the_projects_own_look() -> None:
    """An untouched profile is empty and is the project's own, per scope."""
    for scope in SCOPES:
        profile = default_profile(scope)
        assert profile.is_default()
        assert profile.family == "" and profile.size_pt == 0
        assert not profile.bold and not profile.italic
        assert profile.paragraph == ParagraphStyle()


def test_a_profile_round_trips_through_its_primitive() -> None:
    """A saved profile reads back exactly, field for field."""
    profile = TypographyProfile(
        family="DejaVu Serif",
        mono_family="DejaVu Sans Mono",
        size_pt=16,
        bold=True,
        italic=True,
        paragraph=ParagraphStyle(alignment="center", line_spacing=1.5, space_before=4, space_after=6),
    )
    assert TypographyProfile.from_primitive(profile.to_primitive()) == profile


def test_a_hand_edited_profile_is_clamped_not_trusted() -> None:
    """A bad size, spacing, or alignment falls to a valid value, never a crash."""
    profile = TypographyProfile.from_primitive(
        {
            "size_pt": 10_000,
            "paragraph": {"alignment": "diagonal", "line_spacing": 99, "space_before": -5},
        }
    )
    assert profile.size_pt == MAX_SIZE_PT
    assert profile.paragraph.alignment == "left"
    assert MIN_LINE_SPACING <= profile.paragraph.line_spacing <= MAX_LINE_SPACING
    assert profile.paragraph.space_before == 0


def test_a_malformed_primitive_reads_as_the_default() -> None:
    """A store value that is not even a dict is the project's own, not an error."""
    assert TypographyProfile.from_primitive("nonsense").is_default()
    assert TypographyProfile.from_primitive(None).is_default()
    assert ParagraphStyle.from_primitive([1, 2, 3]) == ParagraphStyle()


def test_the_setters_keep_a_profile_valid() -> None:
    """Each Word-like setter returns a new, validated profile."""
    profile = TypographyProfile()
    assert profile.with_family("Georgia").family == "Georgia"
    assert profile.with_size(14).size_pt == 14
    assert profile.with_bold(True).bold is True
    assert profile.with_italic(True).italic is True
    assert profile.with_size(10_000).size_pt == MAX_SIZE_PT
    assert profile.with_family("Georgia").reset().is_default()


def test_describe_names_the_choices_in_force() -> None:
    """The status line reads as the settings, so a user can see what is applied."""
    profile = TypographyProfile(
        family="Georgia", size_pt=14, bold=True,
        paragraph=ParagraphStyle(alignment="center", line_spacing=1.5),
    )
    text = profile.describe()
    assert "Georgia" in text and "14" in text and "Bold" in text
    assert "Center" in text and "1.5" in text
    assert "Project font" in TypographyProfile().describe()


# --------------------------------------------------------------- persistence


def test_profiles_round_trip_through_the_store(data_root) -> None:
    """Two scopes save and load independently."""
    save_profiles(
        {
            SCOPE_WINDOWS: TypographyProfile(family="Georgia", size_pt=13),
            SCOPE_CHAT: TypographyProfile(paragraph=ParagraphStyle(alignment="right")),
            SCOPE_LISTS: TypographyProfile(size_pt=11),
        }
    )
    loaded = load_profiles()
    assert loaded[SCOPE_WINDOWS].family == "Georgia"
    assert loaded[SCOPE_WINDOWS].size_pt == 13
    assert loaded[SCOPE_CHAT].paragraph.alignment == "right"
    assert loaded[SCOPE_LISTS].size_pt == 11


def test_saving_one_scope_keeps_the_others(data_root) -> None:
    """A change to one scope never clears another's own choice."""
    save_profile(SCOPE_WINDOWS, TypographyProfile(family="Georgia"))
    save_profile(SCOPE_CHAT, TypographyProfile(size_pt=18))
    assert load_profile(SCOPE_WINDOWS).family == "Georgia"
    assert load_profile(SCOPE_CHAT).size_pt == 18


def test_a_broken_store_reads_as_the_default(monkeypatch) -> None:
    """An unreadable store leaves the project's own settings, never a crash."""
    from storage import json_store

    class _Boom:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def load(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("no store")

        def save(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("no store")

    monkeypatch.setattr(json_store, "JsonStore", _Boom)
    loaded = load_profiles()
    assert all(loaded[scope].is_default() for scope in SCOPES)
    # A save that fails is recorded, not raised, so a window still opens.
    save_profiles(loaded)


def test_an_unknown_scope_in_the_store_is_ignored(data_root) -> None:
    """A stale scope name in the file does not invent a profile."""
    from core.paths import data_dir
    from storage.json_store import JsonStore

    JsonStore(data_dir()).save(
        "gui", "typography", {"windows": {"size_pt": 15}, "retired": {"size_pt": 40}}
    )
    loaded = load_profiles()
    assert set(loaded) == set(SCOPES)
    assert loaded[SCOPE_WINDOWS].size_pt == 15


# ---------------------------------------------------- folding onto the scale


def test_a_profile_folds_onto_the_typography_scale() -> None:
    """The family, the size, the styles, and the paragraph reach the font tuple."""
    base = for_root(None)
    profile = TypographyProfile(
        family="DejaVu Serif",
        size_pt=20,
        bold=True,
        italic=True,
        paragraph=ParagraphStyle(alignment="center", line_spacing=1.5, space_before=4, space_after=6),
    )
    typed = base.with_overrides(profile)
    body = typed.font("body")
    assert body[0] == "DejaVu Serif"
    assert "bold" in body and "italic" in body
    assert typed.justify() == "center"
    assert typed.line_spacing == 1.5
    metrics = typed.block_metrics_full("body", gap_px=1)
    assert metrics.spacing1 >= 4 and metrics.spacing3 >= 6


def test_an_empty_profile_leaves_the_scale_exactly_as_it_was() -> None:
    """The project's own look is untouched when nothing was chosen."""
    base = for_root(None)
    typed = base.with_overrides(TypographyProfile())
    assert typed == base
    assert typed.ui_family == base.ui_family
    assert typed.size_scale == 1.0


def test_a_chosen_size_sets_the_body_text_size() -> None:
    """Word's font size is the body size; the other roles scale with it."""
    base = for_root(None)
    typed = base.with_overrides(TypographyProfile(size_pt=16))
    # The requested point size is what the body draws at, within rounding.
    assert typed.points("body") == 16
    # A larger body means a larger heading, not a smaller one.
    assert typed.points("heading1") > typed.points("body")


def test_an_explicit_weight_still_wins_over_the_profile() -> None:
    """A heading stays bold and a muted caption stays normal, whatever the profile."""
    typed = for_root(None).with_overrides(TypographyProfile(bold=True))
    assert "bold" in typed.font("body")
    # An explicit weight still wins: a muted caption can ask for normal.
    assert "bold" not in typed.font("body", weight="normal")
    # A role that is bold anyway is unaffected by a normal body.
    plain = for_root(None).with_overrides(TypographyProfile())
    assert "bold" in plain.font("heading1")


def test_a_bad_paragraph_falls_to_a_drawable_alignment() -> None:
    """An alignment Tk cannot draw is normalised before it reaches a tag."""
    typed = for_root(None).with_overrides(
        TypographyProfile(paragraph=ParagraphStyle(alignment="justify"))
    )
    assert typed.justify() == "left"


def test_the_size_scale_multiplies_every_role() -> None:
    """The scale's own sizes are all multiplied, so the whole window grows."""
    base = for_root(None)
    bigger = base.with_overrides(TypographyProfile(size_pt=base.points("body") * 2))
    assert bigger.points("heading1") >= base.points("heading1") * 2 - 1


def test_resolve_honours_an_overrides_profile() -> None:
    """``resolve`` is the one place a dialog's scale is built."""
    typed = resolve(None, overrides=TypographyProfile(family="DejaVu Serif", size_pt=12))
    assert typed.ui_family == "DejaVu Serif"
    assert typed.points("body") == 12
    # With no profile, resolve is the project's own.
    assert resolve(None).size_scale == 1.0


def test_the_scale_itself_is_unchanged_by_the_settings_layer() -> None:
    """The settings layer never mutates the shared OpenHands scale."""
    before = {role: (step.size, step.leading) for role, step in SCALE.items()}
    for_root(None).with_overrides(TypographyProfile(family="Georgia", size_pt=20))
    after = {role: (step.size, step.leading) for role, step in SCALE.items()}
    assert before == after


def test_the_word_like_presets_are_offered() -> None:
    """The panel's menus draw from the module's own preset lists."""
    assert FONT_SIZES == tuple(sorted(FONT_SIZES))
    assert 11 in FONT_SIZES and 12 in FONT_SIZES
    assert 1.0 in LINE_SPACINGS and 2.0 in LINE_SPACINGS
    assert ALIGNMENTS == ("left", "center", "right")
    assert 0 in SPACE_STEPS
    assert "justification" in ALIGNMENT_NOTE
    assert "list box" in LISTS_NOTE


# --------------------------------------------------------------- the window


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_sidebar_has_a_fonts_section_below_themes(tmp_path, monkeypatch) -> None:
    """The section the user asked for exists, directly after *Themes*."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        assert "fonts" in app._sections
        assert "themes" in app._sections
        order = list(app._sections)
        assert order.index("fonts") == order.index("themes") + 1
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_opening_the_fonts_section_builds_a_block_per_scope(tmp_path, monkeypatch) -> None:
    """Each scope gets its own controls, and the lists block is the shorter one."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._toggle_fonts()
        root.update()
        # A block frame per scope, plus the note labels some blocks pack directly.
        assert len(app._fonts_panel.winfo_children()) >= len(SCOPES)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_windows_change_reaches_the_window_and_keeps_the_chat(tmp_path, monkeypatch) -> None:
    """The two scopes are independent: one moves, the other stays."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        chat_before = app._chat_type.points("body")
        app._set_font_size(SCOPE_WINDOWS, "16")
        root.update()
        assert app._type.points("body") == 16
        assert app._chat_type.points("body") == chat_before
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_chat_change_reaches_only_the_chat_scale(tmp_path, monkeypatch) -> None:
    """A chat-only change never touches the windows scale."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        windows_before = app._type.points("body")
        app._set_alignment(SCOPE_CHAT, "center")
        app._set_line_spacing(SCOPE_CHAT, 1.5)
        root.update()
        assert app._chat_type.justify() == "center"
        assert app._chat_type.line_spacing == 1.5
        assert app._type.points("body") == windows_before
        assert app._type.justify() == "left"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_lists_scope_has_its_own_scale(tmp_path, monkeypatch) -> None:
    """The window lists are tuned apart from the interface and the chat."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._set_font_family(SCOPE_LISTS, "DejaVu Serif")
        root.update()
        assert app._lists_type.ui_family == "DejaVu Serif"
        assert app._type.ui_family != "DejaVu Serif"
        assert app._chat_type.ui_family != "DejaVu Serif"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_change_is_persisted_for_the_next_launch(tmp_path, monkeypatch) -> None:
    """A choice survives a rebuild and a reload from the store."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._set_font_size(SCOPE_CHAT, "18")
        app._set_alignment(SCOPE_WINDOWS, "right")
        root.update()
        saved = load_profiles()
        assert saved[SCOPE_CHAT].size_pt == 18
        assert saved[SCOPE_WINDOWS].paragraph.alignment == "right"
        # A second window on the same data folder starts with the saved choices.
        app2 = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        assert app2._chat_type.points("body") == 18
        assert app2._type.justify() == "right"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_reset_returns_one_scope_to_the_project_default(tmp_path, monkeypatch) -> None:
    """Reset clears the scope it is pressed for and leaves the others alone."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._set_font_size(SCOPE_WINDOWS, "20")
        app._set_font_size(SCOPE_CHAT, "18")
        root.update()
        app._reset_scope(SCOPE_WINDOWS)
        root.update()
        assert app._profiles[SCOPE_WINDOWS].is_default()
        assert app._profiles[SCOPE_CHAT].size_pt == 18
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_bold_and_italic_toggles_reach_the_scale(tmp_path, monkeypatch) -> None:
    """The B / I controls are real toggles, not decoration."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._toggle_font_style(SCOPE_WINDOWS, "bold", True)
        app._toggle_font_style(SCOPE_WINDOWS, "italic", True)
        root.update()
        body = app._type.font("body")
        assert "bold" in body and "italic" in body
        app._toggle_font_style(SCOPE_WINDOWS, "bold", False)
        root.update()
        assert "bold" not in app._type.font("body")
    finally:
        root.destroy()


class _RecordingDialog:
    """A stub dialog: enough of the surface the main window's dispatch touches."""

    def __init__(self) -> None:
        self.applied: list[tuple[Any, Any]] = []
        self.window = cast(Any, None)

    def retype(self, overrides: Any = None, list_overrides: Any = None) -> None:
        self.applied.append((overrides, list_overrides))


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_windows_change_retypes_every_open_dialog(tmp_path, monkeypatch) -> None:
    """One change reaches the main window and every dialog it has open."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        single = _RecordingDialog()
        app._docs_studio_dialog = cast(Any, single)
        chat = _RecordingDialog()
        app._model_chat_dialogs = {"ultron": cast(Any, chat)}
        app._set_font_size(SCOPE_WINDOWS, "16")
        root.update()
        assert single.applied, "the single-window dialog was not re-typed"
        assert chat.applied, "the per-mouth dialog was not re-typed"
        # The lists profile travels with the windows profile.
        assert single.applied[-1][1] == app._profiles[SCOPE_LISTS]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_chat_change_does_not_retype_dialogs(tmp_path, monkeypatch) -> None:
    """A chat-only change leaves the dialogs exactly as they are."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        dialog = _RecordingDialog()
        app._docs_studio_dialog = cast(Any, dialog)
        app._set_alignment(SCOPE_CHAT, "center")
        root.update()
        assert dialog.applied == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_real_dialog_retypes_in_place(tmp_path, monkeypatch) -> None:
    """A real ``ThemedDialog`` rebuilds with the new scale, keeping its messages."""
    import tkinter as tk

    from gui.app import DARK_THEME as APP_DARK
    from gui.image_gen_dialog import ImageGenDialog

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))

    class _Service:
        backend = "procedural"

        def backend_status(self) -> Any:
            return (("procedural", True, "built in"),)

        def for_backend(self, *_a: Any, **_k: Any) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(available=lambda: True)

        def available(self) -> bool:
            return True

        def generate(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("not called")

        def download_model(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("not called")

    root = tk.Tk()
    try:
        dialog = ImageGenDialog(
            root, _Service(), theme=APP_DARK, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        assert dialog._type.size_scale == 1.0
        dialog.retype(TypographyProfile(size_pt=16), TypographyProfile(size_pt=11))
        root.update()
        assert dialog._type.points("body") == 16
        assert dialog._list_type.points("body") == 11
        assert dialog._composer is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_list_box_uses_the_lists_scale(tmp_path, monkeypatch) -> None:
    """The dialog list boxes draw with the lists scope, not the interface one."""
    import tkinter as tk

    from gui.app import DARK_THEME as APP_DARK
    from gui.docs_dialog import DocsDialog

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))

    class _Service:
        def documents(self) -> Any:
            return ()

        def model_status(self) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                available=False, summary=lambda: "no vision model installed"
            )

        def model_names(self) -> Any:
            return ()

        def model_summaries(self) -> Any:
            return ()

    root = tk.Tk()
    try:
        dialog = DocsDialog(
            root, cast(Any, _Service()), theme=APP_DARK, font="DejaVu Sans", mono="DejaVu Sans Mono",
            overrides=TypographyProfile(size_pt=20),
            list_overrides=TypographyProfile(size_pt=11),
        )
        root.update()
        assert dialog._type.points("body") == 20
        assert dialog._list_type.points("body") == 11
        # The real list box draws with the lists scale: Tk returns the font as a
        # string ("family size ..."), so the size is read back out of it.
        drawn = str(dialog._document_list.cget("font")).split()
        assert str(dialog._list_type.points("xs")) in drawn
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_transcript_survives_a_chat_font_change(tmp_path, monkeypatch) -> None:
    """A chat font change rebuilds the window but keeps every message."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._add_bubble("user", "كيف أقيس المسامية؟")
        app._add_bubble("assistant", "Measure it with a **density log**.")
        root.update()
        before = _texts(app._transcript)
        assert len(before) == 2
        app._set_font_size(SCOPE_CHAT, "18")
        root.update()
        assert _texts(app._transcript) == before
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_chat_alignment_reaches_the_bubble_and_the_markdown(tmp_path, monkeypatch) -> None:
    """The alignment is applied to the user's own bubble and to the answer's blocks."""
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.service import ChatService

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=ChatService(data_root=tmp_path))
        root.update()
        app._add_bubble("user", "كيف أقيس المسامية؟")
        app._add_bubble("assistant", "Measure it with a **density log**.")
        root.update()
        app._set_alignment(SCOPE_CHAT, "center")
        root.update()
        widgets = _bubble_widgets(app._transcript)
        user = next(w for w in widgets if "المسامية" in w.get("1.0", "end-1c"))
        answer = next(w for w in widgets if "density log" in w.get("1.0", "end-1c"))
        # The plain bubble carries the alignment tag over its whole content.
        assert user.tag_cget(rich_text.ALIGN_TAG, "justify") == "center"
        # The Markdown answer carries it on its own paragraph tag.
        assert answer.tag_cget("paragraph", "justify") == "center"
    finally:
        root.destroy()
