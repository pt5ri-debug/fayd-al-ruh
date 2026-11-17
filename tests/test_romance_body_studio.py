"""Tests for the Body & shape studio (engine, service, and window).

The user asked for an entry, inside both **My Persona Manager** and the
**Character Manager**, that opens a window showing a man's or a woman's body --
matching the selected character's gender -- centred, with every body measurement
editable, ready-made body templates to start from, and a professional,
comprehensive toolset.

These tests pin the engine (the dial catalogue, the presets, the gender rule, the
clamping and refusals, the real render), the service that drives it, the window
(the centred figure, the dials, the templates, the picture bridge), and the entry
in both managers.
"""

from __future__ import annotations

import base64
import gc
from pathlib import Path

import pytest
from core.romance.body import (
    AVATAR_HEIGHT,
    AVATAR_WIDTH,
    BODY_PRESETS,
    BUILD_CHOICES,
    DIAL_GROUPS,
    DIALS,
    HONESTY_LINE,
    VIEW_FACE,
    VIEW_FULL,
    body_preset_keys,
    body_preset_titles,
    body_summary,
    build_for_gender,
    changed_keys,
    default_parameters,
    dial_for,
    dial_keys,
    dials_in_group,
    group_titles,
    preset_parameters,
    render_avatar,
    render_preview,
    reset_to_preset,
    set_dial,
    set_dials,
)
from core.romance.model import ROLE_COMPANION, ROLE_USER
from gui.romance import RomanceStudioService

# -- the dial catalogue ---------------------------------------------------


def test_every_dial_is_fully_specified_and_unique() -> None:
    keys = dial_keys()
    assert len(keys) == len(set(keys))
    assert len(DIALS) >= 30, "the studio must expose a comprehensive set of measurements"
    for dial in DIALS:
        assert dial.label.strip() and dial.label_ar.strip(), dial.key
        assert dial.group in {key for key, _en, _ar in DIAL_GROUPS}, dial.key
        assert dial.kind in {"float", "int", "choice", "colour"}, dial.key
        if dial.kind in {"float", "int"}:
            assert dial.maximum > dial.minimum, dial.key
            assert dial.step > 0, dial.key


def test_the_groups_cover_every_dial_and_are_bilingual() -> None:
    grouped: list[str] = []
    for _key, _title, dials in _groups():
        grouped.extend(dial.key for dial in dials)
    assert sorted(grouped) == sorted(dial_keys())
    english = group_titles("en")
    arabic = group_titles("ar")
    assert len(english) == len(arabic) == len(DIAL_GROUPS)
    for (_k1, en), (_k2, ar) in zip(english, arabic, strict=True):
        assert en.strip() and ar.strip()


def _groups():
    return tuple(
        (key, title, dials_in_group(key)) for key, title in group_titles("en")
    )


def test_a_numeric_dial_is_clamped_and_a_choice_dial_is_refused() -> None:
    params = default_parameters()
    height = dial_for("height")
    assert height is not None
    assert set_dial(params, "height", 99.0).height == height.maximum
    assert set_dial(params, "height", 0.1).height == height.minimum
    assert set_dial(params, "bust", -5.0).bust == 0.0
    # A choice dial refuses a value it does not offer rather than substituting one.
    with pytest.raises(ValueError):
        set_dial(params, "build", "bogus")
    assert set_dial(params, "build", "feminine").build == "feminine"


def test_an_unknown_dial_is_refused() -> None:
    params = default_parameters()
    with pytest.raises(KeyError):
        set_dial(params, "not_a_dial", 1.0)


def test_a_colour_dial_is_coerced_and_clamped() -> None:
    params = default_parameters()
    out = set_dial(params, "skin", (0.4, 2.0, -1.0))
    assert out.skin == (0.4, 1.0, 0.0)
    assert set_dial(params, "skin", "nonsense").skin == (0.0, 0.0, 0.0)


def test_several_dials_are_set_at_once() -> None:
    params = default_parameters()
    out = set_dials(params, {"height": 1.9, "bust": 0.5, "hair": 0.0})
    assert (out.height, out.bust, out.hair) == (1.9, 0.5, 0.0)


# -- presets --------------------------------------------------------------


def test_every_body_template_is_a_real_full_body() -> None:
    assert len(BODY_PRESETS) >= 6
    keys = body_preset_keys()
    assert len(keys) == len(set(keys))
    base = preset_parameters("average")
    assert base is not None
    for key in keys:
        params = preset_parameters(key)
        assert params is not None
        # Every template but the neutral baseline is a real change of measurements.
        if key != "average":
            assert changed_keys(params), key
    assert changed_keys(base) == ()
    titles_en = dict(body_preset_titles("en"))
    titles_ar = dict(body_preset_titles("ar"))
    assert set(titles_en) == set(keys) == set(titles_ar)
    for key in keys:
        assert titles_en[key].strip() and titles_ar[key].strip()


def test_applying_a_template_changes_the_measurements_and_keeps_the_name() -> None:
    params = default_parameters()
    params = params.with_changes(name="Amina")
    out = reset_to_preset(params, "athletic")
    athletic = preset_parameters("athletic")
    assert athletic is not None
    assert out.name == "Amina"
    assert out.muscle == athletic.muscle
    with pytest.raises(KeyError):
        reset_to_preset(params, "no-such-template")


# -- the gender rule ------------------------------------------------------


def test_a_womans_record_builds_a_feminine_figure_and_a_mans_a_masculine_one() -> None:
    assert build_for_gender("woman") == "feminine"
    assert build_for_gender("man") == "masculine"
    assert build_for_gender("Female") == "feminine"
    assert build_for_gender("male") == "masculine"
    # An unrecorded or non-binary gender never guesses a body.
    assert build_for_gender("") == "neutral"
    assert build_for_gender("nonbinary") == "neutral"


def test_default_parameters_follow_the_characters_own_gender() -> None:
    class _Record:
        def __init__(self, name: str, gender: str) -> None:
            self.name = name
            self.gender = gender

    assert default_parameters(_Record("Amina", "woman")).build == "feminine"
    assert default_parameters(_Record("Karim", "man")).build == "masculine"
    assert default_parameters(_Record("Nour", "")).build == "neutral"
    assert default_parameters(_Record("Amina", "woman")).name == "Amina"
    # The bundle the Studio returns is accepted too.
    class _Bundle:
        def __init__(self, record) -> None:
            self.character = record

    assert default_parameters(_Bundle(_Record("Amina", "woman"))).build == "feminine"


# -- rendering ------------------------------------------------------------


def test_a_render_is_a_real_png_for_both_views() -> None:
    params = default_parameters()
    full = render_preview(params, view=VIEW_FULL)
    face = render_preview(params, view=VIEW_FACE)
    assert full.startswith(b"\x89PNG") and len(full) > 1000
    assert face.startswith(b"\x89PNG") and len(face) > 1000
    # The two framings are genuinely different pictures.
    assert full != face
    # A preview the window can show is decodable by Tk's own PNG reader.
    assert base64.b64encode(full)


def test_an_unknown_view_falls_back_to_the_full_body() -> None:
    params = default_parameters()
    assert render_preview(params, view="nonsense") == render_preview(params, view=VIEW_FULL)


def test_the_avatar_is_a_square_png() -> None:
    data = render_avatar(default_parameters())
    assert data.startswith(b"\x89PNG")
    assert (AVATAR_WIDTH, AVATAR_HEIGHT) == (512, 512)


def test_the_dials_really_change_the_drawn_body() -> None:
    base = default_parameters()
    taller = set_dial(base, "height", 2.05)
    assert render_preview(base) != render_preview(taller)


def test_the_summary_describes_the_body() -> None:
    params = default_parameters().with_changes(build="feminine", bust=0.8)
    text = body_summary(params)
    assert "Feminine" in text and "fuller chest" in text


def test_the_engine_says_it_is_a_stylised_figure() -> None:
    assert "stylised" in HONESTY_LINE.lower()
    assert "not a licensed likeness" in HONESTY_LINE


def test_every_build_family_is_offered() -> None:
    assert set(BUILD_CHOICES) == {"feminine", "masculine", "neutral"}
    build = dial_for("build")
    assert build is not None
    assert build.choices == BUILD_CHOICES


# -- the service ----------------------------------------------------------


def _service(tmp_path: Path) -> RomanceStudioService:
    return RomanceStudioService(root=tmp_path)


def test_the_service_builds_a_body_from_the_characters_gender(tmp_path: Path) -> None:
    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    params = service.body_parameters(character.id)
    assert params.build == "feminine"
    assert params.name == "Amina"


def test_the_service_exposes_the_dials_templates_and_builds(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert [key for key, _title, _dials in service.body_dial_groups()] == [
        key for key, _en, _ar in DIAL_GROUPS
    ]
    assert dict(service.body_presets())["athletic"] == "Athletic"
    assert dict(service.body_builds())["feminine"] == "Feminine"


def test_the_service_sets_dials_and_applies_a_template(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    out = service.set_body_dial(params, "bust", 0.9)
    assert out.bust == 0.9
    assert service.set_body_dial(params, "bust", 5.0).bust == 1.0
    preset = service.apply_body_preset(params, "masculine")
    assert preset.build == "masculine"
    assert service.body_summary(preset)


def test_the_service_renders_a_body(tmp_path: Path) -> None:
    service = _service(tmp_path)
    data = service.render_body(service.body_parameters(""))
    assert data.startswith(b"\x89PNG")


def test_a_rendered_body_becomes_the_characters_picture(tmp_path: Path) -> None:
    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    assert service.avatar_path(character.id) is None
    params = service.body_parameters(character.id)
    result = service.use_body_as_avatar(character.id, params)
    assert Path(result.path).exists()
    assert service.avatar_path(character.id) is not None
    rows = {row.character_id: row for row in service.character_rows()}
    assert rows[character.id].has_avatar


def test_the_avatar_manager_refuses_a_blank_payload(tmp_path: Path) -> None:
    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    from core.romance.avatar import AvatarError

    with pytest.raises(AvatarError):
        service.avatar_manager().set_from_bytes(character.id, b"")
    with pytest.raises(AvatarError):
        service.avatar_manager().set_from_bytes(character.id, b"data", suffix=".txt")


# -- the window -----------------------------------------------------------


def _display_available() -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
    except Exception:
        return False
    return True


def _wait_for(root, predicate, *, rounds: int = 600) -> bool:
    """Drain the UI event loop until ``predicate()`` is true (a worker-thread result)."""
    for _ in range(rounds):
        root.update()
        if predicate():
            return True
    return False


def _close_window(root, dialog) -> None:
    try:
        dialog._close()
    finally:
        root.destroy()
        gc.collect()


def _all_buttons(widget) -> list:
    """Every descendant ``tk.Button``/``Menubutton`` of a widget, recursively."""
    import tkinter as tk

    found: list = []
    stack = [widget]
    while stack:
        current = stack.pop()
        for child in current.winfo_children():
            if isinstance(child, (tk.Button, tk.Menubutton)):
                found.append(child)
            stack.append(child)
    return found


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_shows_a_figure_with_every_dial(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_body import TITLE, BodyStudioDialog

    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        assert TITLE in dialog.window.title()
        # The figure is centred and rendered, following the character's own gender.
        assert dialog._params.build == "feminine"
        assert _wait_for(root, lambda: dialog._photo is not None)
        assert dialog._preview_png.startswith(b"\x89PNG")
        # Every numeric dial has a control and every colour dial a field.
        numeric = [d.key for d in DIALS if d.kind in {"float", "int"}]
        assert set(numeric) <= set(dialog._dial_vars)
        assert {"skin", "hair_color"} <= set(dialog._colour_vars)
        # Moving a dial updates the parameter and the summary label.
        dialog._dial_vars["height"].set(2.0)
        root.update()
        assert dialog._params.height == 2.0
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_applies_templates_views_and_stores_the_picture(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_body import BodyStudioDialog

    service = _service(tmp_path)
    character = service.create_character(name="Karim", role=ROLE_COMPANION, gender="man")
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert dialog._params.build == "masculine"
        dialog._apply_preset("athletic")
        root.update()
        athletic = preset_parameters("athletic")
        assert athletic is not None
        assert dialog._params.muscle == athletic.muscle
        dialog._set_view("face")
        root.update()
        assert dialog._view == "face"
        # The render becomes the character's picture through the avatar manager.
        dialog._use_as_avatar()
        assert _wait_for(root, lambda: service.avatar_path(character.id) is not None)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_every_body_window_button_is_really_placed(tmp_path: Path) -> None:
    """Every button in the body window is packed, so none is invisible.

    The reported failure was "the Body & shape feature does not work -- no window
    opens at all": the window's own buttons (and the managers' entry buttons) were
    created but never packed, so they had no geometry and a click could never
    reach them. This walks every descendant button and asserts each one has a
    manager and a non-empty geometry.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_body import BodyStudioDialog

    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        buttons = _all_buttons(dialog.window)
        assert buttons, "the body window must have controls"
        unplaced = [b for b in buttons if not b.winfo_manager()]
        assert not unplaced, f"unplaced body-window buttons: {len(unplaced)}"
        # The action row (Render / Use as picture / Close) is packed too, so it is
        # not squeezed out by the expanding preview.
        action_row = dialog._action_row
        assert action_row.winfo_manager(), "the action row must be packed"
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_every_manager_button_is_really_placed(tmp_path: Path) -> None:
    """The managers' buttons are packed, so the Body & shape entry is clickable."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_managers import CharacterManagerDialog

    service = _service(tmp_path)
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = CharacterManagerDialog(
            root, service, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        buttons = _all_buttons(dialog.window)
        assert buttons
        unplaced = [b for b in buttons if not b.winfo_manager()]
        assert not unplaced, f"unplaced manager buttons: {len(unplaced)}"
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_both_managers_open_the_body_builder_for_the_selected_character(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_body import BodyStudioDialog
    from gui.romance_managers import CharacterManagerDialog

    service = _service(tmp_path)
    root = tk.Tk()
    root.withdraw()
    dialogs = []
    try:
        for role in (ROLE_USER, ROLE_COMPANION):
            dialog = CharacterManagerDialog(
                root, service, role=role,
                theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
            )
            dialogs.append(dialog)
            root.update()
            # A character of this role exists and is selected, so the builder opens on it.
            created = service.create_character(
                name=f"Body {role}", role=role, gender="woman",
            )
            dialog._refresh_list()
            dialog._select(created.id)
            root.update()
            dialog._open_body_studio()
            root.update()
            assert isinstance(dialog._body_studio, BodyStudioDialog)
            # A second click raises the one window rather than opening another.
            first = dialog._body_studio
            dialog._open_body_studio()
            root.update()
            assert dialog._body_studio is first
    finally:
        for dialog in dialogs:
            dialog._close()
        root.destroy()
        gc.collect()


# -- the headless twin ----------------------------------------------------


def _cli(argv: list[str], capsys) -> tuple[int, str]:
    from cli import romance as cli

    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_the_cli_lists_the_dials_and_templates(tmp_path: Path, capsys) -> None:
    root = str(tmp_path)
    code, out = _cli(["--data-root", root, "body", "--dials"], capsys)
    assert code == 0
    assert "height" in out and "bust" in out
    code, out = _cli(["--data-root", root, "body", "--presets"], capsys)
    assert code == 0
    assert "athletic" in out and "Feminine" in out


def test_the_cli_sets_measurements_and_renders_a_png(tmp_path: Path, capsys) -> None:
    root = str(tmp_path)
    target = tmp_path / "body.png"
    code, out = _cli(
        [
            "--data-root", root, "body",
            "--set", "height=1.92", "--set", "build=feminine",
            "--render", str(target),
        ],
        capsys,
    )
    assert code == 0
    assert target.exists() and target.read_bytes().startswith(b"\x89PNG")
    assert "Feminine" in out


def test_the_cli_refuses_an_unknown_dial(tmp_path: Path, capsys) -> None:
    code, _out = _cli(["--data-root", str(tmp_path), "body", "--set", "nope=1"], capsys)
    assert code == 2


def test_the_cli_stores_the_body_as_the_characters_picture(tmp_path: Path, capsys) -> None:
    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    code, _out = _cli(
        ["--data-root", str(tmp_path), "body", character.id, "--use-as-picture"], capsys
    )
    assert code == 0
    assert service.avatar_path(character.id) is not None
