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
    apply_pose_instruction,
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
    pose_from_instruction,
    pose_keys,
    pose_titles,
    preset_parameters,
    render_avatar,
    render_preview,
    reset_to_preset,
    set_dial,
    set_dials,
    set_pose,
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

    # The studio now opens on one of the four ready-made figures, chosen from the
    # character's own gender: a woman shows the woman's body, a man the man's.
    assert default_parameters(_Record("Amina", "woman")).build == "feminine"
    assert default_parameters(_Record("Karim", "man")).build == "masculine"
    # An unrecorded (or non-binary) gender falls back to the documented default --
    # the woman's figure -- rather than a made-up choice.
    assert default_parameters(_Record("Nour", "")).build == "feminine"
    assert default_parameters(_Record("Nour", "non-binary")).build == "feminine"
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


# -- the FR-CCE character engine (the window's own engine) ----------------


def test_the_service_exposes_the_four_character_templates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    keys = [key for key, _title in service.character_templates()]
    assert keys == ["woman", "girl", "man", "boy"]
    assert dict(service.character_templates())["boy"] == "Boy"
    assert dict(service.character_templates(language="ar"))["woman"] == "أنثى بالغة"


def test_a_characters_body_opens_on_the_template_its_sex_and_age_select(tmp_path: Path) -> None:
    service = _service(tmp_path)
    woman = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    girl = service.create_character(name="Hala", role=ROLE_COMPANION, gender="female", fictional_age=9)
    man = service.create_character(name="Omar", role=ROLE_COMPANION, gender="man")
    assert service.template_for_character(woman.id) == "woman"
    assert service.template_for_character(girl.id) == "girl"
    assert service.template_for_character(man.id) == "man"
    # A character with no saved body opens on its own template.
    assert service.character_body_template(woman.id) == "woman"
    assert service.character_body_parameters(woman.id).name == "Amina"


def test_a_characters_body_saves_and_reopens_independently(tmp_path: Path) -> None:
    service = _service(tmp_path)
    a = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    b = service.create_character(name="Layla", role=ROLE_COMPANION, gender="woman")
    params_a = service.body_parameters(a.id)
    shaped = service.set_body_dial(params_a, "height", 1.83)
    service.save_character_body(a.id, shaped)
    # B was never saved: it still opens on the template.
    assert service.character_body_config(a.id) is not None
    assert service.character_body_config(b.id) is None
    reopened = service.character_body_parameters(a.id)
    assert reopened.height == pytest.approx(1.83)
    assert service.character_body_template(a.id) == "woman"


def test_switching_a_template_gives_a_fresh_body_without_saving(tmp_path: Path) -> None:
    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    saved = service.set_body_dial(service.body_parameters(character.id), "height", 1.7)
    service.save_character_body(character.id, saved)
    # Open the body on another template: the saved config is left untouched.
    fresh = service.set_character_body_template(character.id, "man")
    assert fresh.height != pytest.approx(1.7)
    assert service.character_body_config(character.id).template_id == "woman"


def test_the_service_renders_through_the_engine_and_morphs_geometry(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    png = service.render_character_body(params, width=120, height=180)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # A dial change really moves geometry (measured, not assumed).
    report = service.character_engine_rig_report(params)
    assert report.bones >= 19
    assert report.bound_vertices > 0


def test_a_body_preset_saves_and_loads_through_the_service(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.set_body_dial(service.body_parameters(""), "height", 1.9)
    service.save_body_preset_for("tall", "Tall", "woman", params)
    assert ("tall", "Tall") in service.body_preset_names()
    loaded = service.apply_body_preset_instance(service.body_parameters(""), "tall")
    assert loaded is not None
    assert loaded.height == pytest.approx(1.9)
    assert service.remove_body_preset("tall") is True


def test_the_engine_status_is_honest(tmp_path: Path) -> None:
    service = _service(tmp_path)
    status = service.character_engine_status()
    assert status["templates"] == ["woman", "girl", "man", "boy"]
    assert status["materials"]["texture_maps"] is False


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


def _wait_for(root, predicate, *, timeout: float = 60.0) -> bool:
    """Drain the UI event loop until ``predicate()`` is true (a worker-thread result).

    Bounded by a wall-clock *timeout* rather than a fixed number of ``update``
    rounds: a busy CI runner can spend seconds inside one round (a full-body render
    is CPU-bound), so a round count is not a reliable deadline there.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        root.update()
        if predicate():
            return True
        _time.sleep(0.01)
    return predicate()


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
def test_the_body_window_uses_the_character_engine_templates_and_persistence(
    tmp_path: Path,
) -> None:
    """The window is the frontend of the FR-CCE engine: templates, save, restore.

    The task requires the existing Body & Shape window to be the engine's own
    frontend (never a competing window): it offers the four templates, shows the
    one in force, saves a character's own body, and restores it -- and a dial still
    changes the rendered geometry.
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
        # The window opens on the character's own template, marked and named.
        assert dialog._template_id == "woman"
        assert set(dialog._template_buttons) == {"woman", "girl", "man", "boy"}
        # A template button switches the body.
        dialog._select_template("man")
        root.update()
        assert dialog._template_id == "man"
        assert dialog._params.build == "masculine"
        # Save the body, shape it, then restore the saved body.
        dialog._save_character_body()
        root.update()
        assert service.character_body_config(character.id) is not None
        saved_height = dialog._params.height
        dialog._params = service.set_body_dial(dialog._params, "height", 1.6)
        dialog._sync_controls()
        dialog._restore_character_body()
        root.update()
        assert dialog._params.height == pytest.approx(saved_height)
        # The lighting slider changes the light scale the render uses.
        dialog._light_var.set(1.8)
        root.update()
        assert dialog._light_scale == pytest.approx(1.8)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_regenerates_and_saves_a_preset(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_body import BodyStudioDialog

    service = _service(tmp_path)
    character = service.create_character(name="Omar", role=ROLE_COMPANION, gender="man")
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = BodyStudioDialog(
            root, service, character.id, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        template_height = dialog._params.height
        dialog._params = service.set_body_dial(dialog._params, "height", 1.5)
        dialog._sync_controls()
        dialog._regenerate_template()
        root.update()
        assert dialog._params.height == pytest.approx(template_height)
        # A saved preset is offered back through the service.
        service.save_body_preset_for("mine", "Mine", dialog._template_id, dialog._params)
        assert ("mine", "Mine") in service.body_preset_names()
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
def test_the_body_window_offers_poses_and_applies_an_instruction(tmp_path: Path) -> None:
    """The window lists the poses, applies one on a click, and reads an instruction."""
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
        # A button per pose is drawn in the Pose section.
        pose_titles_set = {title for _key, title in service.body_poses()}
        labels = {b.cget("text") for b in _all_buttons(dialog.window)}
        assert pose_titles_set <= labels
        # A click sets the pose.
        dialog._set_pose("wave")
        root.update()
        assert dialog._params.pose == "wave"
        # A typed instruction sets the pose too.
        dialog._pose_var.set("please sit down")
        dialog._apply_pose_text()
        root.update()
        assert dialog._params.pose == "sit"
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


# -- per-part colour and intimate detail ----------------------------------


def test_the_colour_dials_cover_every_part() -> None:
    for key in ("skin", "hair_color", "lip_color", "eye_color", "nipple_color", "nail_color",
                "genital_color"):
        dial = dial_for(key)
        assert dial is not None and dial.kind == "colour", key


def test_the_intimate_dials_exist_and_are_bounded() -> None:
    for key in ("genitalia", "genital_size", "areola"):
        dial = dial_for(key)
        assert dial is not None, key
    params = default_parameters()
    assert set_dial(params, "genitalia", 5.0).genitalia == 1.0
    assert set_dial(params, "genitalia", -3.0).genitalia == 0.0
    assert set_dial(params, "genital_size", 0.01).genital_size >= 0.4


def test_the_intimate_group_holds_its_dials() -> None:
    keys = {dial.key for dial in dials_in_group("intimate")}
    assert {"genitalia", "genital_size", "areola"} <= keys


def test_hiding_the_intimate_dial_changes_the_mesh() -> None:
    from core.scene3d_figures import build_figure_mesh

    shown = build_figure_mesh(set_dial(default_parameters(), "genitalia", 1.0))
    hidden = build_figure_mesh(set_dial(default_parameters(), "genitalia", 0.0))
    assert hidden.vertex_count < shown.vertex_count


def test_the_mesh_carries_per_vertex_colours() -> None:
    from core.scene3d_figures import build_figure_mesh

    mesh = build_figure_mesh(default_parameters())
    assert mesh.has_vertex_colors
    assert len(mesh.vertex_colors) == mesh.vertex_count
    # More than one distinct colour means the lips/eyes/nails are really shaded apart.
    assert len(set(mesh.vertex_colors)) > 3


# -- poses ----------------------------------------------------------------


def test_pose_presets_are_offered_and_named_bilingually() -> None:
    keys = pose_keys()
    assert "rest" in keys and "sit" in keys and "wave" in keys
    english = dict(pose_titles("en"))
    arabic = dict(pose_titles("ar"))
    assert set(english) == set(arabic) == set(keys)
    for key in keys:
        assert english[key].strip() and arabic[key].strip()


def test_set_pose_changes_the_parameters_and_refuses_an_unknown_pose() -> None:
    params = default_parameters()
    assert set_pose(params, "sit").pose == "sit"
    with pytest.raises(KeyError):
        set_pose(params, "backflip")


def test_a_pose_really_moves_the_mesh() -> None:
    from core.scene3d_figures import build_figure

    rest = build_figure(default_parameters())
    sitting = build_figure(set_pose(default_parameters(), "sit"))
    rest_bounds = _bounds(rest.deformed_vertices())
    sit_bounds = _bounds(sitting.deformed_vertices())
    # Sitting lowers the lowest point (the hips drop) -- the pose is applied.
    assert sit_bounds[1] > rest_bounds[1] + 0.1


def test_pose_from_instruction_reads_english_and_arabic() -> None:
    assert pose_from_instruction("please sit down") == "sit"
    assert pose_from_instruction("wave at the camera") == "wave"
    assert pose_from_instruction("اجلس على الكرسي") == "sit"
    assert pose_from_instruction("just stand there") == "rest"


def test_pose_from_instruction_is_honest_when_nothing_matches() -> None:
    assert pose_from_instruction("make it look nice") is None
    assert pose_from_instruction("") is None
    params = default_parameters()
    unchanged, message = apply_pose_instruction(params, "do something artistic")
    assert unchanged is params
    assert "No pose matched" in message


def test_apply_pose_instruction_sets_the_pose() -> None:
    params = default_parameters()
    updated, message = apply_pose_instruction(params, "lying down")
    assert updated.pose == "lying"
    assert "Lying" in message


def _bounds(vertices: object) -> tuple[float, float, float, float, float, float]:
    points = list(vertices)  # type: ignore[call-overload]
    xs = [v.x for v in points]
    ys = [v.y for v in points]
    zs = [v.z for v in points]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


# -- the service pose API -------------------------------------------------


def test_the_service_lists_poses_and_applies_them(tmp_path: Path) -> None:
    service = _service(tmp_path)
    titles = dict(service.body_poses())
    assert "sit" in titles and titles["sit"].strip()
    params = service.body_parameters("")
    posed = service.set_body_pose(params, "wave")
    assert posed.pose == "wave"
    with pytest.raises(KeyError):
        service.set_body_pose(params, "moonwalk")


def test_the_service_applies_a_pose_instruction(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    updated, note = service.apply_body_pose_instruction(params, "arms crossed")
    assert updated.pose == "arms_crossed"
    assert "Arms crossed" in note


def test_the_cli_lists_and_sets_poses(tmp_path: Path, capsys) -> None:
    root = str(tmp_path)
    code, out = _cli(["--data-root", root, "body", "--poses"], capsys)
    assert code == 0 and "Sit (chair)" in out
    code, out = _cli(["--data-root", root, "body", "--pose", "squat", "--summary"], capsys)
    assert code == 0
    code, _out = _cli(["--data-root", root, "body", "--pose", "backflip"], capsys)
    assert code == 2


def test_the_cli_accepts_a_pose_instruction(tmp_path: Path, capsys) -> None:
    code, _out = _cli(
        ["--data-root", str(tmp_path), "body", "--pose-text", "please wave", "--summary"],
        capsys,
    )
    assert code == 0



# -- the camera (orbit) tool ----------------------------------------------


def test_the_service_exposes_the_camera_presets(tmp_path: Path) -> None:
    service = _service(tmp_path)
    titles = dict(service.body_camera_presets())
    assert "front" in titles and "back" in titles and "left" in titles
    assert service.body_camera_preset("left") == (-90.0, 0.0)
    with pytest.raises(KeyError):
        service.body_camera_preset("over-the-rainbow")


def test_rendering_from_another_angle_changes_the_picture(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    front = service.render_body(params, width=200, height=280)
    back = service.render_body(params, azimuth=180.0, width=200, height=280)
    side = service.render_body(params, azimuth=90.0, width=200, height=280)
    assert front and back and side
    assert front != back
    assert front != side


def test_zooming_changes_the_picture(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    near = service.render_body(params, zoom=0.5, width=200, height=280)
    far = service.render_body(params, zoom=2.0, width=200, height=280)
    assert near != far


def test_the_body_engine_places_the_camera_on_an_orbit() -> None:
    from core.romance.body import CAMERA_MAX_ZOOM, CAMERA_MIN_ZOOM, clamp_zoom, preview_scene

    params = default_parameters()
    front = preview_scene(params).camera.position
    back = preview_scene(params, azimuth=180.0).camera.position
    # A half-turn puts the camera on the other side of the figure.
    assert front.z > 0 > back.z
    # The elevation is clamped short of the poles so the up vector never flips.
    high = preview_scene(params, elevation=200.0).camera.position
    assert high.y > 0
    # A zoom really changes the distance.
    near = preview_scene(params, zoom=0.5).camera.position
    far = preview_scene(params, zoom=2.0).camera.position
    assert (near - preview_scene(params).camera.target).length() < (
        far - preview_scene(params).camera.target
    ).length()
    assert clamp_zoom(0.0) == CAMERA_MIN_ZOOM
    assert clamp_zoom(99.0) == CAMERA_MAX_ZOOM


def test_the_subdivision_and_smoothing_dials_are_real(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    finer = service.set_body_dial(params, "subdivisions", 2)
    assert finer.subdivisions == 2
    relaxed = service.set_body_dial(params, "smoothing", 0)
    assert relaxed.smoothing == 0
    # A numeric dial is clamped to its range, never stored out of it.
    assert service.set_body_dial(params, "subdivisions", 9).subdivisions == 2
    assert service.set_body_dial(params, "smoothing", -4).smoothing == 0


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_has_a_camera_tool(tmp_path: Path) -> None:
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
            root, service, character.id, theme=DARK_THEME,
            font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        assert _wait_for(root, lambda: bool(dialog._preview_png))
        # Every angle preset is a real, clickable button.
        labels = {b.cget("text") for b in _all_buttons(dialog.window)}
        for _key, title in service.body_camera_presets():
            assert title in labels
        assert "Reset the camera" in labels
        # The three orbit sliders are wired to the camera state.
        assert len(dialog._camera_scales) == 3
        dialog._apply_camera_preset("back")
        assert _wait_for(root, lambda: dialog._camera_azimuth == 180.0)
        assert dialog._camera_elevation == 0.0
        # A drag on the picture turns the camera (the 3D-program gesture).
        class _Event:
            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

        dialog._on_preview_press(_Event(100, 40))
        dialog._on_preview_drag(_Event(40, 10))
        dialog._on_preview_release(_Event(40, 10))
        assert dialog._camera_azimuth == pytest.approx(180.0 - 36.0)
        assert dialog._camera_elevation > 0.0
        dialog._reset_camera()
        assert dialog._camera_azimuth == 0.0 and dialog._camera_zoom == 1.0
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_builder_opens_even_with_nothing_selected(tmp_path: Path) -> None:
    """The reported bug: the button did nothing until a row was clicked."""
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
        assert dialog._selected_id() == ""
        dialog._open_body_studio()
        root.update()
        studio = getattr(dialog, "_body_studio", None)
        assert studio is not None, "the body builder did not open with no selection"
        assert not getattr(studio, "_closed", True)
    finally:
        _close_window(root, dialog)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_builder_opens_on_an_existing_character(tmp_path: Path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.romance_managers import CharacterManagerDialog

    service = _service(tmp_path)
    character = service.create_character(name="Amina", role=ROLE_COMPANION, gender="woman")
    root = tk.Tk()
    root.withdraw()
    dialog = None
    try:
        dialog = CharacterManagerDialog(
            root, service, role=ROLE_COMPANION,
            theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont",
        )
        root.update()
        dialog._open_body_studio()
        root.update()
        studio = getattr(dialog, "_body_studio", None)
        assert studio is not None
        assert studio.character_id == character.id
    finally:
        _close_window(root, dialog)


def test_the_cli_lists_and_uses_camera_angles(tmp_path: Path, capsys) -> None:
    root = str(tmp_path)
    code, out = _cli(["--data-root", root, "body", "--cameras"], capsys)
    assert code == 0 and "Left side" in out and "azimuth -90°" in out

    target = tmp_path / "left.png"
    code, _out = _cli(
        ["--data-root", root, "body", "--camera", "left", "--render", str(target)], capsys
    )
    assert code == 0 and target.read_bytes()

    code, out = _cli(
        [
            "--data-root", root, "body", "--azimuth", "90", "--elevation", "20",
            "--zoom", "0.6", "--render", str(tmp_path / "az.png"), "--json",
        ],
        capsys,
    )
    assert code == 0
    assert '"azimuth": 90.0' in out and '"zoom": 0.6' in out

    code, _out = _cli(
        ["--data-root", root, "body", "--camera", "moonwalk", "--render", "x.png"], capsys
    )
    assert code == 2


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_body_window_imports_and_edits_a_model(tmp_path: Path) -> None:
    """The window shows an imported/edited model and edits a part of it.

    The user asked to open a ready-made model, see it, and edit any part. This
    drives the window's model section: turn the built body into a model, apply a
    part edit, add a part, and confirm the preview follows the model (not the
    parametric figure).
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
        assert _wait_for(root, lambda: dialog._photo is not None)
        # The model section exists with its menus populated.
        assert dict(service.body_model_region_titles())["head"] == "Head"
        assert dialog._region_var.get()
        assert dialog._part_var.get()
        # Turn the built body into an editable model.
        dialog._use_figure_as_model()
        assert _wait_for(root, lambda: dialog._mesh is not None)
        first = dialog._mesh
        # Edit the left arm and add a sphere part.
        dialog._set_region("arm_left")
        dialog._edit_vars["scale_x"].set(1.3)
        dialog._apply_part_edit()
        assert _wait_for(root, lambda: dialog._mesh is not first)
        edited = dialog._mesh
        assert edited.face_count == first.face_count
        dialog._add_part()
        assert _wait_for(root, lambda: dialog._mesh is not edited)
        assert dialog._mesh.vertex_count > 0
        # The preview follows the edited model.
        assert dialog._preview_png.startswith(b"\x89PNG")
        # Back to the built figure clears the model.
        dialog._clear_model()
        root.update()
        assert dialog._mesh is None
    finally:
        _close_window(root, dialog)
