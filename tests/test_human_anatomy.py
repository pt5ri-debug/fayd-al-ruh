"""Tests for the FR-CCE anatomical detail engine (areolae, nipples, genitalia).

Every judgement here is about **real geometry**: a dial counts only when it moves
vertices on the generated mesh, never when a slider's stored number changed. The
suite also pins the safety rule that a child's body never receives adult anatomy,
and the honesty rule that a neutral build carries no genital set.
"""

from __future__ import annotations

import math

import pytest
from core.human_anatomy import (
    BreastDetail,
    GenitalDetail,
    describe_anatomy,
    genital_geometry,
    merge_anatomy,
)
from core.romance.body import (
    default_parameters,
    dial_for,
    dials_in_group,
    set_dial,
    set_dials,
    values_of,
)
from core.scene3d_figures import FIGURE_PRESETS, build_figure, build_figure_mesh

BODY_VERTICES = 13_660  # the MakeHuman base mesh's own vertex count


def _mesh(**changes):
    return build_figure_mesh(set_dials(default_parameters(), changes))


def _moved(a, b) -> int:
    """How many vertices really differ between two meshes of equal topology."""
    assert a.vertex_count == b.vertex_count
    return sum(
        1
        for before, after in zip(a.vertices, b.vertices, strict=True)
        if (before - after).length() > 1e-6
    )


# -- the detail is real geometry -------------------------------------------


def test_an_adult_body_carries_generated_anatomy() -> None:
    mesh = _mesh()
    assert mesh.vertex_count > BODY_VERTICES, "no anatomical detail was generated"
    # Every vertex is used by a face -- no stray points left behind.
    used = {index for face in mesh.faces for index in face}
    assert len(used) == mesh.vertex_count
    assert max(used) == mesh.vertex_count - 1


def test_the_detail_is_appended_after_the_untouched_body() -> None:
    hidden = _mesh(genitalia=0.0)
    shown = _mesh(genitalia=1.0)
    # Hiding the genital set removes only that set: the body and the areolae are
    # byte-identical, because the detail is appended rather than welded in.
    assert shown.vertices[: hidden.vertex_count] == hidden.vertices
    assert shown.vertex_colors[: hidden.vertex_count] == hidden.vertex_colors


def test_a_neutral_build_carries_no_genital_set() -> None:
    from core.scene3d_figures import BUILD_NEUTRAL

    mesh = build_figure_mesh(
        FIGURE_PRESETS["woman"].with_changes(build=BUILD_NEUTRAL, minor=False)
    )
    # Areolae are built for any adult body; the genital set is not presumed.
    assert mesh.vertex_count == BODY_VERTICES + 204


# -- the four templates ----------------------------------------------------


@pytest.mark.parametrize("key", ["woman", "man"])
def test_each_adult_template_carries_detail(key: str) -> None:
    mesh = build_figure_mesh(FIGURE_PRESETS[key])
    assert mesh.vertex_count > BODY_VERTICES


@pytest.mark.parametrize("key", ["girl", "boy", "child"])
def test_no_child_template_carries_adult_anatomy(key: str) -> None:
    params = FIGURE_PRESETS[key]
    assert params.is_minor, f"{key} must be flagged as a minor"
    mesh = build_figure_mesh(params)
    # A child's body is the base mesh exactly: no areolae, no nipples, no genitals.
    assert mesh.vertex_count == BODY_VERTICES


def test_a_minor_flag_is_explicit_not_inferred_from_height() -> None:
    # A short adult is not a child: the flag, not the height, decides.
    short_adult = FIGURE_PRESETS["woman"].with_changes(height=1.20)
    assert not short_adult.is_minor
    assert build_figure_mesh(short_adult).vertex_count > BODY_VERTICES


def test_the_two_sexes_get_their_own_genital_set() -> None:
    female = genital_geometry(
        build_figure_mesh(FIGURE_PRESETS["woman"]), GenitalDetail(sex="female")
    )
    male = genital_geometry(
        build_figure_mesh(FIGURE_PRESETS["man"]), GenitalDetail(sex="male")
    )
    assert female.vertex_count > 0 and male.vertex_count > 0
    # The sets are different geometry, not one shape recoloured.
    assert female.vertices != male.vertices


# -- every control moves real geometry ------------------------------------


NIPPLE_DIALS = (
    ("areola", 0.30, 2.20),
    ("areola_raise", 0.0, 1.40),
    ("nipple_size", 0.10, 1.10),
    ("nipple_length", 0.10, 2.20),
    ("nipple_direction", -60.0, 60.0),
    ("nipple_spacing", 0.60, 1.70),
)

GENITAL_DIALS = (
    ("genital_size", 0.40, 2.00),
    ("genital_length", 0.40, 2.00),
    ("genital_fullness", 0.40, 2.00),
    ("genital_direction", -60.0, 60.0),
)


@pytest.mark.parametrize(("key", "low", "high"), NIPPLE_DIALS)
def test_every_nipple_control_moves_vertices(key: str, low: float, high: float) -> None:
    assert _moved(_mesh(**{key: low}), _mesh(**{key: high})) > 0, f"{key} moved nothing"


@pytest.mark.parametrize(("key", "low", "high"), GENITAL_DIALS)
def test_every_genital_control_moves_vertices(key: str, low: float, high: float) -> None:
    assert _moved(_mesh(**{key: low}), _mesh(**{key: high})) > 0, f"{key} moved nothing"


def test_the_nipple_controls_leave_the_body_itself_alone() -> None:
    base = _mesh()
    for key, low, high in NIPPLE_DIALS:
        for value in (low, high):
            moved = _mesh(**{key: value})
            # The body is the same prefix in every mesh, so a nipple dial may not
            # deform the skin it sits on.
            assert moved.vertices[:BODY_VERTICES] == base.vertices[:BODY_VERTICES], key


def test_the_genital_controls_leave_the_body_itself_alone() -> None:
    base = _mesh()
    for key, low, high in GENITAL_DIALS:
        for value in (low, high):
            moved = _mesh(**{key: value})
            assert moved.vertices[:BODY_VERTICES] == base.vertices[:BODY_VERTICES], key


def test_hiding_the_genital_set_removes_it() -> None:
    shown = _mesh(genitalia=1.0)
    hidden = _mesh(genitalia=0.0)
    assert shown.vertex_count > hidden.vertex_count
    # The body plus the areolae remain; only the genital set is gone.
    assert hidden.vertex_count == BODY_VERTICES + 204


def test_the_colour_controls_reach_the_detail_vertices() -> None:
    mesh = _mesh(nipple_color=(0.20, 0.10, 0.30), nipple_tip_color=(0.90, 0.10, 0.10))
    detail = mesh.vertex_colors[BODY_VERTICES:]
    assert (0.20, 0.10, 0.30) in detail, "the areola colour did not reach its vertices"
    assert (0.90, 0.10, 0.10) in detail, "the nipple colour did not reach its vertices"
    # The body's own skin is untouched by an intimate colour.
    assert mesh.vertex_colors[:BODY_VERTICES] == _mesh().vertex_colors[:BODY_VERTICES]


def test_the_genital_colour_reaches_the_genital_vertices() -> None:
    mesh = _mesh(genital_color=(0.15, 0.25, 0.45))
    assert (0.15, 0.25, 0.45) in mesh.vertex_colors[BODY_VERTICES:]


# -- the detail sits where the body is ------------------------------------


def test_the_areolae_sit_on_the_chest() -> None:
    mesh = _mesh()
    low, high = mesh.bounds()
    height = high.y - low.y
    detail = mesh.vertices[BODY_VERTICES:]
    # Every detail vertex is inside the body's own bounding box -- nothing floats
    # off the figure.
    for point in detail:
        assert low.x - 0.02 <= point.x <= high.x + 0.02
        assert low.y <= point.y <= high.y
        assert low.z - 0.02 <= point.z <= high.z + 0.02
    areolae = mesh.vertices[BODY_VERTICES : BODY_VERTICES + 204]
    # The chest is in the upper third of the figure, on both sides of the centre.
    assert all(point.y > low.y + height * 0.6 for point in areolae)
    assert any(point.x > 0 for point in areolae) and any(point.x < 0 for point in areolae)


def test_the_areolae_follow_a_bust_dial() -> None:
    flat = _mesh(bust=0.0)
    full = _mesh(bust=0.9)
    flat_areolae = flat.vertices[BODY_VERTICES : BODY_VERTICES + 204]
    full_areolae = full.vertices[BODY_VERTICES : BODY_VERTICES + 204]
    # A larger bust stands further forward, so its areolae move forward with it --
    # they are placed from the surface the morph actually produced.
    assert max(point.z for point in full_areolae) > max(point.z for point in flat_areolae)


def test_the_genital_set_sits_on_the_pelvis() -> None:
    body = build_figure_mesh(FIGURE_PRESETS["woman"])
    detail = genital_geometry(body, GenitalDetail(sex="female"))
    low, high = body.bounds()
    height = high.y - low.y
    assert detail.vertex_count > 0
    assert all(
        low.y + height * 0.35 < point.y < low.y + height * 0.60 for point in detail.vertices
    ), "the genital set is not on the pelvis"


def test_the_detail_is_symmetric_across_the_centre_line() -> None:
    mesh = _mesh()
    areolae = mesh.vertices[BODY_VERTICES : BODY_VERTICES + 204]
    left = [p for p in areolae if p.x > 0]
    right = [p for p in areolae if p.x < 0]
    assert len(left) == len(right)
    # Mirror symmetry: each side's mean height is the same to a tight tolerance.
    assert math.isclose(
        sum(p.y for p in left) / len(left), sum(p.y for p in right) / len(right), abs_tol=1e-3
    )


# -- the dial catalogue ----------------------------------------------------


def test_the_intimate_group_holds_every_detail_dial() -> None:
    keys = {dial.key for dial in dials_in_group("intimate")}
    assert {
        "genitalia",
        "genital_size",
        "genital_length",
        "genital_fullness",
        "genital_direction",
        "areola",
        "areola_raise",
        "nipple_size",
        "nipple_length",
        "nipple_direction",
        "nipple_spacing",
    } <= keys


def test_the_detail_colour_dials_are_colours() -> None:
    for key in ("nipple_color", "nipple_tip_color", "genital_color"):
        dial = dial_for(key)
        assert dial is not None and dial.kind == "colour", key


def test_a_direction_dial_is_bounded_to_a_tilt() -> None:
    params = default_parameters()
    assert set_dial(params, "nipple_direction", 500.0).nipple_direction == 85.0
    assert set_dial(params, "nipple_direction", -500.0).nipple_direction == -85.0
    assert set_dial(params, "genital_direction", 500.0).genital_direction == 85.0


def test_a_size_dial_is_clamped_to_its_own_range() -> None:
    params = default_parameters()
    assert set_dial(params, "nipple_size", 0.0).nipple_size >= 0.05
    assert set_dial(params, "areola", 99.0).areola <= 2.50
    assert set_dial(params, "genital_fullness", 0.01).genital_fullness >= 0.30


def test_a_negative_geometry_value_is_refused_at_construction() -> None:
    from core.scene3d_figures import FigureParameters

    with pytest.raises(ValueError):
        FigureParameters("Bad", nipple_length=-1.0)
    with pytest.raises(ValueError):
        FigureParameters("Bad", nipple_direction=200.0)


def test_every_detail_dial_is_a_stored_parameter() -> None:
    params = default_parameters()
    values = values_of(params)
    for key in (
        "areola_raise",
        "nipple_size",
        "nipple_length",
        "nipple_direction",
        "nipple_spacing",
        "genital_length",
        "genital_fullness",
        "genital_direction",
        "nipple_tip_color",
    ):
        assert key in values, key


# -- persistence and migration -------------------------------------------


def test_the_detail_survives_a_save_and_load() -> None:
    from core.character_generation import params_from_primitive, params_to_primitive

    saved = set_dials(
        default_parameters(),
        {
            "nipple_size": 0.55,
            "nipple_direction": -25.0,
            "nipple_tip_color": (0.30, 0.15, 0.20),
            "genital_fullness": 1.60,
            "genital_direction": 30.0,
            "areola_raise": 0.90,
        },
    )
    back = params_from_primitive(params_to_primitive(saved))
    for key in (
        "nipple_size",
        "nipple_direction",
        "nipple_tip_color",
        "genital_fullness",
        "genital_direction",
        "areola_raise",
    ):
        assert getattr(back, key) == getattr(saved, key), key


def test_a_configuration_written_before_a_dial_existed_still_loads() -> None:
    from core.character_generation import params_from_primitive, params_to_primitive

    raw = params_to_primitive(default_parameters())
    for key in ("nipple_tip_color", "nipple_spacing", "genital_fullness", "areola_raise"):
        raw.pop(key, None)
    migrated = params_from_primitive(raw)
    # A missing field takes the reference default rather than failing the load.
    assert migrated.nipple_tip_color == (0.66, 0.38, 0.35)
    assert migrated.nipple_spacing == 1.0
    assert migrated.genital_fullness == 1.0
    assert migrated.areola_raise == 0.35


# -- rig compatibility ----------------------------------------------------


def test_the_rig_binds_the_generated_detail() -> None:
    obj = build_figure(FIGURE_PRESETS["woman"])
    assert len(obj.bindings) == obj.mesh.vertex_count
    # No vertex is left unbound, so a pose moves the detail with the skin.
    assert all(len(influences) > 0 for influences in obj.bindings)


def test_posing_moves_the_anatomy_with_the_body() -> None:
    from core.scene3d import Vec3

    obj = build_figure(FIGURE_PRESETS["woman"])
    before = obj.deformed_vertices()
    obj.set_pose("upper_arm.L", Vec3(0.0, 0.0, -60.0))
    after = obj.deformed_vertices()
    assert len(before) == len(after) == obj.mesh.vertex_count
    moved = sum(1 for a, b in zip(before, after, strict=True) if (a - b).length() > 1e-6)
    assert moved > 0


# -- honesty ---------------------------------------------------------------


def test_describe_names_what_the_body_carries() -> None:
    text = describe_anatomy(BreastDetail(), GenitalDetail(sex="female"))
    assert "areolae" in text and "female genitalia" in text
    assert describe_anatomy(None, None) == "No anatomical detail (a neutral body)."


def test_a_hidden_genital_set_is_not_described_as_present() -> None:
    text = describe_anatomy(None, GenitalDetail(sex="female", show=0.0))
    assert "genitalia" not in text


def test_the_detail_is_deterministic() -> None:
    first = _mesh(nipple_direction=17.0, genital_fullness=1.3)
    second = _mesh(nipple_direction=17.0, genital_fullness=1.3)
    assert first.vertices == second.vertices
    assert first.faces == second.faces
    assert first.vertex_colors == second.vertex_colors


def test_generating_the_detail_needs_no_network_or_file(monkeypatch) -> None:
    import socket

    def refuse(*_args, **_kwargs):
        raise AssertionError("the anatomy engine must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    mesh = _mesh()
    assert mesh.vertex_count > BODY_VERTICES


def test_a_merged_detail_never_reuses_a_body_vertex() -> None:
    body = build_figure_mesh(FIGURE_PRESETS["girl"])
    merged = merge_anatomy(body, breast=BreastDetail())
    assert merged.vertex_count > body.vertex_count
    # Every new face indexes a vertex past the body's own count.
    for face in merged.faces[len(body.faces) :]:
        assert all(index >= body.vertex_count for index in face)
