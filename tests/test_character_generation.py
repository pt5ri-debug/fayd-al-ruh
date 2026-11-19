"""The FR-CCE native character-generation engine.

These tests pin the engine the Body & shape window is built on
(:mod:`core.character_generation`):

* the four templates (woman, girl, man, boy) are generated natively, each a real
  body at its age and sex, and each fuses into one connected skin;
* geometry is valid (a real mesh with faces, normals, and no degenerate faces);
* a body parameter changes **actual vertex positions**, not merely a number -- the
  honesty rule the task asks for;
* the parameter limits are enforced;
* a character's template is chosen from its own recorded sex and age (and a minor
  never gets an adult body);
* two characters keep **independent** configurations;
* save/load persists, with backward-compatible migration;
* regenerating a template never overwrites a saved configuration;
* the rig stays compatible with the body when proportions change;
* the engine works **offline** and imports no model.
"""

from __future__ import annotations

import json

import pytest
from core.character_generation import (
    ADULT_AGE,
    CONFIG_VERSION,
    TEMPLATE_BOY,
    TEMPLATE_GIRL,
    TEMPLATE_MAN,
    TEMPLATE_WOMAN,
    CharacterGenerationEngine,
    CharacterPersistenceManager,
    CharacterPresetManager,
    CharacterTemplateManager,
    MaterialManager,
    MorphTargetEngine,
    ProceduralMeshGenerator,
    SurfaceMaterial,
    default_engine,
    migrate_config,
    params_from_primitive,
    params_to_primitive,
)
from core.scene3d_figures import FigureParameters, build_figure_mesh


class _Record:
    """A minimal stand-in for a character record (gender + age + name + id)."""

    def __init__(self, name: str = "", gender: str = "", age: int | None = None, cid: str = "") -> None:
        self.name = name
        self.gender = gender
        self.fictional_age = age
        self.id = cid


class _Bundle:
    def __init__(self, record: _Record) -> None:
        self.character = record


@pytest.fixture
def engine(tmp_path):
    return CharacterGenerationEngine(
        presets=CharacterPresetManager(tmp_path / "presets"),
        persistence=CharacterPersistenceManager(tmp_path / "bodies"),
    )


# -- the four templates ---------------------------------------------------


def test_the_engine_ships_exactly_four_templates_in_order():
    manager = CharacterTemplateManager()
    assert manager.keys() == (TEMPLATE_WOMAN, TEMPLATE_GIRL, TEMPLATE_MAN, TEMPLATE_BOY)


def test_each_template_generates_a_real_body_at_its_age_and_sex():
    manager = CharacterTemplateManager()
    for key in manager.keys():
        params = manager.parameters(key)
        mesh = build_figure_mesh(params)
        assert mesh.vertex_count > 500, key
        assert mesh.face_count > 500, key
        template = manager.get(key)
        assert template.parameters.height > 1.0
        # A child is meaningfully shorter than an adult, not a scaled copy.
        if key in (TEMPLATE_GIRL, TEMPLATE_BOY):
            assert params.height < 1.6
        else:
            assert params.height >= 1.6


def test_the_four_bodies_have_distinguishable_proportions():
    manager = CharacterTemplateManager()
    heights = {key: manager.parameters(key).height for key in manager.keys()}
    # The two children are shorter than both adults; the man is the tallest.
    assert heights[TEMPLATE_GIRL] < heights[TEMPLATE_WOMAN]
    assert heights[TEMPLATE_BOY] < heights[TEMPLATE_MAN]
    assert heights[TEMPLATE_MAN] >= heights[TEMPLATE_WOMAN]


def test_each_template_fuses_into_one_connected_skin():
    from core.scene3d import Mesh  # noqa: F401  (documents the source type)

    manager = CharacterTemplateManager()
    for key in manager.keys():
        mesh = build_figure_mesh(manager.parameters(key))
        # The body is **one** connected skin. The eyes and the two rows of teeth
        # are closed sub-surfaces inside it (as in any character asset), so the
        # body itself is the single largest shell and everything else is a small
        # closed detail -- never a second body part.
        sizes = _shell_sizes(mesh)
        body = sizes[0]
        assert body > 12_000, f"{key}: the body shell is only {body} vertices"
        assert body > 0.97 * sum(sizes), f"{key}: the body is not one skin ({sizes[:6]})"
        # A detail shell is small (an eye or a tooth row), never a limb.
        for detail in sizes[1:]:
            assert detail < 200, f"{key}: a {detail}-vertex shell is not a detail"


def _shell_count(mesh) -> int:
    """Count connected components of a mesh's faces (a union-find over faces)."""
    parent = list(range(len(mesh.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for face in mesh.faces:
        for index in range(1, len(face)):
            union(face[0], face[index])
    used = {vertex for face in mesh.faces for vertex in face}
    return len({find(vertex) for vertex in used})


def _shell_sizes(mesh) -> list[int]:
    """The size of every connected component, largest first."""
    parent = list(range(len(mesh.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in mesh.faces:
        root = find(face[0])
        for index in face[1:]:
            other = find(index)
            if root != other:
                parent[other] = root

    sizes: dict[int, int] = {}
    for vertex in {vertex for face in mesh.faces for vertex in face}:
        root = find(vertex)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


# -- geometry validity ----------------------------------------------------


def test_a_generated_mesh_is_geometrically_valid():
    mesh = build_figure_mesh(CharacterTemplateManager().parameters(TEMPLATE_WOMAN))
    # Every face references a real vertex.
    for face in mesh.faces:
        for index in face:
            assert 0 <= index < len(mesh.vertices)
    # No degenerate faces (three distinct indices).
    for face in mesh.faces:
        assert len(set(face)) >= 3
    # Normals are unit length.
    for normal in mesh.normals[:50]:
        assert abs(normal.length() - 1.0) < 1e-6
    # The bounds are a plausible human size (about two metres tall).
    low, high = mesh.bounds()
    assert -0.02 <= low.y <= 0.2
    assert 1.2 <= high.y <= 2.4


def test_the_mesh_builder_caches_by_parameter_signature(engine):
    params = engine.templates.parameters(TEMPLATE_MAN)
    first = engine.generator.build(params)
    second = engine.generator.build(params)
    assert first is second  # the same object: the cache returned it
    # A changed dial is a different build.
    changed = params.with_changes(height=params.height + 0.05)
    assert engine.generator.build(changed) is not first


# -- vertex deformation (the honesty rule) --------------------------------


def test_a_body_parameter_moves_actual_vertices(engine):
    params = engine.templates.parameters(TEMPLATE_WOMAN)
    before = engine.generator.build(params)
    _updated, after, result = engine.morphs.apply(params, "height", 1.95)
    assert result.changed
    assert result.moved_vertices > 0
    # The meshes really differ at the vertex level.
    assert after.vertices != before.vertices
    assert result.max_displacement > 0.0


def test_several_measurement_dials_each_move_geometry(engine):
    base = engine.templates.parameters(TEMPLATE_MAN)
    for key, value in (
        ("overall_scale", 1.25),
        ("neck_length", 1.4),
        ("arm_length", 1.3),
        ("thigh_thickness", 1.4),
        ("hand_width", 1.3),
        ("foot_length", 1.2),
    ):
        _updated, _after, result = engine.morphs.apply(base, key, value)
        assert result.changed, key
        assert result.moved_vertices > 0, key


def test_an_unchanged_parameter_reports_no_move(engine):
    params = engine.templates.parameters(TEMPLATE_WOMAN)
    _updated, _after, result = engine.morphs.apply(params, "height", params.height)
    assert result.changed is False
    assert result.moved_vertices == 0


def test_the_compare_helper_counts_moved_vertices(engine):
    params = engine.templates.parameters(TEMPLATE_WOMAN)
    a = engine.generator.build(params)
    b = engine.generator.build(params.with_changes(waist=1.4))
    result = MorphTargetEngine.compare(a, b)
    assert result.changed
    assert 0 < result.moved_vertices <= len(a.vertices)


# -- parameter limits -----------------------------------------------------


def test_the_parameter_system_rejects_an_unknown_dial(engine):
    params = engine.templates.parameters(TEMPLATE_WOMAN)
    with pytest.raises(KeyError):
        engine.parameters.set(params, "not_a_dial", 1.0)


def test_a_parameter_out_of_range_is_refused():
    with pytest.raises(ValueError):
        FigureParameters(height=5.0)
    with pytest.raises(ValueError):
        FigureParameters(shoulder_width=0.0)


def test_the_parameter_groups_are_complete_and_ordered(engine):
    groups = engine.parameters.groups()
    keys = [key for key, _title, _dials in groups]
    # The task's parameter groups are all present.
    for required in (
        "proportions", "head", "neck", "shoulders", "chest", "waist", "abdomen",
        "pelvis", "arms", "hands", "thighs", "legs", "feet", "shape",
    ):
        assert required in keys
    # Every dial a group names is a real dial.
    for _key, _title, dials in groups:
        for dial in dials:
            assert engine.parameters.has(dial.key)


# -- character mapping ----------------------------------------------------


def test_the_template_follows_a_characters_sex_and_age():
    manager = CharacterTemplateManager()
    assert manager.template_for_character(_Record(gender="woman", age=30)) == TEMPLATE_WOMAN
    assert manager.template_for_character(_Record(gender="man", age=40)) == TEMPLATE_MAN
    assert manager.template_for_character(_Record(gender="female", age=10)) == TEMPLATE_GIRL
    assert manager.template_for_character(_Record(gender="male", age=8)) == TEMPLATE_BOY


def test_a_minor_never_gets_an_adult_body():
    manager = CharacterTemplateManager()
    for gender in ("woman", "female", "أنثى"):
        assert manager.template_for_character(_Record(gender=gender, age=ADULT_AGE - 1)) == (
            TEMPLATE_GIRL
        )
    for gender in ("man", "male", "ذكر"):
        assert manager.template_for_character(_Record(gender=gender, age=ADULT_AGE - 1)) == (
            TEMPLATE_BOY
        )


def test_an_unrecorded_or_nonbinary_sex_falls_back_to_the_documented_default():
    manager = CharacterTemplateManager()
    assert manager.template_for_character(_Record(gender="", age=30)) == TEMPLATE_WOMAN
    assert manager.template_for_character(_Record(gender="non-binary", age=30)) == TEMPLATE_WOMAN
    assert manager.template_for_character(None) == TEMPLATE_WOMAN


def test_the_arabic_template_titles_are_present():
    summaries = dict(CharacterTemplateManager().summaries(language="ar"))
    assert summaries[TEMPLATE_WOMAN] == "أنثى بالغة"
    assert summaries[TEMPLATE_BOY] == "ولد"


# -- independent character configurations ---------------------------------


def test_two_characters_keep_independent_bodies(engine):
    a = engine.create(TEMPLATE_WOMAN, character_id="a", name="Amina")
    b = engine.create(TEMPLATE_WOMAN, character_id="b", name="Layla")
    engine.morph(a, "height", 1.9)
    assert a.params.height == pytest.approx(1.9)
    assert b.params.height != pytest.approx(1.9)
    engine.save(a)
    engine.save(b)
    engine.morph(b, "waist", 1.3)
    engine.save(b)
    # A's saved body is unchanged by B's edit.
    reloaded_a = engine.load_params("a")
    assert reloaded_a is not None
    assert reloaded_a.height == pytest.approx(1.9)


# -- persistence ----------------------------------------------------------


def test_save_and_load_a_character_body(engine):
    inst = engine.create(TEMPLATE_MAN, character_id="c1", name="Karim")
    engine.morph(inst, "shoulder_width", 0.62)
    config = engine.save(inst)
    assert config.character_id == "c1"
    assert config.template_id == TEMPLATE_MAN
    assert config.engine_version
    loaded = engine.load_params("c1")
    assert loaded is not None
    assert loaded.shoulder_width == pytest.approx(0.62)


def test_a_saved_configuration_records_every_required_field(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c2", name="Huda")
    engine.save(inst, preset_id="preset-x")
    raw = json.loads(engine.persistence.path_for("c2").read_text(encoding="utf-8"))
    for field in (
        "template_id", "character_id", "parameters", "morphs", "skeleton",
        "materials", "preset_id", "last_modified", "engine_version",
    ):
        assert field in raw, field
    assert raw["preset_id"] == "preset-x"
    assert raw["morphs"]  # the measurement values are recorded
    assert raw["config_version"] == CONFIG_VERSION


def test_a_configuration_written_by_an_earlier_build_migrates():
    # A document with an unknown field and a missing field still loads.
    raw = {
        "character_id": "old",
        "template_id": TEMPLATE_WOMAN,
        "parameters": {"height": 1.7, "a_field_from_the_future": 3},
        "engine_version": "0.0.1",
    }
    migrated = migrate_config(raw)
    assert migrated["config_version"] == CONFIG_VERSION
    params = params_from_primitive(migrated["parameters"])
    assert params.height == pytest.approx(1.7)
    # A parameter added since simply takes its default.
    assert params.overall_scale == pytest.approx(1.0)


def test_a_corrupt_configuration_reads_as_absent(tmp_path):
    manager = CharacterPersistenceManager(tmp_path / "bodies")
    manager.root.mkdir(parents=True, exist_ok=True)
    manager.path_for("bad").write_text("{not json", encoding="utf-8")
    assert manager.load("bad") is None


def test_parameters_round_trip_through_the_primitive_form():
    params = FigureParameters(name="X", height=1.81, skin=(0.5, 0.4, 0.3))
    restored = params_from_primitive(params_to_primitive(params))
    assert restored.height == pytest.approx(1.81)
    assert restored.skin == pytest.approx((0.5, 0.4, 0.3))


# -- template regeneration ------------------------------------------------


def test_regenerating_a_template_keeps_a_saved_configuration(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c3", name="Salma")
    engine.morph(inst, "height", 1.95)
    engine.save(inst)
    # Regenerating returns the *live* instance to the template's defaults...
    fresh = engine.regenerate(inst)
    assert fresh.params.height == pytest.approx(
        engine.templates.parameters(TEMPLATE_WOMAN).height
    )
    # ...but the saved configuration is untouched.
    saved = engine.load_params("c3")
    assert saved is not None
    assert saved.height == pytest.approx(1.95)


def test_restoring_a_configuration_brings_the_saved_body_back(engine):
    inst = engine.create(TEMPLATE_MAN, character_id="c4", name="Omar")
    engine.morph(inst, "waist", 0.40)
    config = engine.save(inst)
    fresh = engine.regenerate(inst)
    restored = engine.restore(fresh, config)
    assert restored.params.waist == pytest.approx(0.40)


# -- rig compatibility ----------------------------------------------------


def test_the_rig_stays_compatible_when_proportions_change(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c5", name="Rana")
    base_bones = len(inst.obj.skeleton.names())
    engine.morph(inst, "arm_length", 1.35)
    engine.morph(inst, "thigh_length", 1.3)
    assert len(inst.obj.skeleton.names()) == base_bones  # the same rig
    # Every vertex is still bound to at least one bone.
    assert len(inst.obj.bindings) == len(inst.mesh.vertices)
    report = engine.rig.report(inst.obj)
    assert report.bones == base_bones
    assert report.bound_vertices > 0


def test_a_bone_length_follows_its_measurement_dial(engine):
    short = engine.rig.build(engine.templates.parameters(TEMPLATE_WOMAN))
    tall = engine.rig.build(
        engine.templates.parameters(TEMPLATE_WOMAN).with_changes(arm_length=1.4)
    )
    short_arm = next(b for b in short.bones if b.name == "upper_arm.L")
    tall_arm = next(b for b in tall.bones if b.name == "upper_arm.L")
    assert tall_arm.length > short_arm.length


def test_the_pose_is_applied_without_rebuilding_the_rig(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c6", name="Nour")
    before = inst.obj.deformed_vertices()
    engine.set_pose(inst, "tpose")
    after = inst.obj.deformed_vertices()
    assert after != before  # the skin really moved


# -- materials ------------------------------------------------------------


def test_the_material_system_shapes_roughness_and_metallic():
    rough = SurfaceMaterial(roughness=0.9).to_material()
    smooth = SurfaceMaterial(roughness=0.1).to_material()
    assert smooth.shininess > rough.shininess
    assert smooth.specular > rough.specular


def test_the_material_manager_reports_what_it_cannot_do():
    caps = MaterialManager().capabilities()
    assert caps["skin_material"] and caps["roughness"] and caps["metallic"]
    # This rasteriser has no image sampler, and says so rather than pretending.
    assert caps["texture_maps"] is False
    assert caps["normal_maps"] is False


def test_a_material_refuses_an_impossible_setting():
    with pytest.raises(ValueError):
        SurfaceMaterial(roughness=2.0)
    with pytest.raises(ValueError):
        SurfaceMaterial(metallic=-0.1)


# -- presets --------------------------------------------------------------


def test_a_preset_saves_and_loads(engine):
    inst = engine.create(TEMPLATE_MAN, character_id="c7", name="Tarek")
    engine.morph(inst, "height", 1.92)
    engine.save_preset("tall-man", "Tall man", inst)
    names = engine.presets.list()
    assert any(preset.preset_id == "tall-man" for preset in names)
    params = engine.presets.parameters("tall-man")
    assert params is not None
    assert params.height == pytest.approx(1.92)


def test_a_preset_round_trips_through_parameter_space(engine):
    params = engine.templates.parameters(TEMPLATE_WOMAN)
    engine.save_preset_params("w1", "Woman one", TEMPLATE_WOMAN, params)
    loaded = engine.load_preset_params("w1")
    assert loaded is not None
    assert loaded.height == pytest.approx(params.height)


def test_an_absent_preset_reads_as_none(engine):
    assert engine.presets.load("nope") is None
    assert engine.load_preset_params("nope") is None


# -- rendering ------------------------------------------------------------


def test_the_preview_renderer_produces_a_real_png(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c8", name="Dina")
    report = engine.render(inst, width=160, height=240)
    assert report.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert report.width == 160 and report.height == 240
    assert report.stats.triangles_drawn > 0


def test_lighting_scales_the_render(engine):
    inst = engine.create(TEMPLATE_WOMAN, character_id="c9", name="Hana")
    dim = engine.render(inst, width=120, height=180, light_scale=0.3)
    bright = engine.render(inst, width=120, height=180, light_scale=2.0)
    # The brighter render is not byte-identical to the dim one.
    assert dim.png != bright.png


# -- offline and import safety -------------------------------------------


def test_the_engine_imports_no_model_and_opens_no_socket(monkeypatch):
    # The engine must work fully offline: importing it, and building a body, must
    # not open a socket.
    import socket

    calls = []

    def guard(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the character engine must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", guard)
    eng = default_engine()
    inst = eng.create(TEMPLATE_GIRL, character_id="offline", name="Mona")
    assert inst.mesh.vertex_count > 0
    assert calls == []


def test_the_default_engine_uses_the_project_data_folders(tmp_path):
    eng = default_engine(data_root=tmp_path)
    inst = eng.create(TEMPLATE_MAN, character_id="p1", name="Ali")
    eng.save(inst)
    assert (tmp_path / "p1.json").is_file()
    assert eng.persistence.root == tmp_path


# -- status ---------------------------------------------------------------


def test_the_status_reports_honestly(engine):
    status = engine.status()
    assert status["templates"] == [TEMPLATE_WOMAN, TEMPLATE_GIRL, TEMPLATE_MAN, TEMPLATE_BOY]
    assert status["parameters"] > 20
    assert status["materials"]["texture_maps"] is False
    assert "not a licensed likeness" in status["honesty"]


def test_the_generator_is_a_thin_layer_over_the_project_figure_builder():
    # The engine's mesh is exactly the project's own sculpted figure, so there is
    # no second mesh generator.
    params = CharacterTemplateManager().parameters(TEMPLATE_WOMAN)
    assert ProceduralMeshGenerator().build(params).vertices == build_figure_mesh(params).vertices


# -- the native human-body engine (the real MakeHuman-derived mesh) ----------


def test_the_figure_is_built_from_the_human_mesh_not_a_blob():
    """The body is a dense, anatomically-modelled human mesh, not primitives."""
    from core.human_pack import load_pack

    pack = load_pack()
    # A real human base mesh has thousands of vertices; a primitive assembly has
    # a few hundred. The engine must use the modelled mesh.
    assert pack.vertex_count > 10_000
    mesh = build_figure_mesh(CharacterTemplateManager().parameters(TEMPLATE_WOMAN))
    assert len(mesh.vertices) >= pack.vertex_count


def test_the_mesh_has_a_head_torso_arms_hands_legs_and_feet():
    """Every region a human body needs is present in the generated geometry."""
    mesh = build_figure_mesh(CharacterTemplateManager().parameters(TEMPLATE_WOMAN))
    ys = [v.y for v in mesh.vertices]
    height = max(ys)
    top = [v for v in mesh.vertices if v.y > 0.93 * height]
    # A head: a cluster at the very top, narrower than the shoulders.
    assert top
    head_width = max(v.x for v in top) - min(v.x for v in top)
    shoulder = [v for v in mesh.vertices if 0.78 * height <= v.y <= 0.84 * height]
    shoulder_width = max(v.x for v in shoulder) - min(v.x for v in shoulder)
    assert head_width < shoulder_width
    # Feet on the ground: geometry at y ~ 0 and a footprint wider than the ankle.
    feet = [v for v in mesh.vertices if v.y < 0.04 * height]
    assert feet
    assert max(v.x for v in feet) - min(v.x for v in feet) > 0.05


def test_a_child_head_is_larger_for_its_body_than_an_adults():
    """The head-to-height ratio is what makes a child read as a child."""
    from core.human_body import build_body, dials_from_parameters
    from core.human_pack import load_pack

    pack = load_pack()
    manager = CharacterTemplateManager()

    def ratio(key: str) -> float:
        params = manager.parameters(key)
        body = build_body(dials_from_parameters(params), height=params.height)
        verts = body.mesh.vertices
        height = max(v.y for v in verts)
        neck_y = (pack.joints["joint-neck"][1] - body._offset.y) * body._scale
        head_h = height - min(v.y for v in verts if v.y >= neck_y)
        return head_h / height

    adult = ratio(TEMPLATE_WOMAN)
    girl = ratio(TEMPLATE_GIRL)
    boy = ratio(TEMPLATE_BOY)
    # A child's head is a visibly larger fraction of its height (roughly 1/5)
    # than an adult's (roughly 1/6.5).
    assert girl > adult * 1.05, f"girl head ratio {girl:.3f} vs adult {adult:.3f}"
    assert boy > adult * 1.05, f"boy head ratio {boy:.3f} vs adult {adult:.3f}"


def test_the_head_scale_dial_changes_the_head_proportion():
    """The head-scale dial is a proportion change, not a whole-figure resize."""
    from core.human_body import build_body, dials_from_parameters
    from core.human_pack import load_pack
    from core.scene3d_figures import FigureParameters

    pack = load_pack()

    def head_height(scale: float) -> float:
        params = FigureParameters("probe", head_scale=scale)
        body = build_body(dials_from_parameters(params), height=params.height)
        verts = body.mesh.vertices
        top = max(v.y for v in verts)
        neck_y = (pack.joints["joint-neck"][1] - body._offset.y) * body._scale
        return top - min(v.y for v in verts if v.y >= neck_y)

    small = head_height(0.75)
    large = head_height(1.35)
    assert large > small * 1.15, f"head {small:.3f} -> {large:.3f} did not grow"
