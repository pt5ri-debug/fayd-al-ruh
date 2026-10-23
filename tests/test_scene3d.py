"""The Scene Studio engine: math, meshes, skeletons, skinning, scene, renderer.

These drive the real code paths with no display and no mocks: the linear algebra,
the mesh, forward kinematics, linear-blend skinning, the software rasteriser, the
OBJ/MTL reader and writer, the native scene round-trip, the built-in figures, and
the props. Every test names the property it pins -- determinism, forward
kinematics moving children, skinning actually deforming the mesh, a render
producing a real PNG, and a bounded, honest failure for a bad file.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from core.scene3d import (
    Bone,
    Camera,
    Light,
    Mat4,
    Material,
    Mesh,
    SceneObject,
    Skeleton,
    Transform,
    Vec3,
    compute_skin_weights,
    iter_triangles,
    make_default_scene,
    vec3,
)
from core.scene3d_figures import (
    FIGURE_PRESETS,
    FigureParameters,
    build_figure,
    build_figure_from_preset,
    build_figure_mesh,
    build_figure_scene,
    build_figure_skeleton,
    figure_preset_names,
    morph_figure,
)
from core.scene3d_io import (
    SceneIOError,
    is_mesh_file,
    load_obj,
    load_obj_text,
    load_scene,
    parse_mtl,
    save_obj,
    write_mtl,
)
from core.scene3d_props import (
    MATERIAL_PALETTE,
    PROP_CATALOG,
    build_primitive,
    build_prop,
    material_for,
    primitive_names,
    prop_names,
)
from core.scene3d_render import Framebuffer, SoftwareRenderer, render_scene, shade_point
from core.scene3d_serialize import (
    load_scene_file,
    save_scene,
    scene_filename,
    scene_from_dict,
    scene_to_dict,
)

# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def test_vec3_basic_operations():
    a = vec3(1, 2, 3)
    b = vec3(4, 5, 6)
    assert (a + b) == vec3(5, 7, 9)
    assert (b - a) == vec3(3, 3, 3)
    assert (a * 2) == vec3(2, 4, 6)
    assert a.dot(b) == 32
    assert a.cross(b) == vec3(-3, 6, -3)


def test_vec3_normalized_is_unit_length():
    assert vec3(0, 3, 4).normalized().length() == pytest.approx(1.0)


def test_vec3_zero_normalizes_to_zero_without_dividing_by_zero():
    assert Vec3().normalized() == Vec3()


def test_mat4_identity_leaves_a_point_alone():
    assert Mat4.identity().transform_point(vec3(1, 2, 3)) == vec3(1, 2, 3)


def test_mat4_translation_moves_a_point():
    assert Mat4.translation(vec3(1, 2, 3)).transform_point(Vec3()) == vec3(1, 2, 3)


def test_mat4_rotation_x_rotates_ninety_degrees():
    result = Mat4.rotation_x(90).transform_point(vec3(0, 1, 0))
    assert result.x == pytest.approx(0.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(1.0)


def test_mat4_multiplication_is_associative():
    a = Mat4.translation(vec3(1, 0, 0))
    b = Mat4.rotation_y(30)
    c = Mat4.scaling(vec3(2, 2, 2))
    left = (a @ b) @ c
    right = a @ (b @ c)
    assert left.m == pytest.approx(right.m)


def test_mat4_inverse_round_trips_a_point():
    matrix = Mat4.translation(vec3(2, -1, 4)) @ Mat4.rotation_z(35) @ Mat4.scaling(vec3(2, 3, 4))
    point = vec3(0.5, -1.25, 3.0)
    moved = matrix.transform_point(point)
    restored = matrix.inverse().transform_point(moved)
    assert restored.x == pytest.approx(point.x)
    assert restored.y == pytest.approx(point.y)
    assert restored.z == pytest.approx(point.z)


def test_mat4_singular_matrix_refuses_to_invert():
    with pytest.raises(ValueError):
        Mat4.scaling(vec3(1, 0, 1)).inverse()


def test_mat4_requires_sixteen_components():
    with pytest.raises(ValueError):
        Mat4((1.0, 2.0, 3.0))


def test_perspective_rejects_a_bad_field_of_view():
    with pytest.raises(ValueError):
        Mat4.perspective(0.0, 1.0, 0.1, 10.0)
    with pytest.raises(ValueError):
        Mat4.perspective(180.0, 1.0, 0.1, 10.0)


def test_look_at_rejects_identical_eye_and_target():
    with pytest.raises(ValueError):
        Mat4.look_at(vec3(1, 1, 1), vec3(1, 1, 1))


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def test_mesh_counts_and_bounds():
    mesh = Mesh("tri", (vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)), ((0, 1, 2),))
    assert mesh.vertex_count == 3
    assert mesh.face_count == 1
    assert mesh.triangle_count == 1
    low, high = mesh.bounds()
    assert low == vec3(0, 0, 0)
    assert high == vec3(1, 1, 0)


def test_mesh_quad_fans_into_two_triangles():
    mesh = Mesh("quad", tuple(vec3(x, 0, 0) for x in range(4)), ((0, 1, 2, 3),))
    assert mesh.triangle_count == 2
    assert list(iter_triangles(mesh)) == [(0, 1, 2), (0, 2, 3)]


def test_mesh_rejects_a_face_with_too_few_vertices():
    with pytest.raises(ValueError):
        Mesh("bad", (vec3(0, 0, 0), vec3(1, 0, 0)), ((0, 1),))


def test_mesh_rejects_an_out_of_range_index():
    with pytest.raises(ValueError):
        Mesh("bad", (vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)), ((0, 1, 5),))


def test_mesh_vertex_normals_point_outward_for_a_plane():
    mesh = Mesh(
        "plane",
        (vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1)),
        ((0, 1, 2, 3),),
    )
    for normal in mesh.vertex_normals():
        assert abs(normal.y) == pytest.approx(1.0)


def test_mesh_with_normals_is_idempotent():
    mesh = Mesh("tri", (vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)), ((0, 1, 2),))
    once = mesh.with_normals()
    assert once.with_normals() is once


# ---------------------------------------------------------------------------
# skeleton: forward kinematics
# ---------------------------------------------------------------------------


def test_skeleton_rejects_a_child_before_its_parent():
    with pytest.raises(ValueError):
        Skeleton((Bone("child", "parent"), Bone("parent")))


def test_skeleton_rejects_a_duplicate_bone():
    with pytest.raises(ValueError):
        Skeleton((Bone("a"), Bone("a")))


def test_forward_kinematics_places_a_child_at_the_parent_tail():
    skeleton = Skeleton((Bone("root", length=1.0), Bone("tip", "root", offset=vec3(0, 1, 0))))
    segments = skeleton.segments()
    assert segments["root"][1] == vec3(0.0, 1.0, 0.0)
    assert segments["tip"][0] == vec3(0.0, 1.0, 0.0)


def test_rotating_a_parent_moves_every_child_below_it():
    skeleton = Skeleton(
        (Bone("root", length=1.0), Bone("tip", "root", offset=vec3(0, 1, 0), length=1.0))
    )
    posed = skeleton.with_pose({"root": vec3(0, 0, 90)})
    segments = posed.segments()
    # A 90° Z rotation sends the child's head from (0, 1, 0) to (-1, 0, 0).
    assert segments["tip"][0].x == pytest.approx(-1.0)
    assert segments["tip"][0].y == pytest.approx(0.0, abs=1e-9)


def test_cleared_pose_matches_the_rest_skeleton():
    skeleton = Skeleton((Bone("root", length=1.0),))
    posed = skeleton.with_pose({"root": vec3(10, 20, 30)})
    assert posed.cleared().segments()["root"] == skeleton.segments()["root"]


# ---------------------------------------------------------------------------
# skinning
# ---------------------------------------------------------------------------


def test_skin_weights_sum_to_one_per_vertex():
    skeleton = Skeleton((Bone("a", length=1.0), Bone("b", offset=vec3(0, 1, 0), length=1.0)))
    mesh = Mesh("m", (vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 2, 0)), ((0, 1, 2),))
    bindings = compute_skin_weights(mesh, skeleton)
    for binding in bindings:
        assert sum(weight for _, weight in binding) == pytest.approx(1.0)


def test_skin_weights_honour_the_influence_cap():
    skeleton = Skeleton(
        (Bone("a", length=1.0), Bone("b", offset=vec3(0, 1, 0), length=1.0), Bone("c", offset=vec3(0, 1, 0), length=1.0))
    )
    mesh = Mesh("m", (vec3(0, 0.5, 0), vec3(0, 1.5, 0), vec3(0, 2.5, 0)), ((0, 1, 2),))
    bindings = compute_skin_weights(mesh, skeleton, max_influences=1)
    assert all(len(binding) == 1 for binding in bindings)


def test_skinning_a_rigid_bone_rotates_the_mesh_about_the_bone_head():
    skeleton = Skeleton((Bone("root", length=2.0),))
    mesh = Mesh("m", (vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 2, 0)), ((0, 1, 2),))
    obj = SceneObject("rig", mesh, skeleton=skeleton)
    obj.bind()
    before = obj.world_vertices()
    obj.set_pose("root", vec3(0, 0, 90))
    after = obj.world_vertices()
    # The bone's head is the pivot, so the vertex there stays put and the others
    # swing round: (0, 1, 0) -> (-1, 0, 0) and (0, 2, 0) -> (-2, 0, 0).
    assert (after[0] - before[0]).length() < 1e-9
    assert after[1].to_tuple() == pytest.approx(vec3(-1, 0, 0).to_tuple())
    assert after[2].to_tuple() == pytest.approx(vec3(-2, 0, 0).to_tuple())


def test_a_pose_that_does_not_move_the_bone_leaves_the_vertices_in_place():
    figure = build_figure()
    before = figure.world_vertices()
    figure.set_pose("hand.L", vec3(0, 0, 0))
    after = figure.world_vertices()
    assert all((a - b).length() < 1e-9 for a, b in zip(before, after, strict=False))


def test_clear_pose_returns_a_figure_to_its_rest_shape():
    figure = build_figure()
    rest = figure.world_vertices()
    figure.set_pose("forearm.L", vec3(0, 0, -90))
    figure.clear_pose()
    for actual, expected in zip(figure.world_vertices(), rest, strict=False):
        assert actual.to_tuple() == pytest.approx(expected.to_tuple())


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def test_figure_presets_are_named_and_deterministic():
    assert figure_preset_names() == tuple(FIGURE_PRESETS)
    first = build_figure_from_preset("adult")
    second = build_figure_from_preset("adult")
    assert first.mesh.vertices == second.mesh.vertices


def test_figure_mesh_and_skeleton_agree_on_the_bone_count():
    figure = build_figure()
    assert figure.skeleton is not None
    assert len(figure.skeleton.bones) == 19
    assert figure.is_figure()


def test_figure_is_bound_to_its_skeleton():
    figure = build_figure()
    assert len(figure.bindings) == figure.mesh.vertex_count


def test_figure_parameters_reject_an_impossible_height():
    with pytest.raises(ValueError):
        FigureParameters(height=5.0)


def test_figure_parameters_reject_a_non_positive_measurement():
    with pytest.raises(ValueError):
        FigureParameters(shoulder_width=0.0)


def test_morphing_a_figure_keeps_its_pose():
    figure = build_figure()
    figure.set_pose("forearm.L", vec3(0, 0, -60))
    morphed = morph_figure(figure, FigureParameters("Tall", height=2.0))
    assert morphed.pose == {"forearm.L": vec3(0, 0, -60)}
    assert morphed.mesh.bounds()[1].y > figure.mesh.bounds()[1].y


def test_figure_scene_frames_the_camera_on_the_figure():
    scene = build_figure_scene()
    center, radius = scene.center_and_radius()
    assert radius > 0.5
    distance = (scene.camera.position - center).length()
    assert distance > radius


def test_building_a_figure_with_an_unknown_preset_raises():
    with pytest.raises(KeyError):
        build_figure_from_preset("no-such-figure")


def test_figure_mesh_is_a_single_mesh_of_the_expected_size():
    mesh = build_figure_mesh(FigureParameters("Test"))
    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert build_figure_skeleton(FigureParameters("Test")).names()[0] == "hips"


# ---------------------------------------------------------------------------
# props
# ---------------------------------------------------------------------------


def test_every_primitive_builds_with_faces():
    for kind in primitive_names():
        mesh = build_primitive(kind)
        assert mesh.vertex_count >= 4
        assert mesh.face_count >= 1


def test_every_catalog_prop_builds():
    for key in prop_names():
        obj = build_prop(key)
        assert obj.mesh.face_count >= 1


def test_unknown_primitive_and_prop_and_material_raise():
    with pytest.raises(KeyError):
        build_primitive("dodecahedron")
    with pytest.raises(KeyError):
        build_prop("spaceship")
    with pytest.raises(KeyError):
        material_for("chartreuse")


def test_material_for_default_is_a_neutral_surface():
    assert material_for("default").name == "default"
    assert "skin" in MATERIAL_PALETTE


def test_prop_catalog_has_a_ground_plane():
    assert any(spec.key == "ground" for spec in PROP_CATALOG)


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------


def test_scene_add_and_remove_object_by_name():
    scene = make_default_scene()
    obj = SceneObject("box", build_primitive("cube"))
    scene.add_object(obj)
    assert scene.object_names() == ("box",)
    scene.add_object(SceneObject("box", build_primitive("sphere")))
    assert len(scene.objects) == 1  # replaced, not duplicated
    assert scene.remove_object("box") is True
    assert scene.object_names() == ()


def test_scene_find_object_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        make_default_scene().find_object("ghost")


def test_default_scene_has_three_lights():
    assert len(make_default_scene().lights) == 3


def test_camera_rejects_a_bad_light_kind():
    with pytest.raises(ValueError):
        Light("bad", "spotlight")


def test_light_direction_to_light_is_a_unit_vector():
    light = Light("key", "directional", direction=vec3(-1, -2, -3))
    assert light.direction_to_light(vec3(0, 0, 0)).length() == pytest.approx(1.0)


def test_material_rejects_a_bad_opacity():
    with pytest.raises(ValueError):
        Material(opacity=1.5)


def test_transform_composes_translation_rotation_and_scale():
    matrix = Transform(vec3(1, 0, 0), vec3(0, 0, 0), vec3(2, 2, 2)).to_matrix()
    assert matrix.transform_point(vec3(1, 1, 1)) == vec3(3, 2, 2)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


def test_render_scene_produces_a_real_png_of_the_right_size():
    scene = build_figure_scene()
    buffer, stats = render_scene(scene, width=120, height=90)
    png = buffer.to_png()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (120, 90)
    assert stats.width == 120 and stats.height == 90
    assert stats.triangles_drawn > 0


def test_render_png_decodes_to_the_expected_pixel_count():
    scene = build_figure_scene()
    buffer, _ = render_scene(scene, width=64, height=48)
    png = buffer.to_png()
    # Re-inflate the single IDAT chunk and check the raw scanline size.
    start = png.index(b"IDAT") + 4
    length = struct.unpack(">I", png[start - 8 : start - 4])[0]
    raw = zlib.decompress(png[start : start + length])
    assert len(raw) == 48 * (1 + 64 * 3)


def test_the_figure_covers_a_meaningful_share_of_the_frame():
    scene = build_figure_scene()
    buffer, _ = render_scene(scene, width=320, height=240)
    background = tuple(round(c * 255) for c in scene.background)
    drawn = sum(
        1
        for y in range(buffer.height)
        for x in range(buffer.width)
        if buffer.get_pixel(x, y) != background
    )
    assert drawn > 1500


def test_render_is_deterministic():
    scene = build_figure_scene()
    first, _ = render_scene(scene, width=100, height=80)
    second, _ = render_scene(scene, width=100, height=80)
    assert first.pixels == second.pixels


def test_wireframe_marks_more_pixels_than_the_solid_render():
    scene = build_figure_scene()
    solid, _ = render_scene(scene, width=200, height=160, wireframe=False)
    wired, _ = render_scene(scene, width=200, height=160, wireframe=True)
    background = tuple(round(c * 255) for c in scene.background)
    solid_pixels = sum(
        1 for y in range(solid.height) for x in range(solid.width)
        if solid.get_pixel(x, y) != background
    )
    wired_pixels = sum(
        1 for y in range(wired.height) for x in range(wired.width)
        if wired.get_pixel(x, y) != background
    )
    assert wired_pixels > solid_pixels


def test_a_hidden_object_is_not_drawn():
    scene = build_figure_scene()
    scene.objects[0].visible = False
    _buffer, stats = render_scene(scene, width=100, height=80)
    assert stats.objects == 0
    assert stats.triangles_drawn == 0


def test_framebuffer_clears_to_the_background_colour():
    buffer = Framebuffer(4, 3)
    buffer.clear((1.0, 0.0, 0.0))
    assert buffer.get_pixel(0, 0) == (255, 0, 0)
    assert all(value == float("inf") for value in buffer.depth)


def test_framebuffer_rejects_a_zero_size():
    with pytest.raises(ValueError):
        Framebuffer(0, 10)


def test_shade_point_returns_a_brighter_colour_facing_the_light():
    material = Material("m", (0.5, 0.5, 0.5), 0.0, 16.0)
    light = Light("key", "directional", direction=vec3(0, -1, 0), intensity=1.0)
    # A surface whose normal points straight at the light versus one edge-on to
    # it: the second has no diffuse term, so only the ambient floor remains.
    facing = shade_point(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 1, 0), material, [light])
    edge_on = shade_point(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), material, [light])
    assert sum(facing) > sum(edge_on)


def test_renderer_can_turn_shading_off():
    scene = build_figure_scene()
    buffer, stats = render_scene(scene, width=100, height=80, shaded=False)
    assert stats.shaded is False
    colours = {
        buffer.get_pixel(x, y)
        for y in range(0, buffer.height, 4)
        for x in range(0, buffer.width, 4)
    }
    # Flat shading gives a small, fixed set of colours, not a gradient.
    assert len(colours) <= 2


def test_render_rejects_a_non_positive_size():
    scene = build_figure_scene()
    with pytest.raises(ValueError):
        SoftwareRenderer().render(scene, width=0, height=10)
    with pytest.raises(ValueError):
        render_scene(scene, width=10, height=-5)


# ---------------------------------------------------------------------------
# OBJ / MTL
# ---------------------------------------------------------------------------


def test_load_obj_text_reads_a_triangle():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    scene, report = load_obj_text(text, name="tri")
    assert report.vertices == 3
    assert report.faces == 1
    assert scene.objects[0].mesh.triangle_count == 1


def test_load_obj_handles_negative_indices():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    scene, _ = load_obj_text(text, name="neg")
    assert scene.objects[0].mesh.faces == ((0, 1, 2),)


def test_load_obj_reads_objects_and_groups():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\no first\nf 1 2 3\ng second\nf 1 3 2\n"
    scene, report = load_obj_text(text, name="multi")
    assert report.objects == 2
    assert scene.object_names() == ("first", "second")


def test_load_obj_ignores_vertex_normals_and_texture_coordinates():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1\n"
    scene, _ = load_obj_text(text, name="vt")
    assert scene.objects[0].mesh.vertex_count == 3


def test_load_obj_rejects_a_file_with_no_faces():
    with pytest.raises(SceneIOError):
        load_obj_text("v 0 0 0\nv 1 0 0\n", name="empty")


def test_load_obj_rejects_a_zero_index():
    with pytest.raises(SceneIOError):
        load_obj_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n", name="zero")


def test_load_obj_reports_unsupported_lines_instead_of_failing():
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nnonsense here\nf 1 2 3\n"
    _, report = load_obj_text(text, name="odd")
    assert report.skipped_lines == 1


def test_load_obj_from_a_file_round_trips_through_save(tmp_path: Path):
    scene = build_figure_scene()
    target = save_obj(scene, tmp_path / "figure.obj")
    assert target.is_file()
    loaded, report = load_obj(target)
    assert report.faces == scene.objects[0].mesh.face_count
    assert loaded.objects[0].mesh.vertex_count == scene.objects[0].mesh.vertex_count


def test_save_obj_writes_a_sidecar_mtl(tmp_path: Path):
    scene = build_figure_scene()
    save_obj(scene, tmp_path / "figure.obj")
    mtl = tmp_path / "figure.mtl"
    assert mtl.is_file()
    assert "newmtl" in mtl.read_text(encoding="utf-8")


def test_parse_mtl_reads_colour_and_shininess():
    text = "newmtl gold\nKd 0.8 0.7 0.2\nKs 0.5 0.5 0.5\nNs 64\nd 0.5\n"
    materials = parse_mtl(text)
    assert materials["gold"].base_color == pytest.approx((0.8, 0.7, 0.2))
    assert materials["gold"].shininess == pytest.approx(64.0)
    assert materials["gold"].opacity == pytest.approx(0.5)


def test_write_mtl_round_trips_through_parse_mtl():
    materials = {"cloth": Material("cloth", (0.2, 0.3, 0.4), 0.1, 12.0)}
    parsed = parse_mtl(write_mtl(materials))
    assert parsed["cloth"].base_color == pytest.approx((0.2, 0.3, 0.4))


def test_load_obj_missing_file_reports_the_path(tmp_path: Path):
    with pytest.raises(SceneIOError):
        load_obj(tmp_path / "nope.obj")


def test_load_scene_rejects_an_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "model.stl"
    path.write_text("solid", encoding="utf-8")
    with pytest.raises(SceneIOError):
        load_scene(path)


def test_is_mesh_file_recognises_obj():
    assert is_mesh_file("a.obj")
    assert is_mesh_file("A.OBJ")
    assert not is_mesh_file("a.stl")


def test_load_obj_refuses_an_over_large_file(tmp_path: Path, monkeypatch):
    from core import scene3d_io

    monkeypatch.setattr(scene3d_io, "MAX_FILE_BYTES", 4)
    path = tmp_path / "big.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(SceneIOError):
        load_obj(path)


# ---------------------------------------------------------------------------
# native scene serialization
# ---------------------------------------------------------------------------


def test_scene_round_trip_keeps_the_pose():
    scene = build_figure_scene()
    scene.objects[0].set_pose("forearm.L", vec3(0, 0, -75))
    restored = scene_from_dict(scene_to_dict(scene))
    assert restored.objects[0].pose["forearm.L"] == vec3(0, 0, -75)


def test_scene_round_trip_rebinds_the_figure():
    scene = build_figure_scene()
    restored = scene_from_dict(scene_to_dict(scene))
    assert restored.objects[0].is_figure()
    assert len(restored.objects[0].bindings) == restored.objects[0].mesh.vertex_count


def test_scene_round_trip_keeps_the_geometry_to_sub_micron_precision():
    scene = build_figure_scene()
    scene.objects[0].set_pose("thigh.R", vec3(-25, 0, 0))
    restored = scene_from_dict(scene_to_dict(scene))
    delta = max(
        (a - b).length()
        for a, b in zip(scene.objects[0].world_vertices(), restored.objects[0].world_vertices(), strict=False)
    )
    assert delta < 1e-5


def test_scene_file_round_trips_through_disk(tmp_path: Path):
    scene = build_figure_scene()
    path = save_scene(scene, tmp_path / "scene")
    assert path.suffix == ".faydscene"
    restored = load_scene_file(path)
    assert restored.name == scene.name
    assert len(restored.lights) == len(scene.lights)


def test_scene_from_dict_refuses_an_unknown_version():
    payload = scene_to_dict(build_figure_scene())
    payload["version"] = 999
    with pytest.raises(SceneIOError):
        scene_from_dict(payload)


def test_load_scene_file_reports_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.faydscene"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SceneIOError):
        load_scene_file(path)


def test_load_scene_file_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(SceneIOError):
        load_scene_file(tmp_path / "nope.faydscene")


def test_scene_filename_is_filesystem_safe():
    assert scene_filename("My Scene: v2!") == "My_Scene_v2.faydscene"
    assert scene_filename("") == "scene.faydscene"


def test_scene_round_trip_keeps_camera_and_background():
    scene = build_figure_scene()
    scene.background = (0.1, 0.2, 0.3)
    scene.camera = Camera(position=vec3(1, 2, 3), target=vec3(0, 1, 0))
    restored = scene_from_dict(scene_to_dict(scene))
    assert restored.background == pytest.approx((0.1, 0.2, 0.3))
    assert restored.camera.position == vec3(1, 2, 3)


def test_scene_object_round_trip_keeps_the_transform():
    scene = make_default_scene()
    scene.add_object(
        SceneObject("box", build_primitive("cube"), transform=Transform(vec3(2, 1, 0), vec3(0, 45, 0)))
    )
    restored = scene_from_dict(scene_to_dict(scene))
    assert restored.objects[0].transform.translation == vec3(2, 1, 0)
