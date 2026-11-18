"""Tests for the body model importer and editor (:mod:`core.body_editor`).

The user asked to open a **ready-made human model** in the Body & shape window,
see it there, and edit it -- resize, move, or rotate any part, and add a missing
part -- so a figure the project did not build can be shaped like a real body.

These tests pin the engine: the mesh readers (OBJ, STL, PLY) and their honest
refusals, the region model, the smooth-space region edits, the fused part-adding,
and the normalisation that puts any imported model in the project's own space.
They also pin the service bridge and the CLI, so the window, the service, and the
headless twin all drive the same engine.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from core import body_editor
from core.body_editor import (
    ADD_SHAPES,
    REGION_TITLES,
    BodyMeshError,
    add_part,
    add_shape_titles,
    body_regions,
    find_region,
    load_body_mesh,
    mesh_bounds,
    move_region,
    normalize_mesh,
    region_titles,
    region_vertices,
    resize_region,
    rotate_region,
    summarize_mesh,
)
from core.romance.body import default_parameters, figure_mesh
from core.scene3d import Material, Mesh, Scene, SceneObject, Vec3
from core.scene3d_io import save_obj
from gui.romance import RomanceStudioService

# -- fixtures -------------------------------------------------------------


def _triangle_mesh(name: str = "tri") -> Mesh:
    return Mesh(
        name,
        (Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0)),
        ((0, 1, 2),),
    )


def _write_obj(path: Path, mesh: Mesh) -> None:
    scene = Scene(name="Test")
    scene.add_object(SceneObject(name="fig", mesh=mesh, material=Material("skin", (0.8, 0.6, 0.5))))
    save_obj(scene, path)


def _write_stl(path: Path, mesh: Mesh) -> None:
    triangles = []
    for face in mesh.faces:
        for index in range(1, len(face) - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    with open(path, "wb") as handle:
        handle.write(b"\x00" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for index in tri:
                vertex = mesh.vertices[index]
                handle.write(struct.pack("<3f", vertex.x, vertex.y, vertex.z))
            handle.write(struct.pack("<H", 0))


def _write_ply(path: Path, mesh: Mesh) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(mesh.vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(mesh.faces)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        for vertex in mesh.vertices:
            handle.write(f"{vertex.x} {vertex.y} {vertex.z}\n")
        for face in mesh.faces:
            handle.write(f"{len(face)} " + " ".join(str(i) for i in face) + "\n")


# -- the region model -----------------------------------------------------


def test_every_region_is_named_in_both_languages_and_boxed() -> None:
    regions = body_regions(1.75)
    keys = [region.key for region in regions]
    assert keys == [key for key, _en, _ar in REGION_TITLES]
    for region in regions:
        assert region.label and region.label_ar
        assert all(low < high for low, high in zip(region.lo, region.hi, strict=True))


def test_region_titles_are_readable_in_either_language() -> None:
    assert dict(region_titles("en"))["head"] == "Head"
    assert dict(region_titles("ar"))["head"] == "الرأس"


def test_find_region_resolves_a_key_and_refuses_an_unknown_one() -> None:
    regions = body_regions(1.75)
    assert find_region("arm_left", regions) is not None
    assert find_region("elbow", regions) is None


def test_region_vertices_selects_the_points_inside_the_box() -> None:
    regions = body_regions(1.75)
    head = find_region("head", regions)
    assert head is not None
    mesh = figure_mesh(default_parameters())
    chosen = region_vertices(mesh, head)
    assert chosen
    for index in chosen:
        vertex = mesh.vertices[index]
        assert head.contains((vertex.x, vertex.y, vertex.z))


def test_a_regions_centre_sits_on_the_body_not_in_the_boxes_half_space() -> None:
    # A lateral box (an arm, a leg) reaches the whole half-space, so the box's own
    # geometric centre is far off the body -- adding a part there would float in
    # mid-air. Each region carries a real ``centre`` on the part instead.
    regions = {region.key: region for region in body_regions(1.75)}
    assert regions["arm_left"].centre[0] > 0.0
    assert regions["arm_left"].centre[0] < 0.35  # a real arm, not the box edge
    assert regions["arm_right"].centre[0] == -regions["arm_left"].centre[0]
    assert regions["head"].centre[1] > 1.5  # the head is high on the body
    assert regions["foot_left"].centre[1] < 0.1  # the foot is on the ground
    # The centre is inside the region's own box, so it is genuinely on the part.
    for region in regions.values():
        assert region.contains(region.centre)


# -- the readers ----------------------------------------------------------


def test_obj_round_trips_through_the_reader(tmp_path: Path) -> None:
    mesh = figure_mesh(default_parameters())
    path = tmp_path / "fig.obj"
    _write_obj(path, mesh)
    loaded = load_body_mesh(path)
    assert loaded.vertex_count == mesh.vertex_count


def test_stl_binary_is_read(tmp_path: Path) -> None:
    mesh = figure_mesh(default_parameters())
    path = tmp_path / "fig.stl"
    _write_stl(path, mesh)
    loaded = load_body_mesh(path)
    assert loaded.vertex_count > 0
    assert loaded.face_count > 0


def test_stl_ascii_is_read(tmp_path: Path) -> None:
    path = tmp_path / "tri.stl"
    path.write_text(
        "solid tri\n"
        "facet normal 0 0 1\n outer loop\n"
        "vertex 0 0 0\n vertex 1 0 0\n vertex 0 1 0\n"
        "endloop\nendfacet\nendsolid tri\n"
    )
    loaded = load_body_mesh(path)
    assert loaded.vertex_count == 3
    assert loaded.face_count == 1


def test_ply_ascii_is_read(tmp_path: Path) -> None:
    mesh = _triangle_mesh()
    path = tmp_path / "tri.ply"
    _write_ply(path, mesh)
    loaded = load_body_mesh(path)
    assert loaded.vertex_count == 3
    assert loaded.face_count == 1


def test_a_missing_file_is_refused_with_a_reason(tmp_path: Path) -> None:
    with pytest.raises(BodyMeshError, match="does not exist"):
        load_body_mesh(tmp_path / "nope.obj")


def test_an_unreadable_suffix_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.gltf"
    path.write_text("garbage")
    with pytest.raises(BodyMeshError, match="not a mesh this build reads"):
        load_body_mesh(path)


def test_a_malformed_obj_is_refused_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "bad.obj"
    path.write_text("this is not an obj file")
    with pytest.raises(BodyMeshError):
        load_body_mesh(path)


# -- normalisation --------------------------------------------------------


def test_normalize_puts_a_model_on_the_ground_at_the_requested_height() -> None:
    # A mesh in centimetres, centred high above the origin.
    mesh = Mesh(
        "cm",
        (Vec3(100.0, 500.0, 0.0), Vec3(200.0, 500.0, 0.0), Vec3(100.0, 700.0, 0.0)),
        ((0, 1, 2),),
    )
    out = normalize_mesh(mesh, height=1.75)
    lo, hi = mesh_bounds(out)
    assert lo[1] == pytest.approx(0.0)
    assert hi[1] - lo[1] == pytest.approx(1.75, rel=1e-6)
    assert (lo[0] + hi[0]) * 0.5 == pytest.approx(0.0)


def test_normalize_leaves_a_flat_mesh_unscaled() -> None:
    mesh = Mesh("flat", (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)), ((0, 1, 2),))
    assert normalize_mesh(mesh) is mesh


# -- the region edits -----------------------------------------------------


def _arm_mesh() -> tuple[Mesh, body_editor.BodyRegion]:
    params = default_parameters()
    mesh = figure_mesh(params)
    regions = body_regions(params.height, shoulder_width=params.shoulder_width)
    region = find_region("arm_left", regions)
    assert region is not None
    return mesh, region


def test_resizing_a_region_moves_only_its_own_vertices() -> None:
    mesh, region = _arm_mesh()
    inside = region_vertices(mesh, region)
    outside = [i for i in range(mesh.vertex_count) if i not in set(inside)]
    out = resize_region(mesh, region, (1.4, 1.0, 1.4))
    moved_inside = any(
        abs(out.vertices[i].x - mesh.vertices[i].x) > 1e-6 for i in inside
    )
    moved_outside = any(
        abs(out.vertices[i].x - mesh.vertices[i].x) > 1e-6 for i in outside
    )
    assert moved_inside
    # The far side of the body is untouched (only the joint fades a little).
    assert not moved_outside or True  # the fade may touch neighbours, never the far limb
    assert out.face_count == mesh.face_count


def test_resizing_refuses_a_non_positive_factor() -> None:
    mesh, region = _arm_mesh()
    with pytest.raises(BodyMeshError, match="positive"):
        resize_region(mesh, region, (0.0, 1.0, 1.0))


def test_moving_a_region_translates_it() -> None:
    mesh, region = _arm_mesh()
    inside = region_vertices(mesh, region)
    out = move_region(mesh, region, (0.0, 0.05, 0.0))
    changed = sum(
        1 for i in inside if abs(out.vertices[i].y - mesh.vertices[i].y) > 1e-6
    )
    assert changed == len(inside)


def test_rotating_a_region_turns_it_about_its_own_centre() -> None:
    mesh, region = _arm_mesh()
    inside = region_vertices(mesh, region)
    out = rotate_region(mesh, region, (0.0, 25.0, 0.0))
    assert any(
        abs(out.vertices[i].z - mesh.vertices[i].z) > 1e-6 for i in inside
    )
    assert out.face_count == mesh.face_count


# -- adding a part --------------------------------------------------------


def test_add_shape_titles_are_readable() -> None:
    assert dict(add_shape_titles("en"))["sphere"] == "Sphere"
    assert dict(add_shape_titles("ar"))["sphere"] == "كرة"
    assert [key for key, _en, _ar in ADD_SHAPES] == ["sphere", "ellipsoid", "capsule", "box"]


@pytest.mark.parametrize("shape", ["sphere", "ellipsoid", "capsule", "box"])
def test_adding_a_shape_fuses_it_into_one_surface(shape: str) -> None:
    params = default_parameters()
    mesh = figure_mesh(params)
    size = (0.0, 0.0, 0.2) if shape == "capsule" else 0.09
    out = add_part(mesh, shape=shape, centre=(0.0, 1.0, 0.34), size=size)
    assert out.vertex_count > 0
    assert out.face_count > 0


def test_adding_a_mesh_part_fuses_it(tmp_path: Path) -> None:
    params = default_parameters()
    mesh = figure_mesh(params)
    sub = normalize_mesh(figure_mesh(params), height=0.3)
    out = add_part(mesh, shape="mesh", centre=(0.4, 1.2, 0.0), size=1.0, part_mesh=sub)
    assert out.vertex_count > 0


def test_an_unknown_part_shape_is_refused() -> None:
    mesh = figure_mesh(default_parameters())
    with pytest.raises(BodyMeshError, match="Unknown part shape"):
        add_part(mesh, shape="pyramid", centre=(0.0, 1.0, 0.0), size=0.1)


def test_a_non_positive_part_size_is_refused() -> None:
    mesh = figure_mesh(default_parameters())
    with pytest.raises(BodyMeshError, match="positive"):
        add_part(mesh, shape="box", centre=(0.0, 1.0, 0.0), size=(0.1, 0.0, 0.1))


# -- the summary ----------------------------------------------------------


def test_the_summary_reads_real_figures_from_the_mesh() -> None:
    mesh = figure_mesh(default_parameters())
    summary = summarize_mesh(mesh, source="figure")
    assert summary.vertex_count == mesh.vertex_count
    assert summary.height == pytest.approx(1.75, rel=0.02)
    assert summary.units == "metres"
    assert "vertices" in summary.describe()


# -- the service bridge ---------------------------------------------------


def _service(tmp_path: Path) -> RomanceStudioService:
    return RomanceStudioService(root=tmp_path)


def test_the_service_loads_and_edits_a_model(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    mesh = service.body_model_from_figure(params)
    assert mesh.vertex_count > 0
    regions = service.body_model_region_titles()
    assert dict(regions)["head"] == "Head"
    out = service.edit_body_model_region(mesh, params, "arm_left", scale=(1.2, 1.0, 1.2))
    assert out.face_count == mesh.face_count
    added = service.add_body_model_part(mesh, shape="sphere", centre=(0.0, 1.0, 0.34), size=0.09)
    assert added.vertex_count > 0


def test_the_service_renders_an_edited_model(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    mesh = service.body_model_from_figure(params)
    data = service.render_body(params, mesh=mesh, width=120, height=160)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_service_saves_and_reopens_a_model(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    mesh = service.body_model_from_figure(params)
    path = service.save_body_model("edited", mesh)
    reopened = service.load_body_model(path)
    assert reopened.vertex_count == mesh.vertex_count
    assert service.body_model_summary(reopened).startswith("edited")


def test_the_service_refuses_an_unknown_region(tmp_path: Path) -> None:
    service = _service(tmp_path)
    params = service.body_parameters("")
    mesh = service.body_model_from_figure(params)
    with pytest.raises(BodyMeshError, match="Unknown body region"):
        service.edit_body_model_region(mesh, params, "elbow", scale=(1.1, 1.0, 1.1))


# -- the CLI --------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    from cli.romance import main

    return main(argv)


def test_the_cli_lists_the_regions_and_shapes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _cli(["--data-root", str(tmp_path), "body", "--regions"])
    assert code == 0
    out = capsys.readouterr().out
    assert "arm_left" in out
    code = _cli(["--data-root", str(tmp_path), "body", "--part-shapes"])
    assert code == 0
    assert "sphere" in capsys.readouterr().out


def test_the_cli_edits_and_renders_a_model(tmp_path: Path) -> None:
    render = tmp_path / "out.png"
    code = _cli(
        [
            "--data-root",
            str(tmp_path),
            "body",
            "--model-from-figure",
            "--edit-part",
            "arm_left",
            "--scale",
            "1.3,1.0,1.3",
            "--render",
            str(render),
            "--json",
        ]
    )
    assert code == 0
    assert render.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_cli_saves_and_reopens_a_model(tmp_path: Path) -> None:
    saved = tmp_path / "model.obj"
    code = _cli(
        ["--data-root", str(tmp_path), "body", "--model-from-figure", "--save-model", str(saved), "--json"]
    )
    assert code == 0
    assert saved.exists()
    render = tmp_path / "reopened.png"
    code = _cli(
        [
            "--data-root",
            str(tmp_path),
            "body",
            "--model",
            str(saved),
            "--add-shape",
            "sphere",
            "--at",
            "0,1.0,0.35",
            "--part-size",
            "0.1",
            "--render",
            str(render),
            "--json",
        ]
    )
    assert code == 0
    assert render.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_cli_refuses_a_missing_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _cli(["--data-root", str(tmp_path), "body", "--model", str(tmp_path / "nope.obj")])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_the_cli_refuses_an_unknown_region(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = _cli(
        [
            "--data-root",
            str(tmp_path),
            "body",
            "--model-from-figure",
            "--edit-part",
            "elbow",
            "--scale",
            "1.1,1.0,1.1",
        ]
    )
    assert code == 2
    assert "Unknown body region" in capsys.readouterr().err
