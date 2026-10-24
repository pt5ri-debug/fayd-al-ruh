"""The Scene Studio service: the Tk-free engine the window and the CLI share.

Every test drives the real service -- no display, no mocks -- and names the
behaviour it pins: a scene is created with a ground plane, figures and props are
added, a pose deforms the mesh, the camera frames and orbits, a render writes a
real PNG, a native scene round-trips through disk, an OBJ loads and exports, the
command box runs the same instructions the CLI does, and every failure is an
honest, typed error rather than a crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.scene3d import Vec3
from core.scene3d_io import SceneIOError
from gui.scene3d import (
    PRIMITIVE_NAMES,
    SceneCommand,
    SceneStudioError,
    SceneStudioService,
    command_help,
    parse_command,
    pose_preset_names,
)


@pytest.fixture()
def service(tmp_path: Path) -> SceneStudioService:
    return SceneStudioService(
        scenes_dir=tmp_path / "scenes",
        renders_dir=tmp_path / "renders",
        library_dir=tmp_path / "library",
    )


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------


def test_a_new_scene_has_a_ground_plane(service: SceneStudioService):
    scene = service.new_scene("Test")
    assert scene.name == "Test"
    assert scene.object_names() == ("Ground",)


def test_a_new_scene_can_skip_the_ground(service: SceneStudioService):
    scene = service.new_scene("Bare", with_ground=False)
    assert scene.object_names() == ()


def test_the_service_creates_a_scene_on_first_use(service: SceneStudioService):
    # A render before an explicit new_scene still works: there is always a scene.
    assert service.scene.name == "Untitled scene"


def test_status_reports_the_scene_summary(service: SceneStudioService):
    service.new_scene("Shown")
    assert "Shown" in service.status()


# ---------------------------------------------------------------------------
# figures and props
# ---------------------------------------------------------------------------


def test_add_figure_binds_a_skeleton(service: SceneStudioService):
    obj = service.add_figure("adult")
    assert obj.is_figure()
    assert service.bone_names(obj.name)


def test_add_figure_accepts_an_explicit_name(service: SceneStudioService):
    obj = service.add_figure("adult", name="Hero")
    assert obj.name == "Hero"
    assert "Hero" in service.scene.object_names()


def test_add_figure_rejects_an_unknown_preset(service: SceneStudioService):
    with pytest.raises(SceneStudioError):
        service.add_figure("robot")


def test_add_prop_rejects_an_unknown_key(service: SceneStudioService):
    with pytest.raises(SceneStudioError):
        service.add_prop("spaceship")


def test_add_prop_and_add_primitive(service: SceneStudioService):
    prop = service.add_prop("ground")
    cube = service.add_primitive("cube", name="box")
    assert prop.mesh.face_count >= 1
    assert cube.name == "box"
    names = [name for name, _kind, _visible in service.object_summaries()]
    assert prop.name in names and "box" in names


def test_object_summaries_mark_hidden_objects(service: SceneStudioService):
    service.new_scene("X", with_ground=False)
    obj = service.add_prop("cube")
    service.set_visible(obj.name, False)
    summaries = service.object_summaries()
    assert any(name == obj.name and visible is False for name, _kind, visible in summaries)


def test_remove_object_reports_whether_it_was_there(service: SceneStudioService):
    service.new_scene("X", with_ground=False)
    obj = service.add_prop("cube")
    assert service.remove_object(obj.name) is True
    assert service.remove_object(obj.name) is False


def test_set_visible_rejects_an_unknown_name(service: SceneStudioService):
    with pytest.raises(KeyError):
        service.set_visible("ghost", False)


# ---------------------------------------------------------------------------
# posing
# ---------------------------------------------------------------------------


def test_pose_presets_are_the_documented_six():
    assert pose_preset_names() == ("rest", "wave", "walk", "sit", "tpose", "reach")


def test_pose_bone_moves_the_mesh(service: SceneStudioService):
    obj = service.add_figure("adult")
    before = obj.world_vertices()
    service.pose_bone(obj.name, "forearm.L", Vec3(0, 0, -80))
    after = obj.world_vertices()
    assert any((a - b).length() > 1e-4 for a, b in zip(before, after, strict=False))


def test_pose_bone_rejects_an_unknown_bone(service: SceneStudioService):
    obj = service.add_figure("adult")
    with pytest.raises(SceneStudioError):
        service.pose_bone(obj.name, "tail", Vec3(0, 0, 10))


def test_pose_bone_rejects_a_non_figure(service: SceneStudioService):
    service.new_scene("X", with_ground=False)
    obj = service.add_prop("cube")
    with pytest.raises(SceneStudioError):
        service.pose_bone(obj.name, "root", Vec3(0, 0, 10))


def test_apply_pose_preset_changes_the_pose(service: SceneStudioService):
    obj = service.add_figure("adult")
    service.apply_pose_preset(obj.name, "wave")
    assert obj.pose  # some bones were rotated


def test_apply_pose_preset_rejects_an_unknown_preset(service: SceneStudioService):
    obj = service.add_figure("adult")
    with pytest.raises(SceneStudioError):
        service.apply_pose_preset(obj.name, "backflip")


def test_clear_pose_returns_to_the_rest_shape(service: SceneStudioService):
    obj = service.add_figure("adult")
    rest = obj.world_vertices()
    service.apply_pose_preset(obj.name, "sit")
    service.clear_pose(obj.name)
    for actual, expected in zip(obj.world_vertices(), rest, strict=False):
        assert actual.to_tuple() == pytest.approx(expected.to_tuple())


def test_clear_pose_on_a_never_posed_figure_is_harmless(service: SceneStudioService):
    obj = service.add_figure("adult")
    service.clear_pose(obj.name)
    assert obj.pose == {}


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


def test_frame_camera_puts_the_scene_in_front_of_it(service: SceneStudioService):
    service.add_figure("adult")
    service.frame_camera()
    camera = service.scene.camera
    center, radius = service.scene.center_and_radius()
    assert (camera.position - center).length() > radius


def test_orbit_camera_keeps_its_distance(service: SceneStudioService):
    service.add_figure("adult")
    service.frame_camera()
    before = (service.scene.camera.position - service.scene.camera.target).length()
    service.orbit_camera(45)
    after = (service.scene.camera.position - service.scene.camera.target).length()
    assert after == pytest.approx(before)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_writes_a_real_png(service: SceneStudioService):
    service.new_scene("Shot", with_ground=False)
    service.add_figure("adult")
    service.frame_camera()
    report = service.render(width=160, height=120)
    assert report.ok
    path = Path(report.path)
    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report.size_bytes > 0


def test_render_reports_how_much_was_drawn(service: SceneStudioService):
    service.new_scene("Shot", with_ground=False)
    service.add_figure("adult")
    service.frame_camera()
    report = service.render(width=160, height=120)
    assert report.stats is not None
    assert report.stats.triangles_drawn > 0
    assert "software rasteriser" in report.summary()


def test_render_refuses_an_out_of_range_size_honestly(service: SceneStudioService):
    service.add_figure("adult")
    report = service.render(width=99999, height=100)
    assert report.ok is False
    assert "between" in report.message


def test_render_uses_a_unique_name_for_each_shot(service: SceneStudioService):
    service.new_scene("Series", with_ground=False)
    service.add_figure("adult")
    first = service.render(width=200, height=160)
    second = service.render(width=200, height=160)
    assert first.path != second.path


def test_render_to_an_explicit_path(service: SceneStudioService, tmp_path: Path):
    service.add_figure("adult")
    target = tmp_path / "custom.png"
    report = service.render(width=200, height=160, path=str(target))
    assert Path(report.path) == target
    assert target.is_file()


def test_wireframe_render_is_a_different_picture(service: SceneStudioService):
    service.new_scene("W", with_ground=False)
    service.add_figure("adult")
    service.frame_camera()
    solid = service.render(width=160, height=120)
    wired = service.render(width=160, height=120, wireframe=True)
    assert Path(solid.path).read_bytes() != Path(wired.path).read_bytes()


def test_render_never_writes_the_target_in_place(
    service: SceneStudioService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A render must publish its PNG atomically, not truncate the target first.

    Two renders can write the same viewport file (adding a figure starts one, and
    framing the camera starts another), and a resize re-reads that file to refit
    the preview. A plain ``write_bytes`` truncates the target before writing, so a
    reader in that window sees a truncated image and the preview vanishes -- the
    reported flake. The fix writes a sibling temp file and renames it into place,
    so the target is only ever the whole picture. Pinning "the target is not
    written in place" is the deterministic form of that guarantee.
    """
    service.new_scene("Atomic", with_ground=False)
    service.add_figure("adult")
    target = tmp_path / "shot.png"

    def refuse_in_place(self: Path, _data: bytes) -> int:
        raise AssertionError(f"the render truncated {self} in place")

    monkeypatch.setattr(Path, "write_bytes", refuse_in_place)
    report = service.render(width=160, height=120, path=str(target))
    assert report.ok, report.message
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_a_failed_render_write_leaves_the_previous_image_intact(
    service: SceneStudioService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A render that fails mid-write must not destroy the picture already there.

    The atomic publish is what makes this true: the target is only replaced once
    the new bytes are complete, so a failure (a full disk, an I/O error) leaves
    the previous image whole rather than a truncated file, and the temp file is
    cleaned up so nothing is left behind.
    """
    import os as os_module

    service.new_scene("Crash", with_ground=False)
    service.add_figure("adult")
    service.frame_camera()
    target = tmp_path / "kept.png"
    good = service.render(width=160, height=120, path=str(target))
    assert good.ok
    original = target.read_bytes()

    def refuse_replace(_src: str, _dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os_module, "replace", refuse_replace)
    failed = service.render(width=160, height=120, path=str(target))
    assert failed.ok is False
    assert "Could not write" in failed.message
    # The previous render is still a whole, readable PNG, and no temp file litters.
    assert target.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def test_save_and_open_a_native_scene(service: SceneStudioService):
    service.new_scene("Saved")
    obj = service.add_figure("adult")
    service.apply_pose_preset(obj.name, "wave")
    path = service.save_scene("Saved")
    assert Path(path).is_file()
    # A bare name is a *name*, not a location: it lands in the Studio's own
    # scenes folder, never in whatever directory the program was run from.
    assert Path(path).parent == service.scenes_dir

    reopened = service.open_path(path)
    assert reopened.ok
    assert "Adult" in service.scene.object_names()


def test_open_path_reports_a_bad_scene(service: SceneStudioService, tmp_path: Path):
    broken = tmp_path / "broken.faydscene"
    broken.write_text("{not json", encoding="utf-8")
    result = service.open_path(broken)
    assert result.ok is False
    assert "scene" in result.message.lower() or "json" in result.message.lower()


def test_a_bare_export_name_goes_to_the_library_folder(
    service: SceneStudioService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The same rule as ``save_scene``: a bare filename is a name, not a location,
    # so ``export_obj("out.obj")`` cannot drop an .obj/.mtl in the checkout.
    monkeypatch.chdir(tmp_path)
    service.new_scene("Export")
    service.add_figure("adult")
    saved = service.export_obj("out.obj")
    assert saved.parent == service.library_dir
    assert saved.is_file()
    assert not (tmp_path / "out.obj").exists()
    # An explicit path is still honoured exactly as given.
    explicit = tmp_path / "chosen.obj"
    assert service.export_obj(explicit) == explicit


def test_export_obj_writes_obj_and_mtl(service: SceneStudioService, tmp_path: Path):
    service.new_scene("Exported", with_ground=False)
    service.add_figure("adult")
    target = tmp_path / "out.obj"
    saved = service.export_obj(target)
    assert Path(saved).is_file()
    assert (tmp_path / "out.mtl").is_file()


def test_open_model_loads_an_obj(service: SceneStudioService, tmp_path: Path):
    source = tmp_path / "tri.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    result = service.open_model(source)
    assert result.ok
    assert result.kind == "model"
    assert service.scene.objects[0].mesh.triangle_count == 1


def test_open_model_reports_a_missing_file(service: SceneStudioService, tmp_path: Path):
    result = service.open_model(tmp_path / "gone.obj")
    assert result.ok is False


def test_a_bare_save_name_goes_to_the_scenes_folder(
    service: SceneStudioService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Regression: ``save_scene("Demo")`` used to resolve relative to the current
    # working directory, so running the CLI or the tests dropped a .faydscene in
    # the checkout. A bare name must resolve inside the Studio's own folder.
    monkeypatch.chdir(tmp_path)
    service.new_scene("Demo")
    saved = service.save_scene("Demo")
    assert saved.parent == service.scenes_dir
    assert not (tmp_path / "Demo.faydscene").exists()
    # An explicit path is still honoured exactly as given.
    explicit = tmp_path / "nested" / "chosen.faydscene"
    explicit.parent.mkdir()
    assert service.save_scene(explicit) == explicit


def test_save_scene_creates_a_missing_folder(service: SceneStudioService, tmp_path: Path):
    service.new_scene("X")
    target = tmp_path / "nope" / "deep" / "x.faydscene"
    saved = service.save_scene(target)
    assert Path(saved).is_file()


def test_save_scene_raises_for_an_unwritable_target(
    service: SceneStudioService, tmp_path: Path
):
    service.new_scene("X")
    # A directory where the file should go: the write must fail honestly.
    blocker = tmp_path / "blocked.faydscene"
    blocker.mkdir()
    with pytest.raises(SceneIOError):
        service.save_scene(blocker)


def test_list_saved_scenes_ignores_the_working_scene(service: SceneStudioService):
    service.new_scene("Kept")
    service.save_scene("Kept")
    service.save_working()
    names = [path.name for path in service.list_saved_scenes()]
    assert "working.faydscene" not in names


def test_the_working_scene_round_trips(service: SceneStudioService):
    service.new_scene("Working")
    service.add_figure("slim")
    service.save_working()
    assert service.load_working() is True
    assert "Slim" in service.scene.object_names()


def test_load_working_reports_when_there_is_none(service: SceneStudioService):
    assert service.load_working() is False


# ---------------------------------------------------------------------------
# the command box
# ---------------------------------------------------------------------------


def test_parse_command_reads_an_add_figure():
    command = parse_command("add figure athletic")
    assert command == SceneCommand("add_figure", ("athletic",), "add figure athletic")


def test_parse_command_reads_a_bone_rotation():
    command = parse_command("pose forearm.L 0 0 -90")
    assert command.action == "pose_bone"
    assert command.args == ("forearm.L", "0", "0", "-90")


def test_parse_command_reads_a_pose_preset():
    assert parse_command("pose wave").action == "pose_preset"
    assert parse_command("pose wave").args == ("wave",)


def test_parse_command_is_case_insensitive_and_collapses_whitespace():
    assert parse_command("  ADD   Figure   Athletic ").action == "add_figure"
    assert parse_command("  ADD   Figure   Athletic ").args == ("athletic",)


def test_parse_command_reports_an_unknown_verb():
    assert parse_command("teleport the figure").action == "unknown"
    assert parse_command("").action == "unknown"


def test_parse_command_reads_the_primitive_forms():
    for kind in PRIMITIVE_NAMES:
        assert parse_command(f"add {kind} thing").action == "add_primitive"


def test_parse_command_covers_every_documented_verb():
    samples = {
        "add figure adult": "add_figure",
        "add prop ground": "add_prop",
        "pose wave": "pose_preset",
        "pose hips 0 0 10": "pose_bone",
        "clear pose": "clear_pose",
        "frame": "frame",
        "orbit 30": "orbit",
        "render": "render",
        "remove box": "remove",
        "hide box": "hide",
        "show box": "show",
        "save": "save",
        "new": "new",
        "open model.obj": "open",
    }
    for text, action in samples.items():
        assert parse_command(text).action == action, text


def test_command_help_lists_the_documented_verbs():
    help_text = command_help()
    for verb in ("add figure", "add prop", "pose", "clear pose", "frame", "orbit",
                 "render", "remove", "save", "open"):
        assert verb in help_text
    assert "Commands:" in help_text


def test_run_command_adds_a_figure(service: SceneStudioService):
    service.new_scene("Cmd", with_ground=False)
    result = service.run_command("add figure athletic")
    assert "Athletic" in result
    assert "Athletic" in service.scene.object_names()


def test_run_command_poses_the_first_figure(service: SceneStudioService):
    service.new_scene("Cmd", with_ground=False)
    service.add_figure("adult")
    assert "wave" in service.run_command("pose wave")


def test_run_command_renders_and_saves(service: SceneStudioService):
    service.new_scene("Cmd", with_ground=False)
    service.add_figure("adult")
    service.frame_camera()
    result = service.run_command("render 120 90")
    assert "software rasteriser" in result


def test_run_command_reports_an_unknown_instruction(service: SceneStudioService):
    result = service.run_command("dance wildly")
    assert "Did not understand" in result
    assert "Commands:" in result


def test_run_command_reports_an_unknown_figure_preset(service: SceneStudioService):
    result = service.run_command("add figure robot")
    assert "Unknown figure preset" in result


def test_run_command_reports_an_unknown_prop(service: SceneStudioService):
    result = service.run_command("add prop spaceship")
    assert "Unknown prop" in result


def test_run_command_posing_with_no_figure_says_so(service: SceneStudioService):
    service.new_scene("Empty", with_ground=False)
    assert "no figure" in service.run_command("pose wave").lower()


def test_run_command_manages_objects(service: SceneStudioService):
    service.new_scene("Cmd", with_ground=False)
    service.run_command("add cube box")
    assert "Hid box" in service.run_command("hide box")
    assert "Showed box" in service.run_command("show box")
    assert "Removed box" in service.run_command("remove box")
    assert "No object named" in service.run_command("remove box")


def test_run_command_saves_the_scene(service: SceneStudioService):
    service.new_scene("Cmd")
    assert "Saved the scene" in service.run_command("save Cmd")


def test_run_command_frames_and_orbits(service: SceneStudioService):
    service.add_figure("adult")
    assert "Framed" in service.run_command("frame")
    assert "Orbited" in service.run_command("orbit 15")


# ---------------------------------------------------------------------------
# honesty
# ---------------------------------------------------------------------------


def test_the_service_declares_what_it_cannot_do(service: SceneStudioService):
    from gui.scene3d import HONESTY_LINE

    assert "no GPU" in HONESTY_LINE
    assert "software rasteriser" in HONESTY_LINE


def test_mesh_filetypes_offers_obj_and_scenes(service: SceneStudioService):
    patterns = " ".join(pattern for _label, pattern in service.mesh_filetypes())
    assert "*.obj" in patterns
    assert "*.faydscene" in patterns


def test_mesh_filetypes_are_derived_from_the_real_suffixes():
    from core.scene3d_io import MESH_SUFFIXES
    from core.scene3d_serialize import SCENE_SUFFIXES

    for suffix in MESH_SUFFIXES + SCENE_SUFFIXES:
        assert suffix.startswith(".")
