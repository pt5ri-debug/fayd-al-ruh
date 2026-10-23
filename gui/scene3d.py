"""The Scene Studio service: the Tk-free half of the 3D figure window.

This is where the studio's work actually happens -- build a figure, add props,
pose bones, load a model, render it, save and reopen a scene -- with no display,
so the whole workflow is testable on a fresh clone. The window only collects
clicks and shows the result.

It is deliberately a thin, honest layer over the real engine:

* the scene model and skinning come from :mod:`core.scene3d`;
* figures and props come from :mod:`core.scene3d_figures` /
  :mod:`core.scene3d_props`;
* rendering is the CPU software rasteriser in :mod:`core.scene3d_render`;
* files are read and written by :mod:`core.scene3d_io` (OBJ/MTL) and
  :mod:`core.scene3d_serialize` (native scenes).

Everything is local: nothing here opens a network connection, contacts an asset
store, or downloads anything. A missing or malformed file is reported as a
:class:`SceneStudioError` carrying the real reason, never a silent failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.paths import (
    scene_studio_library_dir,
    scene_studio_renders_dir,
    scene_studio_scenes_dir,
)
from core.scene3d import (
    Scene,
    SceneObject,
    Transform,
    Vec3,
    make_default_scene,
)
from core.scene3d_figures import (
    FIGURE_PRESETS,
    FigureParameters,
    build_figure,
    build_figure_from_preset,
    figure_preset_names,
    morph_figure,
)
from core.scene3d_io import (
    LoadReport,
    SceneIOError,
    load_scene,
    mesh_filetypes,
    save_obj,
)
from core.scene3d_props import (
    build_prop,
    default_ground,
    material_for,
    material_names,
    primitive_names,
    prop_names,
)
from core.scene3d_render import RenderStats, render_scene
from core.scene3d_serialize import (
    SCENE_SUFFIXES,
    load_scene_file,
    save_scene,
    scene_filename,
)

#: The default render size, and the bounds a render may use.
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 640
MIN_DIMENSION = 64
MAX_DIMENSION = 4096

#: The honesty line shown above every render, as everywhere else in the project.
HONESTY_LINE = (
    "Rendered on this machine by the CPU software rasteriser: a real z-buffered "
    "rasteriser with Blinn-Phong shading, but no GPU, no ray tracing, and no "
    "commercial render engine."
)


class SceneStudioError(RuntimeError):
    """Raised when the Studio cannot carry out a request, with the real reason."""


@dataclass(frozen=True)
class RenderReport:
    """The outcome of one render, ready for a window or a terminal."""

    ok: bool
    message: str
    path: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    stats: RenderStats | None = None

    def summary(self) -> str:
        if not self.ok:
            return f"Render failed: {self.message}"
        lines = [f"Render saved to {self.path}"]
        if self.stats is not None:
            lines.append(f"  {self.stats.summary()}")
        lines.append(f"  size    : {self.width}x{self.height}, {self.size_bytes} bytes")
        lines.append(f"  {HONESTY_LINE}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LoadResult:
    """The outcome of loading a model or scene file."""

    ok: bool
    message: str
    kind: str = ""  # "scene" or "model"
    objects: int = 0
    vertices: int = 0
    faces: int = 0

    def summary(self) -> str:
        return self.message if not self.ok else f"{self.message}"


@dataclass
class StudioSession:
    """The scene the Studio currently has open, plus its file and render history."""

    scene: Scene
    scene_path: str = ""
    last_render: str = ""
    report: LoadReport | None = None
    history: list[str] = field(default_factory=list)
    """A short log of the actions taken, newest last, for the window's status line."""

    def note(self, message: str) -> None:
        self.history.append(message)
        del self.history[:-40]

    @property
    def name(self) -> str:
        return self.scene.name


class SceneStudioService:
    """The one engine the window and the CLI share.

    Constructing it opens an empty session (no scene). Call :meth:`new_scene`,
    :meth:`add_figure`, or :meth:`open_model` to put something in it.
    """

    def __init__(
        self,
        *,
        scenes_dir: Path | str | None = None,
        renders_dir: Path | str | None = None,
        library_dir: Path | str | None = None,
    ) -> None:
        self.scenes_dir = Path(scenes_dir) if scenes_dir else scene_studio_scenes_dir()
        self.renders_dir = Path(renders_dir) if renders_dir else scene_studio_renders_dir()
        self.library_dir = Path(library_dir) if library_dir else scene_studio_library_dir()
        self.session: StudioSession | None = None

    # -- session ----------------------------------------------------------

    @property
    def scene(self) -> Scene:
        """The open scene, creating an empty one on first use."""
        if self.session is None:
            self.new_scene()
        assert self.session is not None
        return self.session.scene

    def new_scene(self, name: str = "Untitled scene", *, with_ground: bool = True) -> Scene:
        """Start a fresh scene (optionally with a ground plane), replacing any open one."""
        scene = make_default_scene(name)
        if with_ground:
            scene.add_object(default_ground())
        self.session = StudioSession(scene=scene)
        self.session.note(f"New scene: {name}.")
        return scene

    def open_model(self, path: str | Path) -> LoadResult:
        """Load an OBJ model (or a native scene) into the Studio.

        The suffix chooses the reader: a ``.obj`` is geometry, a ``.faydscene`` /
        ``.json`` is a native scene (rig and pose included). Anything else is
        refused with a message naming what this build reads.
        """
        target = Path(path).expanduser()
        suffix = target.suffix.lower()
        try:
            if suffix in SCENE_SUFFIXES and suffix != ".obj":
                scene = load_scene_file(target)
                self.session = StudioSession(scene=scene, scene_path=str(target))
                self.session.note(f"Opened scene {target.name}.")
                return LoadResult(
                    ok=True,
                    message=f"Opened scene {target.name}: {scene.summary()}",
                    kind="scene",
                    objects=len(scene.objects),
                )
            scene, report = load_scene(target)
        except SceneIOError as error:
            return LoadResult(ok=False, message=str(error))
        self.session = StudioSession(scene=scene, scene_path=str(target), report=report)
        self.session.note(f"Loaded model {target.name}.")
        return LoadResult(
            ok=True,
            message=report.summary(),
            kind="model",
            objects=report.objects,
            vertices=report.vertices,
            faces=report.faces,
        )

    def save_scene(self, path: str | Path | None = None) -> Path:
        """Save the open scene natively (rig and pose included).

        A bare name ("My scene") is a *name*, not a location: it is saved into the
        Studio's own scenes folder with a safe filename, so ``save_scene("Demo")``
        cannot drop a scene in whatever directory the program happened to be run
        from. Anything that names a location -- an absolute path, or a path with a
        directory part -- is honoured as given.
        """
        target = self._scene_target(path)
        saved = save_scene(self.scene, target)
        assert self.session is not None
        self.session.scene_path = str(saved)
        self.session.note(f"Saved scene to {saved.name}.")
        return saved

    def _scene_target(self, path: str | Path | None) -> Path:
        """Resolve a save target: a bare name goes to the scenes folder."""
        if path is None:
            return self.scenes_dir / scene_filename(self.scene.name)
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            return self.scenes_dir / scene_filename(candidate.name)
        return candidate

    def export_obj(self, path: str | Path) -> Path:
        """Export the open scene as Wavefront OBJ (geometry in its current pose).

        Like :meth:`save_scene`, a bare filename is a *name*: it is written into
        the Studio's own ``library/`` folder with a safe filename, so an export can
        never drop an ``.obj``/``.mtl`` in whatever directory the program was run
        from. Anything naming a location is honoured as given.
        """
        saved = save_obj(self.scene, self._export_target(path))
        assert self.session is not None
        self.session.note(f"Exported OBJ to {saved.name}.")
        return saved

    def _export_target(self, path: str | Path) -> Path:
        """Resolve an export target: a bare filename goes to the library folder."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            return self.library_dir / candidate.name
        return candidate

    def list_saved_scenes(self) -> tuple[Path, ...]:
        """Every native scene file in the Studio's scenes folder, newest first.

        The CLI's scratch ``working.faydscene`` is not a scene the user saved, so
        it is left out of the list.
        """
        if not self.scenes_dir.is_dir():
            return ()
        files = [
            path
            for path in self.scenes_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".faydscene"
            and path.name != "working.faydscene"
        ]
        return tuple(sorted(files, key=lambda path: path.stat().st_mtime, reverse=True))

    # -- the working scene ------------------------------------------------

    @property
    def working_path(self) -> Path:
        """The scene file the CLI keeps between commands (a scratch workspace)."""
        return self.scenes_dir / "working.faydscene"

    def load_working(self) -> bool:
        """Load the working scene, if one has been saved. Returns whether it did.

        The window keeps a scene in memory; the CLI is a fresh process per
        command, so it needs somewhere to accumulate: a ``working.faydscene`` in
        the scenes folder. This is what lets ``add-figure`` then ``pose-preset``
        then ``render`` build on one another the way they do in the window.
        """
        if not self.working_path.is_file():
            return False
        try:
            scene = load_scene_file(self.working_path)
        except SceneIOError:
            return False
        self.session = StudioSession(scene=scene, scene_path=str(self.working_path))
        return True

    def save_working(self) -> Path:
        """Save the open scene as the working scene (a no-op when none is open)."""
        if self.session is None:
            raise SceneStudioError("There is no scene to save.")
        return save_scene(self.scene, self.working_path)

    def open_path(self, path: str | Path) -> LoadResult:
        """Load a native scene, keeping the path (used by the window's Open)."""
        target = Path(path).expanduser()
        try:
            scene = load_scene_file(target)
        except SceneIOError as error:
            return LoadResult(ok=False, message=str(error))
        self.session = StudioSession(scene=scene, scene_path=str(target))
        self.session.note(f"Opened scene {target.name}.")
        return LoadResult(
            ok=True,
            message=f"Opened scene {target.name}: {scene.summary()}",
            kind="scene",
            objects=len(scene.objects),
        )

    # -- figures and props ------------------------------------------------

    def figure_preset_names(self) -> tuple[str, ...]:
        return figure_preset_names()

    def add_figure(
        self,
        preset: str = "adult",
        *,
        name: str | None = None,
        translation: Vec3 | None = None,
    ) -> SceneObject:
        """Add a built-in figure to the scene and return it."""
        if preset not in FIGURE_PRESETS:
            raise SceneStudioError(
                f"Unknown figure preset {preset!r}; choose from "
                f"{', '.join(FIGURE_PRESETS)}."
            )
        obj = build_figure_from_preset(preset, name=name)
        if translation is not None:
            obj.transform = Transform(translation, obj.transform.rotation, obj.transform.scale)
        self.scene.add_object(obj)
        assert self.session is not None
        self.session.note(f"Added figure {obj.name} ({preset}).")
        return obj

    def add_custom_figure(self, params: FigureParameters, *, name: str | None = None) -> SceneObject:
        """Add a figure built from explicit parameters (the morph dials)."""
        obj = build_figure(params, name=name)
        self.scene.add_object(obj)
        assert self.session is not None
        self.session.note(f"Added figure {obj.name} ({params.height:.2f} m).")
        return obj

    def morph_object(self, name: str, params: FigureParameters) -> SceneObject:
        """Rebuild a figure from new parameters, keeping its pose and transform."""
        existing = self.scene.find_object(name)
        if not existing.is_figure():
            raise SceneStudioError(f"{name!r} is not a figure and cannot be morphed.")
        rebuilt = morph_figure(existing, params)
        rebuilt.transform = existing.transform
        rebuilt.visible = existing.visible
        self.scene.add_object(rebuilt)
        assert self.session is not None
        self.session.note(f"Morphed figure {name}.")
        return rebuilt

    def prop_names(self) -> tuple[str, ...]:
        return prop_names()

    def add_prop(self, key: str, *, name: str | None = None) -> SceneObject:
        """Add one of the catalog props to the scene."""
        if key not in self.prop_names():
            raise SceneStudioError(
                f"Unknown prop {key!r}; choose from {', '.join(self.prop_names())}."
            )
        obj = build_prop(key, name=name)
        self.scene.add_object(obj)
        assert self.session is not None
        self.session.note(f"Added prop {obj.name}.")
        return obj

    def add_primitive(
        self,
        kind: str,
        *,
        name: str | None = None,
        material: str = "stone",
        translation: Vec3 | None = None,
        scale: Vec3 | None = None,
    ) -> SceneObject:
        """Add a primitive with a chosen material, placed in the scene."""
        from core.scene3d_props import build_primitive

        mesh = build_primitive(kind, name=name or kind.capitalize())
        try:
            surface = material_for(material)
        except KeyError as error:
            raise SceneStudioError(str(error)) from error
        obj = SceneObject(
            name or kind.capitalize(),
            mesh,
            surface,
            Transform(translation or Vec3(), Vec3(), scale or Vec3(1.0, 1.0, 1.0)),
        )
        self.scene.add_object(obj)
        assert self.session is not None
        self.session.note(f"Added {kind} {obj.name}.")
        return obj

    def primitive_names(self) -> tuple[str, ...]:
        return primitive_names()

    def material_names(self) -> tuple[str, ...]:
        return material_names()

    def remove_object(self, name: str) -> bool:
        removed = self.scene.remove_object(name)
        if removed and self.session is not None:
            self.session.note(f"Removed {name}.")
        return removed

    def set_visible(self, name: str, visible: bool) -> SceneObject:
        obj = self.scene.find_object(name)
        obj.visible = visible
        return obj

    def set_material(self, name: str, material: str) -> SceneObject:
        obj = self.scene.find_object(name)
        try:
            obj.material = material_for(material)
        except KeyError as error:
            raise SceneStudioError(str(error)) from error
        return obj

    # -- posing -----------------------------------------------------------

    def bone_names(self, name: str) -> tuple[str, ...]:
        obj = self.scene.find_object(name)
        if obj.skeleton is None:
            return ()
        return obj.skeleton.names()

    def pose_bone(self, object_name: str, bone: str, rotation: Vec3) -> SceneObject:
        """Set one bone's pose rotation on a figure."""
        obj = self.scene.find_object(object_name)
        try:
            obj.set_pose(bone, rotation)
        except ValueError as error:
            raise SceneStudioError(str(error)) from error
        except KeyError as error:
            raise SceneStudioError(f"No bone named {bone!r} on {object_name!r}.") from error
        assert self.session is not None
        self.session.note(f"Posed {object_name}.{bone} = ({rotation.x:.0f}, {rotation.y:.0f}, {rotation.z:.0f}).")
        return obj

    def clear_pose(self, object_name: str) -> SceneObject:
        obj = self.scene.find_object(object_name)
        obj.clear_pose()
        assert self.session is not None
        self.session.note(f"Cleared the pose of {object_name}.")
        return obj

    def set_transform(
        self,
        name: str,
        *,
        translation: Vec3 | None = None,
        rotation: Vec3 | None = None,
        scale: Vec3 | None = None,
    ) -> SceneObject:
        obj = self.scene.find_object(name)
        current = obj.transform
        obj.transform = Transform(
            translation if translation is not None else current.translation,
            rotation if rotation is not None else current.rotation,
            scale if scale is not None else current.scale,
        )
        return obj

    def apply_pose_preset(self, object_name: str, preset: str) -> SceneObject:
        """Apply one of the named pose presets to a figure."""
        obj = self.scene.find_object(object_name)
        if obj.skeleton is None:
            raise SceneStudioError(f"{object_name!r} is not a figure and cannot be posed.")
        try:
            pose = POSE_PRESETS[preset]
        except KeyError as error:
            raise SceneStudioError(
                f"Unknown pose {preset!r}; choose from {', '.join(POSE_PRESETS)}."
            ) from error
        for bone, rotation in pose.items():
            if bone in obj.skeleton.names():
                obj.pose[bone] = rotation
        assert self.session is not None
        self.session.note(f"Applied pose {preset} to {object_name}.")
        return obj

    # -- camera and lights ------------------------------------------------

    def frame_camera(self) -> None:
        """Point the camera at the whole scene."""
        self.scene.frame_camera()
        assert self.session is not None
        self.session.note("Framed the camera on the scene.")

    def set_camera(self, *, position: Vec3 | None = None, target: Vec3 | None = None) -> None:
        camera = self.scene.camera
        self.scene.camera = type(camera)(
            position=position if position is not None else camera.position,
            target=target if target is not None else camera.target,
            up=camera.up,
            fov_degrees=camera.fov_degrees,
            near=camera.near,
            far=camera.far,
        )

    def orbit_camera(self, degrees: float, *, distance_scale: float = 1.0) -> None:
        """Orbit the camera around its target by ``degrees`` about the Y axis."""
        import math

        camera = self.scene.camera
        offset = camera.position - camera.target
        angle = math.radians(degrees)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = Vec3(
            offset.x * cos_a + offset.z * sin_a,
            offset.y,
            -offset.x * sin_a + offset.z * cos_a,
        )
        self.scene.camera = type(camera)(
            position=camera.target + rotated * distance_scale,
            target=camera.target,
            up=camera.up,
            fov_degrees=camera.fov_degrees,
            near=camera.near,
            far=camera.far,
        )
        assert self.session is not None
        self.session.note(f"Orbited the camera {degrees:.0f}°.")

    def set_background(self, color: tuple[float, float, float]) -> None:
        self.scene.background = color

    # -- rendering --------------------------------------------------------

    def render(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        wireframe: bool = False,
        shaded: bool = True,
        path: str | Path | None = None,
    ) -> RenderReport:
        """Render the scene and write a PNG under the renders folder.

        Every render is a real file the user can open; the report carries the
        honest stats from the rasteriser and the path written.
        """
        out_width = int(width or self.scene.width or DEFAULT_WIDTH)
        out_height = int(height or self.scene.height or DEFAULT_HEIGHT)
        if not MIN_DIMENSION <= out_width <= MAX_DIMENSION:
            return RenderReport(
                ok=False,
                message=f"Render width must be between {MIN_DIMENSION} and {MAX_DIMENSION}.",
            )
        if not MIN_DIMENSION <= out_height <= MAX_DIMENSION:
            return RenderReport(
                ok=False,
                message=f"Render height must be between {MIN_DIMENSION} and {MAX_DIMENSION}.",
            )
        if not self.scene.visible_objects():
            return RenderReport(ok=False, message="There is nothing visible to render.")
        try:
            buffer, stats = render_scene(
                self.scene,
                width=out_width,
                height=out_height,
                wireframe=wireframe,
                shaded=shaded,
            )
        except (ValueError, ZeroDivisionError) as error:
            return RenderReport(ok=False, message=f"Could not render the scene: {error}")
        target = Path(path).expanduser() if path else self.renders_dir / self._render_filename()
        target.parent.mkdir(parents=True, exist_ok=True)
        png = buffer.to_png()
        try:
            target.write_bytes(png)
        except OSError as error:
            return RenderReport(ok=False, message=f"Could not write {target}: {error}")
        if self.session is not None:
            self.session.last_render = str(target)
            self.session.note(f"Rendered {out_width}x{out_height} to {target.name}.")
        return RenderReport(
            ok=True,
            message="Render complete.",
            path=str(target),
            width=out_width,
            height=out_height,
            size_bytes=len(png),
            stats=stats,
        )

    def render_to_bytes(
        self, *, width: int | None = None, height: int | None = None
    ) -> bytes:
        """Render the scene and return the PNG bytes (used by the window preview)."""
        out_width = int(width or self.scene.width or DEFAULT_WIDTH)
        out_height = int(height or self.scene.height or DEFAULT_HEIGHT)
        buffer, _ = render_scene(self.scene, width=out_width, height=out_height)
        return buffer.to_png()

    def _render_filename(self) -> str:
        stem = scene_filename(self.scene.name).rsplit(".", 1)[0]
        existing = list(self.renders_dir.glob(f"{stem}_*.png")) if self.renders_dir.is_dir() else []
        return f"{stem}_{len(existing) + 1:03d}.png"

    # -- introspection ----------------------------------------------------

    def object_summaries(self) -> tuple[tuple[str, str, bool], ...]:
        """``(name, kind, visible)`` for each object, for a window's list."""
        result = []
        for obj in self.scene.objects:
            kind = "figure" if obj.is_figure() else "prop"
            result.append((obj.name, kind, obj.visible))
        return tuple(result)

    def status(self) -> str:
        """A short, honest status line for the window and the CLI."""
        if self.session is None:
            return "No scene is open."
        return self.scene.summary()

    def mesh_filetypes(self) -> list[tuple[str, str]]:
        """The file-dialog filters: OBJ models *and* the Studio's own scenes."""
        return [*mesh_filetypes(), ("Scene Studio scene", "*.faydscene"), ("All files", "*.*")]

    # -- commands ---------------------------------------------------------

    def run_command(self, text: str) -> str:
        """Run one plain-language instruction and return a human-readable result.

        This is the *one* interpreter the window's command box and the CLI's
        ``command`` subcommand share, so a typed "add figure athletic" and a
        scripted one do exactly the same thing. An unrecognised word is reported,
        never guessed at.
        """
        command = parse_command(text)
        action = command.action
        try:
            return self._dispatch_command(command)
        except (SceneStudioError, KeyError, ValueError) as error:
            if action == "unknown":
                words = ", ".join(command.args) or "(empty)"
                return f"Did not understand: {words}\n{command_help()}"
            return f"Could not run {action!r}: {error}"

    def _dispatch_command(self, command: SceneCommand) -> str:
        action = command.action
        args = command.args
        if action == "add_figure":
            if args[0] not in self.figure_preset_names():
                return f"Unknown figure preset {args[0]!r}. Choose from {', '.join(self.figure_preset_names())}."
            obj = self.add_figure(args[0])
            return f"Added figure {obj.name}."
        if action == "add_prop":
            if args[0] not in self.prop_names():
                return f"Unknown prop {args[0]!r}. Choose from {', '.join(self.prop_names())}."
            obj = self.add_prop(args[0])
            return f"Added prop {obj.name}."
        if action == "add_primitive":
            obj = self.add_primitive(args[0], name=args[1] if len(args) > 1 else None)
            return f"Added {args[0]} {obj.name}."
        if action == "pose_preset":
            target = self._figure_name()
            if target is None:
                return "There is no figure in the scene to pose."
            self.apply_pose_preset(target, args[0])
            return f"Applied pose {args[0]} to {target}."
        if action == "pose_bone":
            target = self._figure_name()
            if target is None:
                return "There is no figure in the scene to pose."
            numbers = [float(value) for value in args[1:4]]
            while len(numbers) < 3:
                numbers.append(0.0)
            self.pose_bone(target, args[0], Vec3(*numbers[:3]))
            return f"Posed {target}.{args[0]}."
        if action == "clear_pose":
            target = self._figure_name()
            if target is None:
                return "There is no figure in the scene to clear."
            self.clear_pose(target)
            return f"Cleared the pose of {target}."
        if action == "frame":
            self.frame_camera()
            return "Framed the camera on the scene."
        if action == "orbit":
            self.orbit_camera(float(args[0]))
            return f"Orbited the camera {float(args[0]):.0f}°."
        if action == "render":
            numbers = [int(value) for value in args[:2]]
            width = int(numbers[0]) if numbers else None
            height = int(numbers[1]) if len(numbers) > 1 else None
            report = self.render(width=width, height=height)
            return report.summary()
        if action == "remove":
            removed = self.remove_object(args[0])
            return f"Removed {args[0]}." if removed else f"No object named {args[0]!r}."
        if action == "hide":
            self.set_visible(args[0], False)
            return f"Hid {args[0]}."
        if action == "show":
            self.set_visible(args[0], True)
            return f"Showed {args[0]}."
        if action == "save":
            saved = self.save_scene(args[0] if args else None)
            return f"Saved the scene to {saved}."
        if action == "new":
            name = args[0] if args else "Untitled scene"
            self.new_scene(name)
            return f"Started a new scene: {name}."
        if action == "open":
            result = self.open_model(args[0])
            return result.summary()
        words = ", ".join(args) or "(empty)"
        return f"Did not understand: {words}\n{command_help()}"

    def _figure_name(self) -> str | None:
        """The first figure in the scene, or ``None`` when there is none."""
        for obj in self.scene.objects:
            if obj.is_figure():
                return obj.name
        return None


#: Ready-made poses, keyed by name, so a figure can be posed with one call. They
#: are deliberately small rotations of the named bones, and they compose with any
#: further posing the user does.
POSE_PRESETS: dict[str, dict[str, Vec3]] = {
    "rest": {},
    "wave": {
        "shoulder.L": Vec3(0.0, 0.0, -20.0),
        "upper_arm.L": Vec3(0.0, 0.0, -125.0),
        "forearm.L": Vec3(0.0, 0.0, -35.0),
    },
    "walk": {
        "thigh.L": Vec3(-28.0, 0.0, 0.0),
        "shin.L": Vec3(22.0, 0.0, 0.0),
        "thigh.R": Vec3(24.0, 0.0, 0.0),
        "shin.R": Vec3(8.0, 0.0, 0.0),
        "upper_arm.L": Vec3(22.0, 0.0, 0.0),
        "upper_arm.R": Vec3(-22.0, 0.0, 0.0),
    },
    "sit": {
        "thigh.L": Vec3(-85.0, 0.0, 0.0),
        "shin.L": Vec3(85.0, 0.0, 0.0),
        "thigh.R": Vec3(-85.0, 0.0, 0.0),
        "shin.R": Vec3(85.0, 0.0, 0.0),
        "hips": Vec3(0.0, 0.0, 0.0),
    },
    "tpose": {
        "shoulder.L": Vec3(0.0, 0.0, -90.0),
        "shoulder.R": Vec3(0.0, 0.0, 90.0),
        "upper_arm.L": Vec3(0.0, 0.0, -90.0),
        "upper_arm.R": Vec3(0.0, 0.0, 90.0),
    },
    "reach": {
        "shoulder.L": Vec3(0.0, 0.0, -40.0),
        "upper_arm.L": Vec3(-60.0, 0.0, -100.0),
        "forearm.L": Vec3(-25.0, 0.0, 0.0),
        "chest": Vec3(-8.0, 0.0, 0.0),
    },
}


def pose_preset_names() -> tuple[str, ...]:
    return tuple(POSE_PRESETS)


# --------------------------------------------------------------------------- commands


@dataclass(frozen=True)
class SceneCommand:
    """One plain-language instruction, parsed deterministically.

    ``action`` is a stable verb and ``args`` is the rest of the line, so the
    window's command box and the CLI's ``command`` subcommand run the *same*
    parser and do the same thing.
    """

    action: str
    args: tuple[str, ...] = ()
    raw: str = ""

    def describe(self) -> str:
        return self.raw


#: The verbs the parser understands, in the order help should list them.
COMMAND_VERBS: tuple[str, ...] = (
    "add figure",
    "add prop",
    "add cube",
    "add sphere",
    "add cylinder",
    "add cone",
    "add plane",
    "pose",
    "clear pose",
    "frame",
    "orbit",
    "render",
    "remove",
    "hide",
    "show",
    "save",
    "new",
    "open",
)


#: The primitive kinds the command parser accepts after ``add``.
PRIMITIVE_NAMES: tuple[str, ...] = primitive_names()


def parse_command(text: str) -> SceneCommand:
    """Parse one instruction into a :class:`SceneCommand`.

    A deterministic keyword interpreter, not a model: it recognises a small,
    documented vocabulary and returns the words it did not understand as an
    ``unknown`` action, so the caller can report them rather than guess.
    """
    raw = text.strip()
    lowered = " ".join(raw.split())
    if not lowered:
        return SceneCommand("unknown", (), raw)
    words = lowered.split(" ")
    verb = words[0].lower()

    if verb == "add" and len(words) >= 3:
        kind = words[1].lower()
        target = " ".join(words[2:])
        if kind == "figure":
            # The presets are lowercase keys, so the lookup is case-insensitive.
            return SceneCommand("add_figure", (target.lower(),), raw)
        if kind == "prop":
            return SceneCommand("add_prop", (target.lower(),), raw)
        if kind in PRIMITIVE_NAMES:
            # A primitive's trailing word is the object's own name, so its case
            # is the user's and is preserved.
            return SceneCommand("add_primitive", (kind, target), raw)
        return SceneCommand("unknown", (kind,), raw)
    if verb == "pose":
        if len(words) == 2 and words[1].lower() in POSE_PRESETS:
            return SceneCommand("pose_preset", (words[1].lower(),), raw)
        if len(words) >= 3:
            bone = words[1]
            numbers = words[2:]
            return SceneCommand("pose_bone", (bone, *numbers), raw)
        return SceneCommand("unknown", tuple(words[1:]), raw)
    if lowered in {"clear pose", "reset pose"}:
        return SceneCommand("clear_pose", (), raw)
    if verb in {"frame", "focus"}:
        return SceneCommand("frame", (), raw)
    if verb == "orbit" and len(words) >= 2:
        return SceneCommand("orbit", (words[1],), raw)
    if verb == "render":
        return SceneCommand("render", tuple(words[1:]), raw)
    if verb in {"remove", "delete"} and len(words) >= 2:
        return SceneCommand("remove", (" ".join(words[1:]),), raw)
    if verb == "hide" and len(words) >= 2:
        return SceneCommand("hide", (" ".join(words[1:]),), raw)
    if verb == "show" and len(words) >= 2:
        return SceneCommand("show", (" ".join(words[1:]),), raw)
    if verb == "save":
        return SceneCommand("save", tuple(words[1:]), raw)
    if verb in {"new", "scene"}:
        return SceneCommand("new", tuple(words[1:]), raw)
    if verb == "open" and len(words) >= 2:
        return SceneCommand("open", (" ".join(words[1:]),), raw)
    return SceneCommand("unknown", tuple(words), raw)


def command_help() -> str:
    """A short help block listing the commands the parser understands."""
    return (
        "Commands:\n"
        "  add figure <preset>      add a built-in figure (adult, athletic, slim, child)\n"
        "  add prop <key>           add a catalog prop (ground, platform, cube, ...)\n"
        "  add cube|sphere|cylinder|cone|plane [name]\n"
        "  pose <bone> <x> <y> <z>  rotate a bone in degrees\n"
        "  pose <preset>            apply a pose preset (wave, walk, sit, tpose, reach)\n"
        "  clear pose               return the figure to its rest pose\n"
        "  frame                    frame the camera on the whole scene\n"
        "  orbit <degrees>          orbit the camera around its target\n"
        "  render [width height]    render the scene to a PNG\n"
        "  remove|hide|show <name>  manage a scene object\n"
        "  save [name]              save the scene\n"
        "  open <path>              open a model or scene file"
    )


__all__ = [
    "COMMAND_VERBS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "FIGURE_PRESETS",
    "HONESTY_LINE",
    "MAX_DIMENSION",
    "MIN_DIMENSION",
    "POSE_PRESETS",
    "PRIMITIVE_NAMES",
    "LoadResult",
    "RenderReport",
    "SceneCommand",
    "SceneStudioError",
    "SceneStudioService",
    "StudioSession",
    "command_help",
    "parse_command",
    "pose_preset_names",
]
