"""CLI for the Scene Studio: build, pose, and render a 3D figure locally.

    python -m cli scene3d status                 # what the studio can do
    python -m cli scene3d figures                # the built-in figure presets
    python -m cli scene3d props                  # the primitive prop catalog
    python -m cli scene3d poses                  # the pose presets
    python -m cli scene3d new --name Demo        # start a scene
    python -m cli scene3d add-figure athletic    # add a figure
    python -m cli scene3d add-prop ground        # add a prop
    python -m cli scene3d pose Adult forearm.L 0 0 -90
    python -m cli scene3d pose-preset wave
    python -m cli scene3d frame                  # frame the camera
    python -m cli scene3d render --width 900 --height 640
    python -m cli scene3d save                   # save the native scene
    python -m cli scene3d open model.obj         # load an OBJ model
    python -m cli scene3d export-obj out.obj     # export geometry
    python -m cli scene3d command "add figure slim"
    python -m cli scene3d list                   # saved scenes

This is the headless twin of the window (``＋ → Scene Studio``): it drives the
*same* :class:`gui.scene3d.SceneStudioService`, so the CLI and the window share
one scene, one renderer, and one command interpreter, and no workflow logic is
reimplemented here.

Everything runs on this machine: no network, no asset store, no model download.
The renderer is a CPU software rasteriser and the CLI says so.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` nothing to do (no
figure to pose, or an empty/refused render), ``4`` a file could not be read or
written, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from core.scene3d import Vec3
from core.scene3d_figures import FIGURE_PRESETS
from core.scene3d_io import SceneIOError
from core.scene3d_props import PROP_CATALOG, material_names, primitive_names
from gui.scene3d import (
    POSE_PRESETS,
    SceneStudioError,
    SceneStudioService,
    command_help,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_EMPTY = 3
EXIT_FILE = 4
EXIT_FAILURE = 5

__all__ = [
    "EXIT_EMPTY",
    "EXIT_FAILURE",
    "EXIT_FILE",
    "EXIT_OK",
    "EXIT_USAGE",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scene3d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("status", "Report what the studio can do."),
        ("figures", "List the built-in figure presets."),
        ("props", "List the primitive prop catalog."),
        ("materials", "List the material palette."),
        ("primitives", "List the primitive kinds."),
        ("poses", "List the pose presets."),
        ("list", "List the scenes saved by the studio."),
        ("help", "Print the command-box vocabulary."),
    ):
        entry = sub.add_parser(name, help=help_text)
        entry.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    new = sub.add_parser("new", help="Start a new scene.")
    new.add_argument("--name", default="Untitled scene", help="The scene name.")
    new.add_argument("--no-ground", action="store_true", help="Do not add a ground plane.")

    add_figure = sub.add_parser("add-figure", help="Add a built-in figure.")
    add_figure.add_argument("preset", nargs="?", default="adult", help="A figure preset key.")
    add_figure.add_argument("--name", default=None, help="The figure's name.")
    add_figure.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    add_prop = sub.add_parser("add-prop", help="Add a catalog prop.")
    add_prop.add_argument("key", help="A prop key.")
    add_prop.add_argument("--name", default=None, help="The prop's name.")

    add_primitive = sub.add_parser("add-primitive", help="Add a primitive with a material.")
    add_primitive.add_argument("kind", help="cube, sphere, cylinder, cone, or plane.")
    add_primitive.add_argument("--name", default=None, help="The object's name.")
    add_primitive.add_argument("--material", default="stone", help="A palette material key.")

    pose = sub.add_parser("pose", help="Rotate one bone of a figure (degrees).")
    pose.add_argument("object", help="The figure's name.")
    pose.add_argument("bone", help="The bone's name.")
    pose.add_argument("x", type=float)
    pose.add_argument("y", type=float)
    pose.add_argument("z", type=float)

    pose_preset = sub.add_parser("pose-preset", help="Apply a named pose preset.")
    pose_preset.add_argument("preset", help="A pose preset key.")
    pose_preset.add_argument("--object", default=None, help="The figure's name.")

    sub.add_parser("clear-pose", help="Return a figure to its rest pose.").add_argument(
        "--object", default=None, help="The figure's name."
    )
    sub.add_parser("frame", help="Frame the camera on the scene.")

    orbit = sub.add_parser("orbit", help="Orbit the camera around its target.")
    orbit.add_argument("degrees", type=float)

    render = sub.add_parser("render", help="Render the scene to a PNG.")
    render.add_argument("--width", type=int, default=None)
    render.add_argument("--height", type=int, default=None)
    render.add_argument("--wireframe", action="store_true", help="Overlay the wireframe.")
    render.add_argument("--flat", action="store_true", help="Skip shading (flat colour).")
    render.add_argument("--output", default=None, help="Where to write the PNG.")
    render.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    demo = sub.add_parser(
        "demo",
        help="Build a complete scene (a posed figure on a ground plane) and render it.",
    )
    demo.add_argument("--figure", default="adult", help="The figure preset to use.")
    demo.add_argument("--pose", default="wave", help="The pose preset to apply.")
    demo.add_argument("--width", type=int, default=900)
    demo.add_argument("--height", type=int, default=640)
    demo.add_argument("--output", default=None, help="Where to write the PNG.")
    demo.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    save = sub.add_parser("save", help="Save the scene natively (rig and pose included).")
    save.add_argument("path", nargs="?", default=None)

    export = sub.add_parser("export-obj", help="Export the scene as Wavefront OBJ.")
    export.add_argument("path", help="Where to write the OBJ.")

    open_command = sub.add_parser("open", help="Load an OBJ model or a native scene.")
    open_command.add_argument("path", help="The file to open.")

    command = sub.add_parser("command", help="Run one command-box instruction.")
    command.add_argument("text", help="The instruction, e.g. \"add figure athletic\".")
    return parser


def _service() -> SceneStudioService:
    """The one service the CLI drives (a test may replace this).

    The CLI is a fresh process per command, so it *continues the working scene*:
    the scene the last command saved is loaded, the command runs against it, and
    the result is saved back. That is what lets ``add-figure`` then ``pose-preset``
    then ``render`` build on one another, exactly as they do in the window.
    """
    service = SceneStudioService()
    service.load_working()
    return service


def _print_json(payload: object) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=False, indent=2))


#: Commands that do not touch the scene, so the working scene is neither loaded
#: nor written back for them.
_READ_ONLY_COMMANDS = frozenset(
    {"status", "figures", "props", "materials", "primitives", "poses", "list", "help"}
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    command = args.command
    service = _service()

    try:
        code = _dispatch(args, command, service)
    except SceneIOError as error:
        print(f"File error: {error}", file=sys.stderr)
        return EXIT_FILE
    except SceneStudioError as error:
        print(f"Scene error: {error}", file=sys.stderr)
        return EXIT_EMPTY
    except (KeyError, ValueError) as error:
        print(f"Invalid request: {error}", file=sys.stderr)
        return EXIT_USAGE
    # Persist the working scene so the next command continues from here.
    if command not in _READ_ONLY_COMMANDS and service.session is not None and code == EXIT_OK:
        try:
            service.save_working()
        except (SceneIOError, SceneStudioError):
            pass
    return code


def _dispatch(args: argparse.Namespace, command: str, service: SceneStudioService) -> int:
    if command == "status":
        return _status(args, service)
    if command == "figures":
        if args.json:
            _print_json({key: params.height for key, params in FIGURE_PRESETS.items()})
            return EXIT_OK
        print("Built-in figures (generated locally, no download):")
        for key, params in FIGURE_PRESETS.items():
            print(f"  {key:<10} {params.height:.2f} m, shoulders {params.shoulder_width:.2f} m")
        return EXIT_OK
    if command == "props":
        if args.json:
            _print_json([spec.key for spec in PROP_CATALOG])
            return EXIT_OK
        print("Props:")
        for spec in PROP_CATALOG:
            print(f"  {spec.key:<10} {spec.summary}")
        return EXIT_OK
    if command == "materials":
        if args.json:
            _print_json(list(material_names()))
            return EXIT_OK
        print("Materials: " + ", ".join(material_names()))
        return EXIT_OK
    if command == "primitives":
        if args.json:
            _print_json(list(primitive_names()))
            return EXIT_OK
        print("Primitives: " + ", ".join(primitive_names()))
        return EXIT_OK
    if command == "poses":
        if args.json:
            _print_json(list(POSE_PRESETS))
            return EXIT_OK
        print("Pose presets: " + ", ".join(POSE_PRESETS))
        return EXIT_OK
    if command == "help":
        print(command_help())
        return EXIT_OK
    if command == "list":
        scenes = service.list_saved_scenes()
        if args.json:
            _print_json([str(path) for path in scenes])
            return EXIT_OK
        if not scenes:
            print(f"No saved scenes in {service.scenes_dir}.")
            return EXIT_OK
        print(f"Saved scenes in {service.scenes_dir}:")
        for path in scenes:
            print(f"  {path.name}")
        return EXIT_OK

    if command == "new":
        service.new_scene(args.name, with_ground=not args.no_ground)
        print(f"Started a new scene: {args.name}")
        return EXIT_OK
    if command == "add-figure":
        obj = service.add_figure(args.preset, name=args.name)
        if args.json:
            _print_json(
                {
                    "name": obj.name,
                    "preset": args.preset,
                    "vertices": obj.mesh.vertex_count,
                    "triangles": obj.mesh.triangle_count,
                    "bones": list(service.bone_names(obj.name)),
                }
            )
        else:
            print(
                f"Added figure {obj.name} ({obj.mesh.vertex_count} vertices, "
                f"{obj.mesh.triangle_count} triangles)."
            )
        return EXIT_OK
    if command == "add-prop":
        obj = service.add_prop(args.key, name=args.name)
        print(f"Added prop {obj.name}.")
        return EXIT_OK
    if command == "add-primitive":
        obj = service.add_primitive(args.kind, name=args.name, material=args.material)
        print(f"Added {args.kind} {obj.name}.")
        return EXIT_OK
    if command == "pose":
        service.pose_bone(args.object, args.bone, Vec3(args.x, args.y, args.z))
        print(f"Posed {args.object}.{args.bone} = ({args.x:.0f}, {args.y:.0f}, {args.z:.0f}).")
        return EXIT_OK
    if command == "pose-preset":
        target = args.object or _only_figure(service)
        service.apply_pose_preset(target, args.preset)
        print(f"Applied pose {args.preset} to {target}.")
        return EXIT_OK
    if command == "clear-pose":
        target = args.object or _only_figure(service)
        service.clear_pose(target)
        print(f"Cleared the pose of {target}.")
        return EXIT_OK
    if command == "frame":
        service.frame_camera()
        print("Framed the camera on the scene.")
        return EXIT_OK
    if command == "orbit":
        service.orbit_camera(args.degrees)
        print(f"Orbited the camera {args.degrees:.0f}°.")
        return EXIT_OK
    if command == "render":
        report = service.render(
            width=args.width,
            height=args.height,
            wireframe=args.wireframe,
            shaded=not args.flat,
            path=args.output,
        )
        if args.json:
            _print_json(
                {
                    "ok": report.ok,
                    "path": report.path,
                    "width": report.width,
                    "height": report.height,
                    "size_bytes": report.size_bytes,
                    "message": report.message,
                }
            )
        else:
            print(report.summary())
        return EXIT_OK if report.ok else EXIT_EMPTY
    if command == "demo":
        service.new_scene("Demo scene")
        obj = service.add_figure(args.figure)
        if args.pose in POSE_PRESETS:
            service.apply_pose_preset(obj.name, args.pose)
        service.frame_camera()
        report = service.render(
            width=args.width, height=args.height, path=args.output
        )
        if args.json:
            _print_json({"ok": report.ok, "path": report.path, "message": report.message})
        else:
            print(f"Built a {args.figure} figure with the {args.pose} pose.")
            print(report.summary())
        return EXIT_OK if report.ok else EXIT_EMPTY
    if command == "save":
        saved = service.save_scene(args.path)
        print(f"Saved the scene to {saved}")
        return EXIT_OK
    if command == "export-obj":
        saved = service.export_obj(args.path)
        print(f"Exported the scene to {saved}")
        return EXIT_OK
    if command == "open":
        result = service.open_model(args.path)
        print(result.summary())
        return EXIT_OK if result.ok else EXIT_FILE
    if command == "command":
        print(service.run_command(args.text))
        return EXIT_OK
    return EXIT_USAGE


def _status(args: argparse.Namespace, service: SceneStudioService) -> int:
    from gui.scene3d import HONESTY_LINE

    scenes = service.list_saved_scenes()
    if args.json:
        _print_json(
            {
                "figures": list(service.figure_preset_names()),
                "props": list(service.prop_names()),
                "poses": list(POSE_PRESETS),
                "materials": list(material_names()),
                "saved_scenes": len(scenes),
                "scenes_dir": str(service.scenes_dir),
                "renders_dir": str(service.renders_dir),
                "honesty": HONESTY_LINE,
            }
        )
        return EXIT_OK
    print("Scene Studio (local 3D figure studio)")
    print(f"  figures   : {', '.join(service.figure_preset_names())}")
    print(f"  props     : {', '.join(service.prop_names())}")
    print(f"  poses     : {', '.join(POSE_PRESETS)}")
    print(f"  scenes    : {service.scenes_dir} ({len(scenes)} saved)")
    print(f"  renders   : {service.renders_dir}")
    print(f"  {HONESTY_LINE}")
    return EXIT_OK


def _only_figure(service: SceneStudioService) -> str:
    """The name of the scene's only figure, or raise a clear error."""
    figures = [obj.name for obj in service.scene.objects if obj.is_figure()]
    if not figures:
        raise SceneStudioError("There is no figure in the scene; add one first.")
    return figures[0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
