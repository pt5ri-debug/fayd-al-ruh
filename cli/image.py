"""CLI for the optional text-to-image feature.

    python -m cli image generate "a red spiral over a dark sea"
    python -m cli image generate "wellhead schematic 768x512" --output shot.png
    python -m cli image generate "the same again" --seed 42
    python -m cli image generate "a drilling rig at dusk" --backend diffusion
    python -m cli image generate "a drilling rig at dusk" --backend diffusion --preset realistic
    python -m cli image presets                      # list the quality/speed presets
    python -m cli image download --approve           # one-time, ~4 GB
    python -m cli image backends
    python -m cli image edit photo.png "rotate 90 and brighten, then crop square"
    python -m cli image edit photo.png "make the person smile" --backend learned
    python -m cli image edit-model --approve          # one-time, ~5 GB

The default backend renders a PNG with only the standard library and states
plainly that it is a geometric composition, not a learned depiction. The
optional ``diffusion`` backend is a real learned text-to-image model, tuned for
photorealistic, clean pictures (the ``realistic`` preset is the default); it needs
the ``[image]`` extra and a one-time, human-approved weights download. Generated
files are written under the writable data directory (``data/images`` by default,
overridable with ``FAYD_DATA_DIR``) unless ``--output`` names a path.

The ``edit`` subcommand has two backends: ``deterministic`` (default) changes an
image the user already has with the dependency-free keyword interpreter in
:mod:`core.imageedit` (rotate, crop, brightness…); ``learned`` runs an
InstructPix2Pix model that can make a *semantic* change -- an expression, a pose,
the movement -- from an instruction like "make the person smile". It reads PNG
and binary PPM files. The learned backend needs the ``[image]`` extra and its
weights (``python -m cli image edit-model --approve``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gui.image_edit import ImageEditService
from gui.image_gen import ImageGenerationService
from core.imagegen import DEFAULT_HEIGHT, DEFAULT_WIDTH, MAX_DIMENSION, image_backend_status


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="procedural",
        help="Image backend: procedural (default) or diffusion (learned, optional).",
    )
    parser.add_argument(
        "--model", default=None, help="Diffusion model id (overrides the preset's model)."
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Diffusion preset: fast, realistic (default, photorealistic), or detailed.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps (overrides the preset).")
    parser.add_argument(
        "--guidance", type=float, default=None, help="Diffusion guidance scale (overrides the preset)."
    )
    parser.add_argument(
        "--negative",
        default=None,
        help="Negative prompt: what to steer away from (overrides the preset's default).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image", description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for generated files (default: <data>/images).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Render a prompt to a PNG file.")
    generate.add_argument("prompt", help="The text to draw from (a trailing WxH sets the size).")
    generate.add_argument("--width", type=int, default=None, help=f"Width (default {DEFAULT_WIDTH}).")
    generate.add_argument("--height", type=int, default=None, help=f"Height (default {DEFAULT_HEIGHT}).")
    generate.add_argument("--seed", type=int, default=None, help="Reproduce or vary a composition.")
    generate.add_argument("--output", default=None, help="Exact file to write.")
    generate.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the report as JSON."
    )
    _add_backend_options(generate)

    download = sub.add_parser("download", help="Download the learned model's weights (one-time).")
    download.add_argument(
        "--approve",
        action="store_true",
        help="Grant the external capability and record human approval for the download.",
    )
    download.add_argument(
        "--model", default=None, help="Diffusion model id (default: the realistic preset's model)."
    )
    download.add_argument(
        "--preset",
        default=None,
        help="Download the model for this preset: fast, realistic (default), or detailed.",
    )
    download.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the report as JSON."
    )

    sub.add_parser("backends", help="List the image backends available in this build.")
    sub.add_parser("presets", help="List the learned backend's quality/speed presets.")

    edit = sub.add_parser(
        "edit", help="Edit an existing image (PNG or PPM) from a written instruction."
    )
    edit.add_argument("source", help="The image file to edit.")
    edit.add_argument(
        "instruction",
        help=(
            "What to change, e.g. 'rotate 90 and brighten, then crop to a square' "
            "(deterministic) or 'make the person smile' (learned)."
        ),
    )
    edit.add_argument(
        "--backend",
        default="deterministic",
        help="Edit backend: deterministic (default, keyword pixel ops) or learned (model).",
    )
    edit.add_argument(
        "--preset",
        default=None,
        help="Learned change size: subtle, balanced (default), or strong.",
    )
    edit.add_argument("--seed", type=int, default=None, help="Reproduce or vary a learned edit.")
    edit.add_argument(
        "--negative",
        default=None,
        help="Learned: what to steer away from (overrides the default).",
    )
    edit.add_argument(
        "--output", default=None, help="Exact file to write (default: <data>/images)."
    )
    edit.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the report as JSON."
    )

    edit_model = sub.add_parser(
        "edit-model", help="Download the learned editing model's weights (one-time)."
    )
    edit_model.add_argument(
        "--approve",
        action="store_true",
        help="Grant the external capability and record human approval for the download.",
    )
    edit_model.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the report as JSON."
    )
    return parser


def _service(args: argparse.Namespace) -> ImageGenerationService:
    root = Path(args.output_dir) if args.output_dir else None
    return ImageGenerationService(
        root=root,
        backend=getattr(args, "backend", "procedural"),
        model=getattr(args, "model", None),
        preset=getattr(args, "preset", None),
        steps=getattr(args, "steps", None),
        guidance=getattr(args, "guidance", None),
        negative_prompt=getattr(args, "negative", None),
        on_progress=lambda message: print(message, file=sys.stderr),
    )


def _emit(report: Any, *, as_json: bool, summary: bool = True) -> int:
    if as_json:
        import json

        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    else:
        print(report.summary() if summary else report.message)
    return 0 if report.ok else 1


def run_generate(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt=getattr(args, "negative", None),
        output=args.output,
    )
    return _emit(report, as_json=args.as_json)


def run_download(args: argparse.Namespace) -> int:
    service = _service(args)
    if not args.approve:
        print(
            "error: downloading the model weights is an external action and needs "
            "--approve (it fetches ~4 GB from the internet)."
        )
        return 1
    report = service.download_model(approved=True, model=args.model, preset=args.preset)
    return _emit(report, as_json=args.as_json, summary=False)


def run_backends(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else None
    for name, available, note in image_backend_status(root):
        mark = "available" if available else "unavailable"
        print(f"{name}: {mark}")
        print(f"  {note}")
    print(f"  Max dimension: {MAX_DIMENSION}px.")
    return 0


def run_presets(args: argparse.Namespace) -> int:
    from core.imagegen_diffusion import DEFAULT_PRESET, preset_summaries

    print("Learned backend presets (photorealistic by default):")
    for name, model, summary in preset_summaries():
        mark = " (default)" if name == DEFAULT_PRESET else ""
        print(f"  {name}{mark}")
        print(f"    model: {model}")
        print(f"    {summary}")
    print(
        "Usage: python -m cli image generate \"your prompt\" --backend diffusion "
        "--preset realistic"
    )
    return 0


def run_edit(args: argparse.Namespace) -> int:
    root = Path(args.output_dir) if args.output_dir else None
    service = ImageEditService(
        root=root,
        backend=args.backend,
        preset=getattr(args, "preset", None),
        on_progress=lambda message: print(message, file=sys.stderr),
    )
    try:
        session = service.load(args.source)
    except Exception as error:  # noqa: BLE001 - reported, never a traceback
        print(f"error: could not open {args.source}: {error}")
        return 1
    report = service.apply(
        session,
        args.instruction,
        output=args.output,
        seed=getattr(args, "seed", None),
        negative_prompt=getattr(args, "negative", None),
    )
    if args.as_json:
        import json

        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def run_edit_model(args: argparse.Namespace) -> int:
    if not args.approve:
        print(
            "error: downloading the editing model weights is an external action and "
            "needs --approve (it fetches ~4 GB from the internet)."
        )
        return 1
    root = Path(args.output_dir) if args.output_dir else None
    service = ImageEditService(
        root=root,
        backend="learned",
        on_progress=lambda message: print(message, file=sys.stderr),
    )
    report = service.download_model(approved=True)
    return _emit(report, as_json=args.as_json, summary=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "generate":
        return run_generate(args)
    if args.command == "download":
        return run_download(args)
    if args.command == "backends":
        return run_backends(args)
    if args.command == "presets":
        return run_presets(args)
    if args.command == "edit":
        return run_edit(args)
    if args.command == "edit-model":
        return run_edit_model(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
