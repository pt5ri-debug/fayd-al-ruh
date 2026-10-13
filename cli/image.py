"""CLI for the optional text-to-image feature.

    python -m cli image generate "a red spiral over a dark sea"
    python -m cli image generate "wellhead schematic 768x512" --output shot.png
    python -m cli image generate "the same again" --seed 42
    python -m cli image backends

The default backend renders a PNG with only the standard library and states
plainly that it is a geometric composition, not a learned depiction. Generated
files are written under the writable data directory (``data/images`` by default,
overridable with ``FAYD_DATA_DIR``) unless ``--output`` names a path.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gui.image_gen import ImageGenerationService, ImageReport
from core.imagegen import DEFAULT_HEIGHT, DEFAULT_WIDTH, MAX_DIMENSION


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image", description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for generated files (default: <data>/images).",
    )
    parser.add_argument(
        "--backend",
        default="procedural",
        help="Image backend to use (default: procedural).",
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

    sub.add_parser("backends", help="List the image backends available in this build.")
    return parser


def _service(args: argparse.Namespace) -> ImageGenerationService:
    root = Path(args.output_dir) if args.output_dir else None
    return ImageGenerationService(root=root, backend=args.backend)


def _emit(report: ImageReport, *, as_json: bool) -> int:
    if as_json:
        import json

        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def run_generate(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output=args.output,
    )
    return _emit(report, as_json=args.as_json)


def run_backends(_args: argparse.Namespace) -> int:
    from core.imagegen import load_image_generator

    try:
        generator = load_image_generator("procedural")
    except ValueError as error:  # pragma: no cover - the default always loads
        print(error)
        return 1
    print(f"{generator.name}: available")
    print(
        "  Renders a PNG with only the standard library. It is a deterministic "
        "geometric composition, not a learned text-to-image model."
    )
    print(f"  Max dimension: {MAX_DIMENSION}px.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "generate":
        return run_generate(args)
    if args.command == "backends":
        return run_backends(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
