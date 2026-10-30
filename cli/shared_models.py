"""Manage the project's **shared language models** from the command line.

    python -m cli shared-models status
    python -m cli shared-models models
    python -m cli shared-models download --approve              # fetch the default (~1.2 GB)
    python -m cli shared-models download qwen3.5-0.8b --approve
    python -m cli shared-models download-all --approve          # fetch every catalog model
    python -m cli shared-models migrate                          # move a legacy model from Natasha
    python -m cli shared-models list                             # the files present on disk

The headless twin of the window's shared-model controls. It drives the *same*
:mod:`core.shared_models` module the GUI and every character's model layer use,
so the CLI and the window can never disagree about what is installed. A shared
model lives in one public folder (``data/shared_models``) and is offered to
**every** GGUF mouth (Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, Cloud),
so a small general model is downloaded once instead of once per character. It
runs **locally**: nothing is sent off the machine, and the weights download only
through the permission + human-approval gates (``--approve``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` no shared model is
present, ``4`` a download or the migration was refused, ``5`` an unexpected
failure.
"""

from __future__ import annotations

import argparse
import sys

from core.shared_models import (
    DEFAULT_SPEC_KEY,
    SharedModelUnavailable,
    download_all,
    download_model,
    list_models,
    migrate_from_natasha,
    resolve_spec,
    shared_model_status,
    spec_names,
    spec_summaries,
)


def _status(as_json: bool = False) -> int:
    status = shared_model_status()
    if as_json:
        import json

        print(
            json.dumps(
                {
                    "available": status.available,
                    "folder": str(status.folder),
                    "models": list(status.models),
                    "hint": status.hint,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if status.available else 3
    print("Shared language models (every character may run them)")
    print(f"  folder:    {status.folder}")
    print(f"  installed: {'yes' if status.available else 'no'}")
    for model in status.models:
        print(f"    - {model}")
    if not status.available:
        print(f"  next step: {status.hint}")
    return 0 if status.available else 3


def _models(as_json: bool = False) -> int:
    if as_json:
        import json

        print(
            json.dumps(
                [
                    {"key": key, "title": title, "size_gb": size, "summary": summary}
                    for key, title, size, summary in spec_summaries()
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Fetchable shared models (default: {DEFAULT_SPEC_KEY}):")
    for key, title, size, summary in spec_summaries():
        print(f"  {key:16s} {title:34s} ~{size:.1f} GB")
        print(f"  {'':16s} {summary}")
    print("\nFetch one with: python -m cli shared-models download --approve --model <key>")
    return 0


def _list(as_json: bool = False) -> int:
    models = list_models()
    if as_json:
        import json

        print(
            json.dumps(
                [{"file": model.title, "size_gb": model.size_bytes / 1e9} for model in models],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not models:
        print("No shared model file is present yet.")
        print(f"  next step: {shared_model_status().hint}")
        return 3
    for model in models:
        print(f"  {model.describe()}")
    return 0


def _download(spec_key: str | None, approve: bool) -> int:
    if not approve:
        spec = resolve_spec(spec_key)
        print(
            f"error: fetching a model is an external action and needs --approve "
            f"(it fetches ~{spec.approx_size_gb:.1f} GB from the internet)."
        )
        return 4
    try:
        path = download_model(
            spec_key or DEFAULT_SPEC_KEY,
            approved=True,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (SharedModelUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    print(f"Shared model ready: {path}")
    print("Every character can use it now (no per-character download).")
    return 0


def _download_all(approve: bool) -> int:
    if not approve:
        print(
            "error: fetching models is an external action and needs --approve "
            "(it fetches every catalog model, several GB in total)."
        )
        return 4
    try:
        paths = download_all(
            approved=True,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (SharedModelUnavailable, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    for path in paths:
        print(f"  ready: {path}")
    print(f"{len(paths)} shared model(s) ready in {paths[0].parent if paths else ''}.")
    return 0


def _migrate(copy: bool = False, as_json: bool = False) -> int:
    report = migrate_from_natasha(copy=copy)
    if as_json:
        import json

        print(
            json.dumps(
                {
                    "moved": list(report.moved),
                    "skipped": list(report.skipped),
                    "detail": report.detail,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(report.summary())
    for name in report.moved:
        print(f"  moved: {name}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shared-models",
        description="Manage the project's shared language models (data/shared_models).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="report what is installed")
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    models = sub.add_parser("models", help="list the fetchable models")
    models.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    listing = sub.add_parser("list", help="list the model files present on disk")
    listing.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    download = sub.add_parser("download", help="fetch a model (needs --approve)")
    download.add_argument("--model", help=f"which model to fetch, one of: {', '.join(spec_names())}")
    download.add_argument("--approve", action="store_true", help="grant human approval")

    download_all_parser = sub.add_parser(
        "download-all", help="fetch every catalog model (needs --approve)"
    )
    download_all_parser.add_argument("--approve", action="store_true", help="grant human approval")

    migrate = sub.add_parser(
        "migrate", help="move a legacy general model from Natasha's folder to the shared one"
    )
    migrate.add_argument("--copy", action="store_true", help="copy instead of moving")
    migrate.add_argument("--json", dest="as_json", action="store_true", help="print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return _status(getattr(args, "as_json", False))
    if args.command == "models":
        return _models(getattr(args, "as_json", False))
    if args.command == "list":
        return _list(getattr(args, "as_json", False))
    if args.command == "download":
        return _download(args.model, args.approve)
    if args.command == "download-all":
        return _download_all(args.approve)
    if args.command == "migrate":
        return _migrate(getattr(args, "copy", False), getattr(args, "as_json", False))
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
