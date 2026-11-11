"""Import ready-made ``.npz`` weights so the project's trainers continue them.

    python -m cli checkpoints status
    python -m cli checkpoints inspect WEIGHTS.npz
    python -m cli checkpoints install WEIGHTS.npz                # into the trainer's own folder
    python -m cli checkpoints install WEIGHTS.npz --force        # replace, keeping a backup
    python -m cli checkpoints install policy.npz --task drill-well
    python -m cli checkpoints fetch https://…/fayd-llm.npz --approve
    python -m cli checkpoints list                               # what the trainers can resume
    python -m cli checkpoints plan WEIGHTS.npz                   # the command that continues it

The headless twin of the Model Manager window's *Import weights* control. It
drives the *same* :mod:`core.checkpoint_import` engine the window uses, so the
CLI and the window can never disagree about what a file is or where it goes.

Nothing here trains: it reads a file's own structure, installs it into the exact
folder the project's trainer already resumes from, and names the command that
continues it. A file fetched from a URL leaves the machine, so it passes the
permission + human-approval gates (``--approve``).

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` no weight file is
present or the file could not be read, ``4`` a gate refused the import, ``5`` an
unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.checkpoint_import import (
    KIND_BLURBS,
    KINDS,
    CheckpointImportError,
    fetch_weights,
    install_weights,
    list_installed,
    read_weight_info,
    resume_plan,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _status(as_json: bool = False) -> int:
    installed = list_installed()
    if as_json:
        _print_json([item.to_primitive() for item in installed])
        return 0 if installed else 3
    print("Weight files the project's trainers can continue")
    if not installed:
        print("  (none) — import one with `python -m cli checkpoints install WEIGHTS.npz`")
        return 3
    for item in installed:
        print(f"  - {item.describe()}")
    return 0


def _inspect(path: str, as_json: bool = False) -> int:
    info = read_weight_info(path)
    if as_json:
        _print_json(info.to_primitive())
        return 0 if info.valid else 3
    print(info.describe())
    if info.valid:
        plan = resume_plan(info.kind, path=info.path, task=info.task)
        if plan.command:
            print(f"  continue with: {plan.command}")
    return 0 if info.valid else 3


def _install(
    path: str,
    *,
    kind: str = "",
    task: str = "",
    name: str = "",
    destination: str = "",
    force: bool = False,
    no_backup: bool = False,
    as_json: bool = False,
) -> int:
    report = install_weights(
        path,
        kind=kind,
        task=task,
        name=name,
        destination=destination or None,
        force=force,
        backup=not no_backup,
    )
    if as_json:
        _print_json(report.to_primitive())
    else:
        print(report.describe())
    if report.installed:
        return 0
    # A file that is present and in place, or one refused for a real reason, both
    # carry a detail; only a validation failure is "the file could not be read".
    return 3 if not report.info.valid else 4


def _fetch(
    url: str,
    *,
    filename: str = "",
    task: str = "",
    name: str = "",
    force: bool = False,
    approve: bool = False,
    as_json: bool = False,
) -> int:
    if not approve:
        print(
            "Fetching weights leaves this machine, so it needs explicit approval. "
            "Re-run with --approve.",
            file=sys.stderr,
        )
        return 4
    report = fetch_weights(
        url, filename=filename, task=task, name=name, force=force, approved=True
    )
    if as_json:
        _print_json(report.to_primitive())
    else:
        print(report.describe())
    if report.installed:
        return 0
    return 3 if not report.info.valid else 4


def _list(as_json: bool = False) -> int:
    installed = list_installed()
    if as_json:
        _print_json(
            [{"kind": item.kind, "path": str(item.path)} for item in installed]
        )
        return 0
    for item in installed:
        print(item.path)
    return 0


def _kinds(as_json: bool = False) -> int:
    rows = [
        {
            "kind": kind,
            "blurb": KIND_BLURBS.get(kind, ""),
            "command": resume_plan(kind).command,
        }
        for kind in KINDS
    ]
    if as_json:
        _print_json(rows)
        return 0
    print("Weight kinds this build imports (and the trainer that continues each):")
    for row in rows:
        print(f"  {row['kind']:<16} {row['blurb']}")
        if row["command"]:
            print(f"      continue with: {row['command']}")
    return 0


def _plan(path: str, *, task: str = "", as_json: bool = False) -> int:
    info = read_weight_info(path)
    if not info.valid:
        print(info.describe(), file=sys.stderr)
        return 3
    plan = resume_plan(info.kind, path=info.path, task=task or info.task)
    if as_json:
        _print_json(plan.to_primitive())
        return 0
    print(plan.describe() or f"No project trainer continues a {info.kind!r}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkpoints",
        description=(
            "Import ready-made .npz weights so the project's own training tools "
            "continue them (the language model, its state sidecar, or a policy)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="the weight files the trainers can continue")
    status.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    inspect = sub.add_parser("inspect", help="read a .npz file's own structure")
    inspect.add_argument("path", help="the .npz file to read")
    inspect.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    install = sub.add_parser("install", help="install a .npz file into the trainer's folder")
    install.add_argument("path", help="the .npz file to install")
    install.add_argument("--kind", choices=KINDS, help="force the kind (default: detect it)")
    install.add_argument("--task", help="the task a policy belongs to (e.g. drill-well)")
    install.add_argument("--name", help="the file name for a small model")
    install.add_argument("--destination", help="an explicit target path")
    install.add_argument("--force", action="store_true", help="replace an existing file")
    install.add_argument("--no-backup", action="store_true", help="do not keep a backup")
    install.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    fetch = sub.add_parser("fetch", help="fetch a .npz over HTTPS and install it (needs --approve)")
    fetch.add_argument("url", help="the https:// URL of the .npz file")
    fetch.add_argument("--filename", help="the file name to save it as")
    fetch.add_argument("--task", help="the task a policy belongs to")
    fetch.add_argument("--name", help="the file name for a small model")
    fetch.add_argument("--force", action="store_true", help="replace an existing file")
    fetch.add_argument("--approve", action="store_true", help="grant human approval")
    fetch.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    listing = sub.add_parser("list", help="the weight files present on disk")
    listing.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    kinds = sub.add_parser("kinds", help="the weight kinds this build imports")
    kinds.add_argument("--json", dest="as_json", action="store_true", help="print JSON")

    plan = sub.add_parser("plan", help="the command that continues a file's weights")
    plan.add_argument("path", help="the .npz file")
    plan.add_argument("--task", help="the task a policy belongs to")
    plan.add_argument("--json", dest="as_json", action="store_true", help="print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            return _status(getattr(args, "as_json", False))
        if args.command == "inspect":
            return _inspect(args.path, getattr(args, "as_json", False))
        if args.command == "install":
            return _install(
                args.path,
                kind=getattr(args, "kind", "") or "",
                task=getattr(args, "task", "") or "",
                name=getattr(args, "name", "") or "",
                destination=getattr(args, "destination", "") or "",
                force=getattr(args, "force", False),
                no_backup=getattr(args, "no_backup", False),
                as_json=getattr(args, "as_json", False),
            )
        if args.command == "fetch":
            return _fetch(
                args.url,
                filename=getattr(args, "filename", "") or "",
                task=getattr(args, "task", "") or "",
                name=getattr(args, "name", "") or "",
                force=getattr(args, "force", False),
                approve=getattr(args, "approve", False),
                as_json=getattr(args, "as_json", False),
            )
        if args.command == "list":
            return _list(getattr(args, "as_json", False))
        if args.command == "kinds":
            return _kinds(getattr(args, "as_json", False))
        if args.command == "plan":
            return _plan(
                args.path,
                task=getattr(args, "task", "") or "",
                as_json=getattr(args, "as_json", False),
            )
    except CheckpointImportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    parser.error(f"unknown command {args.command!r}")
    return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
