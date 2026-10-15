"""Command line for the Code Studio (the headless twin of the Studio window).

    python -m cli studio status                 # model + workspaces + base revision
    python -m cli studio new "fix the CLI flag" # make a workspace (isolated worktree)
    python -m cli studio list                   # the workspaces so far
    python -m cli studio files <slug>           # the editable files in a workspace
    python -m cli studio read <slug> <path>     # print one file
    python -m cli studio write <slug> <path> --file local.txt  # write a file
    python -m cli studio ask <slug> "add --json to the report command"
    python -m cli studio diff <slug>            # the change as a unified diff
    python -m cli studio check <slug>           # run the project's checks in the workspace
    python -m cli studio merge <slug>           # merge (refused unless checks passed)
    python -m cli studio discard <slug>         # throw the change away
    python -m cli studio coder                  # is the coding model installed?

It drives the *same* :class:`gui.code_studio.CodeStudioService` the window uses,
so the CLI and the window share one engine: one set of workspaces, one diff, one
merge gate. No git or agent logic is reimplemented here.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the coding model is not
available, ``4`` a workspace/gate refused the operation, ``5`` an unexpected
failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.code_studio import DEFAULT_BASE_BRANCH, StudioError
from gui.code_studio import CodeStudioService

#: The repository the Studio edits. ``None`` means "the project this is running
#: from", which is what a user expects from a checkout.
_DEFAULT_ROOT = None


def _service(args: argparse.Namespace) -> CodeStudioService:
    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd()
    return CodeStudioService(
        root, max_steps=args.max_steps, check_command=_check_command(args.check_command)
    )


def _check_command(text: str | None) -> tuple[str, ...] | None:
    """Turn the ``--check-command`` string into an argv tuple, or use the default."""
    if not text or not text.strip():
        return None
    import shlex

    try:
        parts = tuple(shlex.split(text))
    except ValueError as error:
        raise StudioError(f"Could not parse --check-command: {error}") from error
    if not parts:
        return None
    return parts


def _print_report(report: object, *, as_json: bool, extra: dict | None = None) -> None:
    ok = bool(getattr(report, "ok", False))
    message = str(getattr(report, "message", ""))
    detail = str(getattr(report, "detail", ""))
    if as_json:
        payload: dict[str, object] = {"ok": ok, "message": message}
        if detail:
            payload["detail"] = detail
        if extra:
            payload.update(extra)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(message)
    if detail:
        print(detail)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="The project checkout to branch from (default: the current directory).",
    )
    parser.add_argument(
        "--base-branch",
        default=DEFAULT_BASE_BRANCH,
        help=f"The branch workspaces start from and merge into (default: {DEFAULT_BASE_BRANCH}).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=12,
        help="How many model turns one 'ask' may take (default: 12).",
    )
    parser.add_argument(
        "--check-command", default=None,
        help=(
            "The command 'check' runs inside a workspace, as one shell string "
            "(default: 'python -m pytest -q')."
        ),
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Machine-readable output.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show the model status, workspaces, and base revision.")
    sub.add_parser("coder", help="Show whether the coding model is installed.")

    new = sub.add_parser("new", help="Create an isolated workspace for a change.")
    new.add_argument("task", help="A short description of the change.")
    new.add_argument("--slug", default=None, help="Force the workspace name.")

    sub.add_parser("list", help="List the workspaces.")

    files = sub.add_parser("files", help="List the editable files in a workspace.")
    files.add_argument("slug")

    read = sub.add_parser("read", help="Print one file from a workspace.")
    read.add_argument("slug")
    read.add_argument("path")

    write = sub.add_parser("write", help="Write a file into a workspace.")
    write.add_argument("slug")
    write.add_argument("path")
    write.add_argument("--file", default=None, help="Read the content from this local file (default: stdin).")
    write.add_argument("--content", default=None, help="Write this literal content instead.")

    ask = sub.add_parser("ask", help="Ask the coding model to change the workspace.")
    ask.add_argument("slug")
    ask.add_argument("request", help="What the model should change, in plain language.")

    diff = sub.add_parser("diff", help="Show the workspace's change as a unified diff.")
    diff.add_argument("slug")

    check = sub.add_parser("check", help="Run the project's checks inside a workspace.")
    check.add_argument("slug")

    merge = sub.add_parser("merge", help="Merge a workspace into the base branch.")
    merge.add_argument("slug")

    discard = sub.add_parser("discard", help="Discard a workspace and its branch.")
    discard.add_argument("slug")

    return parser


def _run_status(service: CodeStudioService, args: argparse.Namespace) -> int:
    state = service.state()
    if args.as_json:
        print(
            json.dumps(
                {
                    "repository": state.repository,
                    "base_branch": state.base_branch,
                    "base_revision": state.base_revision,
                    "coder": {
                        "available": state.coder.available if state.coder else False,
                        "model": state.coder.model if state.coder else "",
                        "server_reachable": state.coder.server_reachable if state.coder else False,
                    },
                    "workspaces": [workspace.as_dict() for workspace in state.workspaces],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(state.coder.summary() if state.coder else "Coding model status unavailable.")
    print(f"Checkout : {'git' if state.repository else 'NOT a git checkout'}")
    print(f"Base     : {state.base_branch} at {state.base_revision or 'unknown'}")
    if not state.workspaces:
        print("Workspaces: none yet.")
        return 0
    print(f"Workspaces ({len(state.workspaces)}):")
    for workspace in state.workspaces:
        state_text = "checks passed" if workspace.checks_passed else (
            "checks failed" if workspace.checks_passed is False else "checks not run"
        )
        print(f"  - {workspace.slug}  [{workspace.branch}]  from {workspace.base_revision}  ({state_text})")
        if workspace.task:
            print(f"      {workspace.task}")
    return 0


def _run_coder(service: CodeStudioService, args: argparse.Namespace) -> int:
    status = service.status()
    if args.as_json:
        print(
            json.dumps(
                {
                    "available": status.available,
                    "server_reachable": status.server_reachable,
                    "model": status.model,
                    "installed": status.installed,
                    "models": list(status.models),
                    "hint": status.hint,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(status.summary())
        if status.models:
            print("Pulled models: " + ", ".join(status.models))
        if not status.available:
            print(f"Next step: {status.hint}")
    return 0 if status.available else 3


def _run_new(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.create_workspace(args.task, slug=args.slug)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_list(service: CodeStudioService, args: argparse.Namespace) -> int:
    workspaces = service.state().workspaces
    if args.as_json:
        print(json.dumps([workspace.as_dict() for workspace in workspaces], ensure_ascii=False, indent=2))
        return 0
    if not workspaces:
        print("No workspaces yet. Create one with `studio new \"...\"`.")
        return 0
    for workspace in workspaces:
        print(f"{workspace.slug}  [{workspace.branch}]  from {workspace.base_revision}")
        if workspace.task:
            print(f"    {workspace.task}")
    return 0


def _run_files(service: CodeStudioService, args: argparse.Namespace) -> int:
    try:
        entries = service.list_files(args.slug)
    except StudioError as error:
        print(str(error), file=sys.stderr)
        return 4
    if args.as_json:
        print(json.dumps([entry.path for entry in entries], ensure_ascii=False, indent=2))
        return 0
    for entry in entries:
        print(entry.path)
    return 0


def _run_read(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.read_file(args.slug, args.path)
    if not report.ok:
        print(report.message, file=sys.stderr)
        return 4
    print(report.detail, end="" if report.detail.endswith("\n") else "\n")
    return 0


def _run_write(service: CodeStudioService, args: argparse.Namespace) -> int:
    if args.content is not None:
        content = args.content
    elif args.file is not None:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()
    report = service.save_file(args.slug, args.path, content)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_ask(service: CodeStudioService, args: argparse.Namespace) -> int:
    report, trace = service.run_agent(args.slug, args.request)
    if args.as_json:
        payload: dict[str, object] = {
            "ok": report.ok,
            "message": report.message,
            "summary": trace.summary if trace else "",
            "stopped_reason": trace.stopped_reason if trace else "",
            "writes": list(trace.writes) if trace else [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if report.ok else 3
    if trace is not None:
        print(trace.render())
    else:
        print(report.message, file=sys.stderr)
    return 0 if report.ok else 3


def _run_diff(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.diff(args.slug)
    if not report.ok:
        print(report.message, file=sys.stderr)
        return 4
    print(report.detail)
    return 0


def _run_check(service: CodeStudioService, args: argparse.Namespace) -> int:
    report, _result = service.run_checks(args.slug)
    if args.as_json:
        print(json.dumps({"ok": report.ok, "message": report.message}, ensure_ascii=False, indent=2))
    else:
        print(report.message)
        if report.detail:
            print(report.detail)
    return 0 if report.ok else 4


def _run_merge(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.merge(args.slug)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_discard(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.remove_workspace(args.slug)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


_HANDLERS = {
    "status": _run_status,
    "coder": _run_coder,
    "new": _run_new,
    "list": _run_list,
    "files": _run_files,
    "read": _run_read,
    "write": _run_write,
    "ask": _run_ask,
    "diff": _run_diff,
    "check": _run_check,
    "merge": _run_merge,
    "discard": _run_discard,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    service = _service(args)
    try:
        return _HANDLERS[args.command](service, args)
    except StudioError as error:
        print(str(error), file=sys.stderr)
        return 4
    except OSError as error:
        print(f"Could not complete the operation: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
