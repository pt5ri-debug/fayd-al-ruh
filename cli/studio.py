"""Command line for the Code Studio (the headless twin of the Studio window).

    python -m cli studio status                 # model + workspaces + base revision
    python -m cli studio new ["fix the CLI flag"]  # make a workspace (named PR1, PR2, ...)
    python -m cli studio list                   # the workspaces so far
    python -m cli studio files <slug>           # the editable files in a workspace
    python -m cli studio tree <slug>            # the workspace's file tree
    python -m cli studio read <slug> <path>     # print one file
    python -m cli studio write <slug> <path> --file local.txt  # write a file
    python -m cli studio ask <slug> "add --json to the report command"
    python -m cli studio diff <slug>            # the change as a unified diff
    python -m cli studio check <slug>           # run the project's checks in the workspace
    python -m cli studio merge <slug>           # merge (refused unless checks passed)
    python -m cli studio discard <slug>         # throw the change away
    python -m cli studio coder                  # is the coding model installed?

The rest of a developer's day is here too, all inside the workspace:

    python -m cli studio search <slug> "text" [--path core]      # find a literal string
    python -m cli studio replace <slug> --old X --new Y [--apply] # project-wide literal replace
    python -m cli studio outline <slug> <path>                    # a file's symbols
    python -m cli studio notes <slug>                            # TODO/FIXME scan
    python -m cli studio mkdir <slug> <path> | touch <slug> <path>
    python -m cli studio rename <slug> <old> <new> | cp <slug> <old> <new>
    python -m cli studio rm <slug> <path> [--dir]
    python -m cli studio git <slug> status|log|branches|stash|conflicts|remotes
    python -m cli studio git <slug> stage|unstage|commit|restore|stash-pop
    python -m cli studio git <slug> show|blame
    python -m cli studio git <slug> fetch|pull|push --approve     # the network is gated
    python -m cli studio tasks <slug>            # the project's own development commands
    python -m cli studio run <slug> <task> [--command "..."]      # run one, streaming output

It drives the *same* :class:`gui.code_studio.CodeStudioService` the window uses,
so the CLI and the window share one engine: one set of workspaces, one diff, one
merge gate, one task catalog. No git or agent logic is reimplemented here.

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
from typing import Any

from core.code_studio import DEFAULT_BASE_BRANCH, StudioError
from core.studio_files import TreeEntry, iter_tree
from core.studio_tasks import TaskError
from gui.code_studio import CodeStudioService, StudioReport

#: The repository the Studio edits. ``None`` means "the project this is running
#: from", which is what a user expects from a checkout.
_DEFAULT_ROOT = None


def _service(args: argparse.Namespace) -> CodeStudioService:
    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd()
    # The base branch is the branch a workspace is cut from and merged into, so
    # the CLI's --base-branch must reach the engine rather than be accepted and
    # ignored.
    from core.code_studio import CodeStudio

    studio = CodeStudio(root, base_branch=args.base_branch)
    return CodeStudioService(
        root,
        studio=studio,
        max_steps=args.max_steps,
        check_command=_check_command(args.check_command),
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


def _common_flags(parser: argparse.ArgumentParser) -> None:
    """Let the shared flags appear before *or* after the subcommand.

    ``argparse`` only accepts a main-parser option before the subcommand, so
    ``studio list --json`` used to fail. Each sub-parser re-declares the shared
    flags with ``SUPPRESS`` defaults, so a value given after the subcommand wins
    and a value given before it is not overwritten by the sub-parser's default --
    the same "before or after" rule the ingestion CLI uses.
    """
    parser.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--base-branch", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--check-command", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--json", action="store_true", dest="as_json", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def _sub(
    parser: argparse._SubParsersAction, name: str, help: str
) -> argparse.ArgumentParser:
    """A sub-parser that also accepts the shared flags (see :func:`_common_flags`)."""
    child = parser.add_parser(name, help=help)
    _common_flags(child)
    return child


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

    _sub(sub, "status", help="Show the model status, workspaces, and base revision.")
    _sub(sub, "coder", help="Show whether the coding model is installed.")
    _sub(sub, "backend", help="Show which coding runtime is ready (in-process or Ollama).")

    models = _sub(sub, "models", help="List the coding models that can be downloaded.")
    models.set_defaults(_models=True)

    download = _sub(sub, "download", help="Download a coding model for the in-process runtime.")
    download.add_argument("--spec", default="", help="Which model to fetch (default: qwen2.5-coder-7b).")
    download.add_argument("--approve", action="store_true", help="Approve the download (it leaves the machine).")

    new = _sub(sub, "new", help="Create an isolated workspace for a change (named for you).")
    new.add_argument("task", nargs="?", default="", help="A short description of the change (optional).")
    new.add_argument("--slug", default=None, help="Force the workspace name (default: PR1, PR2, ...).")

    _sub(sub, "list", help="List the workspaces.")

    files = _sub(sub, "files", help="List the editable files in a workspace.")
    files.add_argument("slug")

    read = _sub(sub, "read", help="Print one file from a workspace.")
    read.add_argument("slug")
    read.add_argument("path")

    write = _sub(sub, "write", help="Write a file into a workspace.")
    write.add_argument("slug")
    write.add_argument("path")
    write.add_argument("--file", default=None, help="Read the content from this local file (default: stdin).")
    write.add_argument("--content", default=None, help="Write this literal content instead.")

    edit = _sub(sub, "edit", help="Replace one exact snippet in a workspace file.")
    edit.add_argument("slug")
    edit.add_argument("path")
    edit.add_argument("--old", required=True, help="The exact text to replace.")
    edit.add_argument("--new", required=True, help="The replacement text.")
    edit.add_argument("--all", action="store_true", help="Replace every occurrence (default: require a unique match).")

    search = _sub(sub, "search", help="Find a literal string in a workspace.")
    search.add_argument("slug")
    search.add_argument("query")
    search.add_argument("--path", default="", help="Limit the search to one file or subfolder.")

    tree = _sub(sub, "tree", help="Show the workspace's file tree.")
    tree.add_argument("slug")
    tree.add_argument("--path", default="", help="Show only this subfolder.")

    replace = _sub(sub, "replace", help="Replace a literal string across the workspace.")
    replace.add_argument("slug")
    replace.add_argument("--old", required=True, help="The literal text to replace.")
    replace.add_argument("--new", required=True, help="The replacement text.")
    replace.add_argument("--path", default="", help="Limit to one file or subfolder.")
    replace.add_argument("--apply", action="store_true", help="Write the change (default: preview only).")

    outline = _sub(sub, "outline", help="Show a file's definitions (classes, functions, ...).")
    outline.add_argument("slug")
    outline.add_argument("path")

    notes = _sub(sub, "notes", help="Scan the workspace for TODO/FIXME/BUG notes.")
    notes.add_argument("slug")
    notes.add_argument("--path", default="", help="Limit the scan to one file or subfolder.")

    touch = _sub(sub, "touch", help="Create a new empty text file.")
    touch.add_argument("slug")
    touch.add_argument("path")
    touch.add_argument("--content", default="", help="Initial content.")

    mkdir = _sub(sub, "mkdir", help="Create a folder inside the workspace.")
    mkdir.add_argument("slug")
    mkdir.add_argument("path")

    rename = _sub(sub, "rename", help="Rename (or move) a file.")
    rename.add_argument("slug")
    rename.add_argument("old")
    rename.add_argument("new")

    cp = _sub(sub, "cp", help="Copy a file to a new path.")
    cp.add_argument("slug")
    cp.add_argument("old")
    cp.add_argument("new")

    rm = _sub(sub, "rm", help="Delete a file (or, with --dir, a folder).")
    rm.add_argument("slug")
    rm.add_argument("path")
    rm.add_argument("--dir", action="store_true", help="Delete a folder and everything under it.")

    git = _sub(sub, "git", help="Git actions: status/log/branches/commit/stash/fetch/...")
    git.add_argument("slug")
    git.add_argument(
        "action",
        choices=[
            "status", "log", "branches", "stash", "conflicts", "remotes",
            "stage", "unstage", "commit", "restore", "stash-save", "stash-pop",
            "show", "blame", "fetch", "pull", "push",
        ],
        help="Which git action to run.",
    )
    git.add_argument("--path", default="", help="A file path (for show/blame).")
    git.add_argument("--paths", nargs="*", default=[], help="Paths to stage/unstage/restore.")
    git.add_argument("--message", default="", help="A commit/stash message.")
    git.add_argument("--untracked", action="store_true", help="Include untracked files in the stash.")
    git.add_argument("--ref", default="HEAD", help="The revision for show (default: HEAD).")
    git.add_argument("--remote", default="origin", help="The remote for fetch/pull/push.")
    git.add_argument("--approve", action="store_true", help="Approve a network action (fetch/pull/push).")

    tasks = _sub(sub, "tasks", help="List the project's own development commands.")
    tasks.add_argument("slug", nargs="?", default="", help="A workspace (for its location).")

    run = _sub(sub, "run", help="Run one of the project's tasks inside a workspace.")
    run.add_argument("slug")
    run.add_argument("task", help="A task key from `studio tasks` (e.g. tests, lint), or 'custom'.")
    # ``dest`` is explicit: the sub-parser's own name is stored in ``args.command``,
    # so a bare ``--command`` would overwrite it and break the dispatch.
    run.add_argument(
        "--command", dest="custom_command", default="",
        help="The command line when task is 'custom'.",
    )

    ask = _sub(sub, "ask", help="Ask the coding model to change the workspace.")
    ask.add_argument("slug")
    ask.add_argument("request", help="What the model should change, in plain language.")

    diff = _sub(sub, "diff", help="Show the workspace's change as a unified diff.")
    diff.add_argument("slug")

    check = _sub(sub, "check", help="Run the project's checks inside a workspace.")
    check.add_argument("slug")

    merge = _sub(sub, "merge", help="Merge a workspace into the base branch.")
    merge.add_argument("slug")

    discard = _sub(sub, "discard", help="Discard a workspace and its branch.")
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


def _run_edit(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.edit_file(args.slug, args.path, args.old, args.new, replace_all=args.all)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_search(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.search(args.slug, args.query, relative=args.path)
    if args.as_json:
        payload = {"ok": report.ok, "message": report.message, "detail": report.detail}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if report.ok else 4
    print(report.detail or report.message)
    return 0 if report.ok else 4


def _run_tree(service: CodeStudioService, args: argparse.Namespace) -> int:
    entries = service.file_tree(args.slug, relative=args.path)
    if args.as_json:
        print(json.dumps(_tree_payload(entries), ensure_ascii=False, indent=2))
        return 0
    lines = list(iter_tree(entries))
    print("\n".join(lines) if lines else "(no files)")
    return 0


def _tree_payload(entries: Sequence[TreeEntry]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for entry in entries:
        out.append(
            {
                "path": entry.path,
                "name": entry.name,
                "directory": entry.directory,
                "size": entry.size,
                "children": _tree_payload(entry.children),
            }
        )
    return out


def _run_replace(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.replace_across(args.slug, args.old, args.new, relative=args.path, apply=args.apply)
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "message": report.message,
                    "applied": report.applied,
                    "changes": [
                        {"path": c.path, "action": c.action, "detail": c.detail} for c in report.changes
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report.ok else 4
    print(report.message)
    for line in report.summary_lines():
        print(f"  {line}")
    return 0 if report.ok else 4


def _run_outline(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.outline(args.slug, args.path)
    if args.as_json:
        print(
            json.dumps(
                {
                    "path": report.path,
                    "language": report.language,
                    "error": report.error,
                    "symbols": [
                        {"name": s.name, "kind": s.kind, "line": s.line, "detail": s.detail}
                        for s in report.symbols
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not report.error else 4
    print(report.describe())
    return 0 if not report.error else 4


def _run_notes(service: CodeStudioService, args: argparse.Namespace) -> int:
    scan = service.scan_notes(args.slug, relative=args.path)
    if args.as_json:
        print(
            json.dumps(
                {
                    "counts": scan.counts,
                    "notes": [
                        {"path": n.path, "line": n.line, "marker": n.marker, "text": n.text}
                        for n in scan.notes
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(scan.describe())
    return 0


def _run_touch(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.create_file(args.slug, args.path, args.content)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_mkdir(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.create_directory(args.slug, args.path)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_rename(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.rename_file(args.slug, args.old, args.new)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_cp(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.copy_file(args.slug, args.old, args.new)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _run_rm(service: CodeStudioService, args: argparse.Namespace) -> int:
    report = service.delete_file(args.slug, args.path, directory=args.dir)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


_NETWORK_ACTIONS = frozenset({"fetch", "pull", "push"})


def _run_git(service: CodeStudioService, args: argparse.Namespace) -> int:
    """The one git command: read actions never touch the network; the three that
    do (fetch/pull/push) require ``--approve``, so a network call is explicit."""
    slug, action = args.slug, args.action
    if action in _NETWORK_ACTIONS and not args.approve:
        print(
            f"`git {action}` contacts the network, so it needs --approve.",
            file=sys.stderr,
        )
        return 2
    if action == "status":
        status_entries = service.git_status(slug)
        if args.as_json:
            print(json.dumps([_status_dict(e) for e in status_entries], ensure_ascii=False, indent=2))
            return 0
        print("\n".join(e.describe() for e in status_entries) or "(clean)")
        return 0
    if action == "log":
        commits = service.git_log(slug)
        if args.as_json:
            print(json.dumps([_commit_dict(c) for c in commits], ensure_ascii=False, indent=2))
            return 0
        print("\n".join(c.describe() for c in commits) or "(no commits)")
        return 0
    if action == "branches":
        branches = service.git_branches(slug)
        if args.as_json:
            print(json.dumps([_branch_dict(b) for b in branches], ensure_ascii=False, indent=2))
            return 0
        print("\n".join(b.describe() for b in branches) or "(no branches)")
        return 0
    if action == "stash":
        stashes = service.git_stash_list(slug)
        if args.as_json:
            print(json.dumps([{"index": s.index, "ref": s.ref, "subject": s.subject} for s in stashes], ensure_ascii=False, indent=2))
            return 0
        print("\n".join(s.describe() for s in stashes) or "(no stashes)")
        return 0
    if action == "conflicts":
        conflict_report = service.git_conflicts(slug)
        if args.as_json:
            print(json.dumps({"files": list(conflict_report.files), "operation": conflict_report.operation}, ensure_ascii=False, indent=2))
            return 0
        print(conflict_report.describe())
        return 0
    if action == "remotes":
        remotes = service.git_remotes(slug)
        if args.as_json:
            print(json.dumps([{"name": r.name, "url": r.url} for r in remotes], ensure_ascii=False, indent=2))
            return 0
        print("\n".join(r.describe() for r in remotes) or "(no remotes)")
        return 0
    report: StudioReport
    if action == "stage":
        report = service.stage(slug, *args.paths)
    elif action == "unstage":
        report = service.unstage(slug, *args.paths)
    elif action == "restore":
        report = service.discard_changes(slug, *args.paths)
    elif action == "commit":
        report = service.commit(slug, args.message)
    elif action == "stash-save":
        report = service.stash(slug, args.message, include_untracked=args.untracked)
    elif action == "stash-pop":
        report = service.stash_pop(slug)
    elif action == "show":
        report = service.git_show(slug, args.ref)
    elif action == "blame":
        if not args.path:
            print("blame needs --path.", file=sys.stderr)
            return 2
        report = service.git_blame(slug, args.path)
    else:  # fetch/pull/push, approved above
        report = service.remote_action(slug, action, remote=args.remote)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 4


def _status_dict(entry: object) -> dict[str, object]:
    return {
        "path": entry.path,  # type: ignore[attr-defined]
        "code": entry.code,  # type: ignore[attr-defined]
        "staged": entry.staged,  # type: ignore[attr-defined]
        "unstaged": entry.unstaged,  # type: ignore[attr-defined]
        "untracked": entry.untracked,  # type: ignore[attr-defined]
        "conflicted": entry.conflicted,  # type: ignore[attr-defined]
    }


def _commit_dict(commit: object) -> dict[str, object]:
    return {
        "sha": commit.sha,  # type: ignore[attr-defined]
        "short": commit.short_sha,  # type: ignore[attr-defined]
        "author": commit.author,  # type: ignore[attr-defined]
        "date": commit.date,  # type: ignore[attr-defined]
        "subject": commit.subject,  # type: ignore[attr-defined]
        "merge": commit.merge,  # type: ignore[attr-defined]
    }


def _branch_dict(branch: object) -> dict[str, object]:
    return {
        "name": branch.name,  # type: ignore[attr-defined]
        "current": branch.current,  # type: ignore[attr-defined]
        "upstream": branch.upstream,  # type: ignore[attr-defined]
        "ahead": branch.ahead,  # type: ignore[attr-defined]
        "behind": branch.behind,  # type: ignore[attr-defined]
    }


def _run_tasks(service: CodeStudioService, args: argparse.Namespace) -> int:
    specs = service.command_catalog()
    profile = service.project_profile()
    if args.as_json:
        print(
            json.dumps(
                {
                    "project": profile.kind,
                    "markers": list(profile.markers),
                    "tasks": [
                        {
                            "key": s.key,
                            "title": s.title,
                            "category": s.category,
                            "command": s.command_line(),
                            "summary": s.summary,
                        }
                        for s in specs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Project kind: {profile.kind}  ({', '.join(profile.markers) or 'no markers'})")
    print("Tasks you can run in a workspace:")
    for spec in specs:
        print(f"  {spec.key:14s} [{spec.category}] {spec.title}")
        print(f"      {spec.summary}")
    print()
    print("Run one with: python -m cli studio run <slug> <key>")
    return 0


def _run_run(service: CodeStudioService, args: argparse.Namespace) -> int:
    def on_line(line: str) -> None:
        if not args.as_json:
            print(line, flush=True)

    try:
        outcome = service.run_command(
            args.slug, args.task, on_line=None if args.as_json else on_line,
            command_line=args.custom_command,
        )
    except TaskError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": outcome.ok,
                    "returncode": outcome.returncode,
                    "seconds": round(outcome.seconds, 2),
                    "output": outcome.output,
                    "error": outcome.error,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if outcome.ok else 4
    print(outcome.summary())
    return 0 if outcome.ok else 4


def _run_backend(service: CodeStudioService, args: argparse.Namespace) -> int:
    backend = service.backend()
    if args.as_json:
        print(
            json.dumps(
                {
                    "backend": backend.backend,
                    "available": backend.available,
                    "detail": backend.detail,
                    "hint": backend.hint,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if backend.available else 3
    print(backend.summary())
    if not backend.available and backend.hint:
        print(f"Next step: {backend.hint}")
    return 0 if backend.available else 3


def _run_models(service: CodeStudioService, args: argparse.Namespace) -> int:
    rows = service.available_models()
    if args.as_json:
        print(
            json.dumps(
                [
                    {"key": key, "title": title, "size_gb": size, "summary": summary}
                    for key, title, size, summary in rows
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print("Coding models you can download:")
    for key, title, size, summary in rows:
        print(f"  {key:22s} {title}  (~{size:.1f} GB)")
        print(f"      {summary}")
    print()
    print("Download one with: python -m cli studio download --spec <key> --approve")
    return 0


def _run_download(service: CodeStudioService, args: argparse.Namespace) -> int:
    if not args.approve:
        print(
            "Downloading a model leaves the machine, so it needs --approve.",
            file=sys.stderr,
        )
        return 2
    report = service.download_model(args.spec, approved=True)
    _print_report(report, as_json=args.as_json)
    return 0 if report.ok else 3


def _run_ask(service: CodeStudioService, args: argparse.Namespace) -> int:
    # Stream each step as the model takes it, so the terminal shows the work
    # first-by-first like the window does. With --json, stdout stays clean, so the
    # steps are dropped rather than printed.
    def live(step: Any) -> None:
        for line in step.live_lines():
            print(line, flush=True)

    report, trace = service.run_agent(args.slug, args.request, on_step=None if args.as_json else live)
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
    "backend": _run_backend,
    "models": _run_models,
    "download": _run_download,
    "new": _run_new,
    "list": _run_list,
    "files": _run_files,
    "tree": _run_tree,
    "read": _run_read,
    "write": _run_write,
    "edit": _run_edit,
    "search": _run_search,
    "replace": _run_replace,
    "outline": _run_outline,
    "notes": _run_notes,
    "touch": _run_touch,
    "mkdir": _run_mkdir,
    "rename": _run_rename,
    "cp": _run_cp,
    "rm": _run_rm,
    "git": _run_git,
    "tasks": _run_tasks,
    "run": _run_run,
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
    except TaskError as error:
        print(str(error), file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Could not complete the operation: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
