"""The Maintenance CLI: the headless twin of the Maintenance window.

It drives the *same* :class:`gui.maintenance.MaintenanceService` the window
drives, so the CLI and the window share one backup folder, one diagnostics read,
and one shortcut map -- no logic is reimplemented here.

Commands:

* ``status`` -- the stores, the diagnostics, and the shortcuts (the whole page);
* ``stores`` -- the state of each store (present/absent, honestly);
* ``backup <store> [--to PATH]`` -- copy one store into the backup folder;
* ``restore <store> --from PATH [--approve]`` -- restore one store (governed);
* ``diagnostics [--json]`` -- the machine's and the models' figures;
* ``log [--limit N]`` -- the recent Model Manager operation log;
* ``shortcuts [--json]`` -- every keyboard shortcut the program binds.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the store is unknown,
``4`` a gate refused the operation (a restore without ``--approve``), ``5`` an
unexpected failure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from gui.maintenance import STORE_TITLES, MaintenanceService

_EXIT_INVALID = 2
_EXIT_UNKNOWN = 3
_EXIT_REFUSED = 4
_EXIT_FAILURE = 5


def _service(args: argparse.Namespace) -> MaintenanceService:
    root = getattr(args, "data_root", None)
    return MaintenanceService(root)


def _fail(message: str, code: int) -> int:
    print(message)
    return code


# ------------------------------------------------------------------ commands


def _cmd_status(args: argparse.Namespace) -> int:
    service = _service(args)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "stores": [asdict(status) for status in service.store_statuses()],
                    "diagnostics": [asdict(line) for line in service.diagnostics()],
                    "shortcuts": [asdict(shortcut) for shortcut in service.shortcuts()],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(service.describe())
    return 0


def _cmd_stores(args: argparse.Namespace) -> int:
    service = _service(args)
    statuses = service.store_statuses()
    if getattr(args, "json", False):
        print(json.dumps([asdict(status) for status in statuses], ensure_ascii=False, indent=2))
        return 0
    for status in statuses:
        print(status.describe())
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    service = _service(args)
    store = str(args.store or "").strip().lower()
    if store not in STORE_TITLES:
        return _fail(f"Unknown store {args.store!r}. Known: {', '.join(STORE_TITLES)}", _EXIT_UNKNOWN)
    result = service.backup(store, destination=getattr(args, "to", None))
    print(result.describe())
    return 0 if result.ok else _EXIT_FAILURE


def _cmd_restore(args: argparse.Namespace) -> int:
    store = str(args.store or "").strip().lower()
    if store not in STORE_TITLES:
        return _fail(f"Unknown store {args.store!r}. Known: {', '.join(STORE_TITLES)}", _EXIT_UNKNOWN)
    if not getattr(args, "approve", False):
        return _fail(
            "Restoring replaces the current data; re-run with --approve to confirm.",
            _EXIT_REFUSED,
        )
    service = _service(args)
    result = service.restore(store, args.from_path)
    print(result.describe())
    return 0 if result.ok else _EXIT_FAILURE


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    service = _service(args)
    lines = service.diagnostics()
    if getattr(args, "json", False):
        print(json.dumps([asdict(line) for line in lines], ensure_ascii=False, indent=2))
        return 0
    for line in lines:
        print(f"{line.label}: {line.value}")
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    service = _service(args)
    entries = service.recent_log(getattr(args, "limit", 50))
    if getattr(args, "json", False):
        print(json.dumps(list(entries), ensure_ascii=False, indent=2))
        return 0
    print("\n".join(entries) if entries else "No operation has been logged yet.")
    return 0


def _cmd_shortcuts(args: argparse.Namespace) -> int:
    service = _service(args)
    shortcuts = service.shortcuts()
    if getattr(args, "json", False):
        print(json.dumps([asdict(shortcut) for shortcut in shortcuts], ensure_ascii=False, indent=2))
        return 0
    for shortcut in shortcuts:
        print(f"{shortcut.keys:<12} {shortcut.action} ({shortcut.scope})")
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    service = _service(args)
    owner = str(getattr(args, "owner", "") or "").strip()
    if not owner:
        return _fail("Name a personality, e.g. `memory jarvis`.", _EXIT_INVALID)
    lines = service.memory_lines(owner, limit=getattr(args, "limit", 20))
    summary = service.memory_summary(owner)
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "owner": owner,
                    "summary": summary,
                    "entries": [asdict(line) for line in lines],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(summary)
    for line in lines:
        print(f"  {line.describe()}")
    if not lines:
        print("  (nothing remembered yet)")
    return 0


def _cmd_knowledge(args: argparse.Namespace) -> int:
    service = _service(args)
    query = str(getattr(args, "query", "") or "").strip()
    if not query:
        return _fail("Provide a query, e.g. `knowledge porosity`.", _EXIT_INVALID)
    hits = service.knowledge_search(query, limit=getattr(args, "limit", 5))
    if getattr(args, "json", False):
        print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print("Nothing matched -- the knowledge base may be empty, or its index not built.")
        return 0
    for hit in hits:
        print(hit.describe())
        print()
    return 0


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maintenance",
        description="Back up and restore data, read diagnostics, and list shortcuts.",
    )
    parser.add_argument("--data-root", default=None, help="the writable data folder")
    # The shared flag is accepted before *or* after the subcommand, like every
    # other CLI in the project, so `status --data-root X` and `--data-root X status`
    # agree.
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so a subcommand that does not receive the flag does not overwrite
    # the top-level value with its own ``None`` default.
    common.add_argument("--data-root", default=argparse.SUPPRESS, help="the writable data folder")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", parents=[common], help="the whole page: stores, diagnostics, shortcuts")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=_cmd_status)

    p_stores = sub.add_parser("stores", parents=[common], help="the state of each store")
    p_stores.add_argument("--json", action="store_true")
    p_stores.set_defaults(func=_cmd_stores)

    p_backup = sub.add_parser("backup", parents=[common], help="copy one store into the backup folder")
    p_backup.add_argument("store", choices=list(STORE_TITLES))
    p_backup.add_argument("--to", default=None, help="an explicit destination")
    p_backup.set_defaults(func=_cmd_backup)

    p_restore = sub.add_parser("restore", parents=[common], help="restore one store (governed)")
    p_restore.add_argument("store", choices=list(STORE_TITLES))
    p_restore.add_argument("--from", dest="from_path", required=True, help="the backup to restore")
    p_restore.add_argument("--approve", action="store_true", help="confirm the replacement")
    p_restore.set_defaults(func=_cmd_restore)

    p_diag = sub.add_parser("diagnostics", parents=[common], help="the machine's and the models' figures")
    p_diag.add_argument("--json", action="store_true")
    p_diag.set_defaults(func=_cmd_diagnostics)

    p_log = sub.add_parser("log", parents=[common], help="the recent Model Manager operation log")
    p_log.add_argument("--limit", type=int, default=50)
    p_log.add_argument("--json", action="store_true")
    p_log.set_defaults(func=_cmd_log)

    p_keys = sub.add_parser("shortcuts", parents=[common], help="every keyboard shortcut the program binds")
    p_keys.add_argument("--json", action="store_true")
    p_keys.set_defaults(func=_cmd_shortcuts)

    p_mem = sub.add_parser("memory", parents=[common], help="show one personality's memory (read-only)")
    p_mem.add_argument("owner", help="the personality (jarvis, ultron, fayd_al_ruh, ...)")
    p_mem.add_argument("--limit", type=int, default=20)
    p_mem.add_argument("--json", action="store_true")
    p_mem.set_defaults(func=_cmd_memory)

    p_kb = sub.add_parser("knowledge", parents=[common], help="search the bulk knowledge base (read-only)")
    p_kb.add_argument("query", help="what to search for")
    p_kb.add_argument("--limit", type=int, default=5)
    p_kb.add_argument("--json", action="store_true")
    p_kb.set_defaults(func=_cmd_knowledge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        return _cmd_status(argparse.Namespace(json=False, data_root=getattr(args, "data_root", None)))
    try:
        return int(args.func(args))
    except (ValueError, OSError, KeyError) as error:
        return _fail(f"{type(error).__name__}: {error}", _EXIT_FAILURE)


__all__ = ["build_parser", "main"]
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
