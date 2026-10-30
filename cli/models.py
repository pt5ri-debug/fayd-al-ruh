"""Manage the project's language models from the command line.

    python -m cli models list                       # everything the program can run
    python -m cli models status                     # a one-screen summary
    python -m cli models runtimes                   # which runtimes work here
    python -m cli models discover [--root DIR]      # scan for model files
    python -m cli models register PATH [--name N]   # remember a file the user has
    python -m cli models show MODEL                 # one model in detail
    python -m cli models inspect MODEL [--digest]   # re-read the file from disk
    python -m cli models check MODEL                # can it run here, and why not
    python -m cli models load MODEL                 # load it into its runtime
    python -m cli models unload [MODEL]             # release the loaded model(s)
    python -m cli models test MODEL [--prompt P]    # prove it answers
    python -m cli models settings MODEL [--set k=v] # inference settings
    python -m cli models update MODEL [--name N]    # edit a record
    python -m cli models remove MODEL --yes         # forget a record (never a file)
    python -m cli models select [TASK]              # which models suit a task
    python -m cli models log [--limit N]            # what happened recently

The headless twin of the Model Manager window. It drives the *same*
:mod:`core.model_manager` API the GUI and the assistant use, so the two surfaces
can never disagree about which models exist or what state they are in.

The subsystem is **read-mostly by design**: it discovers, registers, inspects,
loads, and tests models, and it reports what a download would be. It never
fetches weights itself -- that stays with the engines that already implement it
behind their own permission and human-approval gates.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the model is unknown or
not usable, ``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from core.model_files import ModelFileError
from core.model_manager import (
    TASKS,
    ModelManager,
    model_manager,
)
from core.model_registry import (
    ConfirmationRequired,
    ModelNotFoundError,
    RegistryError,
)
from core.model_runtime import RuntimeUnavailable
from core.model_settings import SettingsError


def _manager() -> ModelManager:
    return model_manager()


def _print_entries(entries: tuple[Any, ...]) -> None:
    if not entries:
        print("No models. Register one with `python -m cli models register <file>`.")
        return
    for entry in entries:
        marker = "✓" if entry.ready else "·"
        source = "catalog" if entry.from_catalog else entry.record.source
        print(f"{marker} {entry.display_name}  [{entry.state}]")
        print(f"    id:      {entry.model_id}")
        print(f"    runtime: {entry.runtime}   source: {source}")
        if entry.record.path:
            print(f"    file:    {entry.record.path}")
        if not entry.ready and entry.status_line:
            print(f"    note:    {entry.status_line}")


def _cmd_list(args: argparse.Namespace) -> int:
    mgr = _manager()
    entries = mgr.list_models(include_catalog=not args.registered_only)
    if args.json:
        payload = [
            {
                "model_id": entry.model_id,
                "display_name": entry.display_name,
                "state": entry.state,
                "ready": entry.ready,
                "runtime": entry.runtime,
                "source": entry.record.source,
                "from_catalog": entry.from_catalog,
                "path": str(entry.record.path) if entry.record.path else "",
                "status": entry.status_line,
            }
            for entry in entries
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_entries(entries)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    mgr = _manager()
    status = mgr.status()
    if args.json:
        print(
            json.dumps(
                {
                    "registered": status.registered,
                    "catalog": status.catalog,
                    "ready": status.ready,
                    "registry_path": str(status.registry_path),
                    "log_events": status.log_events,
                    "corrupt_records": list(status.corrupt_records),
                    "runtimes": status.runtimes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(status.describe())
    return 0


def _cmd_runtimes(args: argparse.Namespace) -> int:
    mgr = _manager()
    if args.json:
        payload = []
        for runtime in mgr.runtime_names():
            status = mgr.runtime_status(runtime)
            payload.append(
                {
                    "runtime": runtime,
                    "available": status.available,
                    "detail": status.detail,
                    "hint": status.hint,
                    "version": status.version,
                    "gpu_offload": status.gpu_offload,
                    "capabilities": status.capabilities.describe(),
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(mgr.runtime_report())
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    mgr = _manager()
    report = mgr.discover(args.root or None)
    if args.json:
        print(
            json.dumps(
                {
                    "roots": [str(root) for root in report.roots],
                    "files": [
                        {
                            "path": str(probe.path),
                            "format": probe.format,
                            "integrity": probe.integrity,
                            "size_bytes": probe.size_bytes,
                            "usable": probe.usable,
                        }
                        for probe in report.probes
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not report.probes:
        print("No model files found.")
        for root in report.roots:
            print(f"    searched: {root}")
        return 0
    print(f"Found {len(report.probes)} model file(s):")
    for probe in report.probes:
        marker = "✓" if probe.usable else "·"
        print(f"{marker} {probe.path}  [{probe.format}, {probe.integrity}]")
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    mgr = _manager()
    entry = mgr.register(
        args.path,
        display_name=args.name or "",
        notes=args.notes or "",
        engine=args.engine or "",
        digest=args.digest,
        replace_existing=args.replace,
    )
    print(f"Registered {entry.display_name} ({entry.model_id}) — {entry.state}.")
    return 0


def _cmd_import_custom(args: argparse.Namespace) -> int:
    mgr = _manager()
    entries = mgr.import_custom_models(replace_existing=args.replace)
    if not entries:
        print("No hand-added models to import.")
        return 0
    for entry in entries:
        print(f"Imported {entry.display_name} ({entry.model_id}).")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    mgr = _manager()
    entry = mgr.get(args.model)
    if args.json:
        print(json.dumps(_entry_payload(entry), ensure_ascii=False, indent=2))
        return 0
    print(entry.describe())
    record = entry.record
    print(f"  id:         {record.model_id}")
    print(f"  identity:   {record.identity}")
    print(f"  format:     {record.model_format}")
    print(f"  runtime:    {record.runtime}")
    print(f"  source:     {record.source}")
    print(f"  path:       {record.path or '(none)'}")
    print(f"  size:       {record.size_bytes} bytes")
    print(f"  integrity:  {record.integrity or 'unchecked'}")
    if record.tags:
        print(f"  tags:       {', '.join(record.tags)}")
    if record.notes:
        print(f"  notes:      {record.notes}")
    print(f"  registered: {record.registered_at}")
    print(f"  updated:    {record.updated_at}")
    return 0


def _entry_payload(entry: Any) -> dict[str, Any]:
    record = entry.record
    return {
        "model_id": record.model_id,
        "display_name": entry.display_name,
        "identity": record.identity,
        "format": record.model_format,
        "runtime": record.runtime,
        "source": record.source,
        "path": str(record.path) if record.path else "",
        "size_bytes": record.size_bytes,
        "integrity": record.integrity,
        "state": entry.state,
        "ready": entry.ready,
        "status": entry.status_line,
        "tags": list(record.tags),
        "notes": record.notes,
        "registered_at": record.registered_at,
        "updated_at": record.updated_at,
        "from_catalog": entry.from_catalog,
    }


def _cmd_inspect(args: argparse.Namespace) -> int:
    mgr = _manager()
    probe = mgr.inspect(args.model, digest=args.digest)
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(probe.path),
                    "exists": probe.exists,
                    "size_bytes": probe.size_bytes,
                    "format": probe.format,
                    "integrity": probe.integrity,
                    "usable": probe.usable,
                    "digest": probe.digest,
                    "detail": probe.detail,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"{probe.path}")
    print(f"  exists:     {probe.exists}")
    print(f"  size:       {probe.size_bytes} bytes")
    print(f"  format:     {probe.format}")
    print(f"  integrity:  {probe.integrity}")
    print(f"  usable:     {probe.usable}")
    if probe.digest:
        print(f"  digest:     {probe.digest}")
    if probe.gguf is not None:
        print(f"  name:       {probe.gguf.name or '(unnamed)'}")
        print(f"  arch:       {probe.gguf.architecture or '(unknown)'}")
        print(f"  context:    {probe.gguf.context_length or '(unknown)'}")
    if probe.checkpoint is not None:
        print(f"  parameters: {probe.checkpoint.parameter_count}")
    if probe.detail:
        print(f"  note:       {probe.detail}")
    return 0 if probe.usable else 3


def _cmd_check(args: argparse.Namespace) -> int:
    mgr = _manager()
    report = mgr.compatibility(args.model)
    if args.json:
        print(
            json.dumps(
                {
                    "model_id": report.model_id,
                    "runtime": report.runtime,
                    "can_run": report.can_run,
                    "reasons": list(report.reasons),
                    "warnings": list(report.warnings),
                    "gpu_offload": report.gpu_offload,
                    "memory_estimate_bytes": report.memory_estimate_bytes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report.can_run else 3
    print(report.describe())
    return 0 if report.can_run else 3


def _cmd_load(args: argparse.Namespace) -> int:
    mgr = _manager()
    result = mgr.load(args.model)
    print(result.describe())
    return 0 if result.ok else 3


def _cmd_unload(args: argparse.Namespace) -> int:
    mgr = _manager()
    result = mgr.unload(args.model or "")
    print(result.describe())
    return 0 if result.ok else 3


def _cmd_test(args: argparse.Namespace) -> int:
    mgr = _manager()
    result = mgr.test(args.model, prompt=args.prompt)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "model_id": result.model_id,
                    "detail": result.detail,
                    "text": result.inference.text if result.inference else "",
                    "seconds": result.inference.seconds if result.inference else 0.0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.ok else 3
    print(result.describe())
    return 0 if result.ok else 3


def _cmd_settings(args: argparse.Namespace) -> int:
    mgr = _manager()
    if args.reset:
        settings = mgr.reset_settings(args.model)
        print(f"Reset {args.model} to the project defaults: {settings.describe()}")
        return 0
    if args.set:
        changes: dict[str, Any] = {}
        for item in args.set:
            if "=" not in item:
                return parser_error(f"--set needs key=value, got {item!r}")
            key, _, value = item.partition("=")
            changes[key.strip()] = _coerce(value.strip())
        try:
            settings = mgr.update_settings(args.model, **changes)
        except SettingsError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"Saved: {settings.describe()}")
        return 0
    settings = mgr.settings(args.model)
    if args.json:
        print(json.dumps(settings.to_primitive(), ensure_ascii=False, indent=2))
        return 0
    print(settings.describe())
    return 0


def _coerce(value: str) -> Any:
    """Read a ``key=value`` argument as the number it looks like, else a string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _cmd_update(args: argparse.Namespace) -> int:
    mgr = _manager()
    changes: dict[str, Any] = {}
    if args.name is not None:
        changes["display_name"] = args.name
    if args.notes is not None:
        changes["notes"] = args.notes
    if args.engine is not None:
        changes["engine"] = args.engine
    if args.tags is not None:
        changes["tags"] = args.tags
    if not changes:
        return parser_error("nothing to update: pass --name, --notes, --engine, or --tags")
    entry = mgr.update(args.model, **changes)
    print(f"Updated {entry.display_name} ({entry.model_id}).")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    mgr = _manager()
    record = mgr.remove(args.model, confirm=args.yes)
    print(f"Forgot {record.display_name} ({record.model_id}). The file was not touched.")
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    """Show which models suit a task, best first (a *view*, never a new registry)."""
    mgr = _manager()
    task = (args.task or "").strip()
    if not task:
        if args.json:
            print(json.dumps({"tasks": list(TASKS)}, ensure_ascii=False, indent=2))
        else:
            print("Tasks a model can be chosen for:")
            for name in TASKS:
                print(f"  {name}")
        return 0
    if task not in TASKS:
        print(f"Unknown task {task!r}. Known tasks: {', '.join(TASKS)}.", file=sys.stderr)
        return 2
    entries = mgr.models_for_task(task)
    if args.json:
        print(
            json.dumps(
                {"task": task, "models": [_entry_payload(entry) for entry in entries]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not entries:
        print(f"No model can run the {task!r} task here yet.")
        return 3
    print(f"Models for the {task!r} task, best first:")
    _print_entries(entries)
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    mgr = _manager()
    events = mgr.log_tail(args.limit)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "at": event.at,
                        "operation": event.operation,
                        "model_id": event.model_id,
                        "display_name": event.display_name,
                        "runtime": event.runtime,
                        "ok": event.ok,
                        "message": event.message,
                    }
                    for event in events
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not events:
        print("No model operations recorded yet.")
        return 0
    for event in events:
        print(event.describe())
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Filter the listing by free text and/or state -- a *view*, never a new list."""
    mgr = _manager()
    needle = (args.query or "").strip().casefold()
    state = (args.state or "").strip()
    runtime = (args.runtime or "").strip()
    rows: list[Any] = []
    for entry in mgr.list_models(include_catalog=not args.registered_only):
        record = entry.record
        if state and entry.state != state:
            continue
        if runtime and entry.runtime != runtime:
            continue
        if needle:
            haystack = " ".join(
                (
                    entry.display_name,
                    entry.model_id,
                    str(record.path or ""),
                    entry.runtime,
                    record.quantization,
                    record.source,
                )
            ).casefold()
            if needle not in haystack:
                continue
        rows.append(entry)
    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query or "",
                    "state": state,
                    "runtime": runtime,
                    "models": [_entry_payload(entry) for entry in rows],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not rows:
        print("No models matched.")
        return 3
    print(f"{len(rows)} model(s) matched:")
    _print_entries(tuple(rows))
    return 0


def _cmd_active(args: argparse.Namespace) -> int:
    """The models currently *loaded* in their runtime."""
    mgr = _manager()
    entries = mgr.active_models()
    if args.json:
        print(
            json.dumps(
                {"models": [_entry_payload(entry) for entry in entries]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not entries:
        print("No model is loaded right now.")
        return 0
    print(f"{len(entries)} model(s) loaded right now:")
    _print_entries(entries)
    return 0


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    """The whole diagnostics body: machine, runtimes, active models, errors."""
    mgr = _manager()
    report = mgr.diagnostics()
    if args.json:
        resources = report.resources
        print(
            json.dumps(
                {
                    "status": {
                        "registered": report.status.registered,
                        "catalog": report.status.catalog,
                        "ready": report.status.ready,
                        "registry_path": str(report.status.registry_path),
                        "log_events": report.status.log_events,
                        "corrupt_records": list(report.status.corrupt_records),
                    },
                    "resources": (
                        {
                            "system": resources.system,
                            "machine": resources.machine,
                            "python": resources.python,
                            "cpu_count": resources.cpu_count,
                            "total_memory_bytes": resources.total_memory_bytes,
                            "available_memory_bytes": resources.available_memory_bytes,
                            "disk_free_bytes": resources.disk_free_bytes,
                        }
                        if resources is not None
                        else None
                    ),
                    "runtimes": list(report.runtime_lines),
                    "active": list(report.active),
                    "errors": list(report.errors),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(report.describe())
    return 0


def parser_error(message: str) -> int:
    """Report a usage error and return the exit code (never raises)."""
    print(f"error: {message}", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli models",
        description="Discover, register, inspect, and run the project's language models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list the models the program knows about")
    listing.add_argument("--json", action="store_true", help="print JSON")
    listing.add_argument(
        "--registered-only", action="store_true",
        help="only the models the user registered (skip the shipped catalog)",
    )

    status = sub.add_parser("status", help="a one-screen summary of the subsystem")
    status.add_argument("--json", action="store_true", help="print JSON")

    runtimes = sub.add_parser("runtimes", help="which runtimes work on this machine")
    runtimes.add_argument("--json", action="store_true", help="print JSON")

    discover = sub.add_parser("discover", help="scan the project's folders for model files")
    discover.add_argument("--root", action="append", help="a folder to scan (repeatable)")
    discover.add_argument("--json", action="store_true", help="print JSON")

    register = sub.add_parser("register", help="remember a model file the user already has")
    register.add_argument("path", help="the model file to register")
    register.add_argument("--name", help="what to call it")
    register.add_argument("--notes", help="a note to keep with it")
    register.add_argument("--engine", help="the mouth this model is meant for")
    register.add_argument("--digest", action="store_true", help="also compute a content hash")
    register.add_argument("--replace", action="store_true", help="replace an existing record")

    import_custom = sub.add_parser(
        "import-custom", help="import the models added by hand in a mouth window"
    )
    import_custom.add_argument(
        "--replace", action="store_true", help="replace an existing record with the same file"
    )

    show = sub.add_parser("show", help="one model in detail")
    show.add_argument("model", help="the model id")
    show.add_argument("--json", action="store_true", help="print JSON")

    inspect = sub.add_parser("inspect", help="re-read a model's file from disk")
    inspect.add_argument("model", help="the model id")
    inspect.add_argument("--digest", action="store_true", help="also compute a content hash")
    inspect.add_argument("--json", action="store_true", help="print JSON")

    check = sub.add_parser("check", help="whether a model can run here, and why not")
    check.add_argument("model", help="the model id")
    check.add_argument("--json", action="store_true", help="print JSON")

    load = sub.add_parser("load", help="load a model into its runtime")
    load.add_argument("model", help="the model id")

    unload = sub.add_parser("unload", help="release a loaded model (or every loaded model)")
    unload.add_argument("model", nargs="?", default="", help="the model id (default: all)")

    test = sub.add_parser("test", help="ask a model a trivial question to prove it answers")
    test.add_argument("model", help="the model id")
    test.add_argument("--prompt", default="Reply with the single word: ready.")
    test.add_argument("--json", action="store_true", help="print JSON")

    settings = sub.add_parser("settings", help="show or change a model's inference settings")
    settings.add_argument("model", help="the model id")
    settings.add_argument("--set", action="append", metavar="KEY=VALUE", help="set a value")
    settings.add_argument("--reset", action="store_true", help="return to the project defaults")
    settings.add_argument("--json", action="store_true", help="print JSON")

    update = sub.add_parser("update", help="edit a model's record")
    update.add_argument("model", help="the model id")
    update.add_argument("--name", help="a new display name")
    update.add_argument("--notes", help="a new note")
    update.add_argument("--engine", help="the mouth this model is meant for")
    update.add_argument("--tags", nargs="*", help="the tags to keep with it")

    remove = sub.add_parser("remove", help="forget a record (the file itself is never touched)")
    remove.add_argument("model", help="the model id")
    remove.add_argument("--yes", action="store_true", help="confirm the removal")

    log = sub.add_parser("log", help="what happened recently")
    log.add_argument("--limit", type=int, default=30, help="how many events to show")
    log.add_argument("--json", action="store_true", help="print JSON")

    search = sub.add_parser("search", help="filter the listing by text and/or state")
    search.add_argument("query", nargs="?", default="", help="free text to match")
    search.add_argument("--state", help="only models in this state")
    search.add_argument("--runtime", help="only models on this runtime")
    search.add_argument(
        "--registered-only", action="store_true",
        help="only the models the user registered (skip the shipped catalog)",
    )
    search.add_argument("--json", action="store_true", help="print JSON")

    active = sub.add_parser("active", help="the models loaded in their runtime right now")
    active.add_argument("--json", action="store_true", help="print JSON")

    diagnostics = sub.add_parser(
        "diagnostics", help="machine, runtimes, active models, and recent errors"
    )
    diagnostics.add_argument("--json", action="store_true", help="print JSON")

    select = sub.add_parser(
        "select", help="which models suit a task (chat, petroleum, presentation, ...)"
    )
    select.add_argument(
        "task", nargs="?", default="",
        help="the task to choose for (omit to list the known tasks)",
    )
    select.add_argument("--json", action="store_true", help="print JSON")

    return parser


_HANDLERS = {
    "list": _cmd_list,
    "status": _cmd_status,
    "runtimes": _cmd_runtimes,
    "discover": _cmd_discover,
    "register": _cmd_register,
    "import-custom": _cmd_import_custom,
    "show": _cmd_show,
    "inspect": _cmd_inspect,
    "check": _cmd_check,
    "load": _cmd_load,
    "unload": _cmd_unload,
    "test": _cmd_test,
    "settings": _cmd_settings,
    "update": _cmd_update,
    "remove": _cmd_remove,
    "log": _cmd_log,
    "search": _cmd_search,
    "active": _cmd_active,
    "diagnostics": _cmd_diagnostics,
    "select": _cmd_select,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces the set
        parser.error(f"unknown command {args.command!r}")
        return 2
    try:
        return handler(args)
    except ModelNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except ConfirmationRequired as error:
        print(f"refused: {error}", file=sys.stderr)
        print("  re-run with --yes to confirm.", file=sys.stderr)
        return 4
    except (ModelFileError, RegistryError, RuntimeUnavailable) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except Exception as error:  # pragma: no cover - a genuine bug
        print(f"unexpected failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
