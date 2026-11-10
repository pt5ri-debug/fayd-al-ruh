"""CLI for the Workplace: the institute's digital office, headless.

    python -m cli workplace status            # what the Workplace holds right now
    python -m cli workplace seed              # install the defaults (idempotent)
    python -m cli workplace seed --reset      # reinstall the defaults
    python -m cli workplace divisions         # the ten (configurable) divisions
    python -m cli workplace divisions --all   # include deactivated ones
    python -m cli workplace add-division "Metallurgy Lab" --name-ar "مختبر المعادن"
    python -m cli workplace rename-division DIV-… "Welding & NDT"
    python -m cli workplace deactivate-division DIV-…
    python -m cli workplace activate-division DIV-…
    python -m cli workplace hierarchy         # the institution tree
    python -m cli workplace taxonomy          # the professional knowledge taxonomy
    python -m cli workplace backup            # a consistent copy of the database

This is the headless twin of the Workplace window: it drives the *same*
:class:`gui.workplace.WorkplaceService`, so the CLI and the window share one
database, one set of divisions, and one taxonomy, and no workflow logic is
reimplemented here.

The shared flags (``--data-root``, ``--json``) are accepted **before or after**
the subcommand, exactly like the Code Studio CLI, so ``workplace status --json``
and ``workplace --json status`` agree.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the Workplace is not
available, ``4`` an operation was refused, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gui.workplace import WorkplaceError, WorkplaceService

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_REFUSED = 4
EXIT_FAILURE = 5


def _common_flags(parser: argparse.ArgumentParser) -> None:
    """Let the shared flags appear before *or* after the subcommand.

    ``argparse`` only accepts a main-parser option before the subcommand, so
    ``workplace status --json`` would fail. Each sub-parser re-declares the shared
    flags with ``SUPPRESS`` defaults, so a value given after the subcommand wins
    and a value given before it is not overwritten by the sub-parser's default --
    the same "before or after" rule the Code Studio CLI uses.
    """
    parser.add_argument("--data-root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
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
    parser = argparse.ArgumentParser(prog="workplace", description=__doc__)
    parser.add_argument("--data-root", default=None, help="Override the data folder.")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Machine-readable output."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _sub(sub, "status", help="What the Workplace holds right now.")
    _sub(sub, "hierarchy", help="The institution tree: units and their divisions.")
    _sub(sub, "taxonomy", help="The professional knowledge taxonomy.")

    seed = _sub(sub, "seed", help="Install (or reset) the Workplace defaults.")
    seed.add_argument("--reset", action="store_true", help="Reinstall the defaults.")

    divisions = _sub(sub, "divisions", help="List the divisions.")
    divisions.add_argument("--all", action="store_true", help="Include deactivated divisions.")

    add = _sub(sub, "add-division", help="Add a division (no source change).")
    add.add_argument("name", help="The division name (English).")
    add.add_argument("--name-ar", default="", help="The Arabic name.")
    add.add_argument("--key", default="", help="A stable key (default: derived from the name).")
    add.add_argument("--summary", default="", help="A one-line description.")
    add.add_argument("--scope", default="", help="The scope of services.")

    rename = _sub(sub, "rename-division", help="Rename a division (no source change).")
    rename.add_argument("record_id", help="The division id.")
    rename.add_argument("name", help="The new name.")
    rename.add_argument("--name-ar", default="", help="The new Arabic name.")

    deactivate = _sub(sub, "deactivate-division", help="Deactivate a division (soft delete).")
    deactivate.add_argument("record_id", help="The division id.")

    activate = _sub(sub, "activate-division", help="Reactivate a division.")
    activate.add_argument("record_id", help="The division id.")

    backup = _sub(sub, "backup", help="Write a consistent copy of the database.")
    backup.add_argument("--output", default=None, help="Where to write the copy.")

    return parser


def _service(args: argparse.Namespace) -> WorkplaceService:
    root = Path(args.data_root) if args.data_root else None
    return WorkplaceService(root=root)


def _division_payload(summary) -> dict[str, object]:
    return {
        "record_id": summary.record_id,
        "key": summary.key,
        "name": summary.name,
        "name_ar": summary.name_ar,
        "summary": summary.summary,
        "scope": summary.scope,
        "proposed": summary.proposed,
        "order": summary.order,
        "active": summary.active,
        "unit_id": summary.unit_id,
        "unit_name": summary.unit_name,
    }


def run_status(args: argparse.Namespace) -> int:
    status = _service(args).status()
    if args.as_json:
        print(json.dumps({
            "db_path": status.db_path,
            "schema_version": status.schema_version,
            "institutions": status.institutions,
            "units": status.units,
            "divisions": status.divisions,
            "active_divisions": status.active_divisions,
            "taxonomy_areas": status.taxonomy_areas,
            "taxonomy_disciplines": status.taxonomy_disciplines,
            "ready": status.ready,
            "message": status.message,
        }, ensure_ascii=False, indent=2))
    else:
        print(status.describe())
        if not status.ready:
            print(f"  {status.message}")
    return EXIT_OK if status.ready else EXIT_UNAVAILABLE


def run_seed(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.seed(force=args.reset)
    if args.as_json:
        print(json.dumps({
            "institutions": report.institutions,
            "units": report.units,
            "divisions": report.divisions,
            "skipped": report.skipped,
        }, ensure_ascii=False, indent=2))
    else:
        print(report.describe())
    return EXIT_OK


def run_divisions(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.divisions(include_inactive=args.all)
    if args.as_json:
        print(json.dumps([_division_payload(row) for row in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No divisions yet. Run `python -m cli workplace seed` to install the defaults.")
        return EXIT_OK
    for row in rows:
        print(f"  {row.record_id}  {row.describe()}")
    return EXIT_OK


def run_hierarchy(args: argparse.Namespace) -> int:
    service = _service(args)
    view = service.hierarchy()
    if args.as_json:
        print(json.dumps({
            "institution": None if view.institution is None else {
                "record_id": view.institution.record_id,
                "name": view.institution.name,
                "name_ar": view.institution.name_ar,
                "official": view.institution.official,
            },
            "units": [
                {"record_id": u.record_id, "name": u.name, "name_ar": u.name_ar,
                 "kind": u.kind, "official": u.official}
                for u in view.units
            ],
            "divisions_by_unit": {
                unit_id: [_division_payload(d) for d in rows]
                for unit_id, rows in view.divisions_by_unit.items()
            },
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    if view.institution is not None:
        mark = "" if view.institution.official else "  (internal / proposed)"
        print(f"{view.institution.name} / {view.institution.name_ar}{mark}")
    for unit in view.units:
        mark = "" if unit.official else "  (internal / proposed)"
        print(f"  ├─ {unit.name} / {unit.name_ar}{mark}")
        for division in view.divisions_by_unit.get(unit.record_id, []):
            print(f"  │    • {division.describe()}")
    if not view.units:
        print("  (no units yet — run `python -m cli workplace seed`)")
    return EXIT_OK


def run_taxonomy(args: argparse.Namespace) -> int:
    service = _service(args)
    areas = service.taxonomy()
    if args.as_json:
        print(json.dumps([
            {"key": a.key, "name": a.name, "name_ar": a.name_ar, "summary": a.summary,
             "disciplines": list(a.disciplines)}
            for a in areas
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Professional knowledge taxonomy — {service.taxonomy_summary()}")
    for area in areas:
        print(f"  {area.name} / {area.name_ar}")
        for discipline in area.disciplines:
            print(f"      • {discipline}")
    return EXIT_OK


def run_add_division(args: argparse.Namespace) -> int:
    service = _service(args)
    summary = service.add_division(
        name=args.name, name_ar=args.name_ar, key=args.key,
        summary=args.summary, scope=args.scope,
    )
    if args.as_json:
        print(json.dumps(_division_payload(summary), ensure_ascii=False, indent=2))
    else:
        print(f"Added {summary.record_id}: {summary.describe()}")
    return EXIT_OK


def run_rename_division(args: argparse.Namespace) -> int:
    service = _service(args)
    summary = service.rename_division(args.record_id, name=args.name, name_ar=args.name_ar)
    if args.as_json:
        print(json.dumps(_division_payload(summary), ensure_ascii=False, indent=2))
    else:
        print(f"Renamed {summary.record_id}: {summary.describe()}")
    return EXIT_OK


def run_deactivate_division(args: argparse.Namespace) -> int:
    summary = _service(args).deactivate_division(args.record_id)
    print(f"Deactivated {summary.record_id}: {summary.describe()}")
    return EXIT_OK


def run_activate_division(args: argparse.Namespace) -> int:
    summary = _service(args).activate_division(args.record_id)
    print(f"Activated {summary.record_id}: {summary.describe()}")
    return EXIT_OK


def run_backup(args: argparse.Namespace) -> int:
    service = _service(args)
    path = service.backup(Path(args.output) if args.output else None)
    if args.as_json:
        print(json.dumps({"path": path}, ensure_ascii=False, indent=2))
    else:
        print(f"Backed up the Workplace database to {path}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    handlers = {
        "status": run_status,
        "seed": run_seed,
        "divisions": run_divisions,
        "hierarchy": run_hierarchy,
        "taxonomy": run_taxonomy,
        "add-division": run_add_division,
        "rename-division": run_rename_division,
        "deactivate-division": run_deactivate_division,
        "activate-division": run_activate_division,
        "backup": run_backup,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except WorkplaceError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    except Exception as error:  # pragma: no cover - a last-resort guard
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
