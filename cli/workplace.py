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
    python -m cli workplace profile DIV-…     # a division's full technical profile
    python -m cli workplace add-detail DIV-… standard --title "API 1104" --reference "API 1104"
    python -m cli workplace kinds             # the kinds of profile item
    python -m cli workplace add-taxonomy-area "New Area" --name-ar "مجال جديد"
    python -m cli workplace add-taxonomy-discipline TAX-… "New Discipline"
    python -m cli workplace settings --language ar
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

from gui.workplace import WorkplaceDomainError, WorkplaceError, WorkplaceService

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

    move = _sub(sub, "move-division", help="Move a division to another unit.")
    move.add_argument("record_id", help="The division id.")
    move.add_argument("--unit", required=True, help="The target unit id.")
    move.add_argument("--order", type=int, default=None, help="The new display order.")

    profile = _sub(sub, "profile", help="A division's full technical profile.")
    profile.add_argument("record_id", help="The division id.")

    detail = _sub(sub, "add-detail", help="Add an item to a division's profile.")
    detail.add_argument("division_id", help="The division id.")
    detail.add_argument("kind", help="The item kind (see `workplace kinds`).")
    detail.add_argument("--title", default="", help="A short title.")
    detail.add_argument("--detail", default="", help="The detail text.")
    detail.add_argument("--reference", default="", help="A pointer (standard number, file, ...).")

    remove_detail = _sub(sub, "remove-detail", help="Remove a profile item (soft delete).")
    remove_detail.add_argument("record_id", help="The profile item id.")

    _sub(sub, "kinds", help="The kinds of divisional profile item.")

    tax_add = _sub(sub, "add-taxonomy-area", help="Add a taxonomy area (no source change).")
    tax_add.add_argument("name", help="The area name.")
    tax_add.add_argument("--name-ar", default="", help="The Arabic name.")
    tax_add.add_argument("--key", default="", help="A stable key (default: from the name).")
    tax_add.add_argument("--summary", default="", help="A one-line description.")

    tax_disc = _sub(sub, "add-taxonomy-discipline", help="Add a sub-discipline under an area.")
    tax_disc.add_argument("area_id", help="The taxonomy area id.")
    tax_disc.add_argument("name", help="The discipline name.")
    tax_disc.add_argument("--name-ar", default="", help="The Arabic name.")

    tax_rename = _sub(sub, "rename-taxonomy", help="Rename a taxonomy node.")
    tax_rename.add_argument("record_id", help="The taxonomy node id.")
    tax_rename.add_argument("name", help="The new name.")
    tax_rename.add_argument("--name-ar", default="", help="The new Arabic name.")

    tax_remove = _sub(sub, "remove-taxonomy", help="Deactivate a taxonomy node (soft delete).")
    tax_remove.add_argument("record_id", help="The taxonomy node id.")

    settings = _sub(sub, "settings", help="Show or change the Workplace settings.")
    settings.add_argument("--language", choices=("en", "ar"), default=None, help="The interface language.")
    settings.add_argument("--rtl", dest="rtl", action="store_true", default=None, help="Lay out right-to-left.")
    settings.add_argument("--ltr", dest="rtl", action="store_false", help="Lay out left-to-right.")
    settings.add_argument("--default-institution", default=None, help="The default institution key.")
    settings.add_argument("--show-proposed", dest="show_proposed", action="store_true", default=None)
    settings.add_argument("--hide-proposed", dest="show_proposed", action="store_false")

    backup = _sub(sub, "backup", help="Write a consistent copy of the database.")
    backup.add_argument("--output", default=None, help="Where to write the copy.")

    _sub(sub, "dashboard", help="The dashboard figures (real records, never demo data).")

    _sub(sub, "entities", help="The domain entity names (people, projects, ...).")

    domain_add = _sub(sub, "add", help="Create a domain record: `add <entity> key=value ...`.")
    domain_add.add_argument("entity", help="The entity name (see `workplace entities`).")
    domain_add.add_argument("fields", nargs="*", help="Field assignments, e.g. name=Ali role=Inspector.")

    domain_list = _sub(sub, "list", help="List domain records of an entity.")
    domain_list.add_argument("entity", help="The entity name.")
    domain_list.add_argument("--all", action="store_true", help="Include deactivated records.")
    domain_list.add_argument("--limit", type=int, default=None, help="At most this many.")
    domain_list.add_argument(
        "--where", default="", help="A raw SQL filter (advanced), e.g. \"status = 'active'\"."
    )

    domain_show = _sub(sub, "show", help="Show one domain record.")
    domain_show.add_argument("entity", help="The entity name.")
    domain_show.add_argument("record_id", help="The record id.")

    domain_update = _sub(sub, "update", help="Edit a domain record: `update <entity> <id> key=value ...`.")
    domain_update.add_argument("entity", help="The entity name.")
    domain_update.add_argument("record_id", help="The record id.")
    domain_update.add_argument("fields", nargs="+", help="Field assignments.")

    domain_deactivate = _sub(sub, "deactivate", help="Deactivate a domain record (soft delete).")
    domain_deactivate.add_argument("entity", help="The entity name.")
    domain_deactivate.add_argument("record_id", help="The record id.")

    audit = _sub(sub, "audit", help="The append-only audit trail.")
    audit.add_argument("--for", dest="for_id", default="", help="Only rows about one record id.")
    audit.add_argument("--limit", type=int, default=50, help="How many rows (newest first).")

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
    # Prefer the *stored* (editable) taxonomy once it exists, so a user's own
    # areas show up; fall back to the shipped defaults when nothing is stored.
    stored = service.taxonomy_tree()
    if args.as_json:
        if stored:
            print(json.dumps([
                {"key": a.key, "name": a.name, "name_ar": a.name_ar, "summary": a.summary,
                 "disciplines": list(a.metadata.get("disciplines", ()))}
                for a in stored
            ], ensure_ascii=False, indent=2))
        else:
            print(json.dumps([
                {"key": a.key, "name": a.name, "name_ar": a.name_ar, "summary": a.summary,
                 "disciplines": list(a.disciplines)}
                for a in service.taxonomy()
            ], ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Professional knowledge taxonomy — {service.taxonomy_summary()}")
    if stored:
        for area in stored:
            print(f"  {area.name} / {area.name_ar}")
            for discipline in area.metadata.get("disciplines", ()):
                print(f"      • {discipline}")
        return EXIT_OK
    for shipped in service.taxonomy():
        print(f"  {shipped.name} / {shipped.name_ar}")
        for discipline in shipped.disciplines:
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


def _detail_payload(detail) -> dict[str, object]:
    return {
        "record_id": detail.record_id,
        "kind": detail.kind,
        "kind_label": detail.kind_label,
        "title": detail.title,
        "detail": detail.detail,
        "reference": detail.reference,
        "order": detail.order,
        "active": detail.active,
    }


def run_move_division(args: argparse.Namespace) -> int:
    service = _service(args)
    summary = service.set_division_unit(args.record_id, unit_id=args.unit)
    if args.order is not None:
        summary = service.set_division_order(args.record_id, order=args.order)
    if args.as_json:
        print(json.dumps(_division_payload(summary), ensure_ascii=False, indent=2))
    else:
        print(f"Moved {summary.record_id}: {summary.describe()}")
    return EXIT_OK


def run_profile(args: argparse.Namespace) -> int:
    profile = _service(args).division_profile(args.record_id)
    if args.as_json:
        print(json.dumps({
            "division": _division_payload(profile.division),
            "details": [_detail_payload(d) for d in profile.details],
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Profile — {profile.division.describe()}")
    grouped = profile.by_kind()
    if not grouped:
        print("  (no profile items yet — add one with `workplace add-detail`)")
    for _kind, items in grouped.items():
        print(f"  {items[0].kind_label}:")
        for item in items:
            print(f"      • {item.title or item.detail}"
                  + (f" — {item.detail}" if item.title and item.detail else "")
                  + (f"  [{item.reference}]" if item.reference else ""))
    return EXIT_OK


def run_add_detail(args: argparse.Namespace) -> int:
    service = _service(args)
    detail = service.add_division_detail(
        division_id=args.division_id, kind=args.kind,
        title=args.title, detail=args.detail, reference=args.reference,
    )
    if args.as_json:
        print(json.dumps(_detail_payload(detail), ensure_ascii=False, indent=2))
    else:
        print(f"Added {detail.record_id}: {detail.describe()}")
    return EXIT_OK


def run_remove_detail(args: argparse.Namespace) -> int:
    detail = _service(args).remove_division_detail(args.record_id)
    if args.as_json:
        print(json.dumps(_detail_payload(detail), ensure_ascii=False, indent=2))
    else:
        print(f"Removed {detail.record_id}: {detail.describe()}")
    return EXIT_OK


def run_kinds(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.detail_kind_labels()
    if args.as_json:
        print(json.dumps([{"kind": k, "label": label} for k, label in rows],
                         ensure_ascii=False, indent=2))
    else:
        print("Kinds of divisional profile item:")
        for kind, label in rows:
            print(f"  {kind:12s} {label}")
    return EXIT_OK


def run_add_taxonomy_area(args: argparse.Namespace) -> int:
    node = _service(args).add_taxonomy_area(
        name=args.name, name_ar=args.name_ar, key=args.key, summary=args.summary
    )
    if args.as_json:
        print(json.dumps({"record_id": node.record_id, "key": node.key, "name": node.name,
                          "name_ar": node.name_ar}, ensure_ascii=False, indent=2))
    else:
        print(f"Added taxonomy area {node.record_id}: {node.name}")
    return EXIT_OK


def run_add_taxonomy_discipline(args: argparse.Namespace) -> int:
    node = _service(args).add_taxonomy_discipline(
        area_id=args.area_id, name=args.name, name_ar=args.name_ar
    )
    if args.as_json:
        print(json.dumps({"record_id": node.record_id, "key": node.key, "name": node.name,
                          "parent_id": node.parent_id}, ensure_ascii=False, indent=2))
    else:
        print(f"Added discipline {node.record_id}: {node.name}")
    return EXIT_OK


def run_rename_taxonomy(args: argparse.Namespace) -> int:
    node = _service(args).rename_taxonomy_node(
        args.record_id, name=args.name, name_ar=args.name_ar
    )
    if args.as_json:
        print(json.dumps({"record_id": node.record_id, "name": node.name,
                          "name_ar": node.name_ar}, ensure_ascii=False, indent=2))
    else:
        print(f"Renamed {node.record_id}: {node.name}")
    return EXIT_OK


def run_remove_taxonomy(args: argparse.Namespace) -> int:
    node = _service(args).remove_taxonomy_node(args.record_id)
    if args.as_json:
        print(json.dumps({"record_id": node.record_id, "active": node.active},
                         ensure_ascii=False, indent=2))
    else:
        print(f"Deactivated taxonomy node {node.record_id}.")
    return EXIT_OK


def run_settings(args: argparse.Namespace) -> int:
    service = _service(args)
    changes = {
        key: value
        for key, value in (
            ("language", args.language), ("rtl", args.rtl),
            ("default_institution", args.default_institution),
            ("show_proposed", args.show_proposed),
        )
        if value is not None
    }
    settings = service.update_settings(**changes) if changes else service.settings()
    if args.as_json:
        print(json.dumps(settings.to_primitive(), ensure_ascii=False, indent=2))
    else:
        print(f"Language: {settings.language}")
        print(f"Right-to-left: {settings.rtl}")
        print(f"Default institution: {settings.default_institution or '(the first one)'}")
        print(f"Show proposed records: {settings.show_proposed}")
    return EXIT_OK


def run_backup(args: argparse.Namespace) -> int:
    service = _service(args)
    path = service.backup(Path(args.output) if args.output else None)
    if args.as_json:
        print(json.dumps({"path": path}, ensure_ascii=False, indent=2))
    else:
        print(f"Backed up the Workplace database to {path}")
    return EXIT_OK


def run_dashboard(args: argparse.Namespace) -> int:
    service = _service(args)
    board = service.dashboard()
    if args.as_json:
        print(json.dumps({
            "active_projects": board.active_projects,
            "completed_projects": board.completed_projects,
            "total_projects": board.total_projects,
            "pending_assignments": board.pending_assignments,
            "upcoming_inspections": board.upcoming_inspections,
            "outstanding_findings": board.outstanding_findings,
            "reports_awaiting_approval": board.reports_awaiting_approval,
            "documents": board.documents,
            "personnel": board.personnel,
            "clients": board.clients,
            "standards": board.standards,
            "knowledge_records": board.knowledge_records,
            "divisions": board.divisions,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Workplace dashboard (real records):")
    for label, value in board.as_rows():
        print(f"  {label}: {value}")
    return EXIT_OK


def run_entities(args: argparse.Namespace) -> int:
    from core.workplace.records import DOMAIN_RECORDS

    names = sorted(DOMAIN_RECORDS)
    if args.as_json:
        print(json.dumps(names, ensure_ascii=False, indent=2))
    else:
        print("Workplace entities:")
        for name in names:
            print(f"  {name}")
    return EXIT_OK


def _parse_fields(pairs: Sequence[str]) -> dict[str, str]:
    """Turn ``["name=Ali", "role=Inspector"]`` into a field dict."""
    fields: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise WorkplaceError(f"Expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def run_domain_add(args: argparse.Namespace) -> int:
    service = _service(args)
    fields = _parse_fields(args.fields)
    try:
        record = service.domain.create(args.entity, **fields)
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    if args.as_json:
        print(json.dumps(_record_payload(record), ensure_ascii=False, indent=2))
    else:
        print(f"Created {args.entity} {record.record_id}: {_record_title(record)}")
    return EXIT_OK


def run_domain_list(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        rows = service.domain.list_records(
            args.entity, where=args.where, include_inactive=args.all, limit=args.limit
        )
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    if args.as_json:
        print(json.dumps([_record_payload(row) for row in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print(f"No {args.entity} records yet.")
        return EXIT_OK
    for row in rows:
        print(f"  {row.record_id}  {_record_title(row)}")
    return EXIT_OK


def run_domain_show(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        record = service.domain.get(args.entity, args.record_id)
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    if record is None:
        print(f"No {args.entity} with id {args.record_id!r}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps(_record_payload(record), ensure_ascii=False, indent=2))
    else:
        for key, value in _record_payload(record).items():
            print(f"  {key}: {value}")
    return EXIT_OK


def run_domain_update(args: argparse.Namespace) -> int:
    service = _service(args)
    fields = _parse_fields(args.fields)
    try:
        record = service.domain.update(args.entity, args.record_id, **fields)
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    if args.as_json:
        print(json.dumps(_record_payload(record), ensure_ascii=False, indent=2))
    else:
        print(f"Updated {args.entity} {record.record_id}: {_record_title(record)}")
    return EXIT_OK


def run_domain_deactivate(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        record = service.domain.deactivate(args.entity, args.record_id)
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    if args.as_json:
        print(json.dumps({"record_id": record.record_id, "active": record.active},
                         ensure_ascii=False, indent=2))
    else:
        print(f"Deactivated {args.entity} {record.record_id}.")
    return EXIT_OK


def run_audit(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = (
        service.domain.audit_for(args.for_id, limit=args.limit)
        if args.for_id
        else service.domain.audit_tail(limit=args.limit)
    )
    if args.as_json:
        print(json.dumps([_record_payload(row) for row in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("The audit trail is empty.")
        return EXIT_OK
    for row in rows:
        print(f"  {row.created_at}  {row.actor}  {row.action}  {row.entity} {row.entity_id}"
              + (f"  — {row.detail}" if row.detail else ""))
    return EXIT_OK


def _record_title(record: object) -> str:
    for name in ("title", "name", "subject", "number"):
        value = getattr(record, name, "")
        if value:
            return str(value)
    return getattr(record, "record_id", "")


def _record_payload(record: object) -> dict[str, object]:
    import dataclasses

    payload: dict[str, object] = {}
    for spec in dataclasses.fields(record):  # type: ignore[arg-type]
        value = getattr(record, spec.name)
        payload[spec.name] = dict(value) if isinstance(value, dict) else value
    return payload


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
        "move-division": run_move_division,
        "profile": run_profile,
        "add-detail": run_add_detail,
        "remove-detail": run_remove_detail,
        "kinds": run_kinds,
        "add-taxonomy-area": run_add_taxonomy_area,
        "add-taxonomy-discipline": run_add_taxonomy_discipline,
        "rename-taxonomy": run_rename_taxonomy,
        "remove-taxonomy": run_remove_taxonomy,
        "settings": run_settings,
        "backup": run_backup,
        "dashboard": run_dashboard,
        "entities": run_entities,
        "add": run_domain_add,
        "list": run_domain_list,
        "show": run_domain_show,
        "update": run_domain_update,
        "deactivate": run_domain_deactivate,
        "audit": run_audit,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(args)
    except WorkplaceError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    except WorkplaceDomainError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    except Exception as error:  # pragma: no cover - a last-resort guard
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
