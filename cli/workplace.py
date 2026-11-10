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
    python -m cli workplace profiles          # the eleven assistant profiles (§10)
    python -m cli workplace corpus            # rebuild the assistant's corpus
    python -m cli workplace ask "what coating applies to the flowline?" --profile coating
    python -m cli workplace assistant-status
    python -m cli workplace draft-report PRJ-… --kind proposal --with-assistant --save
    python -m cli workplace reports PRJ-…        # a project's reports
    python -m cli workplace approve-report REP-… --by "Dr. Ahmed"
    python -m cli workplace issue-report REP-…   # issue an approved deliverable
    python -m cli workplace deliverables PRJ-…   # the project's issued deliverables
    python -m cli workplace roles                # the roles and their ceilings (§13)
    python -m cli workplace add-user "Dr. Ahmed" --role manager --clearance internal
    python -m cli workplace can approve --user USR-…   # a role + scope decision
    python -m cli workplace refused              # the audited refusals
    python -m cli workplace reopen reports REP-… --note "revise section 3"
    python -m cli workplace assistant-clearance public,internal   # the AI's clearance

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

from core.workplace.knowledge_capture import WorkplaceKnowledgeError
from gui.workplace import (
    WorkplaceDomainError,
    WorkplaceError,
    WorkplaceService,
    reference_kind_label,
)

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

    search = _sub(sub, "search", help="Search every Workplace record kind (the brief's §14).")
    search.add_argument("query", help="The word or phrase to search for.")
    search.add_argument(
        "--kind", dest="kinds", action="append", default=None,
        help="Limit to one record kind (repeatable, e.g. --kind projects).",
    )
    search.add_argument("--limit", type=int, default=20, help="Maximum hits (default 20).")
    search.add_argument(
        "--min-score", type=float, default=0.0,
        help="Drop a hit carrying less than this share of the query (0..1).",
    )

    _sub(sub, "search-kinds", help="The record kinds the search reads.")

    _sub(sub, "knowledge-sources", help="The finished-work source kinds a capture reads.")
    kcand = _sub(sub, "knowledge-candidates", help="Finished sources that yield knowledge.")
    kcand.add_argument("--kind", dest="source_kind", default=None, help="Only this source kind.")
    _sub(sub, "knowledge-status", help="What could still be captured (honest status).")
    klist = _sub(sub, "knowledge", help="The captured institutional knowledge records.")
    klist.add_argument("--project", dest="project_id", default=None, help="Only this project.")
    capture = _sub(sub, "capture-knowledge", help="Capture one finished source as knowledge.")
    capture.add_argument("source_kind", help="The source kind (see `knowledge-sources`).")
    capture.add_argument("source_id", help="The source record id.")
    cap_all = _sub(sub, "capture-all-knowledge", help="Capture every finished source (idempotent).")
    cap_all.add_argument("--kind", dest="source_kind", default=None, help="Only this source kind.")
    cap_proj = _sub(sub, "capture-project-knowledge", help="Capture a project's own finished work.")
    cap_proj.add_argument("project", help="The project id.")

    projects = _sub(sub, "projects", help="List projects with their own real counts.")
    projects.add_argument("--status", default=None, help="Only this project status.")
    projects.add_argument("--division", dest="division_id", default=None, help="Only this division id.")

    project = _sub(sub, "project", help="The full view of one project (references resolved).")
    project.add_argument("record_id", help="The project record id (or its number).")

    people = _sub(sub, "people", help="List personnel (the brief's §8).")
    people.add_argument("--division", dest="division_id", default=None, help="Only this division id.")
    people.add_argument("--discipline", default=None, help="Only this discipline.")
    people.add_argument("--available", action="store_true", help="Only available people.")

    person = _sub(sub, "person", help="One person's profile: qualifications and projects.")
    person.add_argument("record_id", help="The person record id.")

    _sub(sub, "matrix", help="The competency matrix (people × qualifications).")

    select = _sub(sub, "select", help="Rank the people who can cover the required skills.")
    select.add_argument("--skill", dest="skills", action="append", default=[], help="A required skill (repeatable).")
    select.add_argument("--qualification", dest="qualifications", action="append", default=[], help="A required qualification (repeatable).")
    select.add_argument("--division", dest="division_id", default=None, help="Only this division id.")
    select.add_argument("--limit", type=int, default=5, help="At most this many candidates.")

    library = _sub(sub, "library", help="The technical library of standards (the brief's §7).")
    library.add_argument("--body", default=None, help="Only this issuing body (API, ASME, ...).")
    library.add_argument("--kind", dest="reference_kind", default=None, help="Only this reference kind.")
    library.add_argument("--query", default=None, help="A free-text filter.")

    _sub(sub, "library-bodies", help="Every issuing body with how many references it has.")

    _sub(sub, "library-kinds", help="How many references the library holds of each kind.")

    revisions = _sub(sub, "standard-revisions", help="Every revision of one standard, oldest first.")
    revisions.add_argument("body", help="The issuing body.")
    revisions.add_argument("number", help="The standard number.")

    compare = _sub(sub, "compare-standards", help="Compare two revisions of one standard.")
    compare.add_argument("body", help="The issuing body.")
    compare.add_argument("number", help="The standard number.")
    compare.add_argument("--from", dest="from_revision", default="", help="The older revision.")
    compare.add_argument("--to", dest="to_revision", default="", help="The newer revision.")

    add_std = _sub(sub, "add-standard", help="Add a reference to the library (metadata only).")
    add_std.add_argument("--body", required=True, help="The issuing body.")
    add_std.add_argument("--number", required=True, help="The standard number.")
    add_std.add_argument("--title", default="", help="The title.")
    add_std.add_argument("--edition", default="", help="The edition.")
    add_std.add_argument("--revision", default="", help="The revision.")
    add_std.add_argument("--published", dest="published_on", default="", help="The publication date.")
    add_std.add_argument("--kind", dest="reference_kind", default="standard", help="The reference kind.")
    add_std.add_argument("--applicability", default="", help="What it applies to.")
    add_std.add_argument("--source", default="", help="Where the copy came from.")

    documents = _sub(sub, "documents", help="The project document registry (the brief's §9).")
    documents.add_argument("--project", dest="project_id", default=None, help="Only this project id.")
    documents.add_argument("--type", dest="doc_type", default=None, help="Only this document type.")
    documents.add_argument("--query", default=None, help="A free-text filter.")

    document = _sub(sub, "document", help="One document record.")
    document.add_argument("record_id", help="The document record id.")

    doc_hist = _sub(sub, "document-history", help="A document's revision chain.")
    doc_hist.add_argument("record_id", help="The document record id.")

    verify = _sub(sub, "verify-document", help="Re-hash a stored document (integrity check).")
    verify.add_argument("record_id", help="The document record id.")

    doc_text = _sub(sub, "document-text", help="Read a document's text (the project's own reader).")
    doc_text.add_argument("record_id", help="The document record id.")

    register = _sub(sub, "register-document", help="Register a file in the document registry.")
    register.add_argument("path", help="The file to register (it is copied in).")
    register.add_argument("--title", default="", help="The document title.")
    register.add_argument("--type", dest="doc_type", default="other", help="The document type.")
    register.add_argument("--project", dest="project_id", default="", help="The owning project id.")
    register.add_argument("--division", dest="division_id", default="", help="The owning division id.")
    register.add_argument("--revision", default="", help="The revision label.")
    register.add_argument("--confidentiality", default="internal", help="The classification.")

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

    # -- the Workplace assistant (the brief's §10) ------------------------
    profiles = _sub(sub, "profiles", help="The selectable Workplace assistant profiles.")
    profiles.add_argument("--language", choices=("en", "ar"), default="en", help="The label language.")

    ask = _sub(sub, "ask", help="Ask the Workplace assistant (answers cite the records).")
    ask.add_argument("question", help="The question.")
    ask.add_argument("--profile", default="petroleum", help="The assistant profile (see `profiles`).")
    ask.add_argument("--limit", type=int, default=4, help="How many records to cite.")

    corpus = _sub(sub, "corpus", help="Rebuild the assistant's corpus from the records.")
    corpus.add_argument("--refresh", action="store_true", help="Rebuild (the default).")

    _sub(sub, "assistant-status", help="An honest status of the Workplace assistant.")

    # -- reporting and proposal workflows (the brief's §9, §10, §13) ------
    report_kinds = _sub(sub, "report-kinds", help="The report kinds the Workplace drafts.")
    report_kinds.add_argument("--language", choices=("en", "ar"), default="en")

    _sub(sub, "report-statuses", help="The report lifecycle states.")

    draft = _sub(sub, "draft-report", help="Draft a report from the project's own records.")
    draft.add_argument("project", help="The project id.")
    draft.add_argument("--kind", default="technical_report", help="The report kind (see `report-kinds`).")
    draft.add_argument("--title", default="", help="Override the draft title.")
    draft.add_argument("--with-assistant", action="store_true", help="Fill sections from the assistant.")
    draft.add_argument("--profile", default="", help="The assistant profile (with --with-assistant).")
    draft.add_argument("--save", action="store_true", help="Save the draft as a report row.")
    draft.add_argument("--author", default="", help="The author id (with --save).")
    draft.add_argument("--number", default="", help="The report number (with --save).")

    reports = _sub(sub, "reports", help="List a project's reports.")
    reports.add_argument("project", help="The project id.")
    reports.add_argument("--kind", default="", help="Filter by kind.")

    report = _sub(sub, "report", help="Show one report.")
    report.add_argument("report", help="The report id.")

    submit = _sub(sub, "submit-report", help="Submit a draft report for approval.")
    submit.add_argument("report", help="The report id.")

    reject = _sub(sub, "reject-report", help="Reject a submitted report.")
    reject.add_argument("report", help="The report id.")
    reject.add_argument("--note", default="", help="Why it was rejected.")

    approve = _sub(sub, "approve-report", help="Record a person's approval of a report.")
    approve.add_argument("report", help="The report id.")
    approve.add_argument("--by", required=True, dest="approved_by", help="Who approved it (a person).")
    approve.add_argument("--role", default="", help="Their role.")
    approve.add_argument("--note", default="", help="A note.")

    issue = _sub(sub, "issue-report", help="Issue an approved report as a deliverable.")
    issue.add_argument("report", help="The report id.")
    issue.add_argument("--no-document", action="store_true", help="Do not register a deliverable file.")
    issue.add_argument("--issued-on", default="", help="The issue date.")

    supersede = _sub(sub, "supersede-report", help="Mark a report superseded.")
    supersede.add_argument("report", help="The report id.")

    approvals = _sub(sub, "report-approvals", help="The approvals recorded for a report.")
    approvals.add_argument("report", help="The report id.")

    deliverables = _sub(sub, "deliverables", help="A project's issued deliverables.")
    deliverables.add_argument("project", help="The project id.")

    # -- security, permissions, and audit (the brief's §13) ---------------
    _sub(sub, "roles", help="The roles and their permission ceilings.")
    _sub(sub, "permissions", help="The closed set of permission verbs.")

    users = _sub(sub, "users", help="List the Workplace users.")
    users.add_argument("--all", action="store_true", help="Include deactivated users.")

    add_user = _sub(sub, "add-user", help="Create a Workplace user (a role and a clearance).")
    add_user.add_argument("name", help="The user's name.")
    add_user.add_argument("--role", default="viewer", help="The role (see `roles`).")
    add_user.add_argument("--clearance", default="", help="Comma-joined levels the user may read.")
    add_user.add_argument("--divisions", default="", help="Comma-joined division ids (scope).")
    add_user.add_argument("--projects", default="", help="Comma-joined project ids (scope).")
    add_user.add_argument("--permissions", default="", help="Comma-joined per-user grants (narrow only).")
    add_user.add_argument("--email", default="", help="The user's email.")
    add_user.add_argument("--person", default="", help="The linked person id, if any.")

    user = _sub(sub, "user", help="Show one Workplace user.")
    user.add_argument("user", help="The user id.")

    set_role = _sub(sub, "set-role", help="Set a user's role.")
    set_role.add_argument("user", help="The user id.")
    set_role.add_argument("role", help="The new role.")

    set_clearance = _sub(sub, "set-clearance", help="Set the levels a user may read.")
    set_clearance.add_argument("user", help="The user id.")
    set_clearance.add_argument("levels", help="Comma-joined confidentiality levels.")

    set_scope = _sub(sub, "set-scope", help="Set a user's division/project scope.")
    set_scope.add_argument("user", help="The user id.")
    set_scope.add_argument("--divisions", default=None, help="Comma-joined division ids.")
    set_scope.add_argument("--projects", default=None, help="Comma-joined project ids.")

    grant = _sub(sub, "grant", help="Set a user's per-user permission grant (narrows only).")
    grant.add_argument("user", help="The user id.")
    grant.add_argument("permissions", help="Comma-joined permissions.")

    deactivate_user = _sub(sub, "deactivate-user", help="Deactivate a user (soft delete).")
    deactivate_user.add_argument("user", help="The user id.")

    can = _sub(sub, "can", help="Whether the acting user may perform a permission.")
    can.add_argument("permission", help="The permission (see `permissions`).")
    can.add_argument("--user", default="", help="Check as this user id (default: the acting user).")
    can.add_argument("--entity", default="", help="Check scope against this entity.")
    can.add_argument("--record", default="", help="The record id (with --entity).")

    refused = _sub(sub, "refused", help="The most recent refused access decisions.")
    refused.add_argument("--limit", type=int, default=50, help="How many rows (newest first).")

    reopen = _sub(sub, "reopen", help="Reopen an approved record for revision (audited).")
    reopen.add_argument("entity", help="The entity table (e.g. reports).")
    reopen.add_argument("record_id", help="The record id.")
    reopen.add_argument("--note", default="", help="Why it is reopened.")

    member = _sub(sub, "add-member", help="Add a project membership (project-level access).")
    member.add_argument("project", help="The project id.")
    member.add_argument("--person", default="", help="The person id.")
    member.add_argument("--user", default="", help="The user id.")
    member.add_argument("--role", default="member", help="The role on the project.")
    member.add_argument("--organization", default="", help="The member's organization.")

    members = _sub(sub, "members", help="A project's members.")
    members.add_argument("project", help="The project id.")

    clearance = _sub(sub, "assistant-clearance", help="The levels the assistant may read (confidentiality).")
    clearance.add_argument("levels", help="Comma-joined confidentiality levels.")

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


def run_search(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.search(
        args.query, kinds=args.kinds, limit=args.limit, min_score=args.min_score
    )
    if args.as_json:
        print(json.dumps({
            "query": report.query,
            "terms": list(report.terms),
            "count": report.count,
            "searched_kinds": list(report.searched_kinds),
            "hits": [
                {
                    "kind": hit.kind,
                    "kind_label": hit.kind_label,
                    "record_id": hit.record_id,
                    "title": hit.title,
                    "matched": list(hit.matched),
                    "score": hit.score,
                }
                for hit in report.hits
            ],
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(report.describe())
    for hit in report.hits:
        print(f"  [{hit.score:.2f}] {hit.describe()}")
    return EXIT_OK


def run_search_kinds(args: argparse.Namespace) -> int:
    service = _service(args)
    kinds = service.searchable_kinds()
    if args.as_json:
        print(json.dumps(
            [{"kind": kind, "label": label} for kind, label in kinds],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Searchable Workplace record kinds:")
    for kind, label in kinds:
        print(f"  {kind} — {label}")
    return EXIT_OK


def run_knowledge_sources(args: argparse.Namespace) -> int:
    service = _service(args)
    kinds = service.knowledge_source_kinds()
    if args.as_json:
        print(json.dumps(
            [{"kind": kind, "label": label} for kind, label in kinds],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Finished-work source kinds a capture reads:")
    for kind, label in kinds:
        print(f"  {kind} — {label}")
    return EXIT_OK


def run_knowledge_candidates(args: argparse.Namespace) -> int:
    service = _service(args)
    candidates = service.knowledge_candidates(source_kind=args.source_kind)
    if args.as_json:
        print(json.dumps([
            {
                "source_kind": c.source_kind,
                "source_id": c.source_id,
                "title": c.title,
                "kind": c.kind,
                "confidentiality": c.confidentiality,
                "verified": c.verified,
            }
            for c in candidates
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not candidates:
        print("No finished source yields knowledge yet.")
        return EXIT_OK
    print(f"{len(candidates)} finished source(s) yield institutional knowledge:")
    for c in candidates:
        print(f"  {c.describe()}")
    return EXIT_OK


def run_knowledge_status(args: argparse.Namespace) -> int:
    service = _service(args)
    print(service.knowledge_capture_status())
    return EXIT_OK


def run_knowledge(args: argparse.Namespace) -> int:
    service = _service(args)
    records = service.captured_knowledge(project_id=args.project_id)
    if args.as_json:
        print(json.dumps([
            {
                "record_id": r.record_id,
                "title": r.title,
                "kind": r.kind,
                "source": r.source,
                "reference": r.reference,
                "confidentiality": r.confidentiality,
                "verified": r.verified,
                "project_id": r.project_id,
            }
            for r in records
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not records:
        print("No captured knowledge records yet.")
        return EXIT_OK
    print(f"{len(records)} captured knowledge record(s):")
    for r in records:
        flag = " (verified)" if r.verified else ""
        print(f"  [{r.source}] {r.title} ({r.kind}, {r.confidentiality}){flag}")
    return EXIT_OK


def run_capture_knowledge(args: argparse.Namespace) -> int:
    service = _service(args)
    record = service.capture_knowledge(args.source_kind, args.source_id)
    if args.as_json:
        print(json.dumps({
            "record_id": record.record_id, "title": record.title,
            "source": record.source, "reference": record.reference,
            "confidentiality": record.confidentiality, "verified": record.verified,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Captured knowledge: {record.title} ({record.confidentiality}).")
    return EXIT_OK


def _print_capture_report(report, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(report.describe())
    for record in report.captured:
        print(f"  + {record.title}")
    return EXIT_OK


def run_capture_all_knowledge(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.capture_all_knowledge(source_kind=args.source_kind)
    return _print_capture_report(report, as_json=args.as_json)


def run_capture_project_knowledge(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.capture_project_knowledge(args.project)
    return _print_capture_report(report, as_json=args.as_json)


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


def run_projects(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.project_rows(status=args.status, division_id=args.division_id)
    if args.as_json:
        print(json.dumps([
            {
                "record_id": row.project.record_id,
                "number": row.project.number,
                "title": row.project.title,
                "status": row.project.status,
                "client": row.client_name,
                "division": row.division_name,
                "open_findings": row.open_findings,
                "reports_awaiting_approval": row.reports_awaiting_approval,
                "documents": row.documents,
                "assignments": row.assignments,
            }
            for row in rows
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No projects yet. Create one with `workplace add projects number=... title=...`.")
        return EXIT_OK
    for row in rows:
        print(
            f"  {row.project.record_id}  {row.describe()}  "
            f"[client: {row.client_name or '—'}; division: {row.division_name or '—'}; "
            f"findings: {row.open_findings}; docs: {row.documents}; staff: {row.assignments}]"
        )
    return EXIT_OK


def run_project(args: argparse.Namespace) -> int:
    service = _service(args)
    view = service.project_view(args.record_id)
    if view is None:
        by_number = service.projects.project_by_number(args.record_id)
        if by_number is not None:
            view = service.project_view(by_number.record_id)
    if view is None:
        print(f"No project with id or number {args.record_id!r}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        counts = view.counts()
        print(json.dumps({
            "record_id": view.project.record_id,
            "number": view.project.number,
            "title": view.project.title,
            "status": view.project.status,
            "client": view.client.name if view.client else "",
            "contractor": view.contractor.name if view.contractor else "",
            "consultant": view.consultant.name if view.consultant else "",
            "division": view.division.name if view.division else "",
            "manager": view.manager.name if view.manager else "",
            "counts": counts,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(view.describe())
    return EXIT_OK


def run_people(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.personnel_people(
        division_id=args.division_id, discipline=args.discipline, available_only=args.available
    )
    if args.as_json:
        print(json.dumps([
            {
                "record_id": person.record_id,
                "name": person.name,
                "role": person.role,
                "discipline": person.discipline,
                "division_id": person.division_id,
                "available": person.available,
            }
            for person in rows
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No personnel yet. Create one with `workplace add people name=... role=...`.")
        return EXIT_OK
    for person in rows:
        mark = "" if person.available else "  (not available)"
        print(f"  {person.record_id}  {person.name} — {person.role or person.discipline or '—'}{mark}")
    return EXIT_OK


def run_person(args: argparse.Namespace) -> int:
    service = _service(args)
    profile = service.person_view(args.record_id)
    if profile is None:
        print(f"No person with id {args.record_id!r}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps({
            "record_id": profile.person.record_id,
            "name": profile.person.name,
            "role": profile.person.role,
            "discipline": profile.person.discipline,
            "available": profile.person.available,
            "qualifications": [q.title or q.kind for q in profile.qualifications],
            "projects": [f"{p.number} {p.title}".strip() for p in profile.projects],
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(profile.describe())
    return EXIT_OK


def run_matrix(args: argparse.Namespace) -> int:
    service = _service(args)
    matrix = service.competency_matrix()
    if args.as_json:
        print(json.dumps({
            "columns": list(matrix.columns),
            "rows": [{"name": name, "marks": list(marks)} for name, marks in matrix.rows],
            "coverage": matrix.coverage(),
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(matrix.describe())
    return EXIT_OK


def run_select(args: argparse.Namespace) -> int:
    service = _service(args)
    candidates = service.select_personnel(
        skills=tuple(args.skills),
        qualifications=tuple(args.qualifications),
        division_id=args.division_id,
        limit=args.limit,
    )
    if args.as_json:
        print(json.dumps([
            {
                "record_id": c.person.record_id,
                "name": c.person.name,
                "score": round(c.score, 4),
                "matched": list(c.matched),
                "missing": list(c.missing),
            }
            for c in candidates
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not candidates:
        print("No personnel to select from yet.")
        return EXIT_OK
    print("Ranked candidates (score, then name):")
    for candidate in candidates:
        print(f"  {candidate.describe()}")
    return EXIT_OK


def run_library(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.library_standards(
        body=args.body, reference_kind=args.reference_kind, query=args.query
    )
    if args.as_json:
        print(json.dumps([
            {
                "record_id": r.record_id,
                "body": r.body,
                "number": r.number,
                "title": r.title,
                "edition": r.edition,
                "revision": r.revision,
                "published_on": r.published_on,
                "reference_kind": r.reference_kind,
                "applicability": r.applicability,
                "status": r.status,
                "source": r.source,
            }
            for r in rows
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("The library holds no references yet. Add one with `workplace add-standard ...`.")
        return EXIT_OK
    for revision in rows:
        print(f"  {revision.record_id}  {revision.describe()}")
    return EXIT_OK


def run_library_bodies(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.library_bodies()
    if args.as_json:
        print(json.dumps(
            [{"code": code, "title": title, "count": count} for code, title, count in rows],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Issuing bodies (the library's own holdings):")
    for code, title, count in rows:
        print(f"  {code:6}  {count:>3}  {title}")
    return EXIT_OK


def run_library_kinds(args: argparse.Namespace) -> int:
    service = _service(args)
    counts = service.library_counts()
    if args.as_json:
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return EXIT_OK
    print("References by kind:")
    for kind, count in counts.items():
        print(f"  {reference_kind_label(kind):24} {count}")
    return EXIT_OK


def run_standard_revisions(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.standard_revisions(body=args.body, number=args.number)
    if args.as_json:
        print(json.dumps([r.describe() for r in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print(f"No revisions of {args.body} {args.number} in the library.")
        return EXIT_UNAVAILABLE
    for revision in rows:
        print(f"  {revision.record_id}  {revision.describe()}")
    return EXIT_OK


def run_compare_standards(args: argparse.Namespace) -> int:
    service = _service(args)
    comparison = service.compare_standard_revisions(
        body=args.body, number=args.number,
        from_revision=args.from_revision, to_revision=args.to_revision,
    )
    if comparison is None:
        print(f"Nothing to compare for {args.body} {args.number} (fewer than two revisions).")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps({
            "body": comparison.body,
            "number": comparison.number,
            "from": comparison.from_revision.describe(),
            "to": comparison.to_revision.describe(),
            "changed": comparison.changed,
            "changes": comparison.changes(),
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(comparison.describe())
    return EXIT_OK


def run_add_standard(args: argparse.Namespace) -> int:
    service = _service(args)
    revision = service.add_standard(
        body=args.body, number=args.number, title=args.title, edition=args.edition,
        revision=args.revision, published_on=args.published_on,
        reference_kind=args.reference_kind, applicability=args.applicability, source=args.source,
    )
    if args.as_json:
        print(json.dumps({"record_id": revision.record_id, "label": revision.label},
                         ensure_ascii=False, indent=2))
    else:
        print(f"Added {revision.describe()}")
    return EXIT_OK


def run_documents(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.document_rows(
        project_id=args.project_id, doc_type=args.doc_type, query=args.query
    )
    if args.as_json:
        print(json.dumps([
            {
                "record_id": r.record_id,
                "title": r.title,
                "doc_type": r.doc_type,
                "revision": r.revision,
                "project_id": r.project_id,
                "confidentiality": r.confidentiality,
                "content_hash": r.content_hash,
                "supersedes_id": r.supersedes_id,
            }
            for r in rows
        ], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No documents registered yet. Add one with `workplace register-document <path>`.")
        return EXIT_OK
    for record in rows:
        print(f"  {record.record_id}  {record.describe()}")
    return EXIT_OK


def run_document(args: argparse.Namespace) -> int:
    service = _service(args)
    record = service.document_record(args.record_id)
    if record is None:
        print(f"No document with id {args.record_id!r}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps({
            "record_id": record.record_id,
            "title": record.title,
            "doc_type": record.doc_type,
            "revision": record.revision,
            "path": record.path,
            "content_hash": record.content_hash,
            "supersedes_id": record.supersedes_id,
            "confidentiality": record.confidentiality,
            "project_id": record.project_id,
            "division_id": record.division_id,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(record.describe())
    return EXIT_OK


def run_document_history(args: argparse.Namespace) -> int:
    service = _service(args)
    chain = service.document_history(args.record_id)
    if not chain:
        print(f"No document with id {args.record_id!r}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps([r.describe() for r in chain], ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Revision chain (newest first):")
    for record in chain:
        print(f"  {record.record_id}  {record.describe()}")
    return EXIT_OK


def run_verify_document(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        report = service.verify_document(args.record_id)
    except ValueError as error:
        print(str(error))
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps({
            "record_id": report.record_id, "ok": report.ok,
            "recorded_hash": report.recorded_hash, "actual_hash": report.actual_hash,
            "message": report.message,
        }, ensure_ascii=False, indent=2))
    else:
        print(report.describe())
    return EXIT_OK if report.ok else EXIT_UNAVAILABLE


def run_document_text(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        document = service.document_text(args.record_id)
    except ValueError as error:
        print(str(error))
        return EXIT_UNAVAILABLE
    except Exception as error:  # a reader failure (an unsupported file, a missing extra)
        print(f"Could not read the document: {error}")
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps({
            "title": document.title, "kind": document.kind,
            "chars": len(document.text), "words": document.word_count,
        }, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(document.text)
    return EXIT_OK


def run_register_document(args: argparse.Namespace) -> int:
    service = _service(args)
    record = service.register_document_file(
        Path(args.path),
        title=args.title, doc_type=args.doc_type, project_id=args.project_id,
        division_id=args.division_id, revision=args.revision,
        confidentiality=args.confidentiality,
    )
    if args.as_json:
        print(json.dumps({"record_id": record.record_id, "content_hash": record.content_hash},
                         ensure_ascii=False, indent=2))
    else:
        print(f"Registered {record.describe()}  [{record.record_id}]")
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


def run_profiles(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.assistant_profiles(language=args.language)
    if args.as_json:
        print(json.dumps(
            [{"key": key, "name": name, "specialization": spec} for key, name, spec in rows],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Workplace assistant profiles:")
    for key, name, spec in rows:
        print(f"  {key:20}  {name} -- {spec}")
    return EXIT_OK


def run_ask(args: argparse.Namespace) -> int:
    service = _service(args)
    service.refresh_assistant()
    answer = service.ask_workplace(args.question, profile=args.profile, limit=args.limit)
    if args.as_json:
        print(json.dumps({"profile": args.profile, "answer": answer}, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(answer)
    return EXIT_OK


def run_corpus(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.refresh_assistant()
    if args.as_json:
        print(json.dumps(
            {
                "documents": report.documents,
                "by_kind": report.by_kind,
                "withheld": report.withheld,
                "unreadable": report.unreadable,
            },
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print(f"Rebuilt the Workplace assistant's corpus: {report.describe()}.")
    for kind, count in sorted(report.by_kind.items()):
        print(f"  {kind:16} {count}")
    return EXIT_OK


def run_assistant_status(args: argparse.Namespace) -> int:
    service = _service(args)
    service.refresh_assistant()
    status = service.assistant_status()
    if args.as_json:
        print(json.dumps({"status": status}, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(status)
    return EXIT_OK


# -- reporting and proposal workflows (the brief's §9, §10, §13) ----------


def run_report_kinds(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.report_kinds(language=args.language)
    if args.as_json:
        print(json.dumps([{"key": k, "label": v} for k, v in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Report kinds the Workplace drafts:")
    for key, label in rows:
        print(f"  {key:18}  {label}")
    return EXIT_OK


def run_report_statuses(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.report_statuses()
    if args.as_json:
        print(json.dumps([{"key": k, "label": v} for k, v in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Report lifecycle states:")
    for key, label in rows:
        print(f"  {key:18}  {label}")
    return EXIT_OK


def run_draft_report(args: argparse.Namespace) -> int:
    service = _service(args)
    draft = service.draft_report(
        args.project, kind=args.kind, title=args.title,
        use_assistant=args.with_assistant, profile=args.profile,
    )
    saved = ""
    if args.save:
        report = service.save_report_draft(
            args.project, draft, author_id=args.author, number=args.number
        )
        saved = report.record_id
    if args.as_json:
        print(json.dumps(
            {
                "kind": draft.kind,
                "title": draft.title,
                "project_id": draft.project_id,
                "used_assistant": draft.used_assistant,
                "citations": list(draft.citations),
                "sections": [
                    {"title": s.title, "ai": s.ai, "lines": list(s.lines)} for s in draft.sections
                ],
                "text": draft.to_text(),
                "saved_report_id": saved,
            },
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print(draft.to_text())
    if saved:
        print(f"\nSaved as report {saved}.")
    return EXIT_OK


def run_reports(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.project_reports(args.project, kind=args.kind)
    if args.as_json:
        print(json.dumps([_report_payload(r) for r in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Reports for project {args.project}: {len(rows)}")
    for report in rows:
        number = report.number or report.record_id
        print(f"  {number:16}  {report.title} [{report.status}]")
    return EXIT_OK


def _report_payload(report) -> dict[str, object]:
    return {
        "record_id": report.record_id,
        "project_id": report.project_id,
        "kind": report.kind,
        "number": report.number,
        "title": report.title,
        "status": report.status,
        "issued_on": report.issued_on,
        "document_id": report.document_id,
    }


def run_report(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.report_record(args.report)
    if report is None:
        print(f"No report with id {args.report!r}.")
        return EXIT_REFUSED
    if args.as_json:
        payload = _report_payload(report)
        payload["body"] = report.body
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"{report.number or report.record_id} — {report.title} [{report.status}]")
    if report.body:
        print()
        print(report.body)
    return EXIT_OK


def run_submit_report(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.submit_report(args.report)
    print(f"Report {report.record_id} is now {report.status}.")
    return EXIT_OK


def run_reject_report(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.reject_report(args.report, note=args.note)
    print(f"Report {report.record_id} is now {report.status}.")
    return EXIT_OK


def run_approve_report(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.approve_report(
        args.report, approved_by=args.approved_by, role=args.role, note=args.note
    )
    print(f"Report {report.record_id} approved by {args.approved_by}; status {report.status}.")
    return EXIT_OK


def run_issue_report(args: argparse.Namespace) -> int:
    service = _service(args)
    deliverable = service.issue_report(
        args.report, register_document=not args.no_document, issued_on=args.issued_on
    )
    if args.as_json:
        print(json.dumps(
            {
                "report_id": deliverable.report.record_id,
                "status": deliverable.report.status,
                "document_id": deliverable.document_id,
                "document_title": deliverable.document_title,
            },
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print(f"Issued: {deliverable.describe()}")
    return EXIT_OK


def run_supersede_report(args: argparse.Namespace) -> int:
    service = _service(args)
    report = service.supersede_report(args.report)
    print(f"Report {report.record_id} is now {report.status}.")
    return EXIT_OK


def run_report_approvals(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.report_approvals(args.report)
    if args.as_json:
        print(json.dumps(
            [
                {
                    "record_id": a.record_id,
                    "decision": a.decision,
                    "approved_by": a.approved_by,
                    "role": a.role,
                    "note": a.note,
                    "decided_on": a.decided_on,
                }
                for a in rows
            ],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print(f"Approvals for report {args.report}: {len(rows)}")
    for approval in rows:
        print(f"  {approval.decided_on}  {approval.decision} by {approval.approved_by}")
    return EXIT_OK


def run_roles(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = [(role, service.role_permissions(role)) for role in service.role_names()]
    if args.as_json:
        print(json.dumps(
            [{"role": role, "permissions": perms} for role, perms in rows],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print("Workplace roles (the permission ceiling of each):")
    for role, perms in rows:
        print(f"  {role:16}  {', '.join(perms)}")
    return EXIT_OK


def run_permissions(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.permission_names()
    if args.as_json:
        print(json.dumps(list(rows), ensure_ascii=False, indent=2))
        return EXIT_OK
    print("Permissions:")
    for name in rows:
        print(f"  {name}")
    return EXIT_OK


def _user_payload(user: object) -> dict[str, object]:
    return {
        "record_id": user.record_id,  # type: ignore[attr-defined]
        "name": user.name,  # type: ignore[attr-defined]
        "role": user.role,  # type: ignore[attr-defined]
        "clearance": user.clearance,  # type: ignore[attr-defined]
        "division_ids": user.division_ids,  # type: ignore[attr-defined]
        "project_ids": user.project_ids,  # type: ignore[attr-defined]
        "permissions": user.permissions,  # type: ignore[attr-defined]
        "active": user.active,  # type: ignore[attr-defined]
    }


def run_users(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.users(include_inactive=args.all)
    if args.as_json:
        print(json.dumps([_user_payload(u) for u in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No Workplace users yet. Run `python -m cli workplace add-user` to add one.")
        return EXIT_OK
    for user in rows:
        mark = "" if user.active else "  (inactive)"
        print(
            f"  {user.record_id}  {user.name}  [{user.role}]  clearance={user.clearance or '-'}"
            f"{mark}"
        )
    return EXIT_OK


def run_user(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.user(args.user)
    if user is None:
        print(f"No Workplace user with id {args.user!r}.", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if args.as_json:
        print(json.dumps(_user_payload(user), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"{user.record_id}  {user.name}")
    print(f"  role:        {user.role}")
    print(f"  clearance:   {user.clearance or '-'}")
    print(f"  divisions:   {user.division_ids or '-'}")
    print(f"  projects:    {user.project_ids or '-'}")
    print(f"  permissions: {user.permissions or '(the role\'s own set)'}")
    return EXIT_OK


def run_add_user(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.create_user(
        name=args.name,
        role=args.role,
        clearance=args.clearance or None,
        division_ids=args.divisions or None,
        project_ids=args.projects or None,
        permissions=args.permissions or None,
        email=args.email,
        person_id=args.person,
    )
    if args.as_json:
        print(json.dumps(_user_payload(user), ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"Created user {user.record_id}: {user.name} [{user.role}].")
    return EXIT_OK


def run_set_role(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.set_user_role(args.user, args.role)
    print(f"{user.record_id} is now role {user.role}.")
    return EXIT_OK


def run_set_clearance(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.set_user_clearance(args.user, args.levels)
    print(f"{user.record_id} may read: {user.clearance}.")
    return EXIT_OK


def run_set_scope(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.set_user_scope(
        args.user,
        division_ids=args.divisions if args.divisions is not None else None,
        project_ids=args.projects if args.projects is not None else None,
    )
    print(f"{user.record_id} scope updated (divisions={user.division_ids or '-'}, projects={user.project_ids or '-'}).")
    return EXIT_OK


def run_grant(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.grant_permissions(args.user, args.permissions)
    print(f"{user.record_id} grants set to: {user.permissions or '(the role\'s own set)'}.")
    return EXIT_OK


def run_deactivate_user(args: argparse.Namespace) -> int:
    service = _service(args)
    user = service.deactivate_user(args.user)
    print(f"Deactivated user {user.record_id}.")
    return EXIT_OK


def run_can(args: argparse.Namespace) -> int:
    service = _service(args)
    if args.user:
        service.set_current_user(args.user)
    record = None
    if args.entity and args.record:
        record = service.repository.get_domain(args.entity, args.record)
    decision = service.check_access(args.permission, entity=args.entity, record=record)
    if args.as_json:
        print(json.dumps({"allowed": decision.allowed, "reason": decision.reason}, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(("allowed" if decision.allowed else "refused") + (f": {decision.reason}" if decision.reason else ""))
    return EXIT_OK if decision.allowed else EXIT_REFUSED


def run_refused(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.refused_decisions(limit=args.limit)
    if args.as_json:
        print(json.dumps([_record_payload(row) for row in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print("No refused decisions recorded.")
        return EXIT_OK
    for row in rows:
        print(f"  {row.created_at}  {row.actor}  {row.detail}")
    return EXIT_OK


def run_reopen(args: argparse.Namespace) -> int:
    service = _service(args)
    service.reopen_record(args.entity, args.record_id, note=args.note)
    print(f"Reopened {args.entity} {args.record_id} for revision (recorded on the audit trail).")
    return EXIT_OK


def run_add_member(args: argparse.Namespace) -> int:
    service = _service(args)
    member = service.assign_project_member(
        args.project,
        person_id=args.person,
        user_id=args.user,
        role=args.role,
        organization=args.organization,
    )
    print(f"Added member {member.record_id} to project {args.project}.")  # type: ignore[attr-defined]
    return EXIT_OK


def run_members(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.project_members(args.project)
    if args.as_json:
        print(json.dumps([_record_payload(row) for row in rows], ensure_ascii=False, indent=2))
        return EXIT_OK
    if not rows:
        print(f"Project {args.project} has no members recorded.")
        return EXIT_OK
    for row in rows:
        print(f"  {row.record_id}  {row.role}  person={row.person_id or '-'} user={row.user_id or '-'}")
    return EXIT_OK


def run_assistant_clearance(args: argparse.Namespace) -> int:
    service = _service(args)
    service.set_assistant_clearance(args.levels)
    print(f"The assistant may now read: {args.levels}.")
    return EXIT_OK


def run_deliverables(args: argparse.Namespace) -> int:
    service = _service(args)
    rows = service.project_deliverables(args.project)
    if args.as_json:
        print(json.dumps(
            [
                {
                    "report_id": d.report.record_id,
                    "title": d.report.title,
                    "document_id": d.document_id,
                    "document_title": d.document_title,
                }
                for d in rows
            ],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    print(f"Deliverables for project {args.project}: {len(rows)}")
    for deliverable in rows:
        print(f"  {deliverable.describe()}")
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
        "search": run_search,
        "search-kinds": run_search_kinds,
        "knowledge-sources": run_knowledge_sources,
        "knowledge-candidates": run_knowledge_candidates,
        "knowledge-status": run_knowledge_status,
        "knowledge": run_knowledge,
        "capture-knowledge": run_capture_knowledge,
        "capture-all-knowledge": run_capture_all_knowledge,
        "capture-project-knowledge": run_capture_project_knowledge,
        "projects": run_projects,
        "project": run_project,
        "people": run_people,
        "person": run_person,
        "matrix": run_matrix,
        "select": run_select,
        "library": run_library,
        "library-bodies": run_library_bodies,
        "library-kinds": run_library_kinds,
        "standard-revisions": run_standard_revisions,
        "compare-standards": run_compare_standards,
        "add-standard": run_add_standard,
        "documents": run_documents,
        "document": run_document,
        "document-history": run_document_history,
        "verify-document": run_verify_document,
        "document-text": run_document_text,
        "register-document": run_register_document,
        "add": run_domain_add,
        "list": run_domain_list,
        "show": run_domain_show,
        "update": run_domain_update,
        "deactivate": run_domain_deactivate,
        "audit": run_audit,
        "profiles": run_profiles,
        "ask": run_ask,
        "corpus": run_corpus,
        "assistant-status": run_assistant_status,
        "report-kinds": run_report_kinds,
        "report-statuses": run_report_statuses,
        "draft-report": run_draft_report,
        "reports": run_reports,
        "report": run_report,
        "submit-report": run_submit_report,
        "reject-report": run_reject_report,
        "approve-report": run_approve_report,
        "issue-report": run_issue_report,
        "supersede-report": run_supersede_report,
        "report-approvals": run_report_approvals,
        "deliverables": run_deliverables,
        "roles": run_roles,
        "permissions": run_permissions,
        "users": run_users,
        "user": run_user,
        "add-user": run_add_user,
        "set-role": run_set_role,
        "set-clearance": run_set_clearance,
        "set-scope": run_set_scope,
        "grant": run_grant,
        "deactivate-user": run_deactivate_user,
        "can": run_can,
        "refused": run_refused,
        "reopen": run_reopen,
        "add-member": run_add_member,
        "members": run_members,
        "assistant-clearance": run_assistant_clearance,
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
    except WorkplaceKnowledgeError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    except ValueError as error:
        print(f"Workplace refused the operation: {error}")
        return EXIT_REFUSED
    except Exception as error:  # pragma: no cover - a last-resort guard
        print(f"Unexpected failure: {error}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
