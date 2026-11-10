"""The held-out gate for the Workplace Phase 10 (search, optimization, docs).

The brief's §14 asks for *advanced search* over the Workplace, and §15 asks that
each phase be measured rather than asserted. This module is that measurement: a
catalogue of hand-written expectations, run against the **real** search engine and
the **real** repository, each with its own negative control, so a search that
fabricated a hit, searched nothing, ignored a narrowing, or re-introduced the
listing's N+1 reads fails the evaluation instead of passing silently.

The judgements are:

* a query finds a record in **every** kind that carries it (one search, many
  kinds -- the point of the feature);
* a blank query returns **nothing** (not every row);
* narrowing to one kind returns only that kind;
* the strongest match ranks first;
* a query naming nothing returns no hit;
* the search adds **no** table (it is a view over the existing schema);
* the clearance filter withholds a classified record from an uncleared reader;
* the project listing's per-project counts are **correct** and gathered with a
  bounded number of reads (the N+1 fix).

``workplace_phase10_accuracy`` is the share the real behaviour satisfies; with no
repository it is ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from core.workplace.database import WorkplaceDatabase
from core.workplace.records import DOMAIN_RECORDS
from core.workplace.repository import WorkplaceRepository
from core.workplace.search import search_workplace


@dataclass(frozen=True)
class Phase10Judgement:
    """One expectation the real search/listing must satisfy, and why it exists."""

    key: str
    description: str


#: The held-out catalogue. Each key is checked by :func:`_run_judgement`; a key
#: with no check is a failure, so the catalogue and the checks cannot drift.
PHASE10_JUDGEMENTS: tuple[Phase10Judgement, ...] = (
    Phase10Judgement("finds-across-kinds", "one query reaches every kind that carries it"),
    Phase10Judgement("blank-returns-nothing", "a blank query returns no rows (negative control)"),
    Phase10Judgement("kind-narrows", "a kind filter returns only that kind"),
    Phase10Judgement("strongest-first", "the row carrying the whole query ranks first"),
    Phase10Judgement("unknown-query-empty", "a query naming nothing yields no hit (negative control)"),
    Phase10Judgement("no-second-index", "the search adds no table to the schema"),
    Phase10Judgement("clearance-withholds", "a classified record is withheld from an uncleared reader"),
    Phase10Judgement("listing-counts-correct", "the project listing's per-project counts are right"),
    Phase10Judgement("listing-is-bounded", "the listing does not read a table once per project"),
)


def _repository(tmp: Path) -> WorkplaceRepository:
    db = WorkplaceDatabase(tmp / "workplace.db")
    repository = WorkplaceRepository(db)
    return repository


def _seed_rows(repository: WorkplaceRepository) -> None:
    """A small, known dataset: the same term across several kinds."""
    repository.create_domain(
        "clients", name="Petrojet", kind="contractor", summary="Welding and fabrication"
    )
    repository.create_domain(
        "projects", number="PRJ-1", title="Pipeline welding", scope="Welding inspection"
    )
    repository.create_domain(
        "projects", number="PRJ-2", title="Coating works", scope="Coating application"
    )
    repository.create_domain("people", name="Ahmed", role="Welding Inspector", discipline="welding")
    repository.create_domain(
        "standards", body="API", number="1104", title="Welding of pipelines"
    )
    repository.create_domain("findings", number="NCR-1", title="Weld porosity", status="open")


def _run_judgement(repository: WorkplaceRepository, key: str) -> bool:
    """Run one judgement against the real search/listing; return whether it held."""
    if key == "finds-across-kinds":
        report = search_workplace(repository, "welding")
        kinds = {hit.kind for hit in report.hits}
        return {"clients", "projects", "people", "standards"} <= kinds
    if key == "blank-returns-nothing":
        return search_workplace(repository, "   ").count == 0
    if key == "kind-narrows":
        report = search_workplace(repository, "welding", kinds=["people"])
        return report.count >= 1 and all(hit.kind == "people" for hit in report.hits)
    if key == "strongest-first":
        # "pipeline welding" is carried whole by the project's title+scope and only
        # partly by the standard's title, so the project must rank above it.
        report = search_workplace(repository, "pipeline welding")
        assert report.hits, "expected at least one hit"
        return report.hits[0].kind == "projects" and report.hits[0].score == 1.0
    if key == "unknown-query-empty":
        return search_workplace(repository, "zzzznothingzzzz").count == 0
    if key == "no-second-index":
        # The search reads the repository's existing tables; a new table would mean
        # a parallel index, which the brief's §11 forbids.
        from core.workplace.schema import TABLES

        return all(kind in TABLES for kind in DOMAIN_RECORDS) and "search_index" not in TABLES
    if key == "clearance-withholds":
        from core.workplace.domain import WorkplaceDomain
        from core.workplace.security import SecurityManager

        manager = SecurityManager(repository, WorkplaceDomain(repository))
        doc = repository.create_domain(
            "workplace_documents", title="Secret welding record", confidentiality="restricted"
        )
        user = manager.create_user(name="U", role="engineer", clearance="public,internal")
        return not manager.check_clearance(user, doc).allowed
    if key == "listing-counts-correct":
        from core.workplace.projects import ProjectManager

        project = repository.list_domain("projects", where="number = ?", params=("PRJ-1",))[0]
        repository.create_domain("findings", project_id=project.record_id, title="F1", status="open")
        repository.create_domain("findings", project_id=project.record_id, title="F2", status="closed")
        repository.create_domain("reports", project_id=project.record_id, title="R1", status="draft")
        rows = {row.project.number: row for row in ProjectManager(repository).rows()}
        row = rows["PRJ-1"]
        return row.open_findings == 1 and row.reports_awaiting_approval == 1 and row.documents == 0
    if key == "listing-is-bounded":
        from core.workplace.projects import ProjectManager

        # Count the reads the listing issues: with N projects and the fix, the
        # supporting tables are read a fixed number of times, not once per project.
        reads: list[str] = []
        original = repository.list_domain

        def counting(table: str, **kwargs):  # type: ignore[no-untyped-def]
            reads.append(table)
            return original(table, **kwargs)

        repository.list_domain = counting  # type: ignore[method-assign]
        try:
            ProjectManager(repository).rows()
        finally:
            repository.list_domain = original  # type: ignore[method-assign]
        per_table = {table: reads.count(table) for table in set(reads)}
        # Every supporting table is read exactly once regardless of project count.
        return all(count == 1 for count in per_table.values()) and len(per_table) >= 5
    raise KeyError(key)


def workplace_phase10_accuracy(
    judgements: tuple[Phase10Judgement, ...] = PHASE10_JUDGEMENTS,
) -> float:
    """The share of the held-out catalogue the real search/listing satisfies."""
    if not judgements:
        return 0.0
    passed = 0
    with TemporaryDirectory() as tmp:
        repository = _repository(Path(tmp))
        _seed_rows(repository)
        for judgement in judgements:
            try:
                if _run_judgement(repository, judgement.key):
                    passed += 1
            except Exception:  # a check that raises is a failure, never a pass
                pass
    return passed / len(judgements)


__all__ = [
    "PHASE10_JUDGEMENTS",
    "Phase10Judgement",
    "workplace_phase10_accuracy",
]
