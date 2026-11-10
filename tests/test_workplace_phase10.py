"""Tests for the Workplace Phase 10: search, optimization, and documentation (§14).

Phase 10 completes the enterprise module: one **advanced search** over every
Workplace record kind, a performance fix to the project listing (the N+1 reads
removed), and the documentation. Everything here runs on the real service, the
real search engine, and the real repository -- no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.phase10_gate import PHASE10_JUDGEMENTS, workplace_phase10_accuracy
from core.workplace.search import SearchHit, SearchReport, searchable_kinds
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


def _seed(service: WorkplaceService) -> None:
    service.domain.create("clients", name="Petrojet", kind="contractor", summary="Welding works")
    service.domain.create("projects", number="PRJ-1", title="Pipeline welding", scope="Welding")
    service.domain.create("projects", number="PRJ-2", title="Coating works", scope="Coating")
    service.domain.create("people", name="Ahmed", role="Welding Inspector", discipline="welding")
    service.domain.create("standards", body="API", number="1104", title="Welding of pipelines")
    service.domain.create("findings", number="NCR-1", title="Weld porosity", status="open")


# -- the search engine ----------------------------------------------------


def test_the_search_reads_every_kind_that_carries_the_query(service: WorkplaceService):
    _seed(service)
    report = service.search("welding")
    kinds = {hit.kind for hit in report.hits}
    assert {"clients", "projects", "people", "standards"} <= kinds
    assert report.count == len(report.hits)


def test_a_blank_query_returns_nothing(service: WorkplaceService):
    _seed(service)
    report = service.search("   ")
    assert report.count == 0 and report.empty


def test_a_kind_filter_narrows_to_that_kind(service: WorkplaceService):
    _seed(service)
    report = service.search("welding", kinds=["people"])
    assert report.count >= 1
    assert all(hit.kind == "people" for hit in report.hits)
    assert report.searched_kinds == ("people",)


def test_the_strongest_match_ranks_first(service: WorkplaceService):
    _seed(service)
    report = service.search("pipeline welding")
    assert report.hits[0].kind == "projects"
    assert report.hits[0].score == 1.0


def test_a_query_naming_nothing_returns_no_hit(service: WorkplaceService):
    _seed(service)
    assert service.search("zzznothingzzz").count == 0


def test_a_hit_names_the_record_and_what_matched(service: WorkplaceService):
    _seed(service)
    hit = service.search("porosity").hits[0]
    assert isinstance(hit, SearchHit)
    assert hit.record_id and hit.title
    assert hit.matched  # the field(s) that carried the term
    assert "matched" in hit.describe()


def test_the_report_describes_itself(service: WorkplaceService):
    _seed(service)
    report = service.search("welding")
    assert isinstance(report, SearchReport)
    assert "welding" in report.describe()


def test_searchable_kinds_lists_the_domain_kinds():
    kinds = dict(searchable_kinds())
    assert "projects" in kinds and "people" in kinds and "standards" in kinds


def test_the_service_exposes_the_kinds_and_a_record(service: WorkplaceService):
    _seed(service)
    assert ("projects", "Project") in service.searchable_kinds()
    record = service.record("clients", service.search("Petrojet").hits[0].record_id)
    assert record is not None
    assert service.record("clients", "no-such-id") is None


# -- the project-listing performance fix (no N+1) -------------------------


def test_the_project_listing_counts_are_correct(service: WorkplaceService):
    _seed(service)
    project = service.repository.list_domain("projects", where="number = ?", params=("PRJ-1",))[0]
    service.domain.create("findings", project_id=project.record_id, title="F1", status="open")
    service.domain.create("findings", project_id=project.record_id, title="F2", status="closed")
    service.domain.create("reports", project_id=project.record_id, title="R1", status="draft")
    row = next(r for r in service.project_rows() if r.project.number == "PRJ-1")
    assert row.open_findings == 1
    assert row.reports_awaiting_approval == 1


def test_the_project_listing_does_not_read_a_table_per_project(service: WorkplaceService):
    """The N+1 fix: each supporting table is read once, not once per project."""
    for index in range(6):
        service.domain.create("projects", number=f"PRJ-{index}", title=f"Project {index}")
    reads: list[str] = []
    original = service.repository.list_domain

    def counting(table: str, **kwargs):  # type: ignore[no-untyped-def]
        reads.append(table)
        return original(table, **kwargs)

    service.repository.list_domain = counting  # type: ignore[method-assign]
    try:
        service.project_rows()
    finally:
        service.repository.list_domain = original  # type: ignore[method-assign]
    per_table = {table: reads.count(table) for table in set(reads)}
    assert all(count == 1 for count in per_table.values()), per_table


# -- the held-out gate ----------------------------------------------------


def test_the_gate_is_green_on_the_real_search():
    assert workplace_phase10_accuracy() == 1.0


def test_the_gate_is_zero_with_no_judgements():
    assert workplace_phase10_accuracy(()) == 0.0


def test_the_catalogue_has_negative_controls():
    keys = {judgement.key for judgement in PHASE10_JUDGEMENTS}
    assert "blank-returns-nothing" in keys
    assert "unknown-query-empty" in keys
    assert "no-second-index" in keys


# -- the CLI --------------------------------------------------------------


def test_the_cli_searches_and_prints_hits(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _seed(service)
    assert workplace_main(["--data-root", str(tmp_path), "search", "welding"]) == 0
    out = capsys.readouterr().out
    assert "welding" in out
    assert "Pipeline welding" in out


def test_the_cli_search_json_is_machine_readable(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _seed(service)
    assert workplace_main(["--data-root", str(tmp_path), "search", "welding", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "welding"
    assert payload["count"] >= 1
    assert {"kind", "record_id", "title", "score"} <= set(payload["hits"][0])


def test_the_cli_narrows_by_kind(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _seed(service)
    assert workplace_main(
        ["--data-root", str(tmp_path), "search", "welding", "--kind", "people", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(hit["kind"] == "people" for hit in payload["hits"])


def test_the_cli_lists_searchable_kinds(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "search-kinds"]) == 0
    assert "projects" in capsys.readouterr().out


def test_the_cli_blank_query_is_empty(tmp_path: Path, capsys):
    WorkplaceService(root=tmp_path)
    assert workplace_main(["--data-root", str(tmp_path), "search", "   ", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0
