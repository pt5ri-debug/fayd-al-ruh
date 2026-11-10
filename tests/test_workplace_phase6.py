"""Tests for the Workplace Phase 6: the technical library and document management.

Phase 6 adds the two halves the brief's §7 and §9 ask for -- a standards library
that distinguishes an official standard from an internal procedure, a project
specification, and technical guidance, and a document registry that keeps a
document's revision chain, verifies its integrity, and reads it through the
project's *existing* reader. Everything is exercised on the real service, the
real database, and the real repository: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.documents import DocumentManager, content_hash, document_type_label
from core.workplace.library import (
    REFERENCE_KINDS,
    STANDARD_BODIES,
    TechnicalLibrary,
    reference_kind_label,
)
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


@pytest.fixture()
def domain(service: WorkplaceService):
    return service.domain


# -- the technical library (the brief's §7) -------------------------------


def test_reference_kinds_are_the_four_the_brief_names():
    assert REFERENCE_KINDS == (
        "standard",
        "internal_procedure",
        "project_specification",
        "guidance",
    )
    for kind in REFERENCE_KINDS:
        assert reference_kind_label(kind)
        assert reference_kind_label(kind, language="ar")


def test_standard_bodies_are_the_catalogue_the_brief_lists():
    codes = {code for code, _title in STANDARD_BODIES}
    assert {"API", "ASME", "ASTM", "ISO", "AWS", "AMPP", "DNV", "EN", "IEC"} <= codes


def test_add_standard_records_metadata_not_requirements(service):
    revision = service.add_standard(
        body="API", number="1104", title="Welding of Pipelines", edition="22nd",
        revision="2021", published_on="2021-09-01", reference_kind="standard",
        applicability="pipeline welding", source="institute library",
    )
    assert revision.body == "API"
    assert revision.number == "1104"
    assert revision.reference_kind == "standard"
    # The library holds metadata; it never reproduces the standard's text.
    assert revision.label == "API 1104 (22nd)"
    assert "requirement" not in revision.describe().lower()


def test_add_standard_refuses_an_unknown_reference_kind(service):
    with pytest.raises(ValueError):
        service.add_standard(body="API", number="1104", reference_kind="not_a_kind")


def test_add_standard_needs_a_body_or_a_number(service):
    with pytest.raises(ValueError):
        service.add_standard(body="", number="")


def test_library_filters_by_body_kind_and_query(service):
    service.add_standard(body="API", number="1104", title="Welding of Pipelines")
    service.add_standard(body="ISO", number="9001", title="Quality management systems")
    service.add_standard(
        body="EPRI", number="TSTC-001", title="Coating method statement",
        reference_kind="internal_procedure",
    )
    assert len(service.library_standards()) == 3
    assert [r.number for r in service.library_standards(body="API")] == ["1104"]
    assert [r.number for r in service.library_standards(reference_kind="internal_procedure")] == ["TSTC-001"]
    assert [r.number for r in service.library_standards(query="quality")] == ["9001"]


def test_library_distinguishes_the_four_reference_kinds(service):
    service.add_standard(body="API", number="1104", reference_kind="standard")
    service.add_standard(body="EPRI", number="INT-1", reference_kind="internal_procedure")
    service.add_standard(body="GUPCO", number="PS-1", reference_kind="project_specification")
    service.add_standard(body="EPRI", number="GD-1", reference_kind="guidance")
    counts = service.library_counts()
    assert counts == {
        "standard": 1, "internal_procedure": 1,
        "project_specification": 1, "guidance": 1,
    }


def test_library_bodies_include_the_shipped_catalogue_with_zero_counts(service):
    service.add_standard(body="API", number="1104")
    bodies = {code: count for code, _title, count in service.library_bodies()}
    assert bodies["API"] == 1
    # A body the library holds nothing of is still offered, honestly with 0.
    assert bodies.get("DNV", 0) == 0


def test_revisions_are_ordered_and_a_new_revision_keeps_the_original(service):
    first = service.add_standard(
        body="API", number="1104", title="Welding of Pipelines", edition="21st",
        published_on="2013-01-01",
    )
    second = service.record_standard_revision(
        first.record_id, edition="22nd", published_on="2021-09-01", revision="2021",
    )
    chain = service.standard_revisions(body="API", number="1104")
    assert [r.record_id for r in chain] == [first.record_id, second.record_id]
    # The original row is untouched -- a revision is a new record.
    original = service.library.standard(first.record_id)
    assert original is not None and original.edition == "21st"
    assert service.library.current(body="API", number="1104").record_id == second.record_id


def test_compare_revisions_reports_what_changed(service):
    first = service.add_standard(
        body="API", number="1104", title="Welding of Pipelines", edition="21st",
        published_on="2013-01-01",
    )
    service.record_standard_revision(
        first.record_id, edition="22nd", published_on="2021-09-01",
    )
    comparison = service.compare_standard_revisions(body="API", number="1104")
    assert comparison is not None
    assert comparison.changed
    fields = {row[0] for row in comparison.changes()}
    assert "edition" in fields and "published_on" in fields
    assert "22nd" in comparison.describe()


def test_compare_revisions_is_honest_when_there_is_nothing_to_compare(service):
    service.add_standard(body="API", number="1104")
    assert service.compare_standard_revisions(body="API", number="1104") is None


def test_library_is_read_only_without_a_domain(service):
    library = TechnicalLibrary(service.repository)
    with pytest.raises(RuntimeError):
        library.add_standard(body="API", number="1104")


# -- the document registry (the brief's §9) -------------------------------


def test_document_types_are_labelled(service):
    for _key, label in service.document_types():
        assert label
    assert "Nonconformance" in document_type_label("ncr")


def test_register_file_copies_it_in_and_hashes_it(tmp_path: Path, service):
    source = tmp_path / "ms.txt"
    source.write_text("Method statement for coating.\n", encoding="utf-8")
    record = service.register_document_file(
        source, title="Coating method statement", doc_type="method_statement",
    )
    assert record.title == "Coating method statement"
    assert record.content_hash == content_hash(source.read_bytes())
    # The file is copied into the Workplace's own folder, not referenced in place.
    assert Path(record.path).exists()
    assert Path(record.path) != source


def test_register_file_refuses_a_missing_file(service):
    with pytest.raises(ValueError):
        service.register_document_file("/no/such/file.txt")


def test_register_file_refuses_an_unknown_type(tmp_path: Path, service):
    source = tmp_path / "x.txt"
    source.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        service.register_document_file(source, doc_type="not_a_type")


def test_register_text_writes_a_file_the_reader_can_read(service):
    record = service.register_document_text(
        "Coating procedure: surface preparation to Sa 2.5.",
        title="Coating procedure", doc_type="coating_procedure",
    )
    text = service.document_text(record.record_id)
    assert "Sa 2.5" in text.text


def test_document_text_is_read_through_the_project_reader(tmp_path: Path, service):
    source = tmp_path / "report.md"
    source.write_text("# Report\n\nPorosity measured on a core sample.\n", encoding="utf-8")
    record = service.register_document_file(source, title="Core report", doc_type="technical_report")
    text = service.document_text(record.record_id)
    assert "Porosity measured" in text.text


def test_document_chunks_use_the_project_chunker(service):
    record = service.register_document_text(
        " ".join(f"Sentence {i} about coating and inspection." for i in range(200)),
        title="Long document",
    )
    chunks = service.document_chunks(record.record_id)
    assert len(chunks) > 1


def test_documents_filter_by_project_type_and_query(service, domain):
    project = domain.create_project(number="TSTC-1", title="Coating inspection")
    service.register_document_text(
        "A", title="Alpha proposal", doc_type="proposal", project_id=project.record_id
    )
    service.register_document_text("B", title="Beta report", doc_type="technical_report")
    assert len(service.document_rows()) == 2
    assert [r.title for r in service.document_rows(project_id=project.record_id)] == ["Alpha proposal"]
    assert [r.title for r in service.document_rows(doc_type="technical_report")] == ["Beta report"]
    assert [r.title for r in service.document_rows(query="beta")] == ["Beta report"]


def test_revision_chain_is_recorded_and_the_original_is_kept(service):
    first = service.register_document_text("Rev A text", title="ITP", doc_type="itp", revision="A")
    second = service.document_manager.record_revision(first.record_id, revision="B")
    chain = service.document_history(second.record_id)
    ids = [r.record_id for r in chain]
    assert first.record_id in ids and second.record_id in ids
    assert second.supersedes_id == first.record_id
    # The original row is never mutated.
    assert service.document_record(first.record_id).revision == "A"


def test_verify_reports_a_match_and_a_mismatch(service):
    record = service.register_document_text("Integrity", title="Doc")
    assert service.verify_document(record.record_id).ok
    # Change the stored file behind the registry's back.
    Path(record.path).write_text("tampered", encoding="utf-8")
    report = service.verify_document(record.record_id)
    assert not report.ok
    assert "changed" in report.message


def test_verify_reports_a_missing_file(service):
    record = service.register_document_text("Gone", title="Doc")
    Path(record.path).unlink()
    report = service.verify_document(record.record_id)
    assert not report.ok
    assert "missing" in report.message


def test_find_by_hash_reports_a_duplicate(service):
    text = "The same content"
    first = service.register_document_text(text, title="One")
    second = service.register_document_text(text, title="Two")
    matches = service.document_manager.find_by_hash(content_hash(text))
    assert {m.record_id for m in matches} == {first.record_id, second.record_id}


def test_registry_is_read_only_without_a_domain(service):
    manager = DocumentManager(service.repository)
    with pytest.raises(RuntimeError):
        manager.register_text("x", title="x")


def test_document_registration_is_audited(service):
    record = service.register_document_text("Audited", title="Doc")
    tail = service.domain.audit_for(record.record_id)
    assert any(row.action == "create" for row in tail)


# -- the CLI (the brief's §15) --------------------------------------------


def test_cli_library_and_documents(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "add-standard", "--body", "API", "--number", "1104",
                           "--title", "Welding of Pipelines"]) == 0
    capsys.readouterr()
    assert workplace_main(["--data-root", root, "library", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["body"] == "API"

    source = tmp_path / "ms.txt"
    source.write_text("Method statement.", encoding="utf-8")
    assert workplace_main(["--data-root", root, "register-document", str(source),
                           "--title", "Method statement", "--type", "method_statement"]) == 0
    capsys.readouterr()
    assert workplace_main(["--data-root", root, "documents", "--json"]) == 0
    docs = json.loads(capsys.readouterr().out)
    assert docs and docs[0]["title"] == "Method statement"
    record_id = docs[0]["record_id"]
    assert workplace_main(["--data-root", root, "verify-document", record_id]) == 0
    assert "matches" in capsys.readouterr().out


def test_cli_compare_standards_is_honest_when_there_is_nothing(tmp_path: Path, capsys):
    root = str(tmp_path)
    workplace_main(["--data-root", root, "add-standard", "--body", "API", "--number", "1104"])
    capsys.readouterr()
    # Fewer than two revisions: exit code 3, not a fabricated comparison.
    assert workplace_main(["--data-root", root, "compare-standards", "API", "1104"]) == 3


def test_cli_register_refuses_a_missing_file(tmp_path: Path, capsys):
    root = str(tmp_path)
    code = workplace_main(["--data-root", root, "register-document", "/no/such/file.txt"])
    assert code == 4
