"""Tests for the Workplace Phase 7: the assistant and RAG integration (§10, §11).

Phase 7 gives the Workplace its own specialised assistant, built as a **thin
layer over the systems the project already has** -- the brief's §11 forbids a
second RAG, a second document engine, or a second knowledge database. Everything
here is exercised on the real service, the real database, and the real repository,
through the project's own :class:`core.rag.RagRetriever` and
:class:`core.assistant.GroundedAssistant`: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.assistant import (
    DEFAULT_PROFILE,
    DISCLAIMER,
    PROFILE_KEYS,
    PROFILES,
    WorkplaceAssistant,
    build_workplace_documents,
    profile_for,
)
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


@pytest.fixture()
def assistant(service: WorkplaceService) -> WorkplaceAssistant:
    service.open()
    return service.assistant


# -- the eleven profiles (the brief's §10) --------------------------------


def test_the_eleven_profiles_the_brief_names():
    assert len(PROFILES) == 11
    assert len(PROFILE_KEYS) == 11
    assert PROFILE_KEYS == (
        "petroleum",
        "welding",
        "coating",
        "civil",
        "electrical",
        "qa-qc",
        "surveillance",
        "project-management",
        "tender",
        "contract",
        "research",
    )
    for profile in PROFILES:
        assert profile.name and profile.name_ar and profile.specialization
        assert profile.instruction


def test_an_unknown_profile_falls_back_to_the_default():
    assert profile_for("does-not-exist").key == DEFAULT_PROFILE
    assert profile_for("welding").key == "welding"


def test_a_profile_instruction_lays_over_the_base():
    profile = profile_for("welding")
    text = profile.system_instruction("BASE")
    assert text.startswith("BASE")
    assert profile.name in text
    assert "ASME" in text  # its references are named


# -- the corpus is built from the records (never a second copy) ----------


def test_the_corpus_is_built_from_the_records(service: WorkplaceService):
    service.open()
    domain = service.domain
    project = domain.create_project(number="TSTC-2026-001", title="Coating survey", scope="External coating")
    domain.create_person(name="Ali Hassan", role="Coating Inspector", discipline="coating")
    service.add_standard(body="AMPP", number="SP0188", title="Discontinuity testing")
    report = service.refresh_assistant()
    assert report.documents > 0
    assert {"division", "institution", "project", "standard", "person"} <= set(report.by_kind)
    assert report.by_kind["project"] == 1
    # The document is a *view* over the row: it carries the row's own id.
    store = service.assistant.store
    ids = {doc.doc_id for doc in store.documents()}
    assert f"workplace:project:{project.record_id}" in ids


def test_the_corpus_is_cumulative(service: WorkplaceService):
    service.open()
    before = service.refresh_assistant().documents
    service.domain.create_project(number="TSTC-2026-002", title="New project")
    after = service.refresh_assistant().documents
    assert after == before + 1


def test_the_project_corpus_is_reused_not_duplicated(service: WorkplaceService):
    """The Workplace reuses the project's own knowledge (the brief's §11)."""
    service.open()
    service.refresh_assistant()
    store = service.assistant.store
    sources = {doc.source for doc in store.documents()}
    # Both the Workplace's own records and the project's own seeds are present.
    assert any(source.startswith("workplace:") for source in sources)
    assert "domain_knowledge" in sources or "curriculum_knowledge" in sources


def test_a_registered_document_text_becomes_citable(service: WorkplaceService):
    service.open()
    record = service.register_document_text(
        "Surface preparation to Sa 2.5 and coating application to the flowline.",
        title="Coating method statement", doc_type="method_statement",
    )
    report = service.refresh_assistant()
    assert report.by_kind.get("document") == 1
    store = service.assistant.store
    assert f"workplace:document:{record.record_id}" in {d.doc_id for d in store.documents()}


def test_a_confidential_document_is_withheld(service: WorkplaceService):
    service.open()
    service.register_document_text(
        "Restricted content.", title="Restricted", confidentiality="restricted"
    )
    report = service.refresh_assistant()
    assert report.withheld >= 1
    # A cleared caller (default) sees no restricted document.
    store = service.assistant.store
    assert not any(d.doc_id.startswith("workplace:document:") for d in store.documents())


def test_build_documents_reports_withheld_and_unreadable(service: WorkplaceService):
    service.open()
    from core.workplace.assistant import CorpusReport

    report = CorpusReport()
    docs = build_workplace_documents(service.repository, report=report)
    assert report.documents == len(docs)
    assert report.by_kind


# -- the assistant answers with citations --------------------------------


def test_the_assistant_answers_with_citations(service: WorkplaceService):
    service.open()
    service.domain.create_project(
        number="TSTC-2026-003", title="Flowline coating inspection",
        scope="External coating and CWP inspection", status="active",
    )
    service.refresh_assistant()
    answer = service.assistant.ask("What coating inspection applies to the flowline project?")
    assert answer.text.strip()
    assert answer.citations  # grounded in cited records


def test_the_answer_carries_the_honesty_disclaimer(service: WorkplaceService):
    service.open()
    text = service.ask_workplace("what divisions does the center have?")
    assert DISCLAIMER in text


def test_the_assistant_answers_with_no_model(service: WorkplaceService):
    """With no model the assistant still answers, extractively, from the records."""
    service.open()
    service.refresh_assistant()
    answer = service.assistant.ask("what is the institution?")
    assert answer.text.strip()
    assert answer.used_model is False


def test_a_profile_narrows_the_answer(service: WorkplaceService):
    service.open()
    service.refresh_assistant()
    answer = service.assistant.ask("what is welding?", profile="welding")
    assert answer.text.strip()
    # The profile's instruction reaches the assistant's own system prompt.
    inner = service.assistant._assistant_for(profile_for("welding"))
    assert "Welding Engineering Assistant" in inner.system_instruction


# -- the CLI (the brief's §15) --------------------------------------------


def test_cli_profiles(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "profiles", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 11
    assert rows[0]["key"] == "petroleum"


def test_cli_corpus_reports_the_build(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "corpus", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["documents"] > 0
    assert "division" in report["by_kind"]


def test_cli_ask_prints_the_answer_with_the_disclaimer(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "ask", "what is the institution?"]) == 0
    out = capsys.readouterr().out
    assert DISCLAIMER in out


def test_cli_assistant_status(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "assistant-status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "Workplace assistant" in payload["status"]


def test_cli_flags_before_or_after_the_subcommand(tmp_path: Path, capsys):
    root = str(tmp_path)
    assert workplace_main(["--data-root", root, "--json", "profiles"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert workplace_main(["profiles", "--json", "--data-root", root]) == 0
    after = json.loads(capsys.readouterr().out)
    assert before == after


# -- the assistant never fabricates ---------------------------------------


def test_an_unanswerable_question_is_honest(service: WorkplaceService):
    service.open()
    service.refresh_assistant()
    answer = service.assistant.ask("what is the price of tea in 1750?")
    # No fabrication: whatever it returns, every citation names a real document
    # of the corpus it actually holds -- never a record that does not exist.
    titles = {d.title for d in service.assistant.store.documents()}
    for citation in answer.citations:
        assert any(title in citation for title in titles)
    assert isinstance(answer.text, str)
