"""Tests for Presentation Studio: the state machine, the writer, and the CLI.

The feature turns a discussion into a real ``.pptx``. The tests drive the whole
workflow with an *injected* assistant (a small scripted callable), so no model, no
network, and no display are needed -- the same way the Code Studio tests inject a
coding model. The written ``.pptx`` is then opened again with the standard
library's ``zipfile`` to prove it is a real Office Open XML package whose slide
order and text are exactly what was approved.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest
from core.presentation import (
    CONTENT_BULLETS,
    CONTENT_TABLE,
    STATE_CONTENT_APPROVED,
    STATE_DISCUSSION,
    STATE_EXPORTED,
    STATE_OUTLINE_DRAFT,
    STATE_OUTLINE_REVIEW,
    STATE_SLIDE_DRAFTING,
    STATE_SLIDE_REVIEW,
    Deck,
    PresentationError,
    PresentationWriter,
    Slide,
    bullets_from_text,
    dedup_key,
    default_presentation_dir,
    outline_from_prose,
    parse_outline,
    presentation_filename,
    require_transition,
    resolve_content_type,
    strip_discourse_opener,
    topic_sentence,
)
from gui.presentation import PresentationService, SlideSpec

# --- fixtures and helpers --------------------------------------------------


class ScriptedAssistant:
    """A deterministic stand-in for the project's assistant (no model needed)."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or ["A grounded answer."]
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]


def _service(tmp_path: Path, *answers: str) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant(*answers),
    )


def _walk_to_content_approved(service: PresentationService, project_id: str) -> list:
    """Walk the whole workflow and return the drafted slides."""
    service.draft_outline(project_id, text="Intro\n- Reservoir\n- Porosity")
    service.review_outline(project_id)
    service.approve_outline(project_id)
    slides = service.draft_slides_from_outline(project_id)
    service.review_slides(project_id)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    return slides


def _pptx_slide_texts(path: Path) -> list[str]:
    """Every slide's text, in presentation order, from the written package."""
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        names.sort(key=lambda name: int(name[len("ppt/slides/slide") : -len(".xml")]))
        texts = []
        for name in names:
            root = ElementTree.fromstring(archive.read(name))
            texts.append(
                " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
            )
        return texts


# --- the state machine -----------------------------------------------------


def test_the_happy_path_walks_every_required_state(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Reservoir Porosity")
    assert project.state == STATE_DISCUSSION

    revision = service.draft_outline(project.project_id, text="Intro\n- Porosity")
    assert service.get(project.project_id).state == STATE_OUTLINE_DRAFT
    assert revision.revision == 1 and revision.sections

    service.review_outline(project.project_id)
    assert service.get(project.project_id).state == STATE_OUTLINE_REVIEW

    service.approve_outline(project.project_id)
    assert service.get(project.project_id).state == STATE_SLIDE_DRAFTING

    service.draft_slides_from_outline(project.project_id)
    service.review_slides(project.project_id)
    assert service.get(project.project_id).state == STATE_SLIDE_REVIEW

    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    assert service.get(project.project_id).state == STATE_CONTENT_APPROVED

    report = service.export(project.project_id)
    assert report.ok and Path(report.path).exists()
    assert service.get(project.project_id).state == STATE_EXPORTED


def test_a_transition_not_in_the_table_is_refused() -> None:
    with pytest.raises(PresentationError):
        require_transition(STATE_DISCUSSION, STATE_EXPORTED)
    with pytest.raises(PresentationError):
        require_transition(STATE_EXPORTED, STATE_DISCUSSION)


def test_content_approval_is_refused_while_a_slide_is_unapproved(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    _walk_to_content_approved(service, project.project_id)
    first = service.get(project.project_id).ordered_slides()[0]

    service.reopen_content(project.project_id)
    service.unapprove_slide(project.project_id, first.slide_id)
    with pytest.raises(PresentationError):
        service.approve_content(project.project_id)


def test_the_export_gate_refuses_an_unapproved_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    allowed, reason = service.can_export(project.project_id)
    assert not allowed and "approv" in reason.lower()
    with pytest.raises(PresentationError):
        service.export(project.project_id)


def test_editing_an_approved_slide_reopens_the_content_and_rearms_the_gate(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    _walk_to_content_approved(service, project.project_id)
    slide = service.get(project.project_id).ordered_slides()[0]

    service.edit_slide(project.project_id, slide.slide_id, title="Corrected")
    reloaded = service.get(project.project_id)
    assert reloaded.state == STATE_SLIDE_REVIEW
    assert not reloaded.content_approved
    assert not reloaded.slide(slide.slide_id).approved
    allowed, reason = service.can_export(project.project_id)
    assert not allowed and "approv" in reason.lower()


def test_a_restored_revision_must_be_approved_again(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    _walk_to_content_approved(service, project.project_id)
    slide = service.get(project.project_id).ordered_slides()[0]

    service.edit_slide(project.project_id, slide.slide_id, title="Second version")
    service.restore_approved(project.project_id, slide.slide_id)
    restored = service.get(project.project_id).slide(slide.slide_id)
    assert restored.title == slide.title
    assert not restored.approved


def test_an_exported_project_can_be_exported_again(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    _walk_to_content_approved(service, project.project_id)
    first = service.export(project.project_id)
    assert first.ok
    second = service.export(project.project_id)
    assert second.ok and Path(second.path).exists()


# --- slide identity, order, and revisions ----------------------------------


def test_batch_added_slides_get_unique_ids_and_sequential_order(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    service.draft_outline(project.project_id, text="A\nB\nC")
    service.approve_outline(project.project_id)
    specs = [SlideSpec(title=f"Slide {n}") for n in range(6)]
    slides = service.add_slides(project.project_id, specs)
    ids = [slide.slide_id for slide in slides]
    assert len(set(ids)) == len(ids)
    assert [slide.order for slide in slides] == list(range(1, 7))


def test_reordering_changes_the_exported_order_not_the_identity(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    service.draft_outline(project.project_id, text="First\nSecond")
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id)
    service.review_slides(project.project_id)
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)

    moved = slides[-1]
    service.reorder_slide(project.project_id, moved.slide_id, 1)
    # A reorder of approved content reopens it, so the order is approved again.
    assert service.get(project.project_id).state == STATE_SLIDE_REVIEW
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    report = service.export(project.project_id)
    assert report.ok
    texts = _pptx_slide_texts(Path(report.path))
    assert texts[0].strip().startswith(moved.title[:10])

    reloaded = service.get(project.project_id)
    assert {slide.slide_id for slide in reloaded.slides} == {s.slide_id for s in slides}
    assert next(slide.slide_id for slide in reloaded.approved_slides) == moved.slide_id


def test_removing_a_slide_renumbers_the_rest(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    service.draft_outline(project.project_id, text="A\nB\nC")
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id)
    service.remove_slide(project.project_id, slides[1].slide_id)
    remaining = service.get(project.project_id).ordered_slides()
    assert [slide.order for slide in remaining] == list(range(1, len(remaining) + 1))


def test_a_revision_is_recorded_for_every_edit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Deck")
    service.draft_outline(project.project_id, text="A")
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id)
    slide_id = slides[0].slide_id
    service.edit_slide(project.project_id, slide_id, title="One")
    service.edit_slide(project.project_id, slide_id, title="Two")
    history = service.revisions(project.project_id, slide_id)
    assert len(history) >= 2
    assert [item.revision for item in history] == sorted(
        item.revision for item in history
    )


# --- persistence -----------------------------------------------------------


def test_a_project_survives_a_fresh_service(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Persistent", audience="Engineers")
    service.draft_outline(project.project_id, text="Intro\n- Point")

    reopened = _service(tmp_path)
    reloaded = reopened.get(project.project_id)
    assert reloaded.title == "Persistent"
    assert reloaded.audience == "Engineers"
    assert reloaded.state == STATE_OUTLINE_DRAFT
    assert reloaded.outline_revisions[0].sections


def test_listing_orders_the_most_recently_updated_project_first(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.new_project("First")
    second = service.new_project("Second")
    # Updating the first project again makes it the most recently touched one.
    service.draft_outline(first.project_id, text="Intro\n- Point")
    ids = [project.project_id for project in service.list_projects()]
    assert ids.index(first.project_id) < ids.index(second.project_id)


# --- the dependency-free writer --------------------------------------------


def test_the_writer_writes_a_real_pptx_with_no_third_party_package(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Porosity & Permeability")
    service.draft_outline(project.project_id, text="Intro\n- Porosity\n- Permeability")
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id)
    service.review_slides(project.project_id)
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    report = service.export(project.project_id)
    assert report.ok and report.slide_count >= 1

    with zipfile.ZipFile(Path(report.path)) as archive:
        names = set(archive.namelist())
        assert "[Content_Types].xml" in names
        assert "ppt/presentation.xml" in names
        assert "ppt/slides/slide1.xml" in names
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                ElementTree.fromstring(archive.read(name))


def test_the_writer_escapes_markup_in_the_text(tmp_path: Path) -> None:
    writer = PresentationWriter()
    deck = Deck(
        title="Escape",
        slides=(Slide(slide_id="s1", order=1, title="One", content=['Porosity < 20% & "tight"']),),
    )
    written = writer.write(deck, tmp_path / "escaped.pptx")
    with zipfile.ZipFile(written) as archive:
        body = archive.read("ppt/slides/slide1.xml").decode("utf-8")
    assert "&lt; 20%" in body and "&amp;" in body
    assert "<20%" not in body


def test_the_writer_appends_the_pptx_suffix_when_it_is_missing(tmp_path: Path) -> None:
    writer = PresentationWriter()
    deck = Deck(
        title="Escape", slides=(Slide(slide_id="s1", order=1, title="One", content=["Text"]),)
    )
    written = writer.write(deck, tmp_path / "deck.txt")
    assert written.suffix == ".pptx" and written.exists()


def test_the_writer_leaves_no_partial_file_when_the_render_fails(tmp_path: Path) -> None:
    """The write is atomic: a failure never leaves a half-written deck behind."""
    writer = PresentationWriter()
    target = tmp_path / "deck.pptx"

    class ExplodingDeck:
        @property
        def slides(self) -> object:
            raise RuntimeError("render failed")

    with pytest.raises(RuntimeError):
        writer.write(ExplodingDeck(), target)  # type: ignore[arg-type]
    assert not target.exists()
    assert not target.with_suffix(".pptx.tmp").exists()


def test_the_writer_replaces_an_existing_deck_without_leaving_a_temporary(
    tmp_path: Path,
) -> None:
    writer = PresentationWriter()
    target = tmp_path / "deck.pptx"
    first = writer.write(
        Deck(title="First", slides=(Slide(slide_id="s1", order=1, title="One", content=["A"]),)),
        target,
    ).read_bytes()
    second = writer.write(
        Deck(title="Second", slides=(Slide(slide_id="s1", order=1, title="Two", content=["B"]),)),
        target,
    )
    assert second.read_bytes() != first
    assert not target.with_suffix(".pptx.tmp").exists()
    with zipfile.ZipFile(second) as archive:
        assert archive.testzip() is None


# --- parsing helpers --------------------------------------------------------


def test_parse_outline_reads_headings_and_indented_subsections() -> None:
    sections = parse_outline("Introduction\n- Reservoir\n  - Porosity\n- Drilling\n")
    assert [section.heading for section in sections] == ["Introduction", "Reservoir", "Drilling"]
    assert sections[1].subsections == ("Porosity",)


def test_outline_from_prose_derives_a_numbered_skeleton() -> None:
    prose = (
        "Porosity is the fraction of void space in a rock. It controls storage "
        "capacity. Permeability is the ability of a rock to transmit fluids, and "
        "it controls flow. Both are measured in the laboratory."
    )
    sections = outline_from_prose(prose, title="Petrophysics")
    assert len(sections) >= 2
    assert all(section.heading for section in sections)


def test_bullets_from_text_drops_the_extractive_fallback_scaffolding() -> None:
    # The extractive fallback's real shape: a scaffolding line, then the content.
    text = (
        "No web source answered the question about porosity.\n\n"
        "Internal knowledge on file:\n"
        "Porosity is the fraction of void space in a rock. "
        "Permeability governs the flow of fluids.\n\n"
        "Note: see the cited sources."
    )
    bullets = bullets_from_text(text)
    joined = " ".join(bullets).lower()
    assert "porosity is the fraction" in joined and "permeability governs the flow" in joined
    assert not any(
        bullet.lower().startswith(("no web source", "internal knowledge", "note:"))
        for bullet in bullets
    )


def test_resolve_content_type_normalises_but_refuses_an_unknown_name() -> None:
    assert resolve_content_type("bullets") == CONTENT_BULLETS
    assert resolve_content_type("TABLE") == CONTENT_TABLE
    with pytest.raises(PresentationError):
        resolve_content_type("nonsense")


def test_presentation_filename_is_a_path_safe_slug() -> None:
    name = presentation_filename("Reservoir Porosity: A Study")
    assert name.endswith(".pptx")
    assert "/" not in name and ":" not in name and " " not in name


def test_default_presentation_dir_sits_under_the_data_folder() -> None:
    assert default_presentation_dir().name == "presentations"


# --- assistant integration --------------------------------------------------


def test_the_discussion_uses_the_injected_assistant(tmp_path: Path) -> None:
    assistant = ScriptedAssistant("Porosity is the void fraction.")
    service = PresentationService(root=tmp_path / "store", ask=assistant)
    project = service.new_project("Deck")
    result = service.discuss(project.project_id, "What is porosity?")
    assert "void fraction" in result.answer
    assert assistant.asked == ["What is porosity?"]


def test_a_missing_assistant_is_reported_rather_than_faked(tmp_path: Path) -> None:
    service = PresentationService(root=tmp_path / "store")
    project = service.new_project("Deck")
    with pytest.raises(PresentationError):
        service.discuss(project.project_id, "Anything?")
    with pytest.raises(PresentationError):
        service.draft_outline(project.project_id)


def test_prose_outline_sections_become_structured_slide_text(tmp_path: Path) -> None:
    prose = (
        "Porosity is the fraction of void space in a rock and it controls storage. "
        "Permeability is the ability to transmit fluids and it controls flow rate. "
        "Well logs estimate both quantities in the subsurface."
    )
    service = _service(tmp_path, prose)
    project = service.new_project("Petrophysics")
    revision = service.draft_outline(project.project_id)
    assert len(revision.sections) >= 2
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id)
    assert slides
    assert all(slide.title for slide in slides)


# --- the CLI ----------------------------------------------------------------


def test_the_cli_walks_the_workflow_and_exports(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    out = str(tmp_path / "out")
    assert main(["--root", root, "new", "CLI Deck"]) == 0
    capsys.readouterr()

    assert main(["--root", root, "outline", "PRS-001", "--text", "Intro\n- Point"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "review-outline", "PRS-001"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "Point", "--bullet", "Detail"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "review-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-content", "PRS-001"]) == 0
    capsys.readouterr()

    assert main(["--root", root, "--output-dir", out, "--json", "status", "PRS-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == STATE_CONTENT_APPROVED

    assert main(["--root", root, "--output-dir", out, "--json", "export", "PRS-001"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and Path(report["path"]).exists()
    assert Path(report["path"]).suffix == ".pptx"


def test_the_cli_refuses_an_export_before_the_content_is_approved(tmp_path: Path) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Early"]) == 0
    # Exit code 4 means "the gate refused the operation".
    assert main(["--root", root, "export", "PRS-001"]) == 4


def test_the_cli_reports_an_unknown_project_without_crashing(tmp_path: Path) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "status", "PRS-999"]) != 0


def test_the_cli_lists_nothing_before_a_project_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    assert main(["--root", str(tmp_path / "store"), "list"]) == 0
    assert "no presentation" in capsys.readouterr().out.lower()

def test_the_cli_can_withdraw_a_slides_approval(tmp_path: Path) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\n- B"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "One", "--bullet", "X"]) == 0
    assert main(["--root", root, "review-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-content", "PRS-001"]) == 0
    # Withdrawing the approval re-arms the export gate.
    assert main(["--root", root, "unapprove-slide", "PRS-001", "SLD-001"]) == 0
    assert main(["--root", root, "export", "PRS-001"]) == 4


def test_the_cli_prints_a_failed_export_as_exit_code_4(tmp_path: Path) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Deck"]) == 0
    # Nothing is approved yet, so the gate refuses and the CLI says so.
    assert main(["--root", root, "--json", "export", "PRS-001"]) == 4


def test_the_cli_show_reports_the_state_and_the_slides(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "Intro\n- Point"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "Point", "--bullet", "X"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "show", "PRS-001"]) == 0
    printed = capsys.readouterr().out
    assert "Point" in printed
    assert "slide" in printed.lower()


def test_the_cli_records_and_lists_a_slides_revisions(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\n- B"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "One", "--bullet", "X"]) == 0
    assert main(["--root", root, "edit", "PRS-001", "SLD-001", "--title", "Two"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "revisions", "PRS-001", "SLD-001"]) == 0
    printed = capsys.readouterr().out
    assert "r1" in printed and "edited" in printed
    assert "'Two'" in printed


def test_the_cli_exports_a_real_deck_a_second_time(tmp_path: Path) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    out = str(tmp_path / "out")
    assert main(["--root", root, "new", "Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\n- B"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "One", "--bullet", "X"]) == 0
    assert main(["--root", root, "review-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-content", "PRS-001"]) == 0
    first = tmp_path / "out" / "first.pptx"
    assert main(["--root", root, "--output-dir", out, "export", "PRS-001", "--output", str(first)]) == 0
    assert first.exists()
    second = tmp_path / "out" / "second.pptx"
    assert main(["--root", root, "--output-dir", out, "export", "PRS-001", "--output", str(second)]) == 0
    assert second.exists()


def test_the_cli_remembers_a_projects_conversation(tmp_path: Path) -> None:
    """A project's discussion pointer survives the process, so a later run resumes it."""
    from cli.presentation import _load_conversation_id, _save_conversation_id

    root = tmp_path / "store"
    assert _load_conversation_id(root, "PRS-001") == ""
    _save_conversation_id(root, "PRS-001", "CNV-abc123")
    assert _load_conversation_id(root, "PRS-001") == "CNV-abc123"
    # A different project keeps its own conversation.
    assert _load_conversation_id(root, "PRS-002") == ""


def test_topic_sentence_drops_the_answers_own_joining_words() -> None:
    assert topic_sentence("First, casing is steel pipe set in the well.") == (
        "Casing is steel pipe set in"
    )
    assert topic_sentence("In addition, a petroleum system links a source.") == (
        "A petroleum system links a source"
    )
    assert topic_sentence("Finally, Interpretation proceeds in a fixed order.") == (
        "Interpretation proceeds in a fixed order"
    )
    # A sentence that is only an opener keeps its own words rather than vanishing.
    assert topic_sentence("First, porosity.") == "Porosity"


def test_outline_from_prose_headings_do_not_open_with_a_connective() -> None:
    prose = (
        "First, porosity is the fraction of void space in a rock. "
        "In addition, permeability governs the flow of fluids. "
        "Finally, well logs estimate both quantities in the subsurface."
    )
    sections = outline_from_prose(prose, title="Petrophysics")
    for section in sections:
        assert not section.heading.lower().startswith(("first", "in addition", "finally"))

def test_bullets_from_text_drops_the_answers_title_line() -> None:
    """The slide prompt asks for a title line; it is the title, not a bullet."""
    text = "Porosity\n- Fraction of void space\n- Controls storage\n- Measured by logs"
    assert bullets_from_text(text) == [
        "Fraction of void space",
        "Controls storage",
        "Measured by logs",
    ]
    # A reply that is only bullets keeps every one of them.
    assert bullets_from_text("- a\n- b\n- c") == ["a", "b", "c"]
    # A prose reply is still split into sentences.
    assert bullets_from_text("One thing. Two things. Three things.") == [
        "One thing.",
        "Two things.",
        "Three things.",
    ]


def _heading_of(prompt: str) -> str:
    """The outline heading a slide prompt is about (its first line)."""
    return prompt.splitlines()[0].strip()


def test_strip_discourse_opener_removes_only_a_leading_connective() -> None:
    """A joining word is the answer's narration, not the bullet's content."""
    assert strip_discourse_opener("First, Casing is steel pipe set in the well.") == (
        "Casing is steel pipe set in the well."
    )
    assert strip_discourse_opener("In addition, Primary cementing places a sheath.") == (
        "Primary cementing places a sheath."
    )
    assert strip_discourse_opener("Finally, Field development planning chooses.") == (
        "Field development planning chooses."
    )
    # Text with no opener is returned exactly, so nothing is re-cased.
    assert strip_discourse_opener("porosity measures void space") == (
        "porosity measures void space"
    )
    # A connective word inside the sentence is content and is left alone.
    assert strip_discourse_opener("The first step is depth selection") == (
        "The first step is depth selection"
    )
    # A sentence that is only an opener keeps its own words rather than vanishing.
    assert strip_discourse_opener("First, porosity.") == "Porosity."


def test_bullets_from_text_drops_a_leading_discourse_opener() -> None:
    """A bullet reads as a fact, not as the assistant narrating its own list."""
    text = "Porosity\n- First, Fraction of void space\n- In addition, Controls storage"
    assert bullets_from_text(text) == [
        "Fraction of void space",
        "Controls storage",
    ]
    prose = "First, casing is set in sections. Finally, cementing isolates zones."
    assert bullets_from_text(prose) == [
        "Casing is set in sections.",
        "Cementing isolates zones.",
    ]


def test_dedup_key_recognises_a_repeated_bullet_across_slides() -> None:
    """The key ignores case, spacing, a trailing stop, and a leading connective."""
    assert dedup_key("First, Casing is steel pipe.") == dedup_key("casing is steel pipe")
    assert dedup_key("  Porosity   measures  void space  ") == dedup_key(
        "Porosity measures void space."
    )
    # Two genuinely different bullets must not collide.
    assert dedup_key("Porosity and permeability") != dedup_key("Permeability and porosity")


def test_draft_slides_does_not_repeat_a_bullet_on_a_second_slide() -> None:
    """A slide never repeats a line an earlier slide already shows."""
    shared = "Casing is steel pipe set in the well."
    answers = {
        "Well design": f"Title A\n- {shared}\n- A unique line for well design",
        "Cementing": f"Title B\n- {shared}\n- A unique line for cementing",
    }
    service = PresentationService(root=None, ask=lambda q: answers[_heading_of(q)])
    project = service.new_project("Drilling")
    service.draft_outline(project.project_id, text="Well design\nCementing")
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id)
    first, second = slides
    assert first.content == [shared, "A unique line for well design"]
    # The repeat is gone; the second slide keeps only its own new line.
    assert second.content == ["A unique line for cementing"]


def test_draft_slides_flags_a_heading_the_retriever_cannot_distinguish() -> None:
    """When every bullet repeats, one line is kept and the note says why."""
    same = "- The only line the retriever returns"
    service = PresentationService(root=None, ask=lambda q: same)
    project = service.new_project("Drilling")
    service.draft_outline(project.project_id, text="Well design\nCementing")
    service.approve_outline(project.project_id)
    first, second = service.draft_slides_from_outline(project.project_id)
    assert first.content == ["The only line the retriever returns"]
    # A blank slide would be worse than a flagged one, so one line is kept.
    assert second.content == ["The only line the retriever returns"]
    assert "no new material" in second.notes


def test_drafting_never_repeats_the_title_as_the_first_bullet() -> None:
    """A drafted slide's title must not also be its first bullet."""
    from gui.presentation import PresentationService

    class Model:
        """A model that follows the prompt: a title line, then bullets."""

        def __call__(self, question: str) -> str:
            if "outline" in question.lower():
                return "Porosity\n- Fraction of void space\nPermeability\n- Transmissivity"
            return "Porosity\n- Fraction of void space\n- Controls storage"

    service = PresentationService(ask=Model())
    project = service.new_project("Deck")
    service.draft_outline(project.project_id)
    service.review_outline(project.project_id)
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id)
    slides = service.get(project.project_id).ordered_slides()
    assert slides
    for slide in slides:
        assert slide.title
        assert slide.title not in slide.content


def test_drafting_uses_the_section_heading_when_the_answer_is_prose() -> None:
    """A grounded answer is prose, so the section heading is the slide's title."""
    from gui.presentation import PresentationService

    class Prose:
        def __call__(self, question: str) -> str:
            if "outline" in question.lower():
                return "Porosity\n- Fraction of void space\nPermeability\n- Transmissivity"
            return "Porosity is the fraction of void space in a rock and it controls storage."

    service = PresentationService(ask=Prose())
    project = service.new_project("Deck")
    service.draft_outline(project.project_id)
    service.review_outline(project.project_id)
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id)
    slides = service.get(project.project_id).ordered_slides()
    assert slides[0].title == "Porosity"
    # The first sentence is a bullet, and the title is not repeated among them.
    assert any("fraction of void space" in bullet for bullet in slides[0].content)
