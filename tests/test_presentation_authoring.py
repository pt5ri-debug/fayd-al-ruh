"""Tests for structured slide authoring: numbering, ranges, and review.

These cover the *authoring* half of Presentation Studio -- the part that turns an
approved outline into individually numbered, reviewable slides. The workflow
tests (discussion, outline, export) live in ``test_presentation.py``; here the
focus is the slide list itself: dense numbering, section/subsection placement,
inserting and moving by number, range approval, regeneration, rehoming, and the
plain-language command parser shared by the window and the CLI.

Everything runs with an *injected* assistant, so no model, no network, and no
display are needed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from core.presentation import (
    STATE_SLIDE_REVIEW,
    PresentationError,
    parse_slide_command,
    parse_slide_range,
    slide_topic_key,
)
from gui.presentation import PresentationService, SlideSpec


class ScriptedAssistant:
    """A deterministic stand-in for the project's assistant (no model needed)."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers) or ["A grounded answer."]

    def __call__(self, _question: str) -> str:
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0]


def _authoring_service(tmp_path: Path, *answers: str) -> PresentationService:
    """A service whose assistant answers with a distinct bullet each turn."""
    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant(*answers),
    )


def _deck_with_slides(tmp_path: Path, count: int = 6) -> tuple[PresentationService, str]:
    """A project walked to slide review with ``count`` drafted slides."""
    outline = "\n".join(f"Topic {n}" for n in range(1, count + 1))
    service = _authoring_service(tmp_path, "Slide title\nA bullet.\nAnother bullet.")
    project = service.new_project("Authoring Deck")
    service.draft_outline(project.project_id, text=outline)
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id, batch=count)
    service.review_slides(project.project_id)
    return service, project.project_id


def _pptx_slide_texts(path: Path) -> list[str]:
    """Every slide's text from a real .pptx, in slide order."""
    from xml.etree import ElementTree

    texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        names.sort(key=lambda name: int(name.rsplit("slide", 1)[1].split(".", 1)[0]))
        for name in names:
            root = ElementTree.fromstring(archive.read(name))
            texts.append(
                " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
            )
    return texts


# --- the required per-slide fields -----------------------------------------


def test_a_drafted_slide_carries_every_required_field(tmp_path: Path) -> None:
    """Every slide has an id, a number, a section, a title, content, and a status."""
    service, project_id = _deck_with_slides(tmp_path, 3)
    for slide in service.get(project_id).ordered_slides():
        # Stable id, current number, section/subsection, title, content, notes,
        # media, and a review status all exist on the slide.
        assert slide.slide_id.startswith("SLD-")
        assert slide.number == slide.order >= 1
        assert slide.section
        assert slide.title
        assert slide.content
        assert isinstance(slide.notes, str)
        assert isinstance(slide.images, list)
        assert slide.approved is False  # drafted slides start unapproved


def test_slides_are_numbered_densely_and_contiguously(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 6)
    project = service.get(project_id)
    assert [slide.number for slide in project.ordered_slides()] == list(range(1, 7))
    assert project.slide_numbers() == list(range(1, 7))


def test_the_default_batch_size_is_five_slides(tmp_path: Path) -> None:
    """A long outline is drafted five slides at a time, then the next batch."""
    service = _authoring_service(tmp_path, "Title\nA bullet.\nAnother bullet.")
    project = service.new_project("Long Deck")
    outline = "\n".join(f"Topic {n}" for n in range(1, 13))
    service.draft_outline(project.project_id, text=outline)
    service.approve_outline(project.project_id)

    assert len(service.draft_slides_from_outline(project.project_id)) == 5
    assert len(service.draft_slides_from_outline(project.project_id)) == 5
    assert len(service.draft_slides_from_outline(project.project_id)) == 2
    # No fourth call is needed, and a repeat adds nothing.
    assert service.draft_slides_from_outline(project.project_id) == []


def test_the_batch_size_can_be_set_on_the_service(tmp_path: Path) -> None:
    service = _authoring_service(tmp_path, "Title\nA bullet.")
    assert service.set_batch_size(10) == 10
    assert service.batch_size == 10
    with pytest.raises(PresentationError):
        service.set_batch_size(0)


def test_a_section_with_subsections_drafts_one_slide_per_subsection(tmp_path: Path) -> None:
    """Subsections are developed as their own numbered slides, in outline order."""
    service = _authoring_service(tmp_path, "Title\nA bullet.")
    project = service.new_project("Subsection Deck")
    service.draft_outline(
        project.project_id, text="Reservoir\n  - Porosity\n  - Permeability"
    )
    service.approve_outline(project.project_id)
    slides = service.draft_slides_from_outline(project.project_id, batch=10)
    assert [(slide.section, slide.subsection) for slide in slides] == [
        ("Reservoir", "Porosity"),
        ("Reservoir", "Permeability"),
    ]
    assert [slide.number for slide in slides] == [1, 2]


def test_the_slide_topic_key_qualifies_a_subsection_by_its_section() -> None:
    assert slide_topic_key("Reservoir", "Porosity") == "Reservoir / Porosity"
    assert slide_topic_key("Reservoir") == "Reservoir"
    assert slide_topic_key("", "Porosity") == "Porosity"
    assert slide_topic_key("", "") == ""


# --- numbering under insert, move, and renumber ----------------------------


def test_inserting_a_slide_renumbers_the_slides_after_it(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 4)
    inserted = service.insert_slide(
        project_id, SlideSpec(title="Inserted", content=["New point"]), position=2
    )
    ordered = service.get(project_id).ordered_slides()
    assert [slide.number for slide in ordered] == [1, 2, 3, 4, 5]
    assert ordered[1].slide_id == inserted.slide_id
    # The ids stay stable even though the numbers shifted.
    assert len({slide.slide_id for slide in ordered}) == 5


def test_a_slide_can_be_moved_before_another_by_number(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 4)
    last = service.get(project_id).slide_by_number(4)
    service.move_slide(project_id, last.slide_id, before=1)
    ordered = service.get(project_id).ordered_slides()
    assert ordered[0].slide_id == last.slide_id
    assert [slide.number for slide in ordered] == [1, 2, 3, 4]


def test_a_slide_can_be_moved_after_another_by_number(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 4)
    first = service.get(project_id).slide_by_number(1)
    service.move_slide(project_id, first.slide_id, after=3)
    ordered = service.get(project_id).ordered_slides()
    assert ordered[2].slide_id == first.slide_id
    assert [slide.number for slide in ordered] == [1, 2, 3, 4]


def test_renumber_slides_makes_the_numbers_dense_again(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 4)
    project = service.get(project_id)
    # Simulate a gap a manual edit could leave, then renumber.
    project.ordered_slides()[2].order = 9
    service.store.save(project)
    service.renumber_slides(project_id)
    assert [slide.number for slide in service.get(project_id).ordered_slides()] == [1, 2, 3, 4]


def test_a_slide_can_be_rehomed_under_another_section(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 3)
    slide = service.get(project_id).slide_by_number(1)
    service.edit_slide(project_id, slide.slide_id, section="Intro", subsection="Scope")
    reloaded = service.get(project_id).slide(slide.slide_id)
    assert reloaded.section == "Intro" and reloaded.subsection == "Scope"
    assert reloaded.topic == "Intro / Scope"


# --- review: range approval, regeneration, revision ------------------------


def test_approving_a_range_approves_only_those_slides(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 6)
    approved = service.approve_slides_in_range(project_id, 2, 4)
    assert [slide.number for slide in approved] == [2, 3, 4]
    project = service.get(project_id)
    assert [slide.number for slide in project.approved_slides] == [2, 3, 4]
    assert [slide.number for slide in project.outdated_slides] == [1, 5, 6]


def test_approving_a_range_beyond_the_deck_is_clamped(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 4)
    approved = service.approve_slides_in_range(project_id, 3, 12)
    assert [slide.number for slide in approved] == [3, 4]


def test_regenerating_a_slide_keeps_its_id_number_and_heading(tmp_path: Path) -> None:
    """Re-drafting changes the words, never the slide's identity or place."""

    class Changing:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, _question: str) -> str:
            self.n += 1
            return f"Title {self.n}\nBullet number {self.n}."

    service = PresentationService(
        root=tmp_path / "store", output_dir=tmp_path / "out", ask=Changing()
    )
    project = service.new_project("Regenerate Deck")
    service.draft_outline(project.project_id, text="Reservoir\n  - Porosity")
    service.approve_outline(project.project_id)
    (slide,) = service.draft_slides_from_outline(project.project_id)
    before_id, before_number = slide.slide_id, slide.number

    (after,) = service.regenerate_slides(project.project_id, [before_number])
    assert after.slide_id == before_id
    assert after.number == before_number
    assert after.section == "Reservoir" and after.subsection == "Porosity"
    assert after.approved is False


def test_regenerating_clears_approval_and_rearms_the_gate(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 3)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    assert service.can_export(project_id)[0] is True

    service.regenerate_slides(project_id, [2])
    project = service.get(project_id)
    assert project.state == STATE_SLIDE_REVIEW
    assert project.content_approved is False
    allowed, reason = service.can_export(project_id)
    assert allowed is False and "approv" in reason.lower()


def test_revising_a_range_edits_each_named_slide(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 5)
    changed = service.revise_slides(project_id, [2, 3], instruction="add units")
    assert [slide.number for slide in changed] == [2, 3]
    assert all(slide.approved is False for slide in changed)


def test_add_example_after_inserts_the_new_slide_in_place(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 3)
    anchor = service.get(project_id).slide_by_number(2)
    example = service.add_example_after(project_id, 2)
    ordered = service.get(project_id).ordered_slides()
    assert example.number == 3
    assert ordered[2].slide_id == example.slide_id
    assert example.section == anchor.section
    assert [slide.number for slide in ordered] == [1, 2, 3, 4]


def test_the_new_fields_survive_a_save_and_reload(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 2)
    slide = service.get(project_id).slide_by_number(1)
    service.edit_slide(
        project_id,
        slide.slide_id,
        section="Reservoir",
        subsection="Porosity",
        notes="Mention the units.",
        images=["porosity.png"],
    )
    reloaded = _authoring_service(tmp_path).get(project_id).slide(slide.slide_id)
    assert reloaded.section == "Reservoir"
    assert reloaded.subsection == "Porosity"
    assert reloaded.notes == "Mention the units."
    assert reloaded.images == ["porosity.png"]
    assert reloaded.topic == "Reservoir / Porosity"


def test_the_exported_order_is_exactly_the_approved_order(tmp_path: Path) -> None:
    """After a reorder and a re-approval the deck's order is the approved order."""
    service, project_id = _deck_with_slides(tmp_path, 4)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    # Move the last slide to the front, then re-approve.
    last = service.get(project_id).slide_by_number(4)
    service.move_slide(project_id, last.slide_id, before=1)
    service.approve_slides(project_id)
    service.approve_content(project_id)
    report = service.export(project_id)
    assert report.ok
    texts = _pptx_slide_texts(Path(report.path))
    assert len(texts) == 4
    assert texts[0].strip().startswith(last.title[:10])


# --- the plain-language command parser -------------------------------------


def _command(text: str):
    """Parse a slide command, asserting it is one (keeps mypy happy)."""
    command = parse_slide_command(text)
    assert command is not None, text
    return command


def test_parse_slide_command_reads_each_supported_instruction() -> None:
    assert _command("Revise slide 7.").action == "revise"
    assert _command("Revise slide 7.").slides == (7,)
    shorten = _command("Shorten slides 11-15.")
    assert shorten.action == "shorten" and shorten.slides == (11, 12, 13, 14, 15)
    expand = _command("Expand slides 3-4.")
    assert expand.action == "expand" and expand.slides == (3, 4)
    example = _command("Add a worked example after slide 12.")
    assert example.action == "add-example" and example.slides == (12,)
    move = _command("Move slide 18 before slide 17.")
    assert move.action == "move" and move.slides == (18,) and move.anchor == 17 and move.before
    move_after = _command("Move slide 3 after slide 9.")
    assert move_after.action == "move" and move_after.before is False
    assert _command("Renumber all slides after reordering.").action == "renumber"
    regen = _command("Regenerate slides 21-25.")
    assert regen.action == "regenerate" and regen.slides == (21, 22, 23, 24, 25)
    assert _command("Approve slide 2.").action == "approve"
    batch = _command("Approve slides 5-8.")
    assert batch.action == "approve-batch" and batch.slides == (5, 6, 7, 8)


def test_parse_slide_command_returns_none_for_a_question() -> None:
    assert parse_slide_command("What is porosity?") is None
    assert parse_slide_command("") is None


def test_parse_slide_range_reads_the_forms_a_user_writes() -> None:
    assert parse_slide_range("7") == (7, 7)
    assert parse_slide_range("slides 11-15") == (11, 15)
    assert parse_slide_range("11 to 15") == (11, 15)
    assert parse_slide_range("no numbers here") is None


def test_a_renumber_command_is_not_mistaken_for_a_move(tmp_path: Path) -> None:
    """'renumber ... after reordering' contains 'after' but is a renumber."""
    service, project_id = _deck_with_slides(tmp_path, 3)
    result = service.apply_slide_command(project_id, "Renumber all slides after reordering.")
    assert result.ok and result.action == "renumber"
    assert [slide.number for slide in service.get(project_id).ordered_slides()] == [1, 2, 3]


def test_a_command_runs_through_the_shared_service_entry_point(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 6)
    result = service.apply_slide_command(project_id, "Approve slides 2-4.")
    assert result.ok
    assert [slide.number for slide in service.get(project_id).approved_slides] == [2, 3, 4]


def test_an_unrecognised_command_is_reported_not_raised(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 2)
    result = service.apply_slide_command(project_id, "please make it nicer somehow")
    assert result.ok is False and result.action == ""


def test_a_command_names_an_unknown_slide_honestly(tmp_path: Path) -> None:
    service, project_id = _deck_with_slides(tmp_path, 2)
    with pytest.raises(PresentationError):
        service.apply_slide_command(project_id, "Shorten slides 99-100.")


# --- the CLI authoring commands --------------------------------------------


def test_the_cli_lists_and_moves_numbered_slides(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "CLI Authoring"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\nB\nC"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    for title in ("A", "B", "C"):
        assert main(
            ["--root", root, "add", "PRS-001", "--title", title, "--bullet", "x", "--section", title]
        ) == 0
    capsys.readouterr()

    assert main(["--root", root, "slides", "PRS-001"]) == 0
    listing = capsys.readouterr().out
    assert "1." in listing and "[A]" in listing and "3 slide(s)" in listing

    assert main(["--root", root, "move", "PRS-001", "3", "--before", "1"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "--json", "slides", "PRS-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [slide["title"] for slide in payload["slides"]] == ["C", "A", "B"]


def test_the_cli_approves_a_range_and_renumbers(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "CLI Range"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\nB\nC\nD"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    for title in ("A", "B", "C", "D"):
        assert main(
            ["--root", root, "add", "PRS-001", "--title", title, "--bullet", "x", "--section", title]
        ) == 0
    capsys.readouterr()

    assert main(["--root", root, "approve-range", "PRS-001", "2-3"]) == 0
    assert "2, 3" in capsys.readouterr().out
    assert main(["--root", root, "renumber", "PRS-001"]) == 0
    assert "Renumbered 4 slide(s)" in capsys.readouterr().out


def test_the_cli_runs_a_slide_command(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "CLI Command"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\nB\nC"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    for title in ("A", "B", "C"):
        assert main(
            ["--root", root, "add", "PRS-001", "--title", title, "--bullet", "x", "--section", title]
        ) == 0
    capsys.readouterr()

    assert main(["--root", root, "command", "PRS-001", "Approve slides 1-2"]) == 0
    assert "Approved slide(s) 1, 2" in capsys.readouterr().out
    # An unrecognised instruction is exit code 4 (the gate refused it).
    assert main(["--root", root, "command", "PRS-001", "just make it good"]) == 4


def test_the_cli_slide_range_parser_reads_numbers_and_ranges() -> None:
    from cli.presentation import _parse_slide_numbers

    assert _parse_slide_numbers("7") == (7, 7)
    assert _parse_slide_numbers("11-15") == (11, 15)
    assert _parse_slide_numbers("slides 5 to 8") == (5, 8)
    with pytest.raises(PresentationError):
        _parse_slide_numbers("all of them")


def test_the_cli_adds_a_slide_at_a_position(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "CLI Insert"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "A\nB"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    for title in ("A", "B"):
        assert main(
            ["--root", root, "add", "PRS-001", "--title", title, "--bullet", "x", "--section", title]
        ) == 0
    capsys.readouterr()
    assert main(
        ["--root", root, "add", "PRS-001", "--title", "Inserted", "--bullet", "y", "--position", "1"]
    ) == 0
    assert "Added slide 1" in capsys.readouterr().out
