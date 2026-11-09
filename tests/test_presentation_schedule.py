"""Tests for the schedule engine, the schedule build, and its CLI command.

The Studio's headline capability is: paste a training schedule (days and topics),
and the window prepares professional scientific content, split into numbered
slides. These tests pin the deterministic reader (``core.presentation_schedule``),
the one-step build (``PresentationService.build_from_schedule``), and the CLI
(``python -m cli presentation from-schedule``), with **negative controls**: a
blank schedule yields no plan, an unrelated line is never read as a day, and a
bare topic list still parses in order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.presentation import STATE_SLIDE_REVIEW
from core.presentation_schedule import (
    SchedulePlan,
    ScheduleTopic,
    outline_from_schedule,
    parse_schedule,
)
from gui.presentation import PresentationService

SCHEDULE = """Day 1: Porosity and permeability
- Definition of porosity
- Effective vs total porosity
Day 2: Well logging
- Gamma ray
- Density and neutron logs
Day 3: Well testing
"""


class ScriptedAssistant:
    """A deterministic stand-in for the project's assistant (no model needed)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, question: str) -> str:
        self.prompts.append(question)
        return self.answer


def _service(tmp_path: Path, answer: str = "") -> PresentationService:
    ask = ScriptedAssistant(answer) if answer else None
    return PresentationService(root=tmp_path / "store", output_dir=tmp_path / "out", ask=ask)


# --------------------------------------------------------------- the reader


def test_a_day_schedule_parses_into_topics_in_order() -> None:
    plan = parse_schedule(SCHEDULE, title="Reservoir Engineering")
    assert plan.title == "Reservoir Engineering"
    assert [topic.title for topic in plan.topics] == [
        "Porosity and permeability",
        "Well logging",
        "Well testing",
    ]
    assert [topic.number for topic in plan.topics] == [1, 2, 3]
    assert plan.topics[0].details == ("Definition of porosity", "Effective vs total porosity")
    assert plan.topics[2].details == ()
    assert plan.topics[0].heading == "Day 1 — Porosity and permeability"


def test_arabic_days_parse_and_keep_their_ordinal() -> None:
    text = "اليوم الأول: المسامية والنفاذية\n- تعريف المسامية\nاليوم الثاني: تسجيل الآبار\n"
    plan = parse_schedule(text)
    assert [topic.title for topic in plan.topics] == ["المسامية والنفاذية", "تسجيل الآبار"]
    # The Arabic ordinal is used as written, never replaced with a digit.
    assert plan.topics[0].heading == "اليوم الأول — المسامية والنفاذية"


def test_session_and_word_numbers_parse() -> None:
    plan = parse_schedule("Session one - Intro\nSession two - Darcy's law")
    assert [topic.title for topic in plan.topics] == ["Intro", "Darcy's law"]
    assert [topic.number for topic in plan.topics] == [1, 2]


def test_a_bare_topic_list_still_parses_in_order() -> None:
    plan = parse_schedule("Porosity\nPermeability\nWell logs")
    assert [topic.title for topic in plan.topics] == ["Porosity", "Permeability", "Well logs"]
    # A bare list has no day label, so each line is its own numbered topic.
    assert all(topic.label == "" for topic in plan.topics)


def test_a_blank_schedule_yields_no_plan() -> None:
    plan = parse_schedule("   \n\n\t\n")
    assert plan.is_empty
    assert plan.topics == ()


def test_an_unrelated_line_is_never_read_as_a_day() -> None:
    # "Playlist" starts with "Play", not a day noun, so it is a plain topic.
    plan = parse_schedule("Playlist\nSomething else")
    assert [topic.title for topic in plan.topics] == ["Playlist", "Something else"]
    assert all(topic.label == "" for topic in plan.topics)


# --------------------------------------------------------- schedule detection


def test_a_day_schedule_looks_like_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert looks_like_schedule(SCHEDULE)


def test_an_arabic_day_schedule_looks_like_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert looks_like_schedule(
        "اليوم الأول: المسامية والنفاذية\n- تعريف المسامية\nاليوم الثاني: سجلات البئر\n"
    )


def test_a_bare_topic_list_looks_like_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert looks_like_schedule("Porosity\nPermeability\nWell logging\nWell testing\n")


def test_a_question_is_never_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert not looks_like_schedule("What is porosity?")
    assert not looks_like_schedule("How does permeability relate to porosity?")


def test_a_paragraph_is_never_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    prose = (
        "Porosity is the fraction of the bulk volume that is void. Effective porosity "
        "counts only the connected pores, while total porosity counts them all."
    )
    assert not looks_like_schedule(prose)


def test_a_list_of_full_sentences_is_never_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert not looks_like_schedule(
        "Porosity is the void fraction of the rock.\n"
        "Permeability measures how easily fluid flows.\n"
        "Well logs estimate both from the borehole.\n"
    )


def test_a_single_day_with_a_topic_beneath_it_is_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert looks_like_schedule("Day 1: Porosity\n- Definition\n- Measurement\n")


def test_a_single_day_line_alone_is_not_a_schedule() -> None:
    from core.presentation_schedule import looks_like_schedule

    assert not looks_like_schedule("Day 1: Porosity and permeability")



def test_the_outline_has_one_section_per_day_with_its_subtopics() -> None:
    plan = parse_schedule(SCHEDULE)
    sections = outline_from_schedule(plan)
    assert [section.heading for section in sections] == [
        "Day 1 — Porosity and permeability",
        "Day 2 — Well logging",
        "Day 3 — Well testing",
    ]
    assert sections[0].subsections == ("Definition of porosity", "Effective vs total porosity")
    assert sections[0].estimated_slides == 2
    assert sections[2].subsections == ()


def test_the_plan_round_trips_through_its_primitive() -> None:
    plan = parse_schedule(SCHEDULE, title="Deck")
    restored = SchedulePlan.from_primitive(json.loads(json.dumps(plan.to_primitive())))
    assert restored == plan


def test_a_topic_needs_a_title() -> None:
    with pytest.raises(ValueError):
        ScheduleTopic(label="Day", number=1, title="   ")


# --------------------------------------------------------------- the build


def test_build_from_schedule_makes_a_numbered_deck(tmp_path: Path) -> None:
    text = (
        "Porosity\n"
        "- Porosity is the fraction of bulk volume that is void.\n"
        "- Effective porosity counts only interconnected pores.\n"
    )
    service = _service(tmp_path, answer=text)
    report = service.build_from_schedule("Reservoir Engineering", SCHEDULE)
    assert report.topic_count == 3
    assert report.drafted == 7  # 3 day headings + 2 + 2 sub-topics
    assert report.state == STATE_SLIDE_REVIEW
    project = service.get(report.project_id)
    numbers = [slide.number for slide in project.ordered_slides()]
    assert numbers == [1, 2, 3, 4, 5, 6, 7]
    # Every slide keeps the day it came from and the sub-topic it covers.
    assert project.ordered_slides()[0].section == "Day 1 — Porosity and permeability"
    assert project.ordered_slides()[1].subsection == "Definition of porosity"
    # The approved outline is the schedule itself, so the deck is a real structure.
    assert project.outline is not None
    assert len(project.outline.sections) == 3


def test_build_from_schedule_asks_a_professional_scientific_prompt(tmp_path: Path) -> None:
    assistant = ScriptedAssistant("Title\n- One scientific statement.\n")
    service = PresentationService(root=tmp_path / "s", output_dir=tmp_path / "o", ask=assistant)
    service.build_from_schedule("Deck", "Day 1: Porosity\n")
    assert assistant.prompts, "the assistant must have been asked for slide content"
    prompt = assistant.prompts[0]
    assert "professional training deck" in prompt
    assert "Porosity" in prompt
    assert "units" in prompt


def test_build_from_schedule_with_supplied_text_needs_no_model(tmp_path: Path) -> None:
    service = _service(tmp_path)  # no assistant at all
    report = service.build_from_schedule(
        "Deck", "Day 1: Porosity\n", text="Title\n- A bullet.\n- Another bullet.\n"
    )
    assert report.drafted == 1
    assert report.state == STATE_SLIDE_REVIEW


def test_build_from_schedule_flags_a_slide_whose_material_all_repeats(
    tmp_path: Path,
) -> None:
    """A schedule slide that only repeats an earlier one is flagged, not silent.

    The model (or the extractive fallback) can return the same material for two
    related headings. The outline path already keeps the single strongest line
    and sets a note; the schedule path must do the same, so a repeated slide is
    visible in review rather than passed off as fresh content.
    """
    from gui.presentation import REPEATED_HEADING_NOTE

    # The same two bullets for every topic, so every slide after the first has
    # nothing new and must be flagged.
    text = "Title\n- One repeated statement.\n- Another repeated statement.\n"
    service = _service(tmp_path, answer=text)
    report = service.build_from_schedule(
        "Deck", "Day 1: Porosity\n- A subtopic\nDay 2: Logging\n"
    )
    project = service.get(report.project_id)
    slides = project.ordered_slides()
    assert len(slides) >= 3
    # The first slide shows the material; every later one is flagged as repeated.
    assert slides[0].notes == ""
    assert all(slide.notes == REPEATED_HEADING_NOTE for slide in slides[1:])


def test_build_from_schedule_refuses_a_blank_schedule(tmp_path: Path) -> None:
    service = _service(tmp_path, answer="x\n- y\n")
    from core.presentation import PresentationError

    with pytest.raises(PresentationError):
        service.build_from_schedule("Deck", "   \n\n")


def test_parse_schedule_on_the_service_matches_the_engine(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.parse_schedule(SCHEDULE).topics == parse_schedule(SCHEDULE).topics


# --------------------------------------------------------------- the CLI


def test_cli_from_schedule_parse_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cli.presentation import main

    root = tmp_path / "store"
    rc = main(
        [
            "--root", str(root),
            "from-schedule", "Petroleum 101",
            "--schedule", SCHEDULE,
            "--parse-only",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Day 1 — Porosity and permeability" in out
    assert "Well logging" in out


def test_cli_from_schedule_builds_with_text(tmp_path: Path) -> None:
    from cli.presentation import main

    root = tmp_path / "store"
    text_file = tmp_path / "slide.txt"
    text_file.write_text("Porosity\n- A bullet.\n- Another bullet.\n", encoding="utf-8")
    rc = main(
        [
            "--root", str(root),
            "from-schedule", "Petroleum 101",
            "--schedule", "Day 1: Porosity\nDay 2: Permeability\n",
            "--text-file", str(text_file),
        ]
    )
    assert rc == 0
    from gui.presentation import PresentationService

    service = PresentationService(root=root)
    project = service.list_projects()[0]
    assert len(project.slides) == 2


def test_cli_from_schedule_json_is_pure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cli.presentation import main

    root = tmp_path / "store"
    rc = main(
        [
            "--json", "--root", str(root),
            "from-schedule", "Deck",
            "--schedule", "Day 1: A\n",
            "--text", "A\n- bullet one\n- bullet two\n",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drafted"] == 1
    assert payload["state"] == STATE_SLIDE_REVIEW
    assert payload["topics"] == 1


def test_cli_from_schedule_derives_the_title_when_it_is_omitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The title is optional: pasting only a schedule still builds a deck."""
    from cli.presentation import main

    root = tmp_path / "store"
    rc = main(
        [
            "--json", "--root", str(root),
            "from-schedule",
            "--schedule", "Day 1: Porosity\nDay 2: Permeability\n",
            "--text", "A\n- bullet one\n- bullet two\n",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drafted"] == 2
    from gui.presentation import PresentationService

    project = PresentationService(root=root).list_projects()[0]
    # The title came from the schedule's own first topic, not a blank name.
    assert "Porosity" in project.title


def test_cli_from_schedule_parse_only_needs_no_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cli.presentation import main

    root = tmp_path / "store"
    rc = main(
        [
            "--root", str(root),
            "from-schedule",
            "--schedule", "Day 1: Porosity\nDay 2: Permeability\n",
            "--parse-only",
        ]
    )
    assert rc == 0
    assert "Porosity" in capsys.readouterr().out

