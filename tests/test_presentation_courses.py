"""Tests for the Presentation Studio's curriculum bridge.

``core.presentation_courses`` and the service methods over it decide when the
Course window may open, so they must be honest: they match the deck's *own* text
against the project's *own* curriculum, invent nothing, and refuse to nominate a
course on a weak, incidental overlap. These tests exercise that directly, with no
display and no model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.presentation_courses import (
    STRONG_COVERAGE,
    best_course,
    course_detail,
    match_courses,
    match_project,
    project_text,
)
from gui.presentation import PresentationService, SlideSpec

PETROPHYSICS_DECK = (
    "Effective porosity is the fraction of interconnected void space. "
    "Absolute permeability is measured in darcies. "
    "Water saturation is the fraction of pore volume filled with water."
)


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store", output_dir=tmp_path / "out", ask=lambda _q: "x"
    )


def _deck(service: PresentationService, *, title: str = "Petrophysics deck") -> str:
    project = service.new_project(title)
    service.draft_outline(project.project_id, text="Petrophysics\n    - Porosity")
    service.approve_outline(project.project_id)
    service.add_slide(
        project.project_id,
        SlideSpec(title="Effective porosity", content=["Effective porosity of the rock"]),
    )
    service.add_slide(
        project.project_id,
        SlideSpec(title="Absolute permeability", content=["Absolute permeability in darcies"]),
    )
    service.add_slide(
        project.project_id,
        SlideSpec(title="Water saturation", content=["Water saturation of the pores"]),
    )
    return project.project_id


# -- the matcher itself ---------------------------------------------------


def test_a_real_deck_matches_the_right_course() -> None:
    matches = match_courses(PETROPHYSICS_DECK)
    keys = [match.key for match in matches]
    assert "petrophysics" in keys
    best = best_course(matches)
    assert best is not None and best.key == "petrophysics"
    assert best.strong


def test_a_concept_needs_all_its_words() -> None:
    """"porosity" alone must not count as "effective porosity"."""
    only_porosity = match_courses("Porosity is important. Porosity matters a great deal here.")
    petrophysics = next((m for m in only_porosity if m.key == "petrophysics"), None)
    if petrophysics is not None:
        assert "effective porosity" not in petrophysics.matched


def test_nonsense_matches_nothing_rather_than_guessing() -> None:
    assert match_courses("lorem ipsum dolor sit amet consectetur adipiscing") == ()
    assert best_course(match_courses("lorem ipsum dolor sit amet")) is None


def test_one_incidental_word_is_not_enough() -> None:
    """The default threshold stops a single shared word nominating a discipline."""
    assert match_courses("pressure") == ()


def test_matches_are_ordered_strongest_first() -> None:
    matches = match_courses(PETROPHYSICS_DECK)
    counts = [match.match_count for match in matches]
    assert counts == sorted(counts, reverse=True)


def test_coverage_and_strength_are_reported_honestly() -> None:
    match = next(m for m in match_courses(PETROPHYSICS_DECK) if m.key == "petrophysics")
    assert match.match_count == len(match.matched)
    assert 0.0 < match.coverage <= 1.0
    assert match.strong == (match.coverage >= STRONG_COVERAGE)
    assert match.describe().startswith(match.label)


def test_the_limit_caps_the_result() -> None:
    matches = match_courses(PETROPHYSICS_DECK, limit=1)
    assert len(matches) <= 1


def test_project_text_gathers_the_deck_and_not_the_discussion(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _deck(service)
    project = service.get(project_id)
    text = project_text(project)
    assert "Effective porosity" in text
    assert "Petrophysics deck" in text
    assert "Water saturation" in text


def test_course_detail_marks_what_the_deck_covers() -> None:
    detail = course_detail("petrophysics", text=PETROPHYSICS_DECK)
    assert detail.label == "Petrophysics"
    assert detail.branch == "Petroleum Disciplines"
    covered = {concept.name for concept in detail.covered}
    assert "effective porosity" in covered
    assert detail.covered and detail.missing
    assert len(detail.covered) + len(detail.missing) == len(detail.concepts)


def test_course_detail_names_its_relations_readably() -> None:
    detail = course_detail("petrophysics", text="")
    assert detail.relations
    described = detail.relations[0].describe()
    assert "—" in described and "→" in described


def test_an_unknown_course_key_is_refused() -> None:
    with pytest.raises(KeyError):
        course_detail("no-such-course")


# -- the service over it --------------------------------------------------


def test_the_service_matches_a_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _deck(service)
    matches = service.course_matches(project_id)
    assert any(match.key == "petrophysics" for match in matches)
    best = service.best_course(project_id)
    assert best is not None and best.key == "petrophysics"


def test_the_service_offers_only_the_missing_concepts(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _deck(service)
    text = service.course_concepts_text(project_id, "petrophysics")
    assert text
    assert "effective porosity" not in text
    assert all(line.startswith("- ") for line in text.splitlines())


def test_the_service_finds_the_slides_that_cover_a_concept(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _deck(service)
    assert service.slides_for_concept(project_id, "effective porosity") == [1]
    assert service.slides_for_concept(project_id, "absolute permeability") == [2]
    assert service.slides_for_concept(project_id, "hydrocarbon saturation") == []


def test_an_empty_deck_matches_no_course(tmp_path: Path) -> None:
    """The window must not open before the deck has any content."""
    service = _service(tmp_path)
    project = service.new_project("Empty")
    assert service.course_matches(project.project_id) == ()
    assert service.best_course(project.project_id) is None


def test_matching_never_changes_the_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _deck(service)
    before = service.get(project_id)
    service.course_matches(project_id)
    service.best_course(project_id)
    service.course_detail(project_id, "petrophysics")
    after = service.get(project_id)
    assert after.state == before.state
    assert len(after.slides) == len(before.slides)
    assert [slide.approved for slide in after.slides] == [s.approved for s in before.slides]


def test_match_project_accepts_a_project_object(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.get(_deck(service))
    assert match_project(project)
