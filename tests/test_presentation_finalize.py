"""Tests for the Presentation Studio's step 6 -> step 8 workflow.

The Studio's state machine is strict on purpose: the export contains exactly the
approved slides, so ``CONTENT_APPROVED`` is reachable only from ``SLIDE_REVIEW``
and only when every slide is approved. That strictness is what made the window's
*Approve the content* step look broken -- pressing it before approving the slides
raised instead of completing the workflow.

These tests pin the *fixed* behaviour at the service level, which is where the
rule lives:

* approving the whole deck in one call satisfies the gate;
* the content approval then succeeds and the deck is exportable;
* an empty slide is refused by name rather than approved silently;
* an edit after approval re-arms the gate, so the export cannot go stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.presentation import (
    STATE_CONTENT_APPROVED,
    STATE_SLIDE_REVIEW,
    PresentationError,
)
from gui.presentation import PresentationService, SlideSpec

PROSE = (
    "Porosity is the fraction of the bulk volume that is void space. "
    "Permeability is the ability of the rock to transmit a fluid."
)


def _assistant(_question: str) -> str:
    return PROSE


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=_assistant,
    )


def _at_slide_review(service: PresentationService) -> str:
    """Walk a project to SLIDE_REVIEW with two unapproved slides; return its id."""
    project = service.new_project("Workflow deck")
    service.draft_outline(
        project.project_id, text="Reservoir\n    - Porosity\n    - Permeability"
    )
    service.approve_outline(project.project_id)
    service.add_slides(
        project.project_id,
        [
            SlideSpec(title="Porosity", content=["Void fraction"]),
            SlideSpec(title="Permeability", content=["Flow"]),
        ],
    )
    service.review_slides(project.project_id)
    return project.project_id


# ------------------------------------------------------- the bulk-approve gate


def test_approving_every_slide_in_one_call_satisfies_the_content_gate(tmp_path: Path) -> None:
    """This is the fix: step 7 must not require a click per slide to succeed."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)

    assert service.get(project_id).state == STATE_SLIDE_REVIEW
    assert not service.get(project_id).all_slides_approved

    service.approve_slides(project_id)
    assert service.get(project_id).all_slides_approved

    approved = service.approve_content(project_id)
    assert approved.state == STATE_CONTENT_APPROVED
    assert approved.content_approved
    assert len(approved.approved_slides) == 2


def test_the_approved_deck_exports_after_the_bulk_approve(tmp_path: Path) -> None:
    """The whole point of steps 6-8: an approved deck writes a real .pptx."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    allowed, reason = service.can_export(project_id)
    assert allowed, reason

    result = service.export(project_id)
    assert result.ok
    assert result.slide_count == 2
    path = Path(result.path)
    assert path.exists() and path.suffix == ".pptx"


def test_approving_the_content_before_the_slides_is_still_refused(tmp_path: Path) -> None:
    """The gate stays honest: the fix approves the slides, it does not bypass them."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    with pytest.raises(PresentationError):
        service.approve_content(project_id)


def test_approving_the_content_from_the_drafting_state_is_refused(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Early")
    service.draft_outline(project.project_id, text="A\n- one")
    service.approve_outline(project.project_id)
    service.add_slides(project.project_id, [SlideSpec(title="A", content=["one"])])
    # Still SLIDE_DRAFTING: review is the first state an approval is accepted from.
    with pytest.raises(PresentationError):
        service.approve_content(project.project_id)


def test_an_empty_slide_is_named_rather_than_approved_silently(tmp_path: Path) -> None:
    """The service refuses an empty slide by number, so the user knows which one."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.add_slides(project_id, [SlideSpec(title="", content=[])])

    empty = [slide.number for slide in service.get(project_id).ordered_slides() if slide.is_empty]
    assert empty, "the empty slide must be detectable so the window can refuse it"
    with pytest.raises(PresentationError) as excinfo:
        service.approve_slides(project_id)
    assert str(empty[0]) in str(excinfo.value)
    # The refusal is complete: no slide was approved by the failed call.
    assert not service.get(project_id).all_slides_approved


def test_a_slide_with_only_a_title_is_not_empty(tmp_path: Path) -> None:
    """A title is content, so a title-only slide is approvable and exports."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.add_slides(project_id, [SlideSpec(title="Only a title")])
    assert not service.get(project_id).ordered_slides()[-1].is_empty


def test_a_wholly_blank_slide_is_empty(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.add_slides(project_id, [SlideSpec(title="", content=[])])
    assert service.get(project_id).ordered_slides()[-1].is_empty


# --------------------------------------------------- editing after approval


def test_editing_an_approved_slide_re_arms_the_gate(tmp_path: Path) -> None:
    """Nothing changes after the fact by accident: an edit needs a fresh approval."""
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    slide_id = service.get(project_id).ordered_slides()[0].slide_id
    service.edit_slide(project_id, slide_id, content=["A rewritten bullet"])

    project = service.get(project_id)
    # The edit returns the deck to review and clears the content approval, so the
    # export gate re-opens; the edited slide itself is no longer approved.
    assert project.state == STATE_SLIDE_REVIEW
    assert not project.content_approved
    assert not project.slides[0].approved
    allowed, reason = service.can_export(project_id)
    assert not allowed
    assert "approve" in reason.lower()


def test_the_deck_exports_again_after_re_approving(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    slide_id = service.get(project_id).ordered_slides()[0].slide_id
    service.edit_slide(project_id, slide_id, content=["A rewritten bullet"])
    service.approve_slides(project_id)
    service.approve_content(project_id)

    allowed, reason = service.can_export(project_id)
    assert allowed, reason
    assert service.export(project_id).ok


def test_reopening_approved_content_clears_the_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    reopened = service.reopen_content(project_id)
    assert reopened.state == STATE_SLIDE_REVIEW
    assert not reopened.content_approved
    allowed, _reason = service.can_export(project_id)
    assert not allowed


# ------------------------------------------------- the export order guarantee


def test_the_export_keeps_the_approved_order(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _at_slide_review(service)
    service.approve_slides(project_id)
    service.approve_content(project_id)

    ordered = [slide.title for slide in service.get(project_id).ordered_slides()]
    assert ordered == ["Porosity", "Permeability"]

    result = service.export(project_id)
    assert result.ok
    # Reopen the written file and confirm the order survived the round trip.
    from pptx import Presentation

    deck = Presentation(result.path)
    titles = [
        shape.text_frame.text
        for slide in deck.slides
        for shape in slide.shapes
        if shape.has_text_frame and shape.text_frame.text.strip()
    ]
    positions = [titles.index(title) for title in ordered if title in titles]
    assert positions == sorted(positions), f"order lost: {titles}"
