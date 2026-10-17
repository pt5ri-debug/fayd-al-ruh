"""Integration tests for Presentation Studio in the GUI service and the CLI.

These exercise the *whole* feature the way the window and the CLI do: create a
project, walk it to an approved deck, validate it, manage it (rename, duplicate,
delete), and export it. Everything runs through the real
:class:`gui.presentation.PresentationService` and the real
``cli.presentation`` entry point -- no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui.presentation import PresentationService, SlideSpec


def _ready_deck(service: PresentationService, title: str = "Reservoir Porosity") -> str:
    """Build a project all the way to approved content; return its id.

    The slide content is supplied directly (no assistant), so the test needs no
    model: it still walks the real state machine through every gate.
    """
    project = service.new_project(title)
    service.draft_outline(project.project_id, text="Reservoir\n- Porosity\n- Permeability")
    service.approve_outline(project.project_id)
    service.add_slide(
        project.project_id,
        SlideSpec(title="Porosity", content=["Void fraction", "Controls storage"]),
    )
    service.add_slide(
        project.project_id,
        SlideSpec(title="Permeability", content=["Transmits fluids", "Controls flow"]),
    )
    service.review_slides(project.project_id)
    service.approve_slides(project.project_id)
    service.approve_content(project.project_id)
    return project.project_id


def _service(tmp_path: Path) -> PresentationService:
    return PresentationService(root=tmp_path / "store", output_dir=tmp_path / "out")


# --- the service: dashboard, rename, duplicate, validate -------------------


def test_the_status_reports_a_dashboard_row_and_download(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    report = service.export(project_id)
    assert report.ok

    status = service.status(project_id)
    assert status.step == 8
    assert status.created_at and status.updated_at
    assert status.download["filename"].endswith(".pptx")
    assert Path(status.download["path"]).exists()
    row = status.dashboard_row()
    assert project_id in row
    assert "approved" in row


def test_renaming_a_project_keeps_its_slides_and_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    before = len(service.get(project_id).slides)

    renamed = service.rename_project(project_id, "  A Better Title  ")
    assert renamed.project_id == project_id
    assert renamed.title == "A Better Title"
    assert len(renamed.slides) == before
    assert service.get(project_id).title == "A Better Title"


def test_renaming_to_an_empty_title_is_refused(tmp_path: Path) -> None:
    from core.presentation import PresentationError

    service = _service(tmp_path)
    project_id = _ready_deck(service)
    with pytest.raises(PresentationError):
        service.rename_project(project_id, "   ")


def test_duplicating_a_project_copies_slides_but_not_exports(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    service.export(project_id)

    clone = service.duplicate_project(project_id)
    assert clone.project_id != project_id
    assert clone.title == f"{service.get(project_id).title} (copy)"
    assert len(clone.slides) == len(service.get(project_id).slides)
    assert clone.exports == []
    # The original is untouched, and both are stored.
    assert service.get(project_id).exports
    assert set(service.list_ids()) >= {project_id, clone.project_id}


def test_duplicating_with_an_explicit_title(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    clone = service.duplicate_project(project_id, title="Variant A")
    assert clone.title == "Variant A"


def test_validate_reports_a_ready_deck(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    result = service.validate_project(project_id)
    assert result.ok
    assert not result.issues
    assert "ready to export" in result.summary()
    assert result.as_dict()["ok"] is True


def test_validate_flags_an_empty_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Empty")
    result = service.validate_project(project.project_id)
    assert not result.ok
    assert any("no slides" in issue for issue in result.issues)


def test_validate_never_mutates_the_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    before = service.get(project_id).to_primitive()
    service.validate_project(project_id)
    assert service.get(project_id).to_primitive() == before


def test_validate_flags_a_numbering_gap(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = _ready_deck(service)
    project = service.get(project_id)
    # Punch a hole in the numbering without going through a supported edit.
    project.slides[0].order = 5
    service.store.save(project)
    result = service.validate_project(project_id)
    assert not result.ok
    assert any("dense" in issue for issue in result.issues)


# --- the CLI twin ----------------------------------------------------------


def _cli(tmp_path: Path, capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, str]:
    from cli.presentation import main

    root = str(tmp_path / "cli-store")
    out = str(tmp_path / "cli-out")
    code = main(["--root", root, "--output-dir", out, *argv])
    return code, capsys.readouterr().out


def test_the_cli_lists_projects_with_a_dashboard_table(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Alpha"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "list"]) == 0
    out = capsys.readouterr().out
    assert "PRS-001" in out
    assert "STATUS" in out
    assert "Discussion" in out


def test_the_cli_validates_a_ready_deck(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "CLI Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "Reservoir\n- Porosity"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "Porosity", "--bullet", "Void"]) == 0
    assert main(["--root", root, "review-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-content", "PRS-001"]) == 0
    capsys.readouterr()

    assert main(["--root", root, "--json", "validate", "PRS-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["slides"] == 1


def test_the_cli_validate_refuses_an_unready_deck(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Empty"]) == 0
    capsys.readouterr()
    # A non-ready deck is a refused gate (exit code 4), not a crash.
    assert main(["--root", root, "validate", "PRS-001"]) == 4
    out = capsys.readouterr().out
    assert "not ready" in out


def test_the_cli_renames_duplicates_and_deletes(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    assert main(["--root", root, "new", "Original"]) == 0
    capsys.readouterr()

    assert main(["--root", root, "rename", "PRS-001", "Renamed"]) == 0
    assert "Renamed" in capsys.readouterr().out
    assert main(["--root", root, "--json", "duplicate", "PRS-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["project_id"] == "PRS-002"
    assert payload["title"].endswith("(copy)")

    assert main(["--root", root, "delete", "PRS-002"]) == 0
    assert "Removed" in capsys.readouterr().out
    assert main(["--root", root, "--json", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["project_id"] for item in listed] == ["PRS-001"]


def test_the_cli_test_command_runs_the_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cli.presentation import main

    root = str(tmp_path / "store")
    # A single, fast, headless test file keeps this deterministic and quick.
    code = main(
        [
            "--root",
            root,
            "--json",
            "test",
            "tests/test_presentation.py",
            "-k",
            "test_a_transition_not_in_the_table_is_refused",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["returncode"] == 0
    assert code == 0
