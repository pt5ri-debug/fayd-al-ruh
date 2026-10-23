"""Tests for the Presentation Studio PowerPoint export engine.

Three layers are covered:

* the **preparation layer** (:mod:`core.presentation_layout`) -- layout
  classification, the fit estimate, image validation, and safe filenames -- which
  needs no third-party package at all;
* the **render engine** (:mod:`core.pptx_engine`) -- a real ``.pptx`` is written,
  reopened, and checked for slide count, order, titles, notes, images, and
  charts. These tests skip when the optional ``python-pptx`` extra is absent, the
  same way the OCR and PDF tests skip without their extras;
* the **integration** -- the service writes the approved deck, the file is
  reopened, and the result is validated; plus a regression check that the
  standard-library writer still works when the engine is forced to ``stdlib``.
"""

from __future__ import annotations

import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from core.presentation import (
    ASPECT_4_3,
    ASPECT_16_9,
    CONTENT_BULLETS,
    CONTENT_IMAGE,
    CONTENT_SECTION,
    CONTENT_TABLE,
    CONTENT_TITLE,
    Deck,
    PresentationError,
    PresentationSettings,
    PresentationWriter,
    Slide,
    validate_pptx,
)
from core.presentation_layout import (
    LAYOUT_BULLETS,
    LAYOUT_CHART,
    LAYOUT_CONCLUSION,
    LAYOUT_EQUATION,
    LAYOUT_IMAGE,
    LAYOUT_OBJECTIVES,
    LAYOUT_SECTION,
    LAYOUT_SUMMARY,
    LAYOUT_TABLE,
    LAYOUT_TITLE,
    LAYOUT_TWO_COLUMN,
    classify_layout,
    estimate_fit,
    looks_rtl,
    prepare_deck,
    prepare_slide,
    resolve_aspect,
    resolve_layout,
    resolve_theme,
    safe_filename,
    safe_output_path,
    split_columns,
    theme_names,
    unique_export_path,
    validate_image,
)

try:  # pragma: no cover - the presence check itself is trivial
    import pptx as _pptx  # noqa: F401

    PPTX_AVAILABLE = True
except ImportError:  # pragma: no cover
    PPTX_AVAILABLE = False

requires_pptx = pytest.mark.skipif(
    not PPTX_AVAILABLE, reason="the optional python-pptx extra is not installed"
)


# --- helpers ---------------------------------------------------------------


def _png(path: Path, *, width: int = 120, height: int = 80, rgb: tuple[int, int, int] = (200, 80, 80)) -> Path:
    """Write a tiny valid PNG with only the standard library."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    return path


def _deck() -> Deck:
    """A deck that exercises every content type the engine supports."""
    slides = (
        Slide(
            slide_id="S1",
            order=1,
            title="Reservoir Porosity",
            content_type=CONTENT_TITLE,
            content=["A petroleum engineering primer"],
            notes="Welcome the audience.",
        ),
        Slide(
            slide_id="S2",
            order=2,
            title="Learning objectives",
            content=["Define porosity", "Distinguish effective porosity"],
        ),
        Slide(slide_id="S3", order=3, title="Reservoir", content_type=CONTENT_SECTION),
        Slide(
            slide_id="S4",
            order=4,
            title="Porosity vs permeability",
            content=["Storage | Flow", "Void fraction | Darcy's law"],
            section="Reservoir",
            subsection="Porosity",
        ),
        Slide(
            slide_id="S5",
            order=5,
            title="Key values",
            content_type=CONTENT_TABLE,
            table=[["Rock", "Porosity"], ["Sandstone", "0.20"], ["Limestone", "0.15"]],
        ),
        Slide(
            slide_id="S6",
            order=6,
            title="Darcy's law",
            content_type="equation",
            content=["Q = k A (ΔP) / (μ L)"],
        ),
        Slide(
            slide_id="S7",
            order=7,
            title="Conclusion",
            content=["Porosity and permeability differ", "Both control flow"],
        ),
    )
    return Deck(title="Porosity Deck", slides=slides)


# --- preparation: layouts --------------------------------------------------


def test_every_content_type_maps_to_a_layout() -> None:
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type=CONTENT_TITLE)) == LAYOUT_TITLE
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type=CONTENT_SECTION)) == LAYOUT_SECTION
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type=CONTENT_TABLE)) == LAYOUT_TABLE
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type=CONTENT_IMAGE)) == LAYOUT_IMAGE
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type="equation")) == LAYOUT_EQUATION
    assert classify_layout(Slide(slide_id="a", order=1, title="x", content_type=CONTENT_BULLETS)) == LAYOUT_BULLETS


def test_a_title_cue_promotes_a_slide_to_a_semantic_layout() -> None:
    assert classify_layout(Slide(slide_id="a", order=1, title="Learning objectives")) == LAYOUT_OBJECTIVES
    assert classify_layout(Slide(slide_id="a", order=1, title="Summary")) == LAYOUT_SUMMARY
    assert classify_layout(Slide(slide_id="a", order=1, title="Conclusion")) == LAYOUT_CONCLUSION
    # The Arabic equivalents are recognised too.
    assert classify_layout(Slide(slide_id="a", order=1, title="الأهداف")) == LAYOUT_OBJECTIVES


def test_a_pipe_separator_asks_for_a_two_column_slide() -> None:
    slide = Slide(slide_id="a", order=1, title="Compare", content=["left | right"])
    assert classify_layout(slide) == LAYOUT_TWO_COLUMN
    left, right = split_columns(slide.content)
    assert left == ["left"] and right == ["right"]


def test_an_explicit_layout_hint_wins() -> None:
    slide = Slide(slide_id="a", order=1, title="Trend", content_type=CONTENT_TABLE, layout="chart")
    assert classify_layout(slide) == LAYOUT_CHART
    assert resolve_layout("two-column") == LAYOUT_TWO_COLUMN


def test_an_unknown_layout_is_refused() -> None:
    with pytest.raises(PresentationError):
        resolve_layout("hologram")


# --- preparation: fit ------------------------------------------------------


def test_a_short_block_fits_and_a_long_one_is_flagged() -> None:
    fits = estimate_fit(["one line"], size_pt=20, width_emu=9000000, height_emu=3000000)
    assert fits.ok and fits.lines_needed <= fits.lines_available
    over = estimate_fit(["word " * 400], size_pt=20, width_emu=9000000, height_emu=400000)
    assert not over.ok and over.overflow_ratio > 1.0


def test_an_over_full_slide_is_flagged_not_trimmed() -> None:
    slide = Slide(
        slide_id="a",
        order=1,
        title="Dense",
        content=[f"Bullet {index} " + "detail " * 30 for index in range(20)],
    )
    plan = prepare_slide(slide, settings=PresentationSettings(), theme=resolve_theme("professional"))
    # The text is untouched -- every line is still there...
    assert len(plan.blocks[0].lines) == 20
    # ...but the plan says it does not fit, with a warning rather than a clip.
    assert plan.overflows
    assert any("fit" in warning or "splitting" in warning for warning in plan.warnings)


# --- preparation: images ---------------------------------------------------


def test_image_validation_accepts_a_real_file_and_rejects_the_rest(tmp_path: Path) -> None:
    good = _png(tmp_path / "core.png")
    assert validate_image(str(good)).ok
    assert not validate_image(str(tmp_path / "missing.png")).ok
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    assert not validate_image(str(bad)).ok
    assert not validate_image("").ok


def test_a_missing_image_is_reported_but_the_slide_survives(tmp_path: Path) -> None:
    slide = Slide(
        slide_id="a",
        order=1,
        title="Photo",
        content_type=CONTENT_IMAGE,
        images=[str(tmp_path / "nope.png")],
        content=["a caption"],
    )
    plan = prepare_slide(slide, settings=PresentationSettings(), theme=resolve_theme("professional"))
    assert plan.layout == LAYOUT_IMAGE
    assert any("could not be embedded" in warning for warning in plan.warnings)


# --- preparation: names, geometry, themes ----------------------------------


def test_safe_filename_strips_traversal_and_appends_the_id() -> None:
    name = safe_filename("../../etc/passwd", export_id="EXP-001")
    assert "/" not in name and ".." not in name
    assert name.endswith(".pptx") and "EXP-001" in name


def test_safe_output_path_refuses_to_escape_the_directory(tmp_path: Path) -> None:
    inside = safe_output_path(tmp_path, "deck.pptx")
    assert inside.parent == tmp_path.resolve()
    with pytest.raises(PresentationError):
        # A crafted name with separators is reduced to its basename, which is
        # still inside; the guard is exercised by the resolved-parent check.
        safe_output_path(tmp_path, "sub/../../escape.pptx")


def test_unique_export_path_never_overwrites(tmp_path: Path) -> None:
    first = unique_export_path(tmp_path, "Deck")
    first.write_bytes(b"x")
    second = unique_export_path(tmp_path, "Deck")
    assert second != first and not second.exists()


def test_aspects_and_themes_resolve() -> None:
    assert resolve_aspect("") == ASPECT_16_9
    assert resolve_aspect("4:3") == ASPECT_4_3
    assert PresentationSettings().size == (12192000, 6858000)
    assert "professional" in theme_names()
    with pytest.raises(PresentationError):
        resolve_aspect("21:9")


def test_right_to_left_text_is_detected() -> None:
    assert looks_rtl("هندسة النفط")
    assert not looks_rtl("petroleum engineering")


# --- engine: rendering -----------------------------------------------------


@requires_pptx
def test_the_engine_writes_a_real_deck_with_every_slide_in_order(tmp_path: Path) -> None:
    from core.pptx_engine import PptxEngine, validate_deck

    plan = prepare_deck(_deck(), settings=PresentationSettings())
    out = PptxEngine().write(plan, tmp_path / "deck.pptx")
    assert out.exists()
    validation = validate_deck(out, plan)
    assert validation.ok
    assert validation.slide_count == len(plan.slides)
    # The titles come back in the approved order.
    assert validation.titles[0] == "Reservoir Porosity"
    assert validation.titles[-1] == "Conclusion"


@requires_pptx
def test_speaker_notes_survive_saving_and_reopening(tmp_path: Path) -> None:
    from core.pptx_engine import PptxEngine, validate_deck

    plan = prepare_deck(_deck(), settings=PresentationSettings())
    out = PptxEngine().write(plan, tmp_path / "notes.pptx")
    validation = validate_deck(out, plan)
    assert validation.notes_slides == (1,)

    import pptx

    reopened = pptx.Presentation(str(out))
    assert reopened.slides[0].notes_slide.notes_text_frame.text == "Welcome the audience."


@requires_pptx
def test_a_16_by_9_deck_and_a_4_by_3_deck_have_the_right_geometry(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine

    wide = PptxEngine().write(prepare_deck(_deck()), tmp_path / "wide.pptx")
    standard = PptxEngine().write(
        prepare_deck(_deck(), settings=PresentationSettings(aspect=ASPECT_4_3)),
        tmp_path / "standard.pptx",
    )
    assert (pptx.Presentation(str(wide)).slide_width, pptx.Presentation(str(wide)).slide_height) == (
        12192000,
        6858000,
    )
    assert pptx.Presentation(str(standard)).slide_width == 9144000


@requires_pptx
def test_an_image_is_embedded_and_a_missing_one_is_skipped(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine

    image = _png(tmp_path / "core.png")
    deck = Deck(
        title="Images",
        slides=(
            Slide(slide_id="a", order=1, title="Photo", content_type=CONTENT_IMAGE, images=[str(image)]),
            Slide(
                slide_id="b",
                order=2,
                title="Missing",
                content_type=CONTENT_IMAGE,
                images=[str(tmp_path / "gone.png")],
            ),
        ),
    )
    out = PptxEngine().write(prepare_deck(deck), tmp_path / "images.pptx")
    reopened = pptx.Presentation(str(out))
    pictures = [shape for shape in reopened.slides[0].shapes if shape.shape_type == 13]
    assert pictures, "the first slide should carry an embedded picture"
    # The second slide has no picture, but the slide itself is still there.
    assert len(reopened.slides._sldIdLst) == 2


@requires_pptx
def test_a_numeric_table_becomes_a_chart(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine

    deck = Deck(
        title="Chart",
        slides=(
            Slide(
                slide_id="a",
                order=1,
                title="Production",
                content_type=CONTENT_TABLE,
                table=[["Year", "Rate"], ["2020", "10"], ["2021", "20"]],
                layout="chart",
            ),
        ),
    )
    out = PptxEngine().write(prepare_deck(deck), tmp_path / "chart.pptx")
    reopened = pptx.Presentation(str(out))
    has_chart = any(shape.has_chart for shape in reopened.slides[0].shapes)
    assert has_chart


@requires_pptx
def test_a_non_numeric_chart_slide_falls_back_to_a_table(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine

    deck = Deck(
        title="Not a chart",
        slides=(
            Slide(
                slide_id="a",
                order=1,
                title="Labels",
                content_type=CONTENT_TABLE,
                table=[["Name", "Note"], ["A", "text"], ["B", "text"]],
                layout="chart",
            ),
        ),
    )
    out = PptxEngine().write(prepare_deck(deck), tmp_path / "fallback.pptx")
    reopened = pptx.Presentation(str(out))
    has_table = any(shape.has_table for shape in reopened.slides[0].shapes)
    assert has_table


@requires_pptx
def test_arabic_and_mixed_text_round_trip(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine, validate_deck

    deck = Deck(
        title="ثنائي اللغة",
        slides=(
            Slide(
                slide_id="a",
                order=1,
                title="هندسة النفط",
                content=["Petroleum engineering", "هندسة النفط"],
                notes="ملاحظات",
            ),
        ),
    )
    plan = prepare_deck(deck, settings=PresentationSettings(rtl=True))
    out = PptxEngine().write(plan, tmp_path / "rtl.pptx")
    validation = validate_deck(out, plan)
    assert validation.ok and validation.notes_slides == (1,)
    reopened = pptx.Presentation(str(out))
    texts = "\n".join(
        shape.text_frame.text for shape in reopened.slides[0].shapes if shape.has_text_frame
    )
    assert "هندسة النفط" in texts


@requires_pptx
def test_slide_numbers_and_metadata_are_written(tmp_path: Path) -> None:
    import pptx
    from core.pptx_engine import PptxEngine

    settings = PresentationSettings(author="A. Author", company="SPE", footer="Fayd al-Ruh")
    plan = prepare_deck(_deck(), settings=settings)
    out = PptxEngine().write(plan, tmp_path / "meta.pptx")
    reopened = pptx.Presentation(str(out))
    assert reopened.core_properties.author == "A. Author"
    assert reopened.core_properties.comments == "Fayd al-Ruh"
    # A body slide carries its number as a text run.
    body_text = " ".join(
        shape.text_frame.text for shape in reopened.slides[3].shapes if shape.has_text_frame
    )
    assert "4" in body_text


@requires_pptx
def test_the_engine_does_not_overwrite_through_the_service(tmp_path: Path) -> None:
    from core.pptx_engine import PptxEngine

    plan = prepare_deck(_deck())
    target = tmp_path / "deck.pptx"
    PptxEngine().write(plan, target)
    # A second write through the same explicit path replaces atomically (the
    # service is what enforces the no-overwrite rule).
    PptxEngine().write(plan, target)
    assert target.exists()


def test_the_engine_reports_when_python_pptx_is_absent() -> None:
    from core import pptx_engine

    assert pptx_engine.pptx_available() == PPTX_AVAILABLE
    if not PPTX_AVAILABLE:  # pragma: no cover - only when the extra is absent
        with pytest.raises(pptx_engine.PptxUnavailable):
            pptx_engine._require_pptx()


# --- integration: the service ----------------------------------------------


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


def _service(tmp_path: Path):
    from gui.presentation import PresentationService

    return PresentationService(
        root=tmp_path / "store",
        output_dir=tmp_path / "out",
        ask=ScriptedAssistant("Porosity is the void fraction of a rock."),
    )


def _walk(service, project_id: str) -> None:
    service.draft_outline(project_id, text="Reservoir\n  - Porosity\n  - Permeability")
    service.review_outline(project_id)
    service.approve_outline(project_id)
    service.draft_slides_from_outline(project_id, batch=5)
    service.review_slides(project_id)
    service.approve_slides(project_id)
    service.approve_content(project_id)


def test_the_service_exports_through_the_engine_and_reports_the_download(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Porosity Deck")
    _walk(service, project.project_id)
    report = service.export(project.project_id)
    assert report.ok, report.message
    assert report.slide_count >= 1
    assert Path(report.path).exists()
    download = report.download
    assert download["path"] == report.path
    assert download["filename"] == Path(report.path).name
    assert download["directory"] == str(Path(report.path).parent)
    if PPTX_AVAILABLE:
        assert report.engine == "pptx"


def test_the_stdlib_engine_is_always_available_and_writes_a_valid_package(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Fallback Deck")
    _walk(service, project.project_id)
    report = service.export(project.project_id, engine="stdlib")
    assert report.ok and report.engine == "stdlib"
    assert validate_pptx(report.path).ok


def test_the_service_refuses_to_overwrite_an_explicit_file(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Guarded Deck")
    _walk(service, project.project_id)
    target = tmp_path / "out" / "chosen.pptx"
    first = service.export(project.project_id, path=target)
    assert first.ok
    # A second export to the same path is refused rather than silently replacing.
    second = service.export(project.project_id, path=target)
    assert not second.ok and "already exists" in second.message
    # With an explicit overwrite it is allowed.
    third = service.export(project.project_id, path=target, overwrite=True)
    assert third.ok


def test_a_default_export_uses_a_unique_name_and_never_overwrites(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Repeat Deck")
    _walk(service, project.project_id)
    first = service.export(project.project_id)
    second = service.export(project.project_id)
    assert first.ok and second.ok
    assert first.path != second.path


def test_an_explicit_engine_that_is_absent_is_refused_honestly(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Engine Deck")
    _walk(service, project.project_id)
    if PPTX_AVAILABLE:
        assert service.resolve_engine("pptx") == "pptx"
    else:  # pragma: no cover - only when the extra is absent
        with pytest.raises(PresentationError):
            service.export(project.project_id, engine="pptx")
    with pytest.raises(PresentationError):
        service.resolve_engine("hologram")


def test_the_export_warnings_preview_matches_what_an_export_reports(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Warning Deck")
    _walk(service, project.project_id)
    preview = service.export_warnings(project.project_id)
    report = service.export(project.project_id)
    assert report.ok
    assert preview == report.warnings


def test_settings_persist_on_the_project(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Settings Deck")
    updated = service.update_settings(project.project_id, theme="dark", aspect=ASPECT_4_3, footer="Fayd")
    assert updated.theme == "dark" and updated.aspect == ASPECT_4_3
    # A fresh service reads the stored settings back.
    reloaded = _service(tmp_path).get(project.project_id)
    assert reloaded.settings.theme == "dark"
    assert reloaded.settings.aspect == ASPECT_4_3
    assert reloaded.settings.footer == "Fayd"


def test_a_failed_export_preserves_the_project_for_a_retry(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Retry Deck")
    _walk(service, project.project_id)
    # A read-only directory makes the write fail; the project must survive.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        report = service.export(project.project_id, path=blocked / "deck.pptx")
    finally:
        blocked.chmod(0o700)
    assert not report.ok
    # The project is still loadable and can be exported again elsewhere.
    assert service.get(project.project_id).project_id == project.project_id
    retry = service.export(project.project_id)
    assert retry.ok


# --- regression: the original writer still works ---------------------------


def test_the_original_stdlib_writer_still_writes_a_valid_deck(tmp_path: Path) -> None:
    out = PresentationWriter().write(_deck(), tmp_path / "legacy.pptx")
    result = validate_pptx(out)
    assert result.ok and result.slide_count == 7
    with zipfile.ZipFile(out) as archive:
        assert "ppt/presentation.xml" in archive.namelist()


def test_an_exported_project_can_be_exported_again(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project = service.new_project("Twice Deck")
    _walk(service, project.project_id)
    first = service.export(project.project_id)
    second = service.export(project.project_id)
    assert first.ok and second.ok
    assert len(service.export_history(project.project_id)) == 2


# --- the CLI twin ----------------------------------------------------------


def _cli_ready_deck(root: str) -> None:
    from cli.presentation import main

    assert main(["--root", root, "new", "CLI Engine Deck"]) == 0
    assert main(["--root", root, "outline", "PRS-001", "--text", "Reservoir\n- Porosity"]) == 0
    assert main(["--root", root, "review-outline", "PRS-001"]) == 0
    assert main(["--root", root, "approve-outline", "PRS-001"]) == 0
    assert main(["--root", root, "add", "PRS-001", "--title", "Porosity", "--bullet", "Void fraction"]) == 0
    assert main(["--root", root, "review-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-slides", "PRS-001"]) == 0
    assert main(["--root", root, "approve-content", "PRS-001"]) == 0


def test_the_cli_exports_with_the_engine_and_reports_the_download(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    import json

    from cli.presentation import main

    root = str(tmp_path / "store")
    out = str(tmp_path / "out")
    _cli_ready_deck(root)
    capsys.readouterr()

    assert main(["--root", root, "--output-dir", out, "--json", "export", "PRS-001", "--theme", "academic"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    assert report["download"]["filename"].endswith(".pptx")
    assert Path(report["download"]["path"]).exists()
    if PPTX_AVAILABLE:
        assert report["engine"] == "pptx"


def test_the_cli_settings_and_warnings_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    import json

    from cli.presentation import main

    root = str(tmp_path / "store")
    _cli_ready_deck(root)
    capsys.readouterr()

    assert main(["--root", root, "--json", "settings", "PRS-001", "--set-theme", "dark"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["theme"] == "dark"

    assert main(["--root", root, "--json", "warnings", "PRS-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and isinstance(payload["warnings"], list)


def test_the_cli_can_force_the_stdlib_engine(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    from cli.presentation import main

    root = str(tmp_path / "store")
    out = str(tmp_path / "out")
    _cli_ready_deck(root)
    capsys.readouterr()

    assert main(["--root", root, "--output-dir", out, "--json", "export", "PRS-001", "--engine", "stdlib"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["engine"] == "stdlib"
    assert validate_pptx(report["path"]).ok


def test_the_demo_command_builds_and_exports_a_complete_deck(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    import json

    from cli.presentation import main

    root = str(tmp_path / "store")
    out = str(tmp_path / "out")
    assert main(["--root", root, "--output-dir", out, "--json", "demo"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"]
    # The sample deck covers every layout family, so it has a real slide count.
    assert report["slide_count"] >= 8
    assert Path(report["path"]).exists()
    assert report["download"]["filename"].endswith(".pptx")


def test_a_slide_can_carry_an_explicit_layout_hint(tmp_path: Path) -> None:
    from gui.presentation import SlideSpec

    service = _service(tmp_path)
    project = service.new_project("Hint Deck")
    service.draft_outline(project.project_id, text="Reservoir\n- Porosity")
    service.review_outline(project.project_id)
    service.approve_outline(project.project_id)
    service.draft_slides_from_outline(project.project_id, batch=5)
    service.review_slides(project.project_id)
    slide = service.add_slide(
        project.project_id,
        SlideSpec(title="Trend", content_type=CONTENT_TABLE, table=(("A", "1"),), layout="chart"),
    )
    assert slide.layout == "chart"
    # The hint survives a save and reload.
    assert service.get(project.project_id).slide(slide.slide_id).layout == "chart"


def test_the_service_can_export_a_prepared_plan_without_a_project(tmp_path: Path) -> None:
    """The engine and preparation layer work standalone (no project, no store)."""
    plan = prepare_deck(_deck(), settings=PresentationSettings(theme="light"))
    assert plan.slide_count == 7
    assert plan.theme.name == "light"
    primitive = plan.to_primitive()
    assert primitive["slide_count"] == 7
    assert primitive["slides"][0]["layout"] == LAYOUT_TITLE
