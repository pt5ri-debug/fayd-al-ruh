"""Slide-content preparation: turn an approved deck into a render plan.

This module is the **first half** of the Presentation Studio export engine and
holds no rendering dependency at all. It answers three questions about a deck
that has already been approved:

* *What kind of slide is this?* -- :func:`classify_layout` maps a
  :class:`~core.presentation.Slide` onto one of the named layouts (title,
  section divider, objectives, bullets, paragraph, two-column, table, equation,
  chart, image, summary, conclusion).
* *Will the content fit?* -- :func:`estimate_fit` measures the prepared text
  against the slide geometry and reports a :class:`FitEstimate`, so an
  over-full slide is **flagged** rather than silently clipped or shortened.
  Nothing here ever deletes or rewrites approved content.
* *Where do the pictures come from?* -- :func:`validate_image` checks an image
  reference against an extension allow-list and a size cap, so a missing or
  unusable file is reported and skipped instead of crashing the export.

The second half, :mod:`core.pptx_engine`, consumes a :class:`DeckPlan` and
writes the actual ``.pptx``. Keeping the two apart means the classification,
geometry, fit logic, and validation are all unit-testable with no third-party
package, and the renderer stays a thin, mechanical drawing layer.

Design follows the rest of the project: plain frozen dataclasses that round-trip
through ``to_primitive``/``from_primitive``, Unicode text throughout (Arabic,
English, and mixed content are all just strings), and no import of the renderer
in either direction.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.presentation import (
    ASPECT_4_3,
    ASPECT_16_9,
    ASPECTS,
    CONTENT_BULLETS,
    CONTENT_EQUATION,
    CONTENT_IMAGE,
    CONTENT_PARAGRAPH,
    CONTENT_SECTION,
    CONTENT_TABLE,
    CONTENT_TITLE,
    DEFAULT_ASPECT,
    DEFAULT_THEME,
    EMU_PER_INCH,
    EMU_PER_POINT,
    THEMES,
    Deck,
    DeckTheme,
    PresentationError,
    PresentationProject,
    PresentationSettings,
    Slide,
    resolve_aspect,
    resolve_theme,
    slide_size,
    slugify,
    theme_names,
)

# --------------------------------------------------------------------------- #
# Layout names
# --------------------------------------------------------------------------- #

LAYOUT_TITLE = "title"
LAYOUT_SECTION = "section"
LAYOUT_OBJECTIVES = "objectives"
LAYOUT_BULLETS = "bullets"
LAYOUT_PARAGRAPH = "paragraph"
LAYOUT_TWO_COLUMN = "two_column"
LAYOUT_TABLE = "table"
LAYOUT_EQUATION = "equation"
LAYOUT_CHART = "chart"
LAYOUT_IMAGE = "image"
LAYOUT_SUMMARY = "summary"
LAYOUT_CONCLUSION = "conclusion"

LAYOUTS: tuple[str, ...] = (
    LAYOUT_TITLE,
    LAYOUT_SECTION,
    LAYOUT_OBJECTIVES,
    LAYOUT_BULLETS,
    LAYOUT_PARAGRAPH,
    LAYOUT_TWO_COLUMN,
    LAYOUT_TABLE,
    LAYOUT_EQUATION,
    LAYOUT_CHART,
    LAYOUT_IMAGE,
    LAYOUT_SUMMARY,
    LAYOUT_CONCLUSION,
)

LAYOUT_LABELS: dict[str, str] = {
    LAYOUT_TITLE: "Title slide",
    LAYOUT_SECTION: "Section divider",
    LAYOUT_OBJECTIVES: "Learning objectives",
    LAYOUT_BULLETS: "Bulleted list",
    LAYOUT_PARAGRAPH: "Paragraph",
    LAYOUT_TWO_COLUMN: "Two-column comparison",
    LAYOUT_TABLE: "Table",
    LAYOUT_EQUATION: "Equations and worked examples",
    LAYOUT_CHART: "Chart or diagram",
    LAYOUT_IMAGE: "Image-supported slide",
    LAYOUT_SUMMARY: "Summary",
    LAYOUT_CONCLUSION: "Conclusion",
}

#: The single content-type -> layout mapping for a slide with no clearer cue.
_CONTENT_LAYOUT: dict[str, str] = {
    CONTENT_TITLE: LAYOUT_TITLE,
    CONTENT_SECTION: LAYOUT_SECTION,
    CONTENT_PARAGRAPH: LAYOUT_PARAGRAPH,
    CONTENT_BULLETS: LAYOUT_BULLETS,
    CONTENT_TABLE: LAYOUT_TABLE,
    CONTENT_EQUATION: LAYOUT_EQUATION,
    CONTENT_IMAGE: LAYOUT_IMAGE,
}

#: Title cues that promote a slide to a semantic layout (matched on a word
#: boundary, case-insensitively, so "Objectives" and "أهداف" both hit).
_OBJECTIVES_RE = re.compile(
    r"\b(objectives?|learning\s+outcomes?|goals?|aims?)\b|أهداف|الأهداف", re.IGNORECASE
)
_SUMMARY_RE = re.compile(r"\b(summary|recap|takeaways?|key\s+points?)\b|ملخص|الخلاصة", re.IGNORECASE)
_CONCLUSION_RE = re.compile(r"\b(conclusions?|closing|wrap[- ]?up)\b|خاتمة|الخاتمة", re.IGNORECASE)

#: A two-column comparison is requested with a ``|`` between the two sides, or
#: an explicit layout hint. This is deterministic: nothing is guessed.
_COLUMN_SPLIT_RE = re.compile(r"\s*\|\s*")


def resolve_layout(value: str) -> str:
    """Normalise a layout name, raising for an unknown one."""
    key = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LAYOUTS:
        raise PresentationError(
            f"Unknown slide layout {value!r}; expected one of {', '.join(LAYOUTS)}."
        )
    return key


def classify_layout(slide: Slide) -> str:
    """Choose the layout for a slide, deterministically.

    Order of precedence:

    1. An explicit ``slide.layout`` hint (the author's own choice).
    2. A title cue -- "Objectives", "Summary", "Conclusion" (and their Arabic
       equivalents) -- because the same wording should always look the same.
    3. A ``|`` separator in a bullet, which asks for a two-column comparison.
    4. The slide's content type.
    """
    hint = (getattr(slide, "layout", "") or "").strip()
    if hint:
        return resolve_layout(hint)

    title = slide.title or ""
    if _OBJECTIVES_RE.search(title):
        return LAYOUT_OBJECTIVES
    if _CONCLUSION_RE.search(title):
        return LAYOUT_CONCLUSION
    if _SUMMARY_RE.search(title):
        return LAYOUT_SUMMARY

    if slide.content_type == CONTENT_BULLETS and any(
        _COLUMN_SPLIT_RE.search(line) for line in slide.content
    ):
        return LAYOUT_TWO_COLUMN

    return _CONTENT_LAYOUT.get(slide.content_type, LAYOUT_BULLETS)


def split_columns(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ``"left | right"`` bullet lines into two columns, order preserved.

    A line with no separator goes to the left column. The split is at the first
    ``|``, so a right-hand side may itself contain one.
    """
    left: list[str] = []
    right: list[str] = []
    for line in lines:
        parts = _COLUMN_SPLIT_RE.split(line, maxsplit=1)
        if len(parts) == 2:
            if parts[0].strip():
                left.append(parts[0].strip())
            if parts[1].strip():
                right.append(parts[1].strip())
        else:
            left.append(line)
    return left, right


# --------------------------------------------------------------------------- #
# Text measurement and fit
# --------------------------------------------------------------------------- #

#: Mean glyph width as a fraction of the point size, for a Latin text face. Real
#: metrics need a font engine; this is a documented, conservative estimate that
#: errs toward *more* lines, so a slide is flagged slightly early rather than
#: clipped.
_AVG_CHAR_WIDTH_RATIO = 0.52
#: A blank line or a wrapped line still occupies its full leading.
_LINE_SPACING = 1.2


def _line_height_emu(size_pt: float, spacing: float = 1.0) -> float:
    """The EMU height of one line at ``size_pt`` with a ``spacing`` multiplier."""
    return size_pt * _LINE_SPACING * spacing * EMU_PER_POINT


def _chars_per_line(width_emu: float, size_pt: float) -> int:
    """How many characters of ``size_pt`` text fit across ``width_emu``."""
    char_emu = _AVG_CHAR_WIDTH_RATIO * size_pt * EMU_PER_POINT
    if char_emu <= 0:
        return 1
    return max(1, int(width_emu // char_emu))


@dataclass(frozen=True)
class FitEstimate:
    """Whether a block of text is estimated to fit the space prepared for it."""

    lines_needed: int
    lines_available: int
    ok: bool
    overflow_ratio: float

    def to_primitive(self) -> dict[str, Any]:
        return {
            "lines_needed": self.lines_needed,
            "lines_available": self.lines_available,
            "ok": self.ok,
            "overflow_ratio": round(self.overflow_ratio, 3),
        }


def estimate_fit(
    lines: Sequence[str],
    *,
    size_pt: float,
    width_emu: float,
    height_emu: float,
    spacing: float = 1.0,
    space_after: float = 0.0,
) -> FitEstimate:
    """Estimate whether ``lines`` fit a box of ``width_emu`` x ``height_emu``.

    The estimate is deliberately conservative: it assumes an average glyph width
    (no font engine is available to the preparation layer) and counts wrapped
    lines, so a slide that *might* overflow is flagged before the renderer runs.
    It never edits the text.
    """
    per_line = _chars_per_line(width_emu, size_pt)
    line_height = _line_height_emu(size_pt, spacing)
    if line_height <= 0:
        return FitEstimate(0, 0, True, 0.0)
    available = max(1, int(height_emu // line_height))
    needed = 0
    for line in lines:
        text = str(line)
        # A long word cannot wrap below its own length, so the count is at least
        # one line per non-empty string and grows with the character count.
        needed += max(1, math.ceil(len(text) / per_line)) if text else 1
        if space_after:
            needed += 1 if space_after >= 1.0 else 0
    ratio = needed / available if available else 1.0
    return FitEstimate(needed, available, needed <= available, ratio)


# --------------------------------------------------------------------------- #
# Image validation
# --------------------------------------------------------------------------- #

#: Image extensions PowerPoint can embed directly.
IMAGE_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
#: A hard cap on an embedded image, so a mistaken reference cannot bloat a deck.
MAX_IMAGE_BYTES = 40 * 1024 * 1024


@dataclass(frozen=True)
class ImageCheck:
    """The result of validating one image reference."""

    reference: str
    ok: bool
    path: str = ""
    reason: str = ""

    def to_primitive(self) -> dict[str, Any]:
        return {"reference": self.reference, "ok": self.ok, "path": self.path, "reason": self.reason}


def validate_image(reference: str, *, base_dir: str | Path | None = None) -> ImageCheck:
    """Check whether an image reference can be embedded.

    A reference is a filesystem path (absolute or relative to ``base_dir``).
    The check is deliberately forgiving: a missing file or an unsupported format
    returns ``ok=False`` with a readable reason, so the caller can *skip* the
    image and keep the slide rather than failing the whole export. No image is
    ever opened, decoded, or executed here -- only its name and size are read.
    """
    ref = (reference or "").strip()
    if not ref:
        return ImageCheck(reference=reference, ok=False, reason="The image reference is empty.")
    candidate = Path(ref).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir).expanduser() / candidate
    suffix = candidate.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return ImageCheck(
            reference=reference,
            ok=False,
            reason=f"{suffix or 'no extension'} is not an embeddable image format "
            f"({', '.join(IMAGE_SUFFIXES)}).",
        )
    try:
        resolved = candidate.resolve()
        stat = resolved.stat()
    except OSError:
        return ImageCheck(reference=reference, ok=False, reason="The image file was not found.")
    if not resolved.is_file():
        return ImageCheck(reference=reference, ok=False, reason="The image reference is not a file.")
    if stat.st_size > MAX_IMAGE_BYTES:
        return ImageCheck(
            reference=reference,
            ok=False,
            reason=f"The image is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )
    return ImageCheck(reference=reference, ok=True, path=str(resolved))


# --------------------------------------------------------------------------- #
# Filename safety
# --------------------------------------------------------------------------- #

#: Characters that are never allowed in a generated filename.
_UNSAFE_NAME_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def safe_filename(title: str, *, export_id: str = "", suffix: str = ".pptx") -> str:
    """A filesystem-safe, path-traversal-proof filename for an export.

    The title is slugified (Unicode letters, so Arabic survives), any directory
    separators or ``..`` are removed, and a unique ``export_id`` is appended when
    one is given, so two exports never collide and a crafted title can never
    escape the output directory.
    """
    stem = _UNSAFE_NAME_RE.sub("-", (title or "").strip())
    stem = stem.replace("..", "-").strip("-. ")
    stem = re.sub(r"-{2,}", "-", stem)[:80]
    if not stem:
        stem = slugify(title, fallback="presentation")
    # A leading dot would create a hidden file; strip it.
    stem = stem.lstrip(".")
    if export_id:
        ident = _UNSAFE_NAME_RE.sub("", str(export_id))[:24]
        if ident:
            stem = f"{stem}-{ident}"
    if not stem:
        stem = "presentation"
    return f"{stem}{suffix}"


def safe_output_path(
    directory: str | Path,
    filename: str,
) -> Path:
    """Join ``filename`` onto ``directory``, refusing any traversal escape.

    The resolved target must live inside ``directory``. A name that carries a
    path separator, an absolute path, or a ``..`` component is refused outright
    rather than silently reduced, so a crafted title can never write somewhere
    unexpected.
    """
    name = (filename or "").strip()
    if not name:
        raise PresentationError("An export needs a filename.")
    if Path(name).name != name or name in (".", "..") or ".." in Path(name).parts:
        raise PresentationError(
            f"The export name {filename!r} is not a plain filename."
        )
    root = Path(directory).expanduser().resolve()
    candidate = (root / name).resolve()
    if candidate != root and root not in candidate.parents:
        raise PresentationError(
            f"The export name {filename!r} would write outside the output directory."
        )
    return candidate


# --------------------------------------------------------------------------- #
# The render plan
# --------------------------------------------------------------------------- #

BLOCK_TITLE = "title"
BLOCK_BULLETS = "bullets"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_TABLE = "table"
BLOCK_EQUATION = "equation"
BLOCK_IMAGE = "image"
BLOCK_CHART = "chart"
BLOCK_COLUMNS = "columns"


@dataclass(frozen=True)
class ContentBlock:
    """One prepared block of slide content, ready for the renderer."""

    kind: str
    lines: tuple[str, ...] = ()
    #: Table rows (for ``BLOCK_TABLE``), already rectangular.
    rows: tuple[tuple[str, ...], ...] = ()
    #: A validated image path (for ``BLOCK_IMAGE``) plus its alt text.
    image_path: str = ""
    image_reference: str = ""
    #: Column split (for ``BLOCK_COLUMNS``).
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()

    def to_primitive(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lines": list(self.lines),
            "rows": [list(row) for row in self.rows],
            "image_path": self.image_path,
            "image_reference": self.image_reference,
            "left": list(self.left),
            "right": list(self.right),
        }


@dataclass(frozen=True)
class SlidePlan:
    """One slide's prepared content, layout, and fit verdict."""

    slide_id: str
    number: int
    title: str
    layout: str
    section: str = ""
    subsection: str = ""
    blocks: tuple[ContentBlock, ...] = ()
    notes: str = ""
    sources: tuple[str, ...] = ()
    fit: FitEstimate = field(default_factory=lambda: FitEstimate(0, 0, True, 0.0))
    warnings: tuple[str, ...] = ()

    @property
    def topic(self) -> str:
        """The outline heading this slide covers (``"Section / Subsection"``)."""
        head = self.section.strip()
        sub = self.subsection.strip()
        if sub:
            return f"{head} / {sub}" if head else sub
        return head

    @property
    def overflows(self) -> bool:
        """Whether the content is estimated not to fit."""
        return not self.fit.ok

    @property
    def image_paths(self) -> tuple[str, ...]:
        """Every embeddable image path on the slide."""
        return tuple(block.image_path for block in self.blocks if block.kind == BLOCK_IMAGE and block.image_path)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "number": self.number,
            "title": self.title,
            "layout": self.layout,
            "layout_label": LAYOUT_LABELS.get(self.layout, self.layout),
            "section": self.section,
            "subsection": self.subsection,
            "topic": self.topic,
            "blocks": [block.to_primitive() for block in self.blocks],
            "notes": self.notes,
            "sources": list(self.sources),
            "fit": self.fit.to_primitive(),
            "overflows": self.overflows,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class DeckPlan:
    """A whole deck prepared for rendering, with its settings and theme."""

    title: str
    subtitle: str
    slides: tuple[SlidePlan, ...]
    settings: PresentationSettings
    theme: DeckTheme
    author: str = ""

    @property
    def slide_count(self) -> int:
        return len(self.slides)

    @property
    def overflowing(self) -> tuple[SlidePlan, ...]:
        """The slides whose content is estimated not to fit."""
        return tuple(slide for slide in self.slides if slide.overflows)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every slide warning, prefixed with the slide number."""
        out: list[str] = []
        for slide in self.slides:
            for warning in slide.warnings:
                out.append(f"Slide {slide.number}: {warning}")
        return tuple(out)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "slide_count": self.slide_count,
            "settings": self.settings.to_primitive(),
            "theme": self.theme.name,
            "slides": [slide.to_primitive() for slide in self.slides],
            "overflowing": [slide.number for slide in self.overflowing],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Preparing a deck
# --------------------------------------------------------------------------- #

#: Layouts whose body is a list of bullets.
_BULLET_LAYOUTS = frozenset(
    {LAYOUT_BULLETS, LAYOUT_OBJECTIVES, LAYOUT_SUMMARY, LAYOUT_CONCLUSION}
)
#: Layouts whose body is flowing prose.
_PARAGRAPH_LAYOUTS = frozenset({LAYOUT_PARAGRAPH})


def _rectangular(rows: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Pad table rows to a rectangle so every row has the same cell count."""
    cleaned = [tuple(str(cell) for cell in row) for row in rows if row]
    if not cleaned:
        return ()
    width = max(len(row) for row in cleaned)
    return tuple(row + ("",) * (width - len(row)) for row in cleaned)


def _body_box(theme: DeckTheme, size: tuple[int, int], *, below_title: bool) -> tuple[int, int]:
    """The ``(width, height)`` EMU of the body area for a slide."""
    cx, cy = size
    width = cx - 2 * theme.margin_x
    top = theme.margin_y + (theme.title_gap + int(theme.heading_size * 1.4 * EMU_PER_POINT) if below_title else 0)
    height = cy - top - theme.margin_y
    return width, max(1, height)


def prepare_slide(
    slide: Slide,
    *,
    settings: PresentationSettings,
    theme: DeckTheme,
    base_dir: str | Path | None = None,
) -> SlidePlan:
    """Prepare one slide: classify it, build its blocks, and measure the fit."""
    layout = classify_layout(slide)
    warnings: list[str] = []
    blocks: list[ContentBlock] = []
    _cx, _cy = settings.size
    body_cx, body_cy = _body_box(theme, settings.size, below_title=layout not in (LAYOUT_TITLE, LAYOUT_SECTION))
    rtl = settings.rtl

    # --- title-only and section-divider slides -----------------------------
    if layout in (LAYOUT_TITLE, LAYOUT_SECTION):
        lines = tuple(slide.content) or ((slide.title,) if slide.title else ())
        blocks.append(ContentBlock(kind=BLOCK_TITLE, lines=lines))
        fit = estimate_fit(
            [slide.title, *lines],
            size_pt=theme.section_size if layout == LAYOUT_SECTION else theme.title_size,
            width_emu=body_cx,
            height_emu=body_cy,
            spacing=theme.line_spacing,
        )
        return SlidePlan(
            slide_id=slide.slide_id,
            number=slide.number,
            title=slide.title,
            layout=layout,
            section=slide.section,
            subsection=slide.subsection,
            blocks=tuple(blocks),
            notes=slide.notes,
            sources=tuple(slide.sources),
            fit=fit,
            warnings=tuple(warnings),
        )

    # --- tables ------------------------------------------------------------
    if layout == LAYOUT_TABLE and slide.table:
        rows = _rectangular(slide.table)
        blocks.append(ContentBlock(kind=BLOCK_TABLE, rows=rows))
        row_h = _line_height_emu(theme.table_size, theme.line_spacing) * 1.6
        table_height = row_h * len(rows)
        if table_height > body_cy:
            warnings.append(
                f"the table ({len(rows)} rows) is taller than the slide; consider "
                "splitting it across slides."
            )
        fit = FitEstimate(len(rows), max(1, int(body_cy // row_h)), table_height <= body_cy,
                          (table_height / body_cy) if body_cy else 1.0)
        return SlidePlan(
            slide_id=slide.slide_id,
            number=slide.number,
            title=slide.title,
            layout=layout,
            section=slide.section,
            subsection=slide.subsection,
            blocks=tuple(blocks),
            notes=slide.notes,
            sources=tuple(slide.sources),
            fit=fit,
            warnings=tuple(warnings),
        )

    # --- charts ------------------------------------------------------------
    if layout == LAYOUT_CHART:
        rows = _rectangular(slide.table) if slide.table else ()
        if rows:
            blocks.append(ContentBlock(kind=BLOCK_CHART, rows=rows))
        if slide.content:
            blocks.append(ContentBlock(kind=BLOCK_BULLETS, lines=tuple(slide.content)))
        if not rows and not slide.content:
            warnings.append("the chart slide has no data to plot.")
        fit = FitEstimate(1, 1, True, 1.0)
        return SlidePlan(
            slide_id=slide.slide_id,
            number=slide.number,
            title=slide.title,
            layout=layout,
            section=slide.section,
            subsection=slide.subsection,
            blocks=tuple(blocks),
            notes=slide.notes,
            sources=tuple(slide.sources),
            fit=fit,
            warnings=tuple(warnings),
        )

    # --- two-column comparisons -------------------------------------------
    if layout == LAYOUT_TWO_COLUMN:
        left, right = split_columns(slide.content)
        blocks.append(ContentBlock(kind=BLOCK_COLUMNS, left=tuple(left), right=tuple(right)))
        half = (body_cx - theme.margin_x) // 2
        fits = [
            estimate_fit(lines, size_pt=theme.body_size, width_emu=half, height_emu=body_cy)
            for lines in (left, right)
        ]
        needed = max(f.lines_needed for f in fits)
        available = max(1, min(f.lines_available for f in fits))
        fit = FitEstimate(needed, available, all(f.ok for f in fits), needed / available if available else 1.0)
        if not fit.ok:
            warnings.append("a column is longer than the slide; consider splitting it.")
        return SlidePlan(
            slide_id=slide.slide_id,
            number=slide.number,
            title=slide.title,
            layout=layout,
            section=slide.section,
            subsection=slide.subsection,
            blocks=tuple(blocks),
            notes=slide.notes,
            sources=tuple(slide.sources),
            fit=fit,
            warnings=tuple(warnings),
        )

    # --- image slides ------------------------------------------------------
    if layout == LAYOUT_IMAGE:
        check = validate_image(slide.images[0], base_dir=base_dir) if slide.images else None
        if check is None:
            warnings.append("the image slide names no image file.")
        elif not check.ok:
            warnings.append(f"the image could not be embedded: {check.reason}")
        else:
            blocks.append(
                ContentBlock(
                    kind=BLOCK_IMAGE,
                    image_path=check.path,
                    image_reference=check.reference,
                )
            )
        if slide.content:
            blocks.append(ContentBlock(kind=BLOCK_BULLETS, lines=tuple(slide.content)))
        fit = estimate_fit(
            slide.content,
            size_pt=theme.small_size,
            width_emu=body_cx,
            height_emu=max(1, body_cy // 3),
        )
        return SlidePlan(
            slide_id=slide.slide_id,
            number=slide.number,
            title=slide.title,
            layout=layout,
            section=slide.section,
            subsection=slide.subsection,
            blocks=tuple(blocks),
            notes=slide.notes,
            sources=tuple(slide.sources),
            fit=fit,
            warnings=tuple(warnings),
        )

    # --- bullets, objectives, summary, conclusion, paragraph, equation -----
    if layout == LAYOUT_EQUATION:
        lines = tuple(slide.content) + tuple(slide.equations)
        blocks.append(ContentBlock(kind=BLOCK_EQUATION, lines=lines))
        fit = estimate_fit(
            lines,
            size_pt=theme.equation_size,
            width_emu=body_cx,
            height_emu=body_cy,
            spacing=theme.line_spacing,
            space_after=theme.space_after,
        )
    elif layout in _PARAGRAPH_LAYOUTS:
        blocks.append(ContentBlock(kind=BLOCK_PARAGRAPH, lines=tuple(slide.content)))
        fit = estimate_fit(
            slide.content,
            size_pt=theme.body_size,
            width_emu=body_cx,
            height_emu=body_cy,
            spacing=theme.line_spacing,
            space_after=theme.space_after,
        )
    else:
        lines = tuple(slide.content) + tuple(slide.equations)
        blocks.append(ContentBlock(kind=BLOCK_BULLETS, lines=lines))
        fit = estimate_fit(
            lines,
            size_pt=theme.body_size,
            width_emu=body_cx,
            height_emu=body_cy,
            spacing=theme.line_spacing,
            space_after=theme.space_after,
        )
    if not fit.ok:
        warnings.append(
            f"the content needs about {fit.lines_needed} lines but only "
            f"{fit.lines_available} fit; consider splitting the slide or trimming "
            "nonessential wording (with approval)."
        )
    if rtl and not _looks_rtl("\n".join(slide.content)):
        # The caller asked for RTL, so it is honoured; a deck with no Arabic at
        # all is worth noting but nothing is changed.
        warnings.append("right-to-left was requested but the slide text is all left-to-right.")
    return SlidePlan(
        slide_id=slide.slide_id,
        number=slide.number,
        title=slide.title,
        layout=layout,
        section=slide.section,
        subsection=slide.subsection,
        blocks=tuple(blocks),
        notes=slide.notes,
        sources=tuple(slide.sources),
        fit=fit,
        warnings=tuple(warnings),
    )


#: Arabic and Hebrew blocks, used to decide a sensible default text direction.
_RTL_RE = re.compile(r"[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def _looks_rtl(text: str) -> bool:
    """True when ``text`` contains any right-to-left script character."""
    return bool(_RTL_RE.search(text))


def looks_rtl(text: str) -> bool:
    """Public wrapper: whether ``text`` contains right-to-left script."""
    return _looks_rtl(text)


def prepare_deck(
    deck: Deck,
    *,
    settings: PresentationSettings | None = None,
    project: PresentationProject | None = None,
    base_dir: str | Path | None = None,
) -> DeckPlan:
    """Prepare every slide of an approved :class:`~core.presentation.Deck`.

    No content is altered: the plan carries the same titles, lines, tables,
    equations, and notes as the deck, plus a layout choice, a fit estimate, and
    any warnings (an overflowing slide, a missing image). The renderer consumes
    this plan; nothing is written here.
    """
    spec = settings or PresentationSettings()
    theme = spec.theme_spec
    slides = tuple(
        prepare_slide(slide, settings=spec, theme=theme, base_dir=base_dir) for slide in deck.slides
    )
    return DeckPlan(
        title=deck.title,
        subtitle=(project.subtitle if project is not None else ""),
        slides=slides,
        settings=spec,
        theme=theme,
        author=(project.audience if project is not None else "") or spec.author,
    )


# --------------------------------------------------------------------------- #
# Filename helpers re-exported for convenience
# --------------------------------------------------------------------------- #


def unique_export_path(directory: str | Path, title: str, *, export_id: str = "") -> Path:
    """A safe, unique path for a new export inside ``directory``.

    The name is derived from the title and the export id; if a file of that name
    already exists (and the caller has not authorised an overwrite) a numeric
    suffix is added so nothing is ever replaced.
    """
    filename = safe_filename(title, export_id=export_id)
    path = safe_output_path(directory, filename)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = safe_output_path(directory, f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise PresentationError("Could not find a free export filename.")


__all__ = [
    "ASPECTS",
    "ASPECT_4_3",
    "ASPECT_16_9",
    "DEFAULT_ASPECT",
    "DEFAULT_THEME",
    "EMU_PER_INCH",
    "EMU_PER_POINT",
    "LAYOUTS",
    "LAYOUT_BULLETS",
    "LAYOUT_CHART",
    "LAYOUT_CONCLUSION",
    "LAYOUT_EQUATION",
    "LAYOUT_IMAGE",
    "LAYOUT_LABELS",
    "LAYOUT_OBJECTIVES",
    "LAYOUT_PARAGRAPH",
    "LAYOUT_SECTION",
    "LAYOUT_SUMMARY",
    "LAYOUT_TABLE",
    "LAYOUT_TITLE",
    "LAYOUT_TWO_COLUMN",
    "THEMES",
    "ContentBlock",
    "DeckPlan",
    "DeckTheme",
    "FitEstimate",
    "ImageCheck",
    "PresentationSettings",
    "SlidePlan",
    "classify_layout",
    "estimate_fit",
    "looks_rtl",
    "prepare_deck",
    "prepare_slide",
    "resolve_aspect",
    "resolve_layout",
    "resolve_theme",
    "safe_filename",
    "safe_output_path",
    "slide_size",
    "split_columns",
    "theme_names",
    "unique_export_path",
    "validate_image",
]
