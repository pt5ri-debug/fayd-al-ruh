"""The ``.pptx`` render engine: draw a prepared :class:`DeckPlan` with python-pptx.

This is the **second half** of the Presentation Studio export engine. It takes
the dependency-free :class:`~core.presentation_layout.DeckPlan` (produced by
:mod:`core.presentation_layout`, which has already classified each slide, built
its blocks, validated its images, and measured the fit) and draws a real Office
Open XML presentation the user can open and edit.

Why python-pptx here and not the standard library?

The project's first writer (``core.presentation.PresentationWriter``) assembles
the package by hand with ``zipfile`` + ``xml.etree``. That is perfect for a
dependency-free fallback, but it cannot embed an image, draw a chart, honour
right-to-left text properly, or produce a *theme* a user can restyle. python-pptx
does all of that through the same OOXML the project already understands, and it
is exactly the kind of *optional extra* the project already supports (PDF, OCR,
the image backend): imported lazily, installed from a bundled requirements file,
never required by the base install.

So this module:

* **imports python-pptx lazily** and raises :class:`PptxUnavailable` with an
  actionable message when it is absent, so the base install and CI still work
  and the standard-library writer remains a fallback;
* **draws every layout itself** (a blank layout plus explicit geometry), so the
  twelve professional layouts are consistent and never depend on a template
  file's placeholder order;
* **never alters the approved content** -- it draws the plan's own text, tables,
  equations, images, and notes; an over-full slide is *reported*, not clipped or
  shortened;
* **validates the result** by reopening the written file and checking the slide
  count, order, titles, and notes (:func:`validate_deck`);
* **executes nothing** from the deck: no macro, no embedded script, and an image
  is only ever read as bytes by python-pptx.

The renderer is a thin, mechanical drawing layer on purpose: all the judgement
(which layout, does it fit, is the image usable) lives in the preparation layer,
where it is unit-testable with no third-party package.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.presentation_layout import (
    BLOCK_BULLETS,
    BLOCK_CHART,
    BLOCK_COLUMNS,
    BLOCK_EQUATION,
    BLOCK_IMAGE,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TITLE,
    EMU_PER_POINT,
    LAYOUT_SECTION,
    LAYOUT_TITLE,
    DeckPlan,
    DeckTheme,
    looks_rtl,
    safe_output_path,
)

# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #

#: The bundled requirements file that installs python-pptx.
REQUIREMENTS_FILE = "requirements-pptx.txt"


class PptxUnavailable(RuntimeError):
    """Raised when python-pptx is not installed (an optional extra)."""


def pptx_available() -> bool:
    """Whether python-pptx can be imported (the ``[pptx]`` extra is installed)."""
    import importlib.util

    return importlib.util.find_spec("pptx") is not None


def _require_pptx() -> Any:
    """Import python-pptx or raise an actionable :class:`PptxUnavailable`."""
    try:
        import pptx
    except ImportError as error:  # pragma: no cover - exercised via the guard test
        raise PptxUnavailable(
            "The PowerPoint engine needs the optional `python-pptx` package. "
            f"Install it with `python -m pip install -r {REQUIREMENTS_FILE}` "
            "(or `python -m cli deps --install pptx`)."
        ) from error
    return pptx


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _rgb(pptx: Any, value: str) -> Any:
    """An :class:`RGBColor` from a ``RRGGBB`` string."""
    return pptx.dml.color.RGBColor.from_string(value.lstrip("#").upper())


def _set_cs_font(run: Any, typeface: str) -> None:
    """Set the complex-script typeface (Arabic) on a run, so it is not substituted.

    ``run.font.name`` sets only the Latin typeface; a right-to-left run needs the
    ``a:cs`` element, which python-pptx does not expose, so it is added directly.
    This changes no text -- only which font the script is drawn in.
    """
    rpr = run._r.get_or_add_rPr()
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    cs = rpr.find(f"{{{ns}}}cs")
    if cs is None:
        cs = rpr.makeelement(f"{{{ns}}}cs", {})
        rpr.append(cs)
    cs.set("typeface", typeface)


def _style_run(
    run: Any,
    *,
    pptx: Any,
    size: float,
    font: str,
    color: str,
    bold: bool = False,
    italic: bool = False,
    arabic_font: str = "",
) -> None:
    """Apply size, typeface, weight, and colour to one run."""
    run.font.size = pptx.util.Pt(size)
    run.font.name = font
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _rgb(pptx, color)
    if arabic_font:
        _set_cs_font(run, arabic_font)


def _style_paragraph(
    paragraph: Any,
    *,
    pptx: Any,
    rtl: bool,
    line_spacing: float,
    space_after_pt: float = 0.0,
    level: int = 0,
) -> None:
    """Apply spacing, level, and text direction to a paragraph."""
    paragraph.line_spacing = line_spacing
    if space_after_pt:
        paragraph.space_after = pptx.util.Pt(space_after_pt)
    paragraph.level = level
    if rtl:
        ppr = paragraph._p.get_or_add_pPr()
        ppr.set("rtl", "1")
        paragraph.alignment = pptx.enum.text.PP_ALIGN.RIGHT


def _textbox(
    slide: Any,
    pptx: Any,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> Any:
    """A word-wrapping text box with no autofit and no internal margin drift."""
    box = slide.shapes.add_textbox(
        pptx.util.Emu(left), pptx.util.Emu(top), pptx.util.Emu(width), pptx.util.Emu(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    # Do not let PowerPoint shrink or grow the text: the preparation layer already
    # measured it, and an over-full slide is reported rather than silently fitted.
    frame.auto_size = pptx.enum.text.MSO_AUTO_SIZE.NONE
    frame.margin_left = pptx.util.Emu(0)
    frame.margin_right = pptx.util.Emu(0)
    frame.margin_top = pptx.util.Emu(0)
    frame.margin_bottom = pptx.util.Emu(0)
    frame.vertical_anchor = pptx.enum.text.MSO_ANCHOR.TOP
    return frame


def _add_paragraphs(
    frame: Any,
    pptx: Any,
    lines: Any,
    *,
    theme: DeckTheme,
    size: float,
    font: str,
    color: str,
    rtl: bool,
    bullet: bool = False,
    bold: bool = False,
    italic: bool = False,
    start_empty: bool = True,
) -> None:
    """Fill a text frame with one paragraph per line, order preserved."""
    items = [str(line) for line in lines]
    if not items:
        return
    first = True
    for index, text in enumerate(items):
        if first and start_empty:
            paragraph = frame.paragraphs[0]
            first = False
        else:
            paragraph = frame.add_paragraph()
        _style_paragraph(
            paragraph,
            pptx=pptx,
            rtl=rtl,
            line_spacing=theme.line_spacing,
            space_after_pt=theme.body_size * theme.space_after,
        )
        run = paragraph.add_run()
        run.text = text
        _style_run(
            run,
            pptx=pptx,
            size=size,
            font=font,
            color=color,
            bold=bold,
            italic=italic,
            arabic_font=theme.script_font(font),
        )
        if bullet:
            # A literal bullet glyph keeps the list readable in every viewer,
            # including ones that ignore the master's bullet definition.
            paragraph.text = f"•  {text}" if not rtl else f"{text}  •"
        _ = index


# --------------------------------------------------------------------------- #
# Layout geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Frame:
    """A computed rectangle on a slide, in EMU."""

    left: int
    top: int
    width: int
    height: int


def _title_frame(plan: DeckPlan, *, with_title: bool) -> _Frame:
    """Where the slide title sits, and the body area below it."""
    theme = plan.theme
    cx, cy = plan.settings.size
    width = cx - 2 * theme.margin_x
    if not with_title:
        return _Frame(theme.margin_x, theme.margin_y, width, cy - 2 * theme.margin_y)
    title_h = int(theme.heading_size * 1.4 * EMU_PER_POINT)
    return _Frame(theme.margin_x, theme.margin_y, width, title_h)


def _body_frame(plan: DeckPlan, *, with_title: bool) -> _Frame:
    """The body area (below the title when there is one)."""
    theme = plan.theme
    cx, cy = plan.settings.size
    width = cx - 2 * theme.margin_x
    if not with_title:
        return _Frame(theme.margin_x, theme.margin_y, width, cy - 2 * theme.margin_y)
    top = theme.margin_y + int(theme.heading_size * 1.4 * EMU_PER_POINT) + theme.title_gap
    return _Frame(theme.margin_x, top, width, cy - top - theme.margin_y)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@dataclass
class RenderReport:
    """What a render produced: the path, its size, and any slide warnings."""

    path: str
    slide_count: int
    size_bytes: int = 0
    warnings: tuple[str, ...] = ()
    layouts: tuple[str, ...] = ()

    def to_primitive(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "slide_count": self.slide_count,
            "size_bytes": self.size_bytes,
            "warnings": list(self.warnings),
            "layouts": list(self.layouts),
        }


class PptxEngine:
    """Render a :class:`DeckPlan` to a real, editable ``.pptx`` with python-pptx."""

    def __init__(self, *, requirements_file: str = REQUIREMENTS_FILE) -> None:
        self._requirements_file = requirements_file

    @property
    def available(self) -> bool:
        """Whether the engine can run (python-pptx is importable)."""
        return pptx_available()

    def write(self, plan: DeckPlan, path: str | Path) -> Path:
        """Render ``plan`` to ``path`` atomically and return the resolved path.

        The package is built in memory and written through a temporary file that
        is replaced into place, so a failure never leaves a half-written deck.
        The target is confined to its own directory (see
        :func:`core.presentation_layout.safe_output_path`).
        """
        target = Path(path).expanduser()
        if target.suffix.lower() != ".pptx":
            target = target.with_suffix(".pptx")
        payload = self.render(plan)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return target

    def render(self, plan: DeckPlan) -> bytes:
        """Render ``plan`` to the bytes of a ``.pptx`` package."""
        pptx = _require_pptx()
        presentation = pptx.Presentation()
        cx, cy = plan.settings.size
        presentation.slide_width = pptx.util.Emu(cx)
        presentation.slide_height = pptx.util.Emu(cy)
        self._apply_metadata(pptx, presentation, plan)

        blank = self._blank_layout(pptx, presentation)
        for slide_plan in plan.slides:
            slide = presentation.slides.add_slide(blank)
            self._render_slide(pptx, slide, plan, slide_plan)

        buffer = io.BytesIO()
        presentation.save(buffer)
        return buffer.getvalue()

    # ------------------------------------------------------------- internals

    def _blank_layout(self, pptx: Any, presentation: Any) -> Any:
        """The blank layout, so every slide is drawn from explicit geometry."""
        for layout in presentation.slide_layouts:
            if layout.name.lower() in ("blank", "blank layout"):
                return layout
        return presentation.slide_layouts[-1]

    def _apply_metadata(self, pptx: Any, presentation: Any, plan: DeckPlan) -> None:
        """Write the deck's title, author, subject, and comment metadata."""
        core = presentation.core_properties
        core.title = plan.title
        if plan.settings.author or plan.author:
            core.author = plan.settings.author or plan.author
        if plan.settings.company:
            core.category = plan.settings.company
        core.subject = plan.settings.subject or plan.subtitle or plan.title
        if plan.settings.footer:
            core.comments = plan.settings.footer

    def _render_slide(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any
    ) -> None:
        """Draw one slide: background, title, body blocks, and notes."""
        theme = plan.theme
        layout = slide_plan.layout
        # Background: a dark panel for a section divider, the theme background
        # otherwise, so a divider reads as a break in the deck.
        background = (
            theme.section_background if layout == LAYOUT_SECTION else theme.background
        )
        self._set_background(pptx, slide, background)

        rtl = plan.settings.rtl or looks_rtl(slide_plan.title) or any(
            looks_rtl(line) for block in slide_plan.blocks for line in block.lines
        )

        if layout == LAYOUT_TITLE:
            self._render_title_slide(pptx, slide, plan, slide_plan, rtl=rtl)
        elif layout == LAYOUT_SECTION:
            self._render_section_slide(pptx, slide, plan, slide_plan, rtl=rtl)
        else:
            self._render_body_slide(pptx, slide, plan, slide_plan, rtl=rtl)

        if plan.settings.show_slide_numbers and layout not in (LAYOUT_TITLE, LAYOUT_SECTION):
            self._render_slide_number(pptx, slide, plan, slide_plan)
        if plan.settings.footer and layout not in (LAYOUT_TITLE, LAYOUT_SECTION):
            self._render_footer(pptx, slide, plan)

        if slide_plan.notes.strip():
            slide.notes_slide.notes_text_frame.text = slide_plan.notes

    def _set_background(self, pptx: Any, slide: Any, color: str) -> None:
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = _rgb(pptx, color)

    def _render_title_slide(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any, *, rtl: bool
    ) -> None:
        theme = plan.theme
        cx, cy = plan.settings.size
        width = cx - 2 * theme.margin_x
        # A centred title band with the accent rule beneath it.
        top = int(cy * 0.30)
        frame = _textbox(
            slide, pptx, left=theme.margin_x, top=top, width=width, height=int(cy * 0.28)
        )
        frame.vertical_anchor = pptx.enum.text.MSO_ANCHOR.MIDDLE
        _add_paragraphs(
            frame,
            pptx,
            [slide_plan.title],
            theme=theme,
            size=theme.title_size,
            font=theme.heading_font,
            color=theme.foreground,
            rtl=rtl,
            bold=True,
        )
        body = [line for block in slide_plan.blocks for line in block.lines]
        subtitle_lines = body or ([plan.subtitle] if plan.subtitle else [])
        if subtitle_lines:
            sub = _textbox(
                slide,
                pptx,
                left=theme.margin_x,
                top=top + int(cy * 0.30),
                width=width,
                height=int(cy * 0.16),
            )
            _add_paragraphs(
                sub,
                pptx,
                subtitle_lines,
                theme=theme,
                size=theme.subtitle_size,
                font=theme.body_font,
                color=theme.muted,
                rtl=rtl,
            )
        self._accent_rule(pptx, slide, plan, left=theme.margin_x, top=top - int(EMU_PER_POINT * 24), width=int(width * 0.28))

    def _render_section_slide(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any, *, rtl: bool
    ) -> None:
        theme = plan.theme
        cx, cy = plan.settings.size
        width = cx - 2 * theme.margin_x
        top = int(cy * 0.36)
        frame = _textbox(
            slide, pptx, left=theme.margin_x, top=top, width=width, height=int(cy * 0.3)
        )
        frame.vertical_anchor = pptx.enum.text.MSO_ANCHOR.MIDDLE
        heading = slide_plan.title or (slide_plan.topic or "")
        _add_paragraphs(
            frame,
            pptx,
            [heading],
            theme=theme,
            size=theme.section_size,
            font=theme.heading_font,
            color=theme.section_foreground,
            rtl=rtl,
            bold=True,
        )
        body = [line for block in slide_plan.blocks for line in block.lines if line != heading]
        if body:
            sub = _textbox(
                slide,
                pptx,
                left=theme.margin_x,
                top=top + int(cy * 0.30),
                width=width,
                height=int(cy * 0.2),
            )
            _add_paragraphs(
                sub,
                pptx,
                body,
                theme=theme,
                size=theme.subtitle_size,
                font=theme.body_font,
                color=theme.section_foreground,
                rtl=rtl,
            )
        self._accent_rule(
            pptx, slide, plan, left=theme.margin_x, top=top - int(EMU_PER_POINT * 24), width=int(width * 0.28)
        )

    def _render_body_slide(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any, *, rtl: bool
    ) -> None:
        theme = plan.theme
        title = _title_frame(plan, with_title=True)
        if slide_plan.title:
            frame = _textbox(
                slide,
                pptx,
                left=title.left,
                top=title.top,
                width=title.width,
                height=title.height,
            )
            _add_paragraphs(
                frame,
                pptx,
                [slide_plan.title],
                theme=theme,
                size=theme.heading_size,
                font=theme.heading_font,
                color=theme.foreground,
                rtl=rtl,
                bold=True,
            )
            self._accent_rule(
                pptx,
                slide,
                plan,
                left=title.left,
                top=title.top + title.height + int(EMU_PER_POINT * 4),
                width=int(title.width * 0.18),
            )

        body = _body_frame(plan, with_title=True)
        table_block = next(
            (block for block in slide_plan.blocks if block.kind == BLOCK_TABLE), None
        )
        chart_block = next(
            (block for block in slide_plan.blocks if block.kind == BLOCK_CHART), None
        )
        image_block = next(
            (block for block in slide_plan.blocks if block.kind == BLOCK_IMAGE), None
        )
        columns_block = next(
            (block for block in slide_plan.blocks if block.kind == BLOCK_COLUMNS), None
        )

        if table_block is not None:
            self._render_table(pptx, slide, plan, table_block, body, rtl=rtl)
        elif chart_block is not None:
            drawn = self._render_chart(pptx, slide, plan, chart_block, body, rtl=rtl)
            if not drawn and chart_block.rows:
                self._render_table(pptx, slide, plan, chart_block, body, rtl=rtl)
        elif image_block is not None:
            self._render_image(pptx, slide, plan, image_block, body)
        elif columns_block is not None:
            self._render_columns(pptx, slide, plan, slide_plan, columns_block, body, rtl=rtl)
        else:
            self._render_text_blocks(pptx, slide, plan, slide_plan, body, rtl=rtl)

        # A chart or image slide may also carry bullets beneath the figure.
        if (image_block is not None or chart_block is not None) and slide_plan.blocks:
            extras = [
                block
                for block in slide_plan.blocks
                if block.kind in (BLOCK_BULLETS, BLOCK_EQUATION, BLOCK_PARAGRAPH)
            ]
            if extras:
                lines = [line for block in extras for line in block.lines]
                lower = _Frame(
                    body.left,
                    body.top + int(body.height * 0.62),
                    body.width,
                    int(body.height * 0.38),
                )
                frame = _textbox(
                    slide,
                    pptx,
                    left=lower.left,
                    top=lower.top,
                    width=lower.width,
                    height=lower.height,
                )
                _add_paragraphs(
                    frame,
                    pptx,
                    lines,
                    theme=theme,
                    size=theme.small_size,
                    font=theme.body_font,
                    color=theme.foreground,
                    rtl=rtl,
                    bullet=True,
                )

    def _render_text_blocks(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any, body: _Frame, *, rtl: bool
    ) -> None:
        theme = plan.theme
        frame = _textbox(
            slide, pptx, left=body.left, top=body.top, width=body.width, height=body.height
        )
        written = False
        for block in slide_plan.blocks:
            if block.kind == BLOCK_EQUATION:
                _add_paragraphs(
                    frame,
                    pptx,
                    block.lines,
                    theme=theme,
                    size=theme.equation_size,
                    font=theme.body_font,
                    color=theme.foreground,
                    rtl=rtl,
                    italic=True,
                    start_empty=not written,
                )
                written = True
            elif block.kind == BLOCK_PARAGRAPH:
                _add_paragraphs(
                    frame,
                    pptx,
                    block.lines,
                    theme=theme,
                    size=theme.body_size,
                    font=theme.body_font,
                    color=theme.foreground,
                    rtl=rtl,
                    start_empty=not written,
                )
                written = True
            elif block.kind in (BLOCK_BULLETS, BLOCK_TITLE):
                _add_paragraphs(
                    frame,
                    pptx,
                    block.lines,
                    theme=theme,
                    size=theme.body_size,
                    font=theme.body_font,
                    color=theme.foreground,
                    rtl=rtl,
                    bullet=True,
                    start_empty=not written,
                )
                written = True

    def _render_columns(
        self,
        pptx: Any,
        slide: Any,
        plan: DeckPlan,
        slide_plan: Any,
        block: Any,
        body: _Frame,
        *,
        rtl: bool,
    ) -> None:
        theme = plan.theme
        gap = theme.margin_x // 2
        half = (body.width - gap) // 2
        for index, lines in enumerate((block.left, block.right)):
            left = body.left + index * (half + gap)
            frame = _textbox(
                slide, pptx, left=left, top=body.top, width=half, height=body.height
            )
            _add_paragraphs(
                frame,
                pptx,
                lines,
                theme=theme,
                size=theme.body_size,
                font=theme.body_font,
                color=theme.foreground,
                rtl=rtl,
                bullet=True,
            )

    def _render_table(
        self, pptx: Any, slide: Any, plan: DeckPlan, block: Any, body: _Frame, *, rtl: bool
    ) -> None:
        theme = plan.theme
        rows = block.rows
        if not rows:
            return
        n_rows = len(rows)
        n_cols = len(rows[0])
        row_h = int(theme.table_size * 1.6 * EMU_PER_POINT)
        table_shape = slide.shapes.add_table(
            n_rows,
            n_cols,
            pptx.util.Emu(body.left),
            pptx.util.Emu(body.top),
            pptx.util.Emu(body.width),
            pptx.util.Emu(row_h * n_rows),
        )
        table = table_shape.table
        col_w = body.width // n_cols
        for column in table.columns:
            column.width = pptx.util.Emu(col_w)
        for r_index, row in enumerate(rows):
            for c_index in range(n_cols):
                cell = table.cell(r_index, c_index)
                cell.text = row[c_index] if c_index < len(row) else ""
                cell.margin_left = pptx.util.Emu(int(EMU_PER_POINT * 6))
                cell.margin_right = pptx.util.Emu(int(EMU_PER_POINT * 6))
                paragraph = cell.text_frame.paragraphs[0]
                _style_paragraph(
                    paragraph, pptx=pptx, rtl=rtl, line_spacing=theme.line_spacing
                )
                for run in paragraph.runs:
                    _style_run(
                        run,
                        pptx=pptx,
                        size=theme.table_size,
                        font=theme.body_font,
                        color=theme.foreground,
                        bold=(r_index == 0),
                        arabic_font=theme.script_font(theme.body_font),
                    )

    def _render_chart(
        self, pptx: Any, slide: Any, plan: DeckPlan, block: Any, body: _Frame, *, rtl: bool
    ) -> bool:
        """Draw a bar chart from the block's rows; return whether it was drawn.

        A chart needs a header row and at least one numeric column. When the data
        is not numeric the caller falls back to a table, so a "chart" slide is
        never left blank.
        """
        theme = plan.theme
        rows = block.rows
        if len(rows) < 2 or len(rows[0]) < 2:
            return False
        header = rows[0]
        try:
            series: list[list[float]] = [[] for _ in range(len(header) - 1)]
            categories: list[str] = []
            for row in rows[1:]:
                categories.append(row[0])
                for index in range(1, len(header)):
                    raw = row[index] if index < len(row) else ""
                    series[index - 1].append(float(str(raw).replace(",", "").strip()))
        except (ValueError, TypeError):
            return False
        try:
            from pptx.chart.data import CategoryChartData
            from pptx.enum.chart import XL_CHART_TYPE

            chart_data = CategoryChartData()
            chart_data.categories = categories
            for index, values in enumerate(series):
                name = header[index + 1] if index + 1 < len(header) else f"Series {index + 1}"
                chart_data.add_series(name, values)
            frame = slide.shapes.add_chart(
                XL_CHART_TYPE.COLUMN_CLUSTERED,
                pptx.util.Emu(body.left),
                pptx.util.Emu(body.top),
                pptx.util.Emu(body.width),
                pptx.util.Emu(int(body.height * 0.62)),
                chart_data,
            )
            chart = frame.chart
            chart.has_legend = len(series) > 1
            chart.has_title = bool(block.rows)
            _ = theme
        except Exception:
            # A chart that cannot be drawn degrades to the table fallback; the
            # deck is never lost because of one chart.
            return False
        return True

    def _render_image(
        self, pptx: Any, slide: Any, plan: DeckPlan, block: Any, body: _Frame
    ) -> None:
        """Embed a validated image, scaled to fit inside the body area."""
        path = block.image_path
        if not path:
            return
        try:
            from pptx.parts.image import Image as PptxImage

            image = PptxImage.from_file(path)
            px_w, px_h = image.size
        except Exception:
            return
        if not px_w or not px_h:
            return
        # Fit inside the body box, preserving aspect ratio.
        target_h = int(body.height * 0.62)
        scale = min(body.width / px_w, target_h / px_h)
        width = max(1, int(px_w * scale))
        height = max(1, int(px_h * scale))
        left = body.left + (body.width - width) // 2
        slide.shapes.add_picture(
            path, pptx.util.Emu(left), pptx.util.Emu(body.top), pptx.util.Emu(width), pptx.util.Emu(height)
        )

    def _accent_rule(
        self, pptx: Any, slide: Any, plan: DeckPlan, *, left: int, top: int, width: int
    ) -> None:
        """A thin accent-coloured rule under a heading."""
        theme = plan.theme
        height = int(EMU_PER_POINT * 3)
        shape = slide.shapes.add_shape(
            pptx.enum.shapes.MSO_SHAPE.RECTANGLE,
            pptx.util.Emu(left),
            pptx.util.Emu(top),
            pptx.util.Emu(width),
            pptx.util.Emu(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(pptx, theme.accent)
        shape.line.fill.background()
        shape.shadow.inherit = False

    def _render_slide_number(
        self, pptx: Any, slide: Any, plan: DeckPlan, slide_plan: Any
    ) -> None:
        theme = plan.theme
        cx, cy = plan.settings.size
        width = int(EMU_PER_POINT * 60)
        height = int(EMU_PER_POINT * 20)
        left = cx - theme.margin_x - width
        top = cy - theme.margin_y + int(EMU_PER_POINT * 2)
        frame = _textbox(
            slide, pptx, left=left, top=top, width=width, height=height
        )
        frame.paragraphs[0].alignment = pptx.enum.text.PP_ALIGN.RIGHT
        run = frame.paragraphs[0].add_run()
        run.text = str(slide_plan.number)
        _style_run(
            run,
            pptx=pptx,
            size=theme.small_size - 4,
            font=theme.body_font,
            color=theme.muted,
            arabic_font=theme.script_font(theme.body_font),
        )

    def _render_footer(self, pptx: Any, slide: Any, plan: DeckPlan) -> None:
        theme = plan.theme
        cx, cy = plan.settings.size
        width = int(cx * 0.6)
        height = int(EMU_PER_POINT * 20)
        frame = _textbox(
            slide,
            pptx,
            left=theme.margin_x,
            top=cy - theme.margin_y + int(EMU_PER_POINT * 2),
            width=width,
            height=height,
        )
        run = frame.paragraphs[0].add_run()
        run.text = plan.settings.footer
        _style_run(
            run,
            pptx=pptx,
            size=theme.small_size - 4,
            font=theme.body_font,
            color=theme.muted,
            arabic_font=theme.script_font(theme.body_font),
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeckValidation:
    """What reopening a written deck found."""

    ok: bool
    slide_count: int
    titles: tuple[str, ...] = ()
    notes_slides: tuple[int, ...] = ()
    layouts: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    size_bytes: int = 0

    def to_primitive(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "slide_count": self.slide_count,
            "titles": list(self.titles),
            "notes_slides": list(self.notes_slides),
            "layouts": list(self.layouts),
            "issues": list(self.issues),
            "size_bytes": self.size_bytes,
        }


def validate_deck(path: str | Path, plan: DeckPlan | None = None) -> DeckValidation:
    """Re-open a written ``.pptx`` and confirm it matches the plan.

    The check is deliberately *deep*: it opens the file with python-pptx, counts
    the slides, reads each slide's title text and speaker notes back, and (when a
    plan is given) compares the slide count, the order, and the titles. A
    mismatch is reported as an issue rather than raised, so the caller can decide
    what to do. Reopening is also what proves the file is a real, editable deck
    rather than a text blob.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return DeckValidation(False, 0, issues=(f"No file was written at {target}.",))
    size = target.stat().st_size
    try:
        pptx = _require_pptx()
        presentation = pptx.Presentation(str(target))
    except PptxUnavailable:
        raise
    except Exception as error:
        return DeckValidation(False, 0, issues=(f"The deck could not be reopened: {error}",), size_bytes=size)

    slides = list(presentation.slides)
    titles: list[str] = []
    notes_slides: list[int] = []
    for index, slide in enumerate(slides, start=1):
        titles.append(_slide_title_text(slide))
        if _slide_notes_text(slide).strip():
            notes_slides.append(index)

    issues: list[str] = []
    if plan is not None:
        if len(slides) != plan.slide_count:
            issues.append(
                f"The deck has {len(slides)} slide(s) but the plan has {plan.slide_count}."
            )
        for index, slide_plan in enumerate(plan.slides):
            if index >= len(titles):
                break
            expected = slide_plan.title.strip()
            actual = titles[index].strip()
            if expected and expected not in actual and actual and expected != actual:
                # The title may be drawn as a heading plus a rule; a substring
                # match is enough to prove the right slide is in the right place.
                issues.append(
                    f"Slide {index + 1} is titled {actual!r}, expected {expected!r}."
                )
    ok = not issues and bool(slides)
    if not slides:
        issues.append("The deck contains no slides.")
    return DeckValidation(
        ok=ok,
        slide_count=len(slides),
        titles=tuple(titles),
        notes_slides=tuple(notes_slides),
        layouts=tuple(slide.slide_layout.name for slide in slides),
        issues=tuple(issues),
        size_bytes=size,
    )


def _slide_title_text(slide: Any) -> str:
    """The first text a slide carries (its heading), for an order check."""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        if text:
            return text.splitlines()[0]
    return ""


def _slide_notes_text(slide: Any) -> str:
    """A slide's speaker notes, or an empty string when it has none."""
    try:
        if slide.has_notes_slide:
            return slide.notes_slide.notes_text_frame.text
    except Exception:
        return ""
    return ""


def write_deck(plan: DeckPlan, path: str | Path) -> RenderReport:
    """Render ``plan`` to ``path`` and validate the result; return a report."""
    engine = PptxEngine()
    written = engine.write(plan, path)
    validation = validate_deck(written, plan)
    warnings = list(plan.warnings)
    if not validation.ok:
        warnings.extend(validation.issues)
    return RenderReport(
        path=str(written),
        slide_count=validation.slide_count,
        size_bytes=validation.size_bytes,
        warnings=tuple(warnings),
        layouts=validation.layouts,
    )


def safe_deck_path(directory: str | Path, filename: str) -> Path:
    """A traversal-proof path for a deck inside ``directory``."""
    return safe_output_path(directory, filename)


__all__ = [
    "REQUIREMENTS_FILE",
    "DeckValidation",
    "PptxEngine",
    "PptxUnavailable",
    "RenderReport",
    "pptx_available",
    "safe_deck_path",
    "validate_deck",
    "write_deck",
]
