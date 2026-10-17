"""Markdown rendering onto a Tk ``Text``, shared by every window.

The main window (:mod:`gui.app`) and the Code Studio window both draw a
Markdown answer into a read-only ``Text``. The *parsing* lives in
:mod:`gui.markdown`; the *painting* used to be copied into each window, which
is exactly how two windows drift apart in font size and line spacing.

This module owns the painting instead, so every window -- the chat, the Studio,
and any window added later -- draws a heading, a list, a quote, a code block,
and a table with the one type scale from :mod:`gui.typography` and the one
palette from :class:`gui.app.Theme`. A window that wants Markdown calls
:func:`insert_markdown` and nothing else.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, NamedTuple

from gui import markdown as md
from gui.typography import Typography

#: The tag names a rendered message uses. Exposed so a window (or a test) can
#: refer to them by name rather than by string literal.
PARAGRAPH_TAG = "paragraph"
HEADING1_TAG = "heading"
HEADING2_TAG = "heading2"
HEADING3_TAG = "heading3"
BULLET_TAG = "bullet"
QUOTE_TAG = "quote"
CODE_TAG = "code"
TABLE_TAG = "table"
RULE_TAG = "rule"
BOLD_TAG = "bold"
INLINE_CODE_TAG = "inline_code"


class LineSpacing(NamedTuple):
    """The widget-level line rhythm for a Markdown body.

    A tag sets the spacing of the lines it covers, but a *blank* line (the one
    between two Markdown paragraphs) belongs to no block tag, so its height
    comes from the widget's own ``spacing1``/``spacing3``. Giving them the body
    paragraph's own gap keeps the paragraph break even instead of collapsing to
    a bare newline.
    """

    spacing1: int
    spacing3: int


def body_spacing(typography: Typography) -> LineSpacing:
    """The widget-level line rhythm for a Markdown body (see :class:`LineSpacing`)."""
    metrics = typography.block_metrics("body", gap_px=1)
    return LineSpacing(spacing1=metrics.spacing1, spacing3=metrics.spacing3)


def apply_body_spacing(widget: tk.Text, typography: Typography) -> None:
    """Give ``widget`` the body line rhythm, so blank separator lines have height."""
    spacing = body_spacing(typography)
    widget.configure(spacing1=spacing.spacing1, spacing3=spacing.spacing3)


def configure_markdown_tags(widget: tk.Text, *, theme: Any, typography: Typography) -> None:
    """Create the tags :func:`insert_markdown` applies, on this widget.

    Sizes come from :mod:`gui.typography` (OpenHands' scale); each tag's
    ``spacing1``/``spacing2``/``spacing3`` realise the web line-height and the
    block's own margin (``mt-*``/``mb-*``/``py-*``), because a Text *tag* has no
    ``-linespacing`` option.
    """
    t = typography
    # p: py-1.5 leading-6 (paragraph.tsx).
    body = t.block_metrics("body", gap_px=1)
    widget.tag_configure(
        PARAGRAPH_TAG, font=t.font("body"),
        spacing1=body.spacing1, spacing2=body.spacing2, spacing3=body.spacing3,
    )
    # h1: text-xl leading-7, mt-4 mb-3; h2: text-lg leading-6, mt-4 mb-2.5;
    # h3+: mt-3 mb-2 (headings.tsx). OpenHands uses font-medium; Tk's nearest
    # weight is bold.
    h1 = t.block_metrics("heading1", gap_px=14)
    widget.tag_configure(
        HEADING1_TAG, font=t.font("heading1"), foreground=theme.contrast,
        spacing1=h1.spacing1, spacing2=h1.spacing2, spacing3=h1.spacing3,
    )
    h2 = t.block_metrics("heading2", gap_px=13)
    widget.tag_configure(
        HEADING2_TAG, font=t.font("heading2"), foreground=theme.contrast,
        spacing1=h2.spacing1, spacing2=h2.spacing2, spacing3=h2.spacing3,
    )
    h3 = t.block_metrics("heading3", gap_px=10)
    widget.tag_configure(
        HEADING3_TAG, font=t.font("heading3"), foreground=theme.contrast,
        spacing1=h3.spacing1, spacing2=h3.spacing2, spacing3=h3.spacing3,
    )
    # ul/ol: my-2 leading-6, li py-0.5 (list.tsx).
    li = t.block_metrics("body", gap_px=2)
    widget.tag_configure(
        BULLET_TAG, font=t.font("body"), lmargin1=16, lmargin2=30,
        spacing1=li.spacing1, spacing2=li.spacing2, spacing3=li.spacing3,
    )
    # blockquote: my-2, italic, muted, with the 4px accent bar (blockquote.tsx).
    quote = t.block_metrics("body", gap_px=8)
    widget.tag_configure(
        QUOTE_TAG, font=t.font("body", slant="italic"), lmargin1=16, lmargin2=16,
        foreground=theme.text_secondary, background=theme.surface_raised,
        spacing1=quote.spacing1, spacing2=quote.spacing2, spacing3=quote.spacing3,
    )
    code = t.block_metrics("code", gap_px=2)
    widget.tag_configure(
        CODE_TAG, font=t.mono("code"), background=theme.code_bg,
        foreground=theme.code_fg, lmargin1=10, lmargin2=10,
        spacing1=code.spacing1, spacing2=code.spacing2, spacing3=code.spacing3,
    )
    # table: my-4, text-sm, cells px-3 py-2 (table.tsx).
    cell = t.block_metrics("table", gap_px=3)
    widget.tag_configure(
        TABLE_TAG, font=t.mono("table"), foreground=theme.foreground, lmargin1=8,
        spacing1=cell.spacing1, spacing2=cell.spacing2, spacing3=cell.spacing3,
    )
    # hr: my-4 (horizontal-rule.tsx).
    widget.tag_configure(RULE_TAG, foreground=theme.border, spacing1=t.px(16), spacing3=t.px(16))
    widget.tag_configure(BOLD_TAG, font=t.font("body", weight="bold"), foreground=theme.contrast)
    widget.tag_configure(
        INLINE_CODE_TAG, font=t.mono("code"), background=theme.code_bg, foreground=theme.code_fg
    )
    widget.tag_configure("kw", foreground=theme.primary)
    widget.tag_configure("str", foreground=theme.success)
    widget.tag_configure("cmt", foreground=theme.text_dim)


def insert_markdown(widget: tk.Text, text: str, *, theme: Any, typography: Typography) -> None:
    """Render ``text`` into ``widget`` as Markdown, tags included.

    Newlines are inserted *between* lines only, so the widget's content is the
    message exactly -- no trailing newline -- which keeps a copied message
    byte-for-byte identical to what was shown.
    """
    configure_markdown_tags(widget, theme=theme, typography=typography)
    for index, styled in enumerate(md.iter_styled_lines(text)):
        if index:
            widget.insert("end", "\n")
        _insert_styled_line(widget, styled)


def _insert_styled_line(widget: tk.Text, styled: md.StyledLine) -> None:
    start = widget.index("end-1c")
    widget.insert("end", styled.text)
    end = widget.index("end-1c")
    tag = _tag_for(styled)
    widget.tag_add(tag, start, end)
    if styled.in_code:
        _highlight_code_line(widget, styled.text, start)
    elif styled.kind not in (md.CODE, md.RULE):
        _inline_tags(widget, styled.text, start)


def _tag_for(styled: md.StyledLine) -> str:
    """The tag one styled line is drawn with."""
    if styled.in_code:
        return CODE_TAG
    if styled.kind == md.HEADING:
        if styled.level == 1:
            return HEADING1_TAG
        if styled.level == 2:
            return HEADING2_TAG
        return HEADING3_TAG
    if styled.kind in (md.BULLET, md.NUMBER):
        return BULLET_TAG
    if styled.kind == md.QUOTE:
        return QUOTE_TAG
    if styled.kind == md.TABLE:
        return TABLE_TAG
    if styled.kind == md.RULE:
        return RULE_TAG
    return PARAGRAPH_TAG


def _inline_tags(widget: tk.Text, line: str, line_start: str) -> None:
    """Add the bold / inline-code tags to the recognised spans of one line."""
    row = line_start.split(".")[0]
    for segment in md.parse_inline(line):
        if segment.kind == "bold":
            token = f"**{segment.text}**"
            idx = line.find(token)
            if idx >= 0:
                widget.tag_add(BOLD_TAG, f"{row}.{idx}", f"{row}.{idx + len(token)}")
    position = 0
    while True:
        opening = line.find("`", position)
        if opening < 0:
            break
        closing = line.find("`", opening + 1)
        if closing < 0:
            break
        widget.tag_add(INLINE_CODE_TAG, f"{row}.{opening}", f"{row}.{closing + 1}")
        position = closing + 1


def _highlight_code_line(widget: tk.Text, line: str, start: str) -> None:
    """Colour the keywords/strings/comments inside a fenced code line."""
    row = start.split(".")[0]
    for segment in md.highlight_code(line):
        if segment.kind == "plain" or segment.text == "\n":
            continue
        tag = {"keyword": "kw", "string": "str", "comment": "cmt"}.get(segment.kind)
        if tag is None:
            continue
        idx = line.find(segment.text)
        if idx >= 0:
            widget.tag_add(tag, f"{row}.{idx}", f"{row}.{idx + len(segment.text)}")
