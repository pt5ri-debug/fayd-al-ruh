"""A small, dependency-free Markdown renderer for the message transcript.

Tk has no Markdown engine, and the project refuses to add a runtime dependency
for one. This module parses the subset of Markdown a chat answer actually uses
into a flat list of :class:`Block`s (headings, paragraphs, lists, quotes, fenced
code, tables, rules), with inline emphasis and code split into
:class:`Segment`s. The Tk window walks that structure and applies its own tags,
so the *logic* is testable here without a display and the *look* stays in
``gui/app.py``.

A lightweight keyword/string/comment/number highlighter is included for fenced
code, so code answers get real syntax colouring without pulling in Pygments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Block kinds the window knows how to draw.
HEADING = "heading"
PARAGRAPH = "paragraph"
BULLET = "bullet"
NUMBER = "number"
QUOTE = "quote"
CODE = "code"
TABLE = "table"
RULE = "rule"

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_BULLET = re.compile(r"^\s*[-*+•]\s+(.*)$")
_NUMBER = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_FENCE = re.compile(r"^\s*```\s*([\w+-]*)\s*$")
_INLINE = re.compile(r"(\*\*.+?\*\*|__.+?__|`[^`]+`)")
_CODE_KEYWORD = re.compile(
    r"\b(?:def|class|return|if|else|elif|for|while|import|from|as|with|try|except|"
    r"finally|raise|yield|lambda|and|or|not|in|is|None|True|False|pass|break|continue|"
    r"function|const|let|var|new|public|private|static|void|int|float|double|string|"
    r"select|from|where|insert|update|delete|create|table|join|group|order|by)\b"
)
_CODE_STRING = re.compile(r"(\"[^\"\n]*\"|'[^'\n]*')")
_CODE_COMMENT = re.compile(r"(#.*$|//.*$)")
_CODE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class Segment:
    """A run of inline text with one style: ``text``, ``bold``, or ``code``."""

    kind: str
    text: str


@dataclass(frozen=True)
class Block:
    """One Markdown block, ready for the window to draw."""

    kind: str
    text: str = ""
    level: int = 0
    marker: str = ""
    language: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    segments: tuple[Segment, ...] = field(default_factory=tuple)


def parse_inline(text: str) -> tuple[Segment, ...]:
    """Split a line into ``text`` / ``bold`` / ``code`` segments.

    Only the two inline styles a chat answer relies on are recognised -- bold
    (``**x**`` / ``__x__``) and inline code (`` `x` ``). Italic is deliberately
    left alone so a formula like ``2 * 3`` is never mistaken for emphasis.
    """
    segments: list[Segment] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            segments.append(Segment("text", text[position:match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            segments.append(Segment("code", token[1:-1]))
        else:
            segments.append(Segment("bold", token[2:-2]))
        position = match.end()
    if position < len(text):
        segments.append(Segment("text", text[position:]))
    if not segments:
        segments.append(Segment("text", ""))
    return tuple(segments)


def _is_table_row(line: str) -> bool:
    return "|" in line.strip().strip("|")


def _split_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return tuple(cell.strip() for cell in stripped.split("|"))


def render_markdown(text: str) -> tuple[Block, ...]:
    """Parse ``text`` into an ordered tuple of :class:`Block`s.

    The parser is line-based and forgiving: anything it does not recognise stays
    an ordinary paragraph, so model prose is never lost or mangled.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        fence = _FENCE.match(line)
        if fence:
            index, block = _consume_code(lines, index, fence.group(1))
            blocks.append(block)
            continue

        if not line.strip():
            index += 1
            continue

        if _RULE.match(line):
            blocks.append(Block(RULE))
            index += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(
                Block(
                    HEADING,
                    text=heading.group(2).strip(),
                    level=len(heading.group(1)),
                    segments=parse_inline(heading.group(2).strip()),
                )
            )
            index += 1
            continue

        if _is_table_row(line) and index + 1 < len(lines) and _TABLE_SEP.match(lines[index + 1]):
            index, block = _consume_table(lines, index)
            blocks.append(block)
            continue

        bullet = _BULLET.match(line)
        if bullet:
            blocks.append(
                Block(BULLET, text=bullet.group(1).strip(), marker="•", segments=parse_inline(bullet.group(1).strip()))
            )
            index += 1
            continue

        number = _NUMBER.match(line)
        if number:
            blocks.append(
                Block(
                    NUMBER,
                    text=number.group(2).strip(),
                    marker=f"{number.group(1)}.",
                    segments=parse_inline(number.group(2).strip()),
                )
            )
            index += 1
            continue

        quote = _QUOTE.match(line)
        if quote:
            blocks.append(
                Block(QUOTE, text=quote.group(1).strip(), segments=parse_inline(quote.group(1).strip()))
            )
            index += 1
            continue

        index, block = _consume_paragraph(lines, index)
        blocks.append(block)
    return tuple(blocks)


def _consume_code(lines: list[str], start: int, language: str) -> tuple[int, Block]:
    body: list[str] = []
    index = start + 1
    while index < len(lines) and not lines[index].strip().startswith("```"):
        body.append(lines[index])
        index += 1
    index += 1  # skip the closing fence (or run off the end)
    return index, Block(CODE, text="\n".join(body), language=language)


def _consume_table(lines: list[str], start: int) -> tuple[int, Block]:
    rows: list[tuple[str, ...]] = [_split_row(lines[start])]
    index = start + 2  # skip the header and the separator
    while index < len(lines) and _is_table_row(lines[index]):
        rows.append(_split_row(lines[index]))
        index += 1
    return index, Block(TABLE, rows=tuple(rows))


def _consume_paragraph(lines: list[str], start: int) -> tuple[int, Block]:
    collected: list[str] = []
    index = start
    while index < len(lines) and lines[index].strip() and not _starts_block(lines[index]):
        collected.append(lines[index].strip())
        index += 1
    text = " ".join(collected)
    return index, Block(PARAGRAPH, text=text, segments=parse_inline(text))


def _starts_block(line: str) -> bool:
    return bool(
        _FENCE.match(line)
        or _RULE.match(line)
        or _HEADING.match(line)
        or _BULLET.match(line)
        or _NUMBER.match(line)
        or _QUOTE.match(line)
        or _is_table_row(line)
    )


@dataclass(frozen=True)
class StyledLine:
    """One source line with the style the window should apply to it.

    ``text`` is the line exactly as it appeared, so a Text widget that inserts
    every :class:`StyledLine` in order, separated by newlines, round-trips the
    original message character for character -- which keeps the transcript fully
    selectable and copyable while still being styled.
    """

    text: str
    kind: str
    level: int = 0
    in_code: bool = False
    language: str = ""


def iter_styled_lines(text: str) -> tuple[StyledLine, ...]:
    """Classify each line of ``text`` for tag-based rendering in a Text widget.

    Mirrors :func:`render_markdown`'s block grammar but preserves the raw lines,
    so the window can apply Markdown styling without altering the message text.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    styled: list[StyledLine] = []
    in_code = False
    language = ""
    for index, line in enumerate(lines):
        if in_code:
            if line.strip().startswith("```"):
                in_code = False
                language = ""
                styled.append(StyledLine(line, CODE, in_code=True))
                continue
            styled.append(StyledLine(line, CODE, in_code=True, language=language))
            continue
        fence = _FENCE.match(line)
        if fence:
            in_code = True
            language = fence.group(1)
            styled.append(StyledLine(line, CODE, in_code=True, language=language))
            continue
        if not line.strip():
            styled.append(StyledLine(line, PARAGRAPH))
            continue
        if _RULE.match(line):
            styled.append(StyledLine(line, RULE))
            continue
        heading = _HEADING.match(line)
        if heading:
            styled.append(StyledLine(line, HEADING, level=len(heading.group(1))))
            continue
        if _is_table_row(line) and index + 1 < len(lines) and _TABLE_SEP.match(lines[index + 1]):
            styled.append(StyledLine(line, TABLE))
            continue
        if _TABLE_SEP.match(line) and index > 0 and _is_table_row(lines[index - 1]):
            styled.append(StyledLine(line, TABLE))
            continue
        if _BULLET.match(line):
            styled.append(StyledLine(line, BULLET))
            continue
        if _NUMBER.match(line):
            styled.append(StyledLine(line, NUMBER))
            continue
        if _QUOTE.match(line):
            styled.append(StyledLine(line, QUOTE))
            continue
        styled.append(StyledLine(line, PARAGRAPH))
    return tuple(styled)


def highlight_code(code: str, language: str = "") -> tuple[Segment, ...]:
    """Split a code block into plain/keyword/string/comment/number segments.

    A deliberately small highlighter: enough to make code answers readable
    (keywords, strings, comments, numbers) with no third-party dependency. The
    window maps each segment kind to a colour; unknown text stays ``plain``.
    """
    spans: list[Segment] = []
    for line in code.split("\n"):
        comment = _CODE_COMMENT.search(line)
        body = line
        tail = ""
        if comment:
            body = line[: comment.start()]
            tail = line[comment.start():]
        spans.extend(_highlight_line(body))
        if tail:
            spans.append(Segment("comment", tail))
        spans.append(Segment("plain", "\n"))
    if spans and spans[-1].text == "\n":
        spans.pop()
    return tuple(spans)


def _highlight_line(line: str) -> list[Segment]:
    tokens = re.split(r"(\"[^\"\n]*\"|'[^'\n]*')", line)
    spans: list[Segment] = []
    for token in tokens:
        if not token:
            continue
        if _CODE_STRING.fullmatch(token):
            spans.append(Segment("string", token))
            continue
        spans.extend(_highlight_words(token))
    return spans


def _highlight_words(token: str) -> list[Segment]:
    spans: list[Segment] = []
    position = 0
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b|\w+", token):
        if match.start() > position:
            spans.append(Segment("plain", token[position:match.start()]))
        word = match.group(0)
        if _CODE_NUMBER.fullmatch(word):
            spans.append(Segment("number", word))
        elif _CODE_KEYWORD.fullmatch(word):
            spans.append(Segment("keyword", word))
        else:
            spans.append(Segment("plain", word))
        position = match.end()
    if position < len(token):
        spans.append(Segment("plain", token[position:]))
    return spans


__all__ = [
    "BULLET",
    "CODE",
    "HEADING",
    "NUMBER",
    "PARAGRAPH",
    "QUOTE",
    "RULE",
    "TABLE",
    "Block",
    "Segment",
    "StyledLine",
    "highlight_code",
    "iter_styled_lines",
    "parse_inline",
    "render_markdown",
]
