"""Dependency-free builders for synthetic DOCX files used by the Stage 2B tests.

The tests must not depend on a Word-generation library, so this module writes
minimal, valid Office Open XML packages by hand with the standard library's
:mod:`zipfile` -- enough of the format for :mod:`core.docx_extract` to parse. It
covers the cases the ingestion stage has to handle:

* ordinary documents (paragraphs, runs, tabs, line breaks);
* headings (``Heading1`` .. ``Heading3`` styles);
* tables (rows and cells, including an empty cell);
* Arabic (and other non-Latin) text, stored as Unicode in the XML;
* empty documents (a body with no text);
* documents with no ``word/document.xml`` (a plain ZIP) and malformed XML;
* an OLE/CFB header (an encrypted or legacy ``.doc`` file);
* a macro part (``word/vbaProject.bin``), to prove it is never executed;
* archive-bomb fixtures (too many members, an over-large part) built on request.

Only the pieces a reader touches are emitted: ``word/document.xml`` and, when a
test asks for it, ``docProps/core.xml``. Nothing here executes or follows the
content -- the bytes are inert text.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: A non-ZIP byte string (the header of a corrupt/legacy file).
GARBAGE = b"this is not a zip archive at all"

#: The CFB/OLE2 magic an encrypted or legacy binary Word document starts with.
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

#: A short Arabic sentence used by the bilingual fixture.
_ARABIC = "ضغط الخزان والنفاذية"


def document_xml(body: str) -> str:
    """Wrap ``body`` XML in a minimal ``<w:document>`` envelope."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document xmlns:w="{_WORD_NAMESPACE}"><w:body>{body}</w:body></w:document>'
    )


def paragraph(text: str) -> str:
    """One paragraph holding a single run, with XML-escaped text."""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f"<w:p><w:r><w:t xml:space='preserve'>{escaped}</w:t></w:r></w:p>"


def heading(text: str, level: int = 1) -> str:
    """A heading paragraph carrying a ``HeadingN`` style."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<w:p><w:pPr><w:pStyle w:val='Heading{level}'/></w:pPr>"
        f"<w:r><w:t>{escaped}</w:t></w:r></w:p>"
    )


def table(rows: list[list[str]]) -> str:
    """A ``<w:tbl>`` with one ``<w:tc>`` per cell, each holding a paragraph."""
    row_xml: list[str] = []
    for row in rows:
        cells = "".join(f"<w:tc>{paragraph(cell)}</w:tc>" for cell in row)
        row_xml.append(f"<w:tr>{cells}</w:tr>")
    return "<w:tbl>" + "".join(row_xml) + "</w:tbl>"


def write_docx(
    path: Path,
    body: str | None = None,
    *,
    document_part: bool = True,
    extra_members: dict[str, bytes] | None = None,
    core_title: str | None = None,
    compresslevel: int | None = None,
) -> Path:
    """Write a DOCX package.

    ``body`` is the ``<w:body>`` XML (default: one paragraph). ``document_part``
    is False to omit ``word/document.xml`` (a plain ZIP). ``extra_members`` adds
    arbitrary archive members (e.g. a macro part or a large filler). ``core_title``
    writes ``docProps/core.xml`` with a title. ``compresslevel`` is forced to 0
    (stored, not deflated) when a test needs a *large declared size* without the
    file itself being large.
    """
    if body is None:
        body = paragraph("Reservoir porosity measures the void space in rock.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if compresslevel == 0:
        archive_cm = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED)
    elif compresslevel is None:
        archive_cm = zipfile.ZipFile(path, "w")
    else:
        archive_cm = zipfile.ZipFile(path, "w", compresslevel=compresslevel)
    with archive_cm as archive:
        if document_part:
            archive.writestr("word/document.xml", document_xml(body))
        if core_title is not None:
            archive.writestr(
                "docProps/core.xml",
                '<?xml version="1.0"?>'
                '<cp:coreProperties '
                'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/">'
                f"<dc:title>{core_title}</dc:title></cp:coreProperties>",
            )
        for name, payload in (extra_members or {}).items():
            archive.writestr(name, payload)
    return path


def write_plain_zip(path: Path, members: dict[str, bytes] | None = None) -> Path:
    """A ZIP that is *not* a DOCX (no ``word/document.xml``)."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in (members or {"readme.txt": b"just a zip"}).items():
            archive.writestr(name, payload)
    return path


def write_ole(path: Path, *, prefix: bytes = OLE_MAGIC) -> Path:
    """A file that begins with the OLE/CFB magic (encrypted or legacy Word)."""
    path.write_bytes(prefix + b"\x00" * 64)
    return path


def write_garbage(path: Path) -> Path:
    path.write_bytes(GARBAGE)
    return path


def write_many_members(path: Path, count: int, *, member_size: int = 4) -> Path:
    """A ZIP with ``count`` members, to exercise the member-count bomb limit."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml(paragraph("Body text.")))
        for index in range(count):
            archive.writestr(f"word/filler{index}.bin", b"x" * member_size)
    return path


def write_large_part(path: Path, size: int) -> Path:
    """A DOCX whose ``word/document.xml`` *declares* ``size`` bytes (stored).

    With compression off the declared size is the real size, so a modest file can
    present an over-large part without wasting disk or memory on a real bomb.
    """
    body = paragraph("x" * max(0, size - 200))
    return write_docx(path, body, compresslevel=0)


def latin_and_arabic_docx(path: Path) -> Path:
    """A small bilingual document, for the Arabic-preservation test."""
    body = (
        heading("Reservoir", 1)
        + paragraph("Porosity measures the voids in rock.")
        + paragraph(_ARABIC)
    )
    return write_docx(path, body)


__all__ = [
    "GARBAGE",
    "OLE_MAGIC",
    "document_xml",
    "heading",
    "latin_and_arabic_docx",
    "paragraph",
    "table",
    "write_docx",
    "write_garbage",
    "write_large_part",
    "write_many_members",
    "write_ole",
    "write_plain_zip",
]
