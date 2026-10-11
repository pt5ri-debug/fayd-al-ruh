"""Dependency-free builders for synthetic PDFs used by the Stage 2A tests.

The tests must not depend on a PDF-generation library, so this module writes
minimal, valid PDFs by hand -- enough of the format for ``pypdf`` to parse. It
covers the cases the ingestion stage has to handle:

* multi-page documents with English text (a built-in ``Helvetica`` font);
* Arabic (and other non-Latin) text, via a ``Type0``/``Identity-H`` font with a
  ``ToUnicode`` map so the extractor can recover the characters;
* blank pages (an empty content stream);
* image-only pages (an image XObject and no text) -- a synthetic "scan";
* a truncated/garbage byte string that is not a PDF at all.

Encrypted fixtures are built with ``pypdf`` itself (the optional dependency) in
the one test that needs them, since hand-rolling RC4 is not worth it; that test
is skipped when ``pypdf`` is absent.

Only the pieces a reader touches are emitted: a catalog, a page tree, fonts, a
``ToUnicode`` CMap, and content streams. Nothing here executes or follows the
content -- the bytes are inert text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: A page is either text (one or more runs), blank, or an image-only page.
_TEXT = "text"
_BLANK = "blank"
_IMAGE = "image"


@dataclass
class Run:
    """One run of text on a page: ``kind`` is ``"latin"`` or ``"unicode"``."""

    kind: str
    text: str


@dataclass
class Page:
    """One synthetic PDF page."""

    kind: str = _TEXT
    runs: list[Run] = field(default_factory=list)


def latin_page(*lines: str) -> Page:
    return Page(runs=[Run("latin", line) for line in lines])


def arabic_page(*lines: str) -> Page:
    return Page(runs=[Run("unicode", line) for line in lines])


def mixed_page(*runs: tuple[str, str]) -> Page:
    return Page(runs=[Run(kind, text) for kind, text in runs])


def blank_page() -> Page:
    return Page(kind=_BLANK)


def image_page() -> Page:
    """A page that shows only an image and carries no text (a synthetic scan)."""
    return Page(kind=_IMAGE)


def _escape_latin(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_content(page: Page) -> bytes:
    if page.kind == _BLANK:
        return b""
    if page.kind == _IMAGE:
        return b"q 200 0 0 200 72 480 cm /Im0 Do Q"
    parts: list[str] = []
    y = 720
    for run in page.runs:
        if run.kind == "unicode":
            codes = "".join(f"{ord(ch):04X}" for ch in run.text)
            parts.append(f"BT /F2 18 Tf 72 {y} Td <{codes}> Tj ET")
        else:
            parts.append(f"BT /F1 14 Tf 72 {y} Td ({_escape_latin(run.text)}) Tj ET")
        y -= 28
    return "\n".join(parts).encode("latin-1")


def _tounicode(pages: list[Page]) -> bytes:
    codes: list[int] = []
    for page in pages:
        for run in page.runs:
            if run.kind == "unicode":
                codes.extend(ord(ch) for ch in run.text)
    unique = sorted(set(codes))
    bfchar = "\n".join(f"<{c:04X}> <{c:04X}>" for c in unique)
    body = (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CMapName /FaydTest def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
    )
    if unique:
        body += f"{len(unique)} beginbfchar\n{bfchar}\nendbfchar\n"
    body += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    return body.encode("latin-1")


def build_pdf(
    pages: list[Page], *, title: str | None = None, open_action_js: str | None = None
) -> bytes:
    """Serialise ``pages`` into PDF bytes (dependency-free, minimal but valid).

    ``open_action_js`` embeds a JavaScript *open action* in the catalog -- used by
    the safety test to prove the reader never executes embedded scripts.
    """
    # Object numbers are fixed up front so references are simple integers.
    catalog_no = 1
    pages_no = 2
    helvetica_no = 3
    unicode_font_no = 4
    cid_font_no = 5
    tounicode_no = 6
    image_no = 7
    first_page_no = 8  # page/content pairs follow, two objects per page
    # Auxiliary objects live *after* every page object, so a large document can
    # never collide with them (a fixed high number would clash past ~46 pages).
    aux_no = first_page_no + len(pages) * 2
    font_descriptor_no = aux_no
    info_no = aux_no + 1
    action_no = aux_no + 2

    objects: dict[int, bytes] = {}
    page_refs: list[int] = []
    for index, page in enumerate(pages):
        page_no = first_page_no + index * 2
        content_no = page_no + 1
        page_refs.append(page_no)
        resources = "<< /Font << /F1 3 0 R /F2 4 0 R >>"
        if page.kind == _IMAGE:
            resources += f" /XObject << /Im0 {image_no} 0 R >>"
        resources += " >>"
        content = _page_content(page)
        objects[page_no] = (
            f"<< /Type /Page /Parent {pages_no} 0 R /MediaBox [0 0 612 792] "
            f"/Resources {resources} /Contents {content_no} 0 R >>"
        ).encode("latin-1")
        objects[content_no] = (
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
        )

    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    action = ""
    if open_action_js is not None:
        objects[action_no] = (
            f"<< /S /JavaScript /JS ({_escape_latin(open_action_js)}) >>".encode("latin-1")
        )
        action = f" /OpenAction {action_no} 0 R"
    objects[catalog_no] = (
        f"<< /Type /Catalog /Pages {pages_no} 0 R{action} >>"
    ).encode("latin-1")
    objects[pages_no] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>"
    ).encode("latin-1")
    objects[helvetica_no] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects[unicode_font_no] = (
        f"<< /Type /Font /Subtype /Type0 /BaseFont /FaydUnicode /Encoding /Identity-H "
        f"/DescendantFonts [{cid_font_no} 0 R] /ToUnicode {tounicode_no} 0 R >>"
    ).encode("latin-1")
    objects[cid_font_no] = (
        f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /FaydUnicode "
        f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        f"/FontDescriptor {font_descriptor_no} 0 R /DW 1000 >>"
    ).encode("latin-1")
    tounicode = _tounicode(pages)
    objects[tounicode_no] = (
        b"<< /Length " + str(len(tounicode)).encode() + b" >>\nstream\n" + tounicode + b"\nendstream"
    )
    image_data = bytes([0, 128, 255, 64])
    objects[image_no] = (
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        b"/BitsPerComponent 8 /Length "
        + str(len(image_data)).encode()
        + b" >>\nstream\n"
        + image_data
        + b"\nendstream"
    )
    # A minimal font descriptor for the CID font (the reader never renders it).
    objects[font_descriptor_no] = (
        b"<< /Type /FontDescriptor /FontName /FaydUnicode /Flags 4 "
        b"/FontBBox [0 0 1000 1000] /CapHeight 700 /Ascent 700 /Descent -200 /StemV 80 "
        b"/ItalicAngle 0 >>"
    )

    info = ""
    if title:
        objects[info_no] = (
            f"<< /Title ({_escape_latin(title)}) /Producer (fayd-tests) >>".encode("latin-1")
        )
        info = f" /Info {info_no} 0 R"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode("latin-1") + objects[number] + b"\nendobj\n"

    size = max(objects) + 1
    xref = len(out)
    out += f"xref\n0 {size}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        if number in offsets:
            out += f"{offsets[number]:010d} 00000 n \n".encode("latin-1")
        else:
            # A hole in the object numbering is a free entry, not a bad offset.
            out += b"0000000000 65535 f \n"
    trailer = f"<< /Size {size} /Root {catalog_no} 0 R{info} >>"
    out += f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def write_pdf(path: str | Path, pages: list[Page], *, title: str | None = None) -> Path:
    target = Path(path)
    target.write_bytes(build_pdf(pages, title=title))
    return target


#: Bytes that are not a PDF at all (a malformed-input fixture).
GARBAGE = b"%PDF-1.4\nthis is not a real pdf, just bytes pretending to be one\n"


__all__ = [
    "GARBAGE",
    "Page",
    "Run",
    "arabic_page",
    "blank_page",
    "build_pdf",
    "image_page",
    "latin_page",
    "mixed_page",
    "write_pdf",
]
