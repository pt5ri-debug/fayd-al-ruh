"""Read text out of a Word ``.docx`` file safely, with no third-party dependency.

Stage 2B of file ingestion adds **Word documents**. A ``.docx`` is an Office Open
XML package: a ZIP archive whose main document is ``word/document.xml``. This
module turns that package into an ordered list of text *blocks* -- paragraphs
(with their heading level when the paragraph is a heading) and table rows
(rendered as TSV) -- so a downstream connector can preserve the document's
structure as far as the format exposes it.

Why a dedicated module rather than growing ``core.document_reader``: a DOCX is an
**untrusted ZIP-based format**, and reading one safely is its own job. This module
owns the ZIP/XML handling, the resource limits, and the classification of
unusable files, so ``document_reader`` stays a thin delegator and the ingestion
connector never touches the archive itself.

Design points that matter for ingestion:

* **Bounded memory.** The archive is not extracted to disk; only
  ``word/document.xml`` is read into memory, and the total text kept is capped at
  ``max_chars``. A DOCX is small (text) so this is honest without pretending to
  be constant-memory.
* **Structure is preserved.** Blocks keep paragraph order; a table's rows become
  blocks rendered as TSV, and the rows of one table are grouped so a reader sees
  them together rather than interleaved with stray paragraphs.
* **Arabic and English are preserved** as characters. DOCX stores text as Unicode
  in XML, so there is no visual-order problem like PDF has.
* **Honest classification.** A non-ZIP file, a ZIP without ``word/document.xml``,
  a malformed XML part, an archive-bomb, an over-large part, an over-long
  document, an encrypted (OLE/CFB) file, and a document with no text are each
  reported distinctly.
* **No execution.** Macros (``word/vbaProject.bin``), embedded objects,
  hyperlink targets, and any instructions inside the text are never run or
  followed -- the XML is parsed into strings and nothing else.
* **Header/footer/comment/textbox content is deliberately excluded.** Only the
  main document body is read, because that is the document's own flow; see the
  "Limitations" section in the module table below.

Known limitations (documented on purpose, not silent):

* **Headers, footers, footnotes, endnotes, and comments** live in sibling parts
  (``word/header1.xml``, ``word/footnotes.xml``, ...) and are **not** read.
* **Tracked changes**: only the accepted text (``<w:t>`` runs outside
  ``<w:del>``) is read; deleted text stored in ``<w:delText>`` is skipped, and
  whether inserted text is present depends on how the producer saved it.
* **Images, charts, and other embedded drawings** contribute no text.
* **Text boxes and shapes** (``w:txbxContent`` in drawing markup) are not
  traversed.
* **Macro-enabled ``.docm``** is read for its body text; macros are never run.
* **Legacy binary ``.doc``** is a different format and is not supported.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_WORD_NAMESPACE}}}"
_DOCUMENT_PART = "word/document.xml"

#: Cap on the total text kept from one DOCX, mirroring the reader's own cap so a
#: pathological document cannot exhaust memory.
DEFAULT_MAX_CHARS = 2_000_000
#: Cap on the decompressed size of ``word/document.xml``. A real document is far
#: below this; a larger part is treated as suspicious (a possible archive bomb).
DEFAULT_MAX_PART_BYTES = 32 * 1024 * 1024
#: Cap on the number of archive members. A DOCX holds a handful; thousands is a
#: sign of a malicious package.
DEFAULT_MAX_ENTRIES = 512
#: Cap on a single member's decompressed size, so a bomb in any part is refused
#: before it is inflated.
DEFAULT_MAX_ENTRY_BYTES = 64 * 1024 * 1024
#: Cap on the total decompressed size across members, for the same reason.
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024

#: The CFB magic prefix (OLE2 compound file) -- what an encrypted or legacy
#: binary Word file starts with. A real DOCX starts with the ZIP magic instead.
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_MAGIC = b"PK\x03\x04"

#: Heading styles map to level via their *end* number: "Heading2" -> 2,
#: "heading 3" -> 3, "Titel" -> 1. Only a positive integer is kept.
_HEADING_STYLE_PREFIXES = ("heading", "titel", "title", "berschrift", "rubrik")


class DocxExtractionError(ValueError):
    """A DOCX could not be read as text (malformed, encrypted, empty, too large, ...)."""


@dataclass(frozen=True)
class DocxBlock:
    """One ordered block of a DOCX: a paragraph, or one row of a table.

    ``kind`` is ``"paragraph"`` or ``"table_row"``. ``level`` is the heading
    level (1-9) when the paragraph carries a heading style, else ``0``.
    ``table`` is the 1-based index of the table a row belongs to (``0`` for a
    paragraph), so the rows of one table can be kept together.
    """

    kind: str
    text: str
    level: int = 0
    table: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"paragraph", "table_row"}:
            raise ValueError(f"Unknown DocxBlock kind {self.kind!r}.")


@dataclass(frozen=True)
class DocxDocument:
    """The blocks of a DOCX plus a few honest facts about how it was read."""

    blocks: tuple[DocxBlock, ...]
    title: str
    truncated: bool = False
    """True when the accumulated text hit :data:`DEFAULT_MAX_CHARS`."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Text-free metadata (title, paragraph/table counts), when available."""

    @property
    def paragraphs(self) -> tuple[str, ...]:
        """The text of every block that carries text, in order (flattened)."""
        return tuple(block.text for block in self.blocks if block.text.strip())

    @property
    def text(self) -> str:
        """Every block's text joined by a newline, so block boundaries survive."""
        return "\n".join(block.text for block in self.blocks if block.text.strip())

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def heading_count(self) -> int:
        return sum(1 for block in self.blocks if block.level > 0)

    @property
    def table_count(self) -> int:
        return max((block.table for block in self.blocks), default=0)


def _require_zipfile() -> Any:
    return zipfile


def _looks_encrypted_or_legacy(path: Path) -> bool:
    """Whether the file starts with the OLE/CFB magic (encrypted or legacy Word).

    An encrypted Office document is an OLE2 compound file, as is the legacy
    binary ``.doc`` format. Either way the bytes are not a ZIP package, so this
    is checked before ``zipfile`` to give a specific message.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    return head.startswith(_OLE_MAGIC)


def _safe_archive_members(
    archive: zipfile.ZipFile,
    *,
    max_entries: int,
    max_entry_bytes: int,
    max_total_bytes: int,
) -> None:
    """Reject a ZIP whose size or member count suggests an archive bomb.

    Every member's *declared* decompressed size is checked before anything is
    read, and the total is checked as well, so a small archive that inflates to
    gigabytes is refused rather than expanded.
    """
    infos = archive.infolist()
    if len(infos) > max_entries:
        raise DocxExtractionError(
            f"The archive holds {len(infos)} members, above the {max_entries} limit; "
            "refusing it as a possible archive bomb."
        )
    total = 0
    for info in infos:
        size = info.file_size
        if size > max_entry_bytes:
            raise DocxExtractionError(
                f"Archive member {info.filename!r} declares {size / 1024 / 1024:.1f} MB, "
                f"above the {max_entry_bytes / 1024 / 1024:.0f} MB per-member limit; "
                "refusing it as a possible archive bomb."
            )
        total += size
        if total > max_total_bytes:
            raise DocxExtractionError(
                f"The archive declares {total / 1024 / 1024:.1f} MB in total, above the "
                f"{max_total_bytes / 1024 / 1024:.0f} MB limit; refusing it as a "
                "possible archive bomb."
            )


def _read_document_part(
    archive: zipfile.ZipFile, *, max_part_bytes: int
) -> bytes:
    """Read just ``word/document.xml``, bounded by ``max_part_bytes``."""
    try:
        info = archive.getinfo(_DOCUMENT_PART)
    except KeyError as error:
        raise DocxExtractionError(
            f"The package is not a valid Word document (missing {_DOCUMENT_PART}). "
            "It may be a plain ZIP or a different Office file."
        ) from error
    if info.file_size > max_part_bytes:
        raise DocxExtractionError(
            f"{_DOCUMENT_PART} is {info.file_size / 1024 / 1024:.1f} MB, above the "
            f"{max_part_bytes / 1024 / 1024:.0f} MB limit; refusing an over-large document."
        )
    try:
        # read() honours the declared size; the check above bounds it.
        return archive.read(_DOCUMENT_PART)
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise DocxExtractionError(f"Could not read {_DOCUMENT_PART}: {error}") from error


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """The visible text of one paragraph, honouring tabs and line breaks.

    Only ``<w:t>`` runs contribute characters; ``<w:tab>`` becomes a tab and
    ``<w:br>``/``<w:cr>`` a line break. ``<w:delText>`` (deleted tracked text) is
    ignored because its tag differs, so a tracked deletion does not reappear.
    """
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _heading_level(paragraph: ElementTree.Element) -> int:
    """The heading level of a paragraph (1-9) when it carries a heading style.

    Reads ``<w:pPr><w:pStyle w:val="Heading2"/>`` and maps a trailing integer to
    a level. A paragraph without a heading style -- or with a non-heading one --
    returns ``0``.
    """
    properties = paragraph.find(f"{_W}pPr")
    if properties is None:
        return 0
    style = properties.find(f"{_W}pStyle")
    if style is None:
        return 0
    value = style.get(f"{_W}val") or style.get("val") or ""
    lowered = value.casefold()
    if not lowered.startswith(_HEADING_STYLE_PREFIXES):
        return 0
    digits = "".join(ch for ch in lowered if ch.isdigit())
    if not digits:
        return 1  # a plain "Heading"/"Title" style is a top-level heading
    level = int(digits)
    return level if 1 <= level <= 9 else 0


def _cell_text(cell: ElementTree.Element) -> str:
    """The text of one table cell, flattening its paragraphs with a space."""
    pieces = [_paragraph_text(p) for p in cell.iter(f"{_W}p")]
    return " ".join(piece.strip() for piece in pieces if piece.strip())


def _iter_table_rows(table: ElementTree.Element) -> Iterator[list[str]]:
    """Yield each row of a table as a list of its cell strings."""
    for row in table.iter(f"{_W}tr"):
        cells = [_cell_text(cell) for cell in row.findall(f"{_W}tc")]
        if any(cell.strip() for cell in cells):
            yield cells


def _blocks_from_body(body: ElementTree.Element, *, max_chars: int) -> tuple[list[DocxBlock], bool]:
    """Walk a document body in order, emitting paragraph and table-row blocks.

    Iterating the body's direct children (rather than ``iter``) keeps the order
    the reader sees: a table sits between the paragraphs that surround it. When
    the accumulated text would exceed ``max_chars`` the walk stops and the
    truncation is reported.
    """
    blocks: list[DocxBlock] = []
    total = 0
    truncated = False
    table_index = 0
    for child in body:
        if child.tag == f"{_W}p":
            text = _paragraph_text(child).strip()
            if not text:
                continue
            remaining = max_chars - total
            if len(text) + 1 > remaining:
                # Keep what fits (a document is capped, never dropped whole).
                text = text[: max(0, remaining - 1)]
                truncated = True
            if text:
                blocks.append(DocxBlock(kind="paragraph", text=text, level=_heading_level(child)))
                total += len(text) + 1
            if truncated:
                break
        elif child.tag == f"{_W}tbl":
            table_index += 1
            for cells in _iter_table_rows(child):
                line = "\t".join(cells)
                remaining = max_chars - total
                if len(line) + 1 > remaining:
                    line = line[: max(0, remaining - 1)]
                    truncated = True
                if line:
                    blocks.append(DocxBlock(kind="table_row", text=line, table=table_index))
                    total += len(line) + 1
                if truncated:
                    break
            if truncated:
                break
    return blocks, truncated


def _package_metadata(archive: zipfile.ZipFile) -> dict[str, Any]:
    """A couple of text-free facts from the package (e.g. the core title)."""
    meta: dict[str, Any] = {}
    try:
        if "docProps/core.xml" in archive.namelist():
            payload = archive.read("docProps/core.xml")
            root = ElementTree.fromstring(payload)
            for node in root.iter():
                if node.tag.endswith("}title") and node.text:
                    value = node.text.strip()
                    if value:
                        meta["title"] = value
    except (KeyError, ElementTree.ParseError, OSError, RuntimeError):
        pass
    return meta


class DocxReader:
    """Opens a DOCX package once and exposes its ordered blocks.

    Opening validates the file up front (exists, is a ZIP, not an encrypted/legacy
    OLE file, within the archive resource limits, with a readable
    ``word/document.xml``), so an unusable file fails fast with a specific reason.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1.")
        if max_part_bytes < 1:
            raise ValueError("max_part_bytes must be at least 1.")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self.path = Path(path).expanduser()
        self.max_chars = max_chars
        self.max_part_bytes = max_part_bytes
        self.max_entries = max_entries
        self.max_entry_bytes = max_entry_bytes
        self.max_total_bytes = max_total_bytes
        self.truncated = False
        self.metadata: dict[str, Any] = {}
        self.title = self.path.stem or self.path.name
        self._blocks: tuple[DocxBlock, ...] = ()
        self._read()

    def _read(self) -> None:
        source = self.path
        if not source.exists():
            raise DocxExtractionError(f"No such file: {source}")
        if not source.is_file():
            raise DocxExtractionError(f"Not a file: {source}")
        if _looks_encrypted_or_legacy(source):
            raise DocxExtractionError(
                f"{source.name!r} looks like an encrypted or legacy binary Word file, "
                "not a .docx package. Password-protected and .doc files are not supported; "
                "save it as .docx (unencrypted) and try again."
            )
        try:
            with zipfile.ZipFile(source) as archive:
                _safe_archive_members(
                    archive,
                    max_entries=self.max_entries,
                    max_entry_bytes=self.max_entry_bytes,
                    max_total_bytes=self.max_total_bytes,
                )
                payload = _read_document_part(archive, max_part_bytes=self.max_part_bytes)
                self.metadata = _package_metadata(archive)
        except zipfile.BadZipFile as error:
            raise DocxExtractionError(
                f"{source.name!r} is not a valid .docx package (it is not a ZIP archive)."
            ) from error
        except OSError as error:
            raise DocxExtractionError(f"Could not open {source.name!r}: {error}") from error

        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as error:
            raise DocxExtractionError(
                f"{source.name!r} has malformed document XML: {error}"
            ) from error

        body = root.find(f"{_W}body")
        if body is None:
            raise DocxExtractionError(
                f"{source.name!r} has no document body ({_W}body missing)."
            )
        blocks, truncated = _blocks_from_body(body, max_chars=self.max_chars)
        self.truncated = truncated
        if not any(block.text.strip() for block in blocks):
            raise DocxExtractionError(
                f"{source.name!r} contains no text (it may be empty, or hold only "
                "images, drawings, or empty tables)."
            )
        self._blocks = tuple(blocks)
        meta_title = self.metadata.get("title")
        if isinstance(meta_title, str) and meta_title.strip():
            self.title = meta_title.strip()

    def blocks(self) -> tuple[DocxBlock, ...]:
        return self._blocks

    def document(self) -> DocxDocument:
        return DocxDocument(
            blocks=self._blocks,
            title=self.title,
            truncated=self.truncated,
            metadata={
                **self.metadata,
                "paragraphs": sum(1 for b in self._blocks if b.kind == "paragraph"),
                "table_rows": sum(1 for b in self._blocks if b.kind == "table_row"),
                "tables": max((b.table for b in self._blocks), default=0),
                "headings": sum(1 for b in self._blocks if b.level > 0),
            },
        )


def extract_docx(
    path: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> DocxDocument:
    """Materialise ``path`` into a :class:`DocxDocument` (all blocks in memory).

    Raises :class:`DocxExtractionError` for a missing file, a non-ZIP file, a
    package without ``word/document.xml``, malformed XML, an encrypted/legacy
    OLE file, an archive-bomb, an over-large part or document, and a document
    that holds no text.
    """
    return DocxReader(
        path,
        max_chars=max_chars,
        max_part_bytes=max_part_bytes,
        max_entries=max_entries,
        max_entry_bytes=max_entry_bytes,
        max_total_bytes=max_total_bytes,
    ).document()


def iter_blocks(path: str | Path, **kwargs: Any) -> Iterator[DocxBlock]:
    """Yield one :class:`DocxBlock` at a time, for a caller that wants to stream."""
    yield from DocxReader(path, **kwargs).blocks()


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_ENTRY_BYTES",
    "DEFAULT_MAX_PART_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DocxBlock",
    "DocxDocument",
    "DocxExtractionError",
    "DocxReader",
    "extract_docx",
    "iter_blocks",
]
