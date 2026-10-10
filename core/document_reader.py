"""Read plain text out of a Word ``.docx`` file, with no third-party dependency.

A ``.docx`` is an Office Open XML package: a ZIP archive whose main document is
``word/document.xml``. The text of the document is the concatenation of its
``<w:t>`` runs, with ``<w:p>`` paragraph boundaries and ``<w:tab>``/``<w:br>``
markers honoured, so the extracted text keeps the paragraph and line structure a
reader expects. The standard library's :mod:`zipfile` and :mod:`xml.etree` are
enough, so the project keeps its no-mandatory-dependency promise.

The reader is deliberately strict about what it accepts: only ``.docx`` (and the
macro-enabled ``.docm``) are supported. The legacy binary ``.doc`` format is a
different, undocumented layout, so it is refused with a clear message rather than
scrambling its bytes into nonsense. A ``.txt`` file is read as-is, since plain
text is already the thing this module produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_WORD_NAMESPACE}}}"
_DOCUMENT_PART = "word/document.xml"

#: Extensions this reader can turn into text.
SUPPORTED_EXTENSIONS = frozenset({".docx", ".docm", ".txt"})
#: Recognised but deliberately unsupported (binary, undocumented layout).
LEGACY_EXTENSIONS = frozenset({".doc", ".rtf"})

#: A cap so a pathological file cannot exhaust memory; generous for real documents.
MAX_TEXT_CHARS = 2_000_000


class DocumentReadError(ValueError):
    """A document could not be read as text (unsupported, missing, or corrupt)."""


@dataclass(frozen=True)
class DocumentText:
    """The plain text of a document, plus a few facts about where it came from."""

    text: str
    paragraphs: tuple[str, ...]
    title: str
    kind: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise DocumentReadError("The document has no readable text.")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """The visible text of one ``<w:p>`` paragraph, with tabs and breaks honoured."""
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _read_docx(path: Path) -> DocumentText:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            try:
                payload = archive.read(_DOCUMENT_PART)
            except KeyError as error:
                raise DocumentReadError(
                    f"{path.name!r} is not a valid Word document (missing {_DOCUMENT_PART})."
                ) from error
    except zipfile.BadZipFile as error:
        raise DocumentReadError(f"{path.name!r} is not a valid Word document.") from error
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise DocumentReadError(f"{path.name!r} has malformed document XML.") from error

    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        text = _paragraph_text(paragraph)
        if text.strip():
            paragraphs.append(text.strip())

    if not paragraphs:
        raise DocumentReadError(
            f"{path.name!r} contains no text (it may be empty, or hold only images or tables)."
        )
    return _assemble(paragraphs, path, "docx")


def _read_text(path: Path) -> DocumentText:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error
    paragraphs = [line.strip() for line in raw.splitlines() if line.strip()]
    if not paragraphs:
        raise DocumentReadError(f"{path.name!r} is empty.")
    return _assemble(paragraphs, path, "text")


def _assemble(paragraphs: list[str], path: Path, kind: str) -> DocumentText:
    text = "\n".join(paragraphs)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        paragraphs = text.split("\n")
    return DocumentText(
        text=text,
        paragraphs=tuple(paragraphs),
        title=path.stem or path.name,
        kind=kind,
    )


def read_document(path: str | Path) -> DocumentText:
    """Read ``path`` into :class:`DocumentText`, or raise :class:`DocumentReadError`.

    ``.docx``/``.docm`` are parsed from their XML; ``.txt`` is read directly.
    Legacy ``.doc``/``.rtf`` are refused with an actionable message.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise DocumentReadError(f"No such file: {source}")
    if not source.is_file():
        raise DocumentReadError(f"Not a file: {source}")
    suffix = source.suffix.lower()
    if suffix in LEGACY_EXTENSIONS:
        raise DocumentReadError(
            f"{source.suffix} is an older Word format this reader cannot parse. "
            "Save it as .docx (Word: File > Save As > Word Document) and try again."
        )
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentReadError(
            f"Unsupported file type {source.suffix or '(none)'}. "
            "Attach a .docx or .txt file."
        )
    if suffix in {".docx", ".docm"}:
        return _read_docx(source)
    return _read_text(source)


__all__ = [
    "DocumentReadError",
    "DocumentText",
    "LEGACY_EXTENSIONS",
    "MAX_TEXT_CHARS",
    "SUPPORTED_EXTENSIONS",
    "read_document",
]
