"""Read text out of a digital PDF, one page at a time, with ``pypdf``.

Stage 2A of file ingestion adds **digitally generated** PDFs. Extraction is
delegated to `pypdf <https://pypi.org/project/pypdf/>`_ -- a maintained,
pure-Python, BSD-licensed library with no compiled dependency -- which the
project already declares as the optional ``[pdf]`` extra. This module wraps it so
the rest of the system never imports ``pypdf`` directly and so its failure modes
are turned into honest, actionable messages.

Design points that matter for ingestion:

* **Page by page, bounded memory.** :class:`PdfPageStream` keeps the file on disk
  and lets ``pypdf`` load a page's content only when it is reached, so pages are
  yielded one at a time. Only one page's text is held while a caller iterates --
  a 500-page PDF is never materialised whole.
* **Page boundaries are preserved.** Every page is a :class:`PdfPage` carrying its
  1-based number, so a downstream connector can cite "page N" and the page split
  survives into chunks.
* **Honest classification.** Encrypted, malformed, empty, and image-only PDFs are
  distinguished, not lumped together. An image-only (scanned) PDF raises with a
  message pointing at the future OCR stage instead of pretending to read it.
* **No execution.** ``pypdf`` only *parses* the document structure. Embedded
  JavaScript, launch actions, and instructions inside the text are never run or
  followed -- the bytes are turned into a string and nothing else.
* **Arabic is preserved as far as the format permits.** When a PDF carries a
  ``ToUnicode`` map (as reportlab and most modern generators do), Arabic decodes
  correctly. PDFs store glyphs in *visual* order, so extracted Arabic is often
  reversed -- a limitation of the PDF format and of any text extractor, not of
  this module. ``rtl_hint`` reports when a page is mostly Arabic so a caller can
  see that the caveat applies.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The optional dependency the ``[pdf]`` extra installs, kept in one place so the
#: absence message and the capability check cannot drift apart.
PDF_DEPENDENCY = "pypdf"
PDF_EXTRA = "fayd-al-ruh[pdf]"

#: Cap on the total text kept from one PDF, mirroring the reader's own cap so a
#: pathological document cannot exhaust memory.
DEFAULT_MAX_CHARS = 2_000_000

#: Pages whose extracted text is at least this fraction Arabic are flagged, since
#: their visual (possibly reversed) order is worth knowing about.
RTL_HINT_THRESHOLD = 0.5


class PdfExtractionError(ValueError):
    """A PDF could not be read as text (missing dependency, encrypted, corrupt, ...)."""


@dataclass(frozen=True)
class PdfPage:
    """The extracted text of one PDF page, with its 1-based page number."""

    number: int
    text: str
    char_count: int = 0
    rtl_hint: bool = False

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("PdfPage number is 1-based and must be >= 1.")


@dataclass(frozen=True)
class PdfDocument:
    """The pages of a PDF plus a few honest facts about how it was read."""

    pages: tuple[PdfPage, ...]
    page_count: int
    title: str
    truncated: bool = False
    """True when the accumulated text hit :data:`DEFAULT_MAX_CHARS`."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Text-free document metadata (title/author/...), when the PDF carries it."""

    @property
    def text(self) -> str:
        """Every page's text joined by a blank line, so page boundaries survive."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def empty_pages(self) -> tuple[int, ...]:
        """The 1-based numbers of pages that produced no text (blank or image-only)."""
        return tuple(page.number for page in self.pages if not page.text.strip())


def pdf_available() -> bool:
    """Whether the optional ``pypdf`` dependency can be imported."""
    return importlib.util.find_spec(PDF_DEPENDENCY) is not None


def _require_pypdf() -> Any:
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the environment
        from core.optional_deps import FEATURE_PDF, pip_install_command

        raise PdfExtractionError(
            f"PDF support needs the optional '{PDF_DEPENDENCY}' package. Install it with "
            f"`pip install {PDF_EXTRA}` or, from a source checkout, "
            f"`{pip_install_command(FEATURE_PDF)}` and try again."
        ) from error
    return pypdf


def _clean_page_text(raw: str) -> str:
    """Tidy one page's extracted text without altering its characters.

    Only whitespace is normalized (trailing spaces, runs of blank lines); letters,
    diacritics, and Arabic text are left exactly as the library produced them.
    """
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = "\n".join(lines).strip()
    # Collapse runs of 3+ newlines to a single blank line, never touching letters.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _is_rtl_hint(text: str) -> bool:
    """Whether the text is mostly Arabic-script (so visual order is a caveat)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for ch in letters if "ARABIC" in unicodedata.name(ch, ""))
    return arabic / len(letters) >= RTL_HINT_THRESHOLD


def _document_metadata(reader: Any) -> dict[str, Any]:
    """Text-free metadata (title/author/producer/...), best effort."""
    try:
        raw = reader.metadata
    except Exception:
        return {}
    if not raw:
        return {}
    meta: dict[str, Any] = {}
    for key in ("title", "author", "subject", "creator", "producer"):
        try:
            value = raw.get(f"/{key.capitalize()}") or raw.get(f"/{key}")
        except Exception:
            continue
        if value:
            meta[key] = str(value)
    return meta


class PdfPageStream:
    """A lazily-iterated view over a digital PDF's pages.

    Opening validates the document up front (dependency present, file exists,
    not encrypted or decryptable, a readable page tree with at least one page), so
    an unusable PDF fails fast with a specific reason. Pages are then yielded one
    at a time; the accumulated text is capped at ``max_chars``. If no page yields
    any text the iteration ends with a :class:`PdfExtractionError` describing the
    document as empty or image-only (a scan).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        password: str | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1.")
        self.path = Path(path).expanduser()
        self.max_chars = max_chars
        self.truncated = False
        self._reader = self._open(password)

    def _open(self, password: str | None) -> Any:
        pypdf = _require_pypdf()
        source = self.path
        if not source.exists():
            raise PdfExtractionError(f"No such file: {source}")
        if not source.is_file():
            raise PdfExtractionError(f"Not a file: {source}")
        try:
            reader = pypdf.PdfReader(str(source))
        except Exception as error:
            raise PdfExtractionError(
                f"{source.name!r} could not be read as a PDF: {error}"
            ) from error
        if reader.is_encrypted:
            unlocked = False
            for candidate in (password or "", ""):
                try:
                    if reader.decrypt(candidate):
                        unlocked = True
                        break
                except Exception:
                    continue
            if not unlocked:
                raise PdfExtractionError(
                    f"{source.name!r} is encrypted (password-protected). Supply the password "
                    "to read it; encrypted PDFs are refused by default."
                )
        try:
            self.page_count = reader.get_num_pages()
        except Exception as error:
            raise PdfExtractionError(
                f"{source.name!r} has an unreadable page tree: {error}"
            ) from error
        if self.page_count < 1:
            raise PdfExtractionError(f"{source.name!r} contains no pages.")
        self.metadata = _document_metadata(reader)
        self.title = self.metadata.get("title") or source.stem or source.name
        return reader

    def __iter__(self) -> Iterator[PdfPage]:
        total = 0
        found_text = False
        for number, page in enumerate(self._reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as error:  # one unreadable page must not sink the document
                raise PdfExtractionError(f"Could not read page {number}: {error}") from error
            text = _clean_page_text(raw)
            if text:
                found_text = True
                total += len(text) + 2  # the blank-line join later
                if total > self.max_chars:
                    remaining = self.max_chars - (total - len(text) - 2)
                    text = text[: max(0, remaining)]
                    self.truncated = True
            yield PdfPage(
                number=number, text=text, char_count=len(text), rtl_hint=_is_rtl_hint(text)
            )
            if self.truncated:
                # Keep page numbering honest for the pages we stopped reading.
                for extra in range(number + 1, self.page_count + 1):
                    yield PdfPage(number=extra, text="", char_count=0)
                break
        if not found_text:
            raise PdfExtractionError(
                f"{self.path.name!r} contains no extractable text. It is likely a scanned "
                "image-only PDF that needs OCR (a later stage), or a blank document -- not a "
                "digital PDF with a text layer."
            )


def extract_pdf(
    path: str | Path,
    *,
    password: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> PdfDocument:
    """Materialise ``path`` into a :class:`PdfDocument` (all pages in memory).

    Prefer :class:`PdfPageStream` for a large PDF; this convenience is for callers
    that want the whole document at once.

    Raises :class:`PdfExtractionError` for a missing dependency, a missing file,
    an encrypted file (unless ``password`` decrypts it), a malformed file, an
    empty document, or a PDF whose pages hold no extractable text (a scan).
    """
    stream = PdfPageStream(path, password=password, max_chars=max_chars)
    pages = tuple(stream)
    return PdfDocument(
        pages=pages,
        page_count=stream.page_count,
        title=stream.title,
        truncated=stream.truncated,
        metadata=stream.metadata,
    )


def iter_pages(
    path: str | Path,
    *,
    password: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Iterator[PdfPage]:
    """Yield one :class:`PdfPage` at a time, for a caller that wants to stream."""
    yield from PdfPageStream(path, password=password, max_chars=max_chars)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "PDF_DEPENDENCY",
    "PDF_EXTRA",
    "RTL_HINT_THRESHOLD",
    "PdfDocument",
    "PdfExtractionError",
    "PdfPage",
    "PdfPageStream",
    "extract_pdf",
    "iter_pages",
    "pdf_available",
]
