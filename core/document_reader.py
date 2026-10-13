"""Read plain text out of documents: text, Markdown, Word, and digital PDF.

The reader is the project's single, dependency-light front door to a file's text:

* ``.txt``/``.md``/``.markdown`` -- read with **safe encoding detection** (a BOM,
  then strict UTF-8, then a short list of legacy encodings), preserving Arabic and
  English; a very large file is streamed line by line under a hard character cap.
* ``.docx``/``.docm`` -- a Word Office Open XML package (a ZIP holding
  ``word/document.xml``), read through :mod:`core.docx_extract`, which owns the
  untrusted-ZIP handling (resource limits, encrypted/legacy detection, malformed
  reporting) and returns paragraphs **and table rows** in document order. The
  standard library's :mod:`zipfile` and :mod:`xml.etree` are enough, so the
  project keeps its no-mandatory-dependency promise.
* ``.pdf`` -- a *digital* PDF, read page by page through :mod:`core.pdf_extract`
  (the optional ``pypdf`` extra), preserving page numbers; a scanned PDF with no
  text layer falls back to the separate optional OCR extra.

The reader is deliberately strict about what it accepts: the legacy binary
``.doc``/``.rtf`` formats are different, undocumented layouts, so they are refused
with a clear message rather than scrambling their bytes into nonsense.
"""

from __future__ import annotations

import codecs
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.text_encoding import (
    EncodingDetectionError,
    decode_bytes,
    detect_encoding,
)
from core.text_normalize import normalize_text

if TYPE_CHECKING:  # the OCR module is optional and imported lazily at runtime
    from core.pdf_ocr import OcrConfig, PageOcrEngine

#: Extensions this reader can turn into text.
SUPPORTED_EXTENSIONS = frozenset({".docx", ".docm", ".txt", ".md", ".markdown"})
#: Recognised but deliberately unsupported (binary, undocumented layout).
LEGACY_EXTENSIONS = frozenset({".doc", ".rtf"})
#: Extracted through an optional dependency (``fayd-al-ruh[pdf]``) when present.
OPTIONAL_EXTENSIONS = frozenset({".pdf"})
#: Scanned PDFs need the separate optional OCR extra (``fayd-al-ruh[ocr]``).
OCR_EXTRA = "fayd-al-ruh[ocr]"

#: A cap so a pathological file cannot exhaust memory; generous for real documents.
MAX_TEXT_CHARS = 2_000_000
#: Plain-text files larger than this are read as a bounded stream, not whole.
STREAM_THRESHOLD_BYTES = 4 * 1024 * 1024
#: Bytes sampled from the head of a large file to detect its encoding.
ENCODING_SAMPLE_BYTES = 64 * 1024

#: An OCR engine reads an image-producing PDF and returns its text ("" when none).
OCREngine = Callable[[Path], str]


class DocumentReadError(ValueError):
    """A document could not be read as text (unsupported, missing, or corrupt)."""


class NormalizationError(ValueError):
    """A requested text normalization could not be applied."""


@dataclass(frozen=True)
class DocumentText:
    """The plain text of a document, plus a few facts about where it came from."""

    text: str
    paragraphs: tuple[str, ...]
    title: str
    kind: str
    truncated: bool = False
    """True when the text hit :data:`MAX_TEXT_CHARS` and was cut off.

    Honest reporting: a caller can tell a complete document from a capped one
    rather than treating silent truncation as the whole file.
    """

    encoding: str = "utf-8"
    """The encoding the bytes were decoded with (``"utf-8"`` for ``.docx`` XML).

    Reported so an import can record *how* a file was read; an explicit decode
    failure raises :class:`DocumentReadError` rather than being hidden here.
    """

    page_numbers: tuple[int, ...] = ()
    """The 1-based page numbers that produced text (PDFs only; empty otherwise).

    Preserved so a citation can point at "page N" and so the page split is not
    lost when the text is chunked.
    """

    scanned: bool = False
    """True when the text came from OCR of a scanned PDF, not a digital text layer."""

    ocr_pages: tuple[int, ...] = ()
    """The 1-based page numbers whose text came from OCR (Stage 3; empty otherwise).

    Kept separate from ``page_numbers`` so OCR-derived text is never mistaken for
    natively extracted PDF text -- a caller can label or debug it.
    """

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise DocumentReadError("The document has no readable text.")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _read_docx(path: Path, *, max_chars: int = MAX_TEXT_CHARS) -> DocumentText:
    """Read a Word ``.docx``/``.docm`` body through :mod:`core.docx_extract`.

    The extractor owns the untrusted-ZIP handling (resource limits, encrypted/
    legacy detection, malformed-XML reporting) and returns blocks in document
    order -- paragraphs and table rows -- so this stays a thin delegator. The
    flattened block text becomes :class:`DocumentText` (tables included, which the
    earlier paragraph-only reader dropped).
    """
    from core.docx_extract import DocxExtractionError, extract_docx

    try:
        document = extract_docx(path, max_chars=max_chars)
    except DocxExtractionError as error:
        raise DocumentReadError(str(error)) from error
    paragraphs = list(document.paragraphs)
    if not paragraphs:
        raise DocumentReadError(
            f"{path.name!r} contains no text (it may be empty, or hold only images or tables)."
        )
    result = _assemble(paragraphs, path, "docx")
    if not document.title or document.title == path.stem:
        return result
    return DocumentText(
        text=result.text,
        paragraphs=result.paragraphs,
        title=document.title,
        kind=result.kind,
        truncated=result.truncated,
        encoding=result.encoding,
    )


def _read_text(path: Path, *, encoding: str | None = None) -> DocumentText:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error
    if size >= STREAM_THRESHOLD_BYTES:
        return _read_text_streamed(path, encoding=encoding)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error
    try:
        decoded = decode_bytes(raw, encoding=encoding)
    except EncodingDetectionError as error:
        raise DocumentReadError(f"Could not decode {path.name!r}: {error}") from error
    paragraphs = [line.strip() for line in decoded.text.splitlines() if line.strip()]
    if not paragraphs:
        raise DocumentReadError(f"{path.name!r} is empty.")
    return _assemble(paragraphs, path, "text", encoding=decoded.encoding)


def _read_text_streamed(path: Path, *, encoding: str | None = None) -> DocumentText:
    """Read a very large plain-text file line by line, capping at ``MAX_TEXT_CHARS``.

    Memory stays bounded: only the kept lines, the accumulated text (itself
    capped), and a running character count are held -- never the whole file plus
    a copy of its text. Once the cap is reached the remainder is not read, so a
    multi-gigabyte log cannot be pulled into memory.

    The encoding is detected from a bounded prefix, then the file is decoded
    *incrementally* with the same codec, so a wrong guess cannot smuggle in
    replacement characters and the whole file is never held as bytes.
    """
    try:
        with path.open("rb") as handle:
            sample = handle.read(ENCODING_SAMPLE_BYTES)
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error
    chosen = encoding
    if chosen is None:
        try:
            chosen = detect_encoding(sample, allow_truncated=True)
        except EncodingDetectionError as error:
            raise DocumentReadError(f"Could not decode {path.name!r}: {error}") from error

    paragraphs: list[str] = []
    total = 0
    truncated = False
    try:
        decoder = codecs.getincrementaldecoder(chosen)(errors="strict")
    except LookupError as error:
        raise DocumentReadError(f"Unknown encoding {chosen!r} for {path.name!r}.") from error
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    line = decoder.decode(raw_line)
                except UnicodeDecodeError as error:
                    raise DocumentReadError(
                        f"Could not decode {path.name!r} as {chosen}: {error}"
                    ) from error
                stripped = line.strip()
                if not stripped:
                    continue
                paragraphs.append(stripped)
                total += len(stripped) + 1  # the newline later joined back
                if total >= MAX_TEXT_CHARS:
                    truncated = True
                    break
    except OSError as error:
        raise DocumentReadError(f"Could not open {path.name!r}: {error}") from error
    if not paragraphs:
        raise DocumentReadError(f"{path.name!r} is empty.")
    return _assemble(paragraphs, path, "text", truncated=truncated, encoding=chosen)


def _auto_ocr_config() -> "OcrConfig | None":
    """An OCR config for scanned pages, or ``None`` when OCR is not installed.

    Returns ``None`` (importing nothing heavy) when the optional ``[ocr]`` extra
    or the Tesseract engine is absent, so the base install keeps its
    dependency-free behaviour and a scanned PDF is reported rather than mis-read.
    When the extra *is* present, this is what makes scanned pages readable by
    default instead of requiring an opt-in flag.

    ``min_chars_per_page=1`` means only a page with **no** text layer is a
    candidate, so a digital page that merely carries short text (a title, a
    number) is left to native extraction; the explicit ``--ocr`` path keeps the
    larger default threshold for genuinely low-text pages.
    """
    from core.pdf_ocr import OcrConfig, ocr_available

    if not ocr_available():
        return None
    return OcrConfig(enabled=True, min_chars_per_page=1)


def _read_pdf(
    path: Path,
    *,
    ocr: OCREngine | None = None,
    allow_ocr: bool = True,
    ocr_config: "OcrConfig | None" = None,
    ocr_engine: "PageOcrEngine | None" = None,
    password: str | None = None,
) -> DocumentText:
    """Extract text from a PDF, OCR-ing the low-text (scanned) pages.

    Digital extraction is delegated to :mod:`core.pdf_extract`. The per-page OCR
    path (Stage 3, ``core.pdf_ocr``) runs when ``ocr_config`` is enabled, an
    ``ocr_engine`` is supplied, or -- with ``allow_ocr`` (the default) -- the
    optional OCR extra is installed. It OCRs only the pages whose native text is
    too thin and keeps ``ocr_pages`` provenance. A whole-file ``ocr`` callable is
    still accepted for compatibility and takes precedence when given.

    A page that already carries real text is never sent to OCR, so valid extracted
    text is never replaced with a lower-quality guess. Because low-text pages are
    OCR'd rather than skipped, a *mixed* PDF (some digital pages, some scanned)
    no longer silently drops its scanned pages. OCR reads pixels; it never
    executes the document.
    """
    from core.pdf_ocr import OcrConfig, extract_pdf_pages

    wants_stage3_ocr = bool(ocr_config and ocr_config.enabled) or ocr_engine is not None
    if ocr_config is None and allow_ocr and ocr is None:
        # Read scanned pages by default when the optional OCR extra is present. This
        # is the conservative config (only pages with no text layer), so it is also
        # what an injected engine gets when the caller supplied no explicit config.
        ocr_config = _auto_ocr_config()
        wants_stage3_ocr = wants_stage3_ocr or ocr_config is not None
    if wants_stage3_ocr:
        config = ocr_config or OcrConfig(enabled=True)
        try:
            result = extract_pdf_pages(path, config=config, engine=ocr_engine, password=password)
        except DocumentReadError:
            raise
        except Exception as error:  # an OCR/dependency failure is actionable, not fatal
            raise DocumentReadError(f"{path.name!r} could not be read: {error}") from error
        if not result.text.strip():
            raise DocumentReadError(
                f"{path.name!r} contains no extractable text"
                + (" even after OCR." if config.enabled else " (it may be a scanned image).")
            )
        paragraphs = [page.text for page in result.pages if page.text.strip()]
        document = _assemble(paragraphs, path, "pdf-ocr" if result.used_ocr else "pdf")
        return DocumentText(
            text=document.text,
            paragraphs=document.paragraphs,
            title=result.title or document.title,
            kind=document.kind,
            truncated=result.truncated,
            encoding="utf-8",
            page_numbers=tuple(page.number for page in result.pages if page.text.strip()),
            scanned=result.used_ocr,
            ocr_pages=result.ocr_pages,
        )

    from core.pdf_extract import PdfExtractionError, PdfPageStream

    try:
        stream = PdfPageStream(path, password=password)
        pages = list(stream)
    except PdfExtractionError as error:
        message = str(error)
        # A PDF with no extractable text may be a scan: offer the OCR path.
        if "no extractable text" not in message:
            raise DocumentReadError(message) from error
        return _pdf_ocr_fallback(path, error, ocr=ocr, allow_ocr=allow_ocr)

    paragraphs = [page.text for page in pages if page.text.strip()]
    if paragraphs:
        document = _assemble(paragraphs, path, "pdf")
        return _with_pdf_facts(document, pages, stream)
    return _pdf_ocr_fallback(path, None, ocr=ocr, allow_ocr=allow_ocr)


def _with_pdf_facts(document: DocumentText, pages: list, stream) -> DocumentText:
    """Attach page numbers and a scanned-PDF flag to an extracted PDF document."""
    return DocumentText(
        text=document.text,
        paragraphs=document.paragraphs,
        title=stream.title or document.title,
        kind=document.kind,
        truncated=document.truncated,
        encoding=document.encoding,
        page_numbers=tuple(page.number for page in pages if page.text.strip()),
        scanned=False,
    )


def _pdf_ocr_fallback(
    path: Path, error: Exception | None, *, ocr: OCREngine | None, allow_ocr: bool
) -> DocumentText:
    """Fall back to whole-file OCR for a PDF with no text layer, or explain how to enable it.

    This is the compatibility path for a caller that supplies its own
    ``ocr`` callable. The modern per-page path lives in :func:`_read_pdf` and
    :mod:`core.pdf_ocr`.
    """
    engine = ocr
    if engine is None and allow_ocr:
        engine = _default_ocr_engine()
    if engine is None:
        if not allow_ocr:
            raise DocumentReadError(
                f"{path.name!r} contains no extractable text (it may be a scanned image). "
                "Re-run with OCR enabled to read scanned PDFs."
            )
        raise DocumentReadError(
            f"{path.name!r} contains no extractable text (it may be a scanned image). "
            f"For scanned PDFs install the optional OCR extra "
            f"(`pip install {OCR_EXTRA}`, which also needs the Tesseract engine)."
        )
    try:
        ocr_text = engine(path)
    except DocumentReadError:
        raise
    except Exception as error:  # an OCR failure is reported, never fatal to a batch
        raise DocumentReadError(f"{path.name!r} could not be OCR'd: {error}") from error
    ocr_paragraphs = [line.strip() for line in (ocr_text or "").splitlines() if line.strip()]
    if not ocr_paragraphs:
        raise DocumentReadError(f"{path.name!r} produced no text even after OCR.")
    document = _assemble(ocr_paragraphs, path, "pdf-ocr")
    return DocumentText(
        text=document.text,
        paragraphs=document.paragraphs,
        title=document.title,
        kind=document.kind,
        truncated=document.truncated,
        encoding=document.encoding,
        page_numbers=(),
        scanned=True,
    )


def _default_ocr_engine() -> OCREngine | None:
    """The optional whole-file OCR engine, or ``None`` when it is not installed.

    Delegates to :mod:`core.pdf_ocr` (the Stage 3 per-page pipeline) so the
    optional packages are imported in exactly one place. Imported lazily so the
    base install stays dependency-light.
    """
    from core.pdf_ocr import OcrConfig, extract_pdf_pages, ocr_available

    if not ocr_available():
        return None

    def _run(path: Path) -> str:
        result = extract_pdf_pages(path, config=OcrConfig(enabled=True))
        return result.text

    return _run


def _assemble(
    paragraphs: list[str],
    path: Path,
    kind: str,
    *,
    truncated: bool = False,
    encoding: str = "utf-8",
) -> DocumentText:
    text = "\n".join(paragraphs)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        paragraphs = text.split("\n")
        truncated = True
    return DocumentText(
        text=text,
        paragraphs=tuple(paragraphs),
        title=path.stem or path.name,
        kind=kind,
        truncated=truncated,
        encoding=encoding,
    )


def read_document(
    path: str | Path,
    *,
    normalize: bool = False,
    fold_arabic: bool = False,
    ocr: OCREngine | None = None,
    allow_ocr: bool = True,
    ocr_config: "OcrConfig | None" = None,
    ocr_engine: "PageOcrEngine | None" = None,
    encoding: str | None = None,
    password: str | None = None,
) -> DocumentText:
    """Read ``path`` into :class:`DocumentText`, or raise :class:`DocumentReadError`.

    ``.docx``/``.docm`` are parsed from their XML; ``.txt``/``.md`` are read with
    safe encoding detection (a BOM, then UTF-8, then a short list of legacy
    encodings, each strict) so a mis-decoded file raises rather than filling with
    replacement characters. ``encoding`` pins the codec when the caller knows it.
    ``.pdf`` is read page by page through :mod:`core.pdf_extract` (the optional
    ``pypdf`` extra); ``password`` decrypts a protected PDF. Legacy ``.doc``/``.rtf``
    are refused with an actionable message.

    ``normalize=True`` applies :func:`core.text_normalize.normalize_text` to the
    extracted text (``fold_arabic=True`` additionally unifies Arabic letter
    forms). Normalization is off by default because it changes the text and hence
    its content hash.

    Scanned PDFs (pages with no text layer) are read automatically when the
    optional OCR extra is installed: ``allow_ocr`` defaults to ``True`` and the
    reader enables the Stage 3 per-page OCR path on its own, so a mixed PDF's
    scanned pages are recovered instead of silently dropped. Pass ``allow_ocr=False``
    to refuse OCR (a scanned page is then reported, never mis-read), an enabled
    :class:`core.pdf_ocr.OcrConfig` (or a ready ``ocr_engine``) to control it
    explicitly, or a whole-file ``ocr`` callable for compatibility. A page with
    real text is never sent to OCR.
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
    if suffix in OPTIONAL_EXTENSIONS:
        document = _read_pdf(
            source,
            ocr=ocr,
            allow_ocr=allow_ocr,
            ocr_config=ocr_config,
            ocr_engine=ocr_engine,
            password=password,
        )
    elif suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentReadError(
            f"Unsupported file type {source.suffix or '(none)'}. "
            "Supported: .txt, .md, .markdown, .docx, .docm, .pdf, .csv, .json."
        )
    elif suffix in {".docx", ".docm"}:
        document = _read_docx(source)
    else:
        document = _read_text(source, encoding=encoding)
    if normalize:
        return _normalized(document, fold_arabic=fold_arabic)
    return document


def _normalized(document: DocumentText, *, fold_arabic: bool) -> DocumentText:
    """Return ``document`` with its text normalized, or raise ``NormalizationError``."""
    try:
        text = normalize_text(document.text, fold_arabic_letters=fold_arabic)
    except Exception as error:  # a normalization failure must be actionable
        raise NormalizationError(f"Could not normalize {document.title!r}: {error}") from error
    if not text.strip():
        raise NormalizationError(
            f"Normalization left {document.title!r} with no text."
        )
    paragraphs = tuple(line for line in text.split("\n") if line.strip())
    return DocumentText(
        text=text,
        paragraphs=paragraphs,
        title=document.title,
        kind=document.kind,
        truncated=document.truncated,
        encoding=document.encoding,
        page_numbers=document.page_numbers,
        scanned=document.scanned,
        ocr_pages=document.ocr_pages,
    )


__all__ = [
    "DocumentReadError",
    "DocumentText",
    "ENCODING_SAMPLE_BYTES",
    "LEGACY_EXTENSIONS",
    "MAX_TEXT_CHARS",
    "NormalizationError",
    "OCREngine",
    "OCR_EXTRA",
    "OPTIONAL_EXTENSIONS",
    "STREAM_THRESHOLD_BYTES",
    "SUPPORTED_EXTENSIONS",
    "read_document",
]
