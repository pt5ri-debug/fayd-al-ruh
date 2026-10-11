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

import codecs
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from core.text_encoding import (
    EncodingDetectionError,
    decode_bytes,
    detect_encoding,
)
from core.text_normalize import normalize_text

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_WORD_NAMESPACE}}}"
_DOCUMENT_PART = "word/document.xml"

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


def _read_pdf(
    path: Path,
    *,
    ocr: OCREngine | None = None,
    allow_ocr: bool = True,
    password: str | None = None,
) -> DocumentText:
    """Extract text from a PDF, optionally falling back to OCR for scans.

    Digital text extraction is delegated to :mod:`core.pdf_extract`, which wraps
    the optional ``pypdf`` dependency and reports encrypted / malformed / empty /
    image-only documents distinctly. When a PDF yields no text it is likely
    scanned images; if an ``ocr`` engine was supplied it is used, otherwise the
    caller is told which optional extra to install. OCR never executes the
    document -- it reads pixels.
    """
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
    """Fall back to OCR for a PDF with no text layer, or explain how to enable it."""
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
    """The optional PyTesseract OCR engine, or ``None`` when it is not installed.

    Imported lazily so the base install stays dependency-light. The PDF is
    rasterised in memory by ``pypdfium2`` (no external poppler binary); both
    packages live in the ``ocr`` extra.
    """
    try:
        import pypdfium2  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
    except ImportError:
        return None

    def _run(path: Path) -> str:
        pdf = pypdfium2.PdfDocument(str(path))
        try:
            parts: list[str] = []
            for page in pdf:
                bitmap = page.render(scale=2.0)
                image = bitmap.to_pil()
                try:
                    parts.append(pytesseract.image_to_string(image))
                finally:
                    image.close()
            return "\n".join(parts)
        finally:
            pdf.close()

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
    allow_ocr: bool = False,
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
    its content hash. ``ocr`` supplies an OCR engine for scanned PDFs; when it is
    omitted the optional PyTesseract engine is used if installed.
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
        document = _read_pdf(source, ocr=ocr, allow_ocr=allow_ocr, password=password)
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
