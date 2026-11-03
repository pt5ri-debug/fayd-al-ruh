"""The document processing pipeline (section 4).

This is the general reader for the text-bearing formats the project supports:
plain text, Markdown, CSV/TSV, JSON/JSONL, Word (``.docx``/``.docm``), and PDF.
It is deliberately a **thin dispatcher** over the readers that already exist --
:mod:`core.document_reader` (text/Word/PDF), :mod:`core.docx_extract`,
:mod:`core.pdf_extract` -- and adds only what perception needs:

* **Structure preservation.** Word paragraphs and headings keep their order and
  level; a document's blocks become segments, so a heading survives into the
  result rather than being flattened into one string.
* **Honest encoding handling.** A mis-decoded text file raises rather than filling
  with replacement characters (the reader's own rule, reused).
* **Incremental/bounded reads.** A large file is read through the reader's
  streaming path (it caps at ``MAX_TEXT_CHARS``) and perception adds its own cap
  and cancellation, so a huge document never exhausts memory.
* **Failure reporting.** A legacy ``.doc``/``.rtf`` is refused with "save as
  .docx"; an unreadable file returns a ``FAILED`` result naming the reason; a
  scanned PDF is routed to :mod:`perception.pdf` so OCR can recover it.

Nothing here executes a file's contents: a document is parsed into text and
nothing else, and any instruction inside it is inert.
"""

from __future__ import annotations

from pathlib import Path

from perception.detector import SourceKind, detect_file, file_digest
from perception.limits import CancelToken, PerceptionLimits, cancelled
from perception.metadata import extract_metadata
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionResult,
    PerceptionStatus,
    PerceptionWarning,
    Segment,
    SourceType,
)

#: The kinds the document pipeline reads. An image or audio file is routed
#: elsewhere; an unsupported/unknown suffix is refused with a clear message.
DOCUMENT_KINDS: tuple[SourceKind, ...] = (
    SourceKind.TEXT,
    SourceKind.WORD,
    SourceKind.PDF,
)

#: Suffixes whose content is *structured* (rows, records) rather than prose. They
#: are read by the ingestion connector's own streaming readers -- the project's
#: one implementation of "CSV/JSON to text" -- rather than reimplemented here.
STRUCTURED_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")


def perceive_document(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    normalize: bool = False,
    fold_arabic: bool = False,
    ocr: bool | None = None,
    languages: tuple[str, ...] = ("eng", "ara"),
    token: CancelToken | None = None,
) -> PerceptionResult:
    """Read any supported document into a normalized perception result.

    Dispatches by the detected kind: a PDF goes through :func:`perception.pdf.perceive_pdf`
    (so a scanned PDF is recovered by OCR), a Word file through the reader's
    paragraph-aware path, and a text file through the safe-encoding reader. An
    unsupported suffix (``.doc``/``.rtf``) and an unknown suffix each return a
    result that says so, rather than raising into an unrelated traceback.
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    detected = detect_file(source)

    if not source.exists() or not source.is_file():
        return _failed(detected, "the file does not exist.")
    if detected.kind is SourceKind.UNSUPPORTED:
        return PerceptionResult(
            source_type=SourceType.DOCUMENT,
            source_reference=str(source),
            extraction_method=ExtractionMethod.NONE,
            status=PerceptionStatus.UNSUPPORTED,
            errors=(
                f"{source.suffix} is a legacy format this build does not read. "
                "Save it as .docx (Word) or .pdf and try again.",
            ),
        )
    if detected.kind not in DOCUMENT_KINDS:
        return PerceptionResult(
            source_type=SourceType.UNKNOWN,
            source_reference=str(source),
            extraction_method=ExtractionMethod.NONE,
            status=PerceptionStatus.UNSUPPORTED,
            errors=(
                f"{source.name!r} is not a document format the perception pipeline reads "
                f"(detected: {detected.kind.value}).",
            ),
        )

    if detected.kind is SourceKind.PDF:
        from perception.pdf import PdfPerceptionError, perceive_pdf

        try:
            result = perceive_pdf(
                source,
                limits=settings,
                ocr=ocr,
                languages=languages,
                token=token,
                normalize=normalize,
            )
        except PdfPerceptionError as error:
            return _failed(detected, str(error))
        return result

    if cancelled(token):
        return _failed(detected, "cancelled before the document was read.")

    if detected.suffix in STRUCTURED_SUFFIXES:
        return _read_structured(
            detected,
            settings=settings,
            normalize=normalize,
            fold_arabic=fold_arabic,
            token=token,
        )

    return _read_text_document(
        detected, settings=settings, normalize=normalize, fold_arabic=fold_arabic
    )


def _read_structured(
    detected,
    *,
    settings: PerceptionLimits,
    normalize: bool,
    fold_arabic: bool,
    token: CancelToken | None,
) -> PerceptionResult:
    """Read a CSV/TSV/JSON/JSONL source through the ingestion connector's streamer.

    The connector's readers are the project's one implementation of "structured
    data to text" (a CSV row becomes ``header: value`` lines, a JSON value is
    rendered recursively), so perception reuses them rather than writing a second
    one. Rows/records become segments, so a citation can point at the row a fact
    came from.
    """
    from ingest.connectors.files import FileConnector
    from ingest.records import SourceSpec

    source: Path = detected.path
    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass

    connector = FileConnector()
    segments: list[Segment] = []
    warnings: list[PerceptionWarning] = []
    truncated = False
    try:
        for item in connector.plan(SourceSpec(str(source))):
            if cancelled(token):
                break
            for record in connector.stream_item(item):
                if cancelled(token):
                    break
                text = record.text.strip()
                if not text:
                    continue
                segments.append(
                    Segment(
                        index=len(segments) + 1,
                        text=text,
                        label=record.title or f"record {len(segments) + 1}",
                        content_type=ContentType.STRUCTURED,
                    )
                )
                if len(segments) >= settings.max_pages:
                    truncated = True
                    warnings.append(
                        PerceptionWarning("record limit reached; the rest of the file was not read")
                    )
                    break
            if truncated:
                break
    except Exception as error:
        return _failed(detected, f"the structured file could not be read: {error}")

    content = "\n\n".join(segment.text for segment in segments)
    if len(content) > settings.max_content_chars:
        content = content[: settings.max_content_chars]
        truncated = True
        warnings.append(PerceptionWarning("content capped at the character limit"))
    if normalize and content:
        from perception.normalize import normalize_content

        content, report = normalize_content(content, fold_arabic=fold_arabic)
    else:
        report = None

    from core.voices import detect_language

    language = detect_language(content) if content.strip() else ""
    metadata = extract_metadata(source)
    metadata["records"] = len(segments)
    status = PerceptionStatus.OK if content.strip() else PerceptionStatus.EMPTY
    if content.strip() and warnings:
        status = PerceptionStatus.PARTIAL
    return PerceptionResult(
        source_type=SourceType.SPREADSHEET,
        source_reference=str(source),
        extraction_method=ExtractionMethod.READER,
        content=content,
        content_type=ContentType.STRUCTURED,
        language=language,
        segments=tuple(segments),
        confidence=0.95,
        status=status,
        warnings=tuple(warnings),
        provenance={"source_digest": digest},
        metadata={**metadata, "normalised": bool(report and report.changed), "truncated": truncated},
    )


def _read_text_document(
    detected,
    *,
    settings: PerceptionLimits,
    normalize: bool,
    fold_arabic: bool,
) -> PerceptionResult:
    """Read a text or Word document through the project's reader, block by block."""
    from core.document_reader import DocumentReadError, read_document

    source: Path = detected.path
    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass

    try:
        document = read_document(source, normalize=False)
    except DocumentReadError as error:
        return _failed(detected, str(error))
    except Exception as error:  # a reader bug on a hostile file: report, do not crash
        return _failed(detected, f"the document could not be read: {error}")

    segments: list[Segment] = []
    warnings: list[PerceptionWarning] = []
    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.strip()
        if not text:
            continue
        segments.append(
            Segment(
                index=len(segments) + 1,
                text=text,
                label=_block_label(detected.kind, index),
                content_type=ContentType.NATIVE,
            )
        )

    content = document.text
    if normalize:
        from perception.normalize import normalize_content

        content, report = normalize_content(content, fold_arabic=fold_arabic)
    else:
        report = None

    truncated = document.truncated
    if len(content) > settings.max_content_chars:
        content = content[: settings.max_content_chars]
        truncated = True
        warnings.append(PerceptionWarning("content capped at the character limit"))
    if document.truncated:
        warnings.append(
            PerceptionWarning("the reader capped the text; the document is larger than read")
        )

    from core.voices import detect_language

    language = detect_language(content)
    metadata = extract_metadata(source)
    metadata["encoding"] = document.encoding
    metadata["kind"] = document.kind
    if document.page_numbers:
        metadata["page_numbers"] = list(document.page_numbers)

    if not content.strip():
        status = PerceptionStatus.EMPTY
    elif warnings:
        status = PerceptionStatus.PARTIAL
    else:
        status = PerceptionStatus.OK

    source_type = SourceType.WORD if detected.kind is SourceKind.WORD else SourceType.DOCUMENT
    method = ExtractionMethod.DOCX if detected.kind is SourceKind.WORD else ExtractionMethod.READER
    if document.ocr_pages:
        method = ExtractionMethod.PDF_OCR
    return PerceptionResult(
        source_type=source_type,
        source_reference=str(source),
        extraction_method=method,
        content=content,
        content_type=ContentType.OCR if document.ocr_pages else ContentType.NATIVE,
        language=language,
        segments=tuple(segments),
        confidence=0.9 if document.ocr_pages else 1.0,
        status=status,
        warnings=tuple(warnings),
        provenance={"source_digest": digest},
        metadata={**metadata, "normalised": bool(report and report.changed), "truncated": truncated},
    )


def _block_label(kind: SourceKind, index: int) -> str:
    return f"paragraph {index}" if kind is SourceKind.WORD else f"block {index}"


def _failed(detected, reason: str) -> PerceptionResult:
    source: Path = detected.path
    return PerceptionResult(
        source_type=SourceType.DOCUMENT,
        source_reference=str(source),
        extraction_method=ExtractionMethod.NONE,
        status=PerceptionStatus.FAILED,
        errors=(reason,),
    )


def read_text_with_structure(path: str | Path) -> tuple[list[tuple[str, str]], str]:
    """Read a Word file as ordered ``(label, text)`` blocks, reusing the extractor.

    A convenience for a caller that wants the structure (a heading, then its
    paragraph) without building a whole result -- e.g. the Presentation Studio's
    outline import. Returns the blocks and the document's encoding name. A
    non-Word file raises :class:`ValueError` rather than guessing.
    """
    source = Path(path).expanduser()
    if source.suffix.lower() not in (".docx", ".docm"):
        raise ValueError(f"{source.name} is not a Word document.")
    from core.docx_extract import extract_docx

    document = extract_docx(source)
    blocks: list[tuple[str, str]] = []
    for block in document.blocks:
        label = "heading" if getattr(block, "level", 0) else "paragraph"
        blocks.append((label, block.text))
    return blocks, "utf-8"


__all__ = [
    "DOCUMENT_KINDS",
    "perceive_document",
    "read_text_with_structure",
]
