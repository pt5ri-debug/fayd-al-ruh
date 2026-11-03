"""The PDF extraction engine (sections 4 and 5).

A PDF is not one thing: it may carry a **text layer** (a digital PDF), be a
**scan** (page images with no text), or be **mixed** (some pages of each). The
engine's job is to read each page with the right method and to *say which method
it used*, so a scanned page's OCR text is never mistaken for a native text layer.

It is a thin layer over the project's existing readers -- :mod:`core.pdf_extract`
for the native text layer and :mod:`core.pdf_ocr` for the per-page OCR path -- and
adds the classification and the normalized :class:`~perception.schema.PerceptionResult`.

Design rules, matching the rest of the project:

* **OCR only where needed.** A page with real text is never sent to OCR and never
  overwritten; only a low-text page is. This is :func:`core.pdf_ocr.extract_pdf_pages`'s
  own rule, reused.
* **No silent omission.** A page that produced nothing (a blank page, a scan with
  no OCR installed) is recorded as a segment with an honest status, and a page
  that failed is named in the warnings -- never dropped.
* **Bounded.** The page count and the character total are capped by
  :class:`perception.limits.PerceptionLimits`, and cancellation is polled between
  pages.
* **No execution.** ``pypdf``/``pypdfium2`` only parse the file; embedded
  JavaScript and launch actions are never run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from perception.detector import file_digest
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

if TYPE_CHECKING:  # the OCR extra is optional, so this is a type-only import
    from core.pdf_ocr import OcrConfig

#: Below this many characters a page is treated as a scan (no usable text layer),
#: matching :data:`core.pdf_ocr.OcrConfig.min_chars_per_page` and Doc's Studio.
MIN_CHARS_PER_PAGE = 16


class PdfPerceptionError(ValueError):
    """A PDF could not be perceived (missing extra, unreadable file)."""


def classify_pdf(path: str | Path, *, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> str:
    """Classify a PDF as ``"text"``, ``"scanned"``, or ``"mixed"`` (or ``"empty"``).

    A *text* PDF has a real text layer on every page; a *scanned* PDF has none on
    any page; a *mixed* PDF has both. This is what tells a caller whether OCR is
    needed at all, before any expensive rasterisation runs. Uses the project's own
    page reader, so it agrees with the extractor about what counts as text.
    """
    from core.pdf_extract import PdfExtractionError, PdfPageStream

    source = Path(path)
    text_pages = 0
    empty_pages = 0
    try:
        for page in PdfPageStream(source):
            if len(page.text.strip()) >= min_chars_per_page:
                text_pages += 1
            else:
                empty_pages += 1
    except PdfExtractionError as error:
        if "no extractable text" not in str(error):
            raise PdfPerceptionError(str(error)) from error
        # A fully scanned PDF: no page yielded text, but the count is known.
        empty_pages = 1
    if text_pages and empty_pages:
        return "mixed"
    if text_pages:
        return "text"
    if empty_pages:
        return "scanned"
    return "empty"


def _usable_languages(config: object) -> tuple[str, ...]:
    """The requested OCR languages the engine actually has (its own resolver)."""
    from core.pdf_ocr import usable_languages

    languages = getattr(config, "languages", ("eng",))
    try:
        return usable_languages(languages)
    except Exception:
        return ()


def perceive_pdf(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    ocr: bool | None = None,
    languages: tuple[str, ...] = ("eng", "ara"),
    token: CancelToken | None = None,
    dpi: int = 200,
    normalize: bool = False,
) -> PerceptionResult:
    """Read a PDF into a normalized perception result, page by page.

    ``ocr`` selects the OCR policy: ``True`` forces the OCR path on, ``False``
    refuses it (native text only), and ``None`` (the default) enables it
    automatically when the OCR engine is installed -- the same
    auto-when-available rule the rest of the project uses. A scanned page is
    OCR'd; a page with real text is left exactly as it was read.

    Raises :class:`PdfPerceptionError` when the PDF cannot be read at all; a page
    that fails is recorded in the result's warnings, never dropped.
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    if not source.exists() or not source.is_file():
        raise PdfPerceptionError(f"No such file: {source}")

    from core.pdf_extract import pdf_available

    if not pdf_available():
        return PerceptionResult(
            source_type=SourceType.PDF,
            source_reference=str(source),
            extraction_method=ExtractionMethod.NONE,
            status=PerceptionStatus.FAILED,
            errors=(
                "Reading a PDF needs the optional [pdf] extra. Install it with: "
                "python -m pip install -r requirements-pdf.txt",
            ),
        )

    from core.pdf_ocr import OcrConfig, ocr_available

    use_ocr = ocr if ocr is not None else ocr_available()
    config = OcrConfig(
        enabled=bool(use_ocr),
        languages=languages,
        dpi=dpi,
        max_pages=settings.max_pages,
    )

    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass

    kind = classify_pdf(source)
    result = _read_pdf_pages(source, config, settings=settings, token=token, kind=kind)
    metadata = extract_metadata(source)
    metadata["pdf_kind"] = kind
    result = replace(result, metadata={**result.metadata, **metadata})

    if normalize and result.content:
        from perception.normalize import normalize_content

        text, report = normalize_content(result.content)
        result = replace(
            result,
            content=text,
            language=report.language,
            metadata={**result.metadata, "normalised": report.changed},
        )
    if digest:
        result = replace(result, provenance={**result.provenance, "source_digest": digest})
    return result


def _read_pdf_pages(
    source: Path,
    config: OcrConfig,
    *,
    settings: PerceptionLimits,
    token: CancelToken | None,
    kind: str,
) -> PerceptionResult:
    """Read the pages of a PDF, OCR'ing only the low-text ones, and build the result."""
    from core.pdf_extract import PdfExtractionError
    from core.pdf_ocr import OcrPdfResult, extract_pdf_pages

    warnings: list[PerceptionWarning] = []
    errors: list[str] = []
    segments: list[Segment] = []
    ocr_used = False
    languages_used: tuple[str, ...] = ()

    try:
        ocr_result: OcrPdfResult = extract_pdf_pages(source, config=config)
    except PdfExtractionError as error:
        # A PDF with no text layer and OCR refused: report it as scanned, not broken.
        if "no extractable text" in str(error) and kind in ("scanned", "mixed"):
            return PerceptionResult(
                source_type=SourceType.SCANNED,
                source_reference=str(source),
                extraction_method=ExtractionMethod.PDF_OCR,
                status=PerceptionStatus.EMPTY,
                errors=(
                    f"{source.name!r} is a scanned PDF with no text layer and OCR is "
                    "not enabled. Enable OCR (install the [ocr] extra) to read it.",
                ),
                metadata={"pdf_kind": kind},
            )
        raise PdfPerceptionError(str(error)) from error

    ocr_pages = set(ocr_result.ocr_pages)
    languages_used = tuple(ocr_result.languages_used)
    truncated = False

    for page in ocr_result.pages:
        if cancelled(token):
            warnings.append(PerceptionWarning("cancelled before every page was read"))
            truncated = True
            break
        settings.check_timeout(None)
        text = page.text or ""
        content_type = ContentType.OCR if page.number in ocr_pages else ContentType.NATIVE
        if page.number in ocr_pages:
            ocr_used = True
        segments.append(
            Segment(
                index=page.number,
                text=text,
                label=f"page {page.number}",
                content_type=content_type,
                confidence=0.75 if page.number in ocr_pages else 1.0,
            )
        )

    for failure in ocr_result.failures:
        warnings.append(
            PerceptionWarning(f"page could not be read: {failure.reason}", f"page {failure.number}")
        )
    for number in ocr_result.skipped_pages:
        warnings.append(
            PerceptionWarning("page was not OCR'd (a cap or time budget was hit)", f"page {number}")
        )

    content = "\n\n".join(segment.text for segment in segments if segment.has_text)
    if len(content) > settings.max_content_chars:
        content = content[: settings.max_content_chars]
        warnings.append(PerceptionWarning("content capped at the character limit"))
        truncated = True

    has_text = bool(content.strip())
    if not has_text:
        status = PerceptionStatus.EMPTY
    elif warnings or truncated:
        status = PerceptionStatus.PARTIAL
    else:
        status = PerceptionStatus.OK

    content_type = ContentType.OCR if ocr_used else ContentType.NATIVE
    method = ExtractionMethod.PDF_OCR if ocr_used else ExtractionMethod.PDF_TEXT
    confidence = _pdf_confidence(ocr_pages, segments)
    language = ""
    if content.strip():
        from core.voices import detect_language

        language = detect_language(content)

    source_type = SourceType.SCANNED if kind == "scanned" else SourceType.PDF
    return PerceptionResult(
        source_type=source_type,
        source_reference=str(source),
        extraction_method=method,
        content=content,
        content_type=content_type,
        language=language,
        segments=tuple(segments),
        confidence=confidence,
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        provenance={"pdf_kind": kind, "ocr_languages": list(languages_used)},
        metadata={
            "page_count": len(segments),
            "pdf_kind": kind,
            "ocr_pages": sorted(ocr_pages),
            "native_pages": list(ocr_result.native_pages),
            "truncated": truncated,
        },
    )


def _pdf_confidence(ocr_pages: set[int], segments: list[Segment]) -> float:
    """A quality signal: high for a pure text layer, lower when OCR carried pages."""
    total = len(segments)
    if total == 0:
        return 0.0
    ocr_share = len(ocr_pages) / total
    # A native text layer is near-certain; an OCR'd page is good but not perfect.
    return round(1.0 - 0.3 * ocr_share, 4)


__all__ = [
    "MIN_CHARS_PER_PAGE",
    "PdfPerceptionError",
    "classify_pdf",
    "perceive_pdf",
]
