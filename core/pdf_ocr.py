"""Optional OCR for scanned PDFs (Stage 3), off unless explicitly enabled.

Stage 2A reads a *digital* PDF through :mod:`core.pdf_extract`; a PDF whose pages
hold no text layer (a scan) is reported distinctly, pointing at this stage. OCR
is a **separate, optional processing path**: the base install never imports the
OCR packages, never needs the Tesseract engine, and never runs OCR unless a
caller builds an :class:`OcrConfig` with ``enabled=True``.

Design points, matched to the stage requirements:

* **Detect low-text pages.** :func:`page_needs_ocr` flags a page whose extracted
  text is shorter than :attr:`OcrConfig.min_chars_per_page` (a scanned page, or one
  holding only a page number). Only those pages are sent to OCR; a page that
  already carries real text is never replaced (:func:`extract_pdf_pages`).
* **A maintained engine, honest dependency report.** The default engine is
  :class:`TesseractOcrEngine` -- `pypdfium2 <https://pypi.org/project/pypdfium2/>`_
  to rasterise in memory (no external poppler binary) and `pytesseract
  <https://pypi.org/project/pytesseract/>`_ to drive the maintained Tesseract
  engine. The *system* packages (``tesseract-ocr`` and the language data
  ``tesseract-ocr-eng`` / ``tesseract-ocr-ara``) are documented in
  :data:`OCR_SYSTEM_PACKAGES` and named in the error messages.
* **Arabic + English** when their language models are installed; the requested
  set is intersected with what Tesseract actually has, so a missing model degrades
  to the ones present and is reported rather than crashing.
* **Bounded batches and honest limits.** Pages are processed ``batch_size`` at a
  time; a page is refused *before* rasterising if its pixel dimensions or pixel
  count would exceed :attr:`OcrConfig.max_page_dimension` /
  :attr:`OcrConfig.max_pixels_per_page`; the raster handed to the engine is capped
  at :attr:`OcrConfig.max_temp_bytes` (written to a temp file only after it is
  measured, then removed); and OCR stops once :attr:`OcrConfig.time_budget_seconds`
  elapses, leaving the remaining pages honestly marked as not attempted.
* **Provenance and separate failures.** Every OCR'd page keeps its 1-based number,
  and :class:`OcrPdfResult` reports which pages came from OCR (``ocr_pages``)
  separately from its native text. A per-page failure is recorded as an
  :class:`OcrFailure` and never aborts the file; :attr:`OcrConfig.retries` retries
  a *transient* page failure before giving up.
* **Never executes the document.** The engine reads pixels; embedded JavaScript,
  launch actions, and instructions in the text are never run or followed.
"""

from __future__ import annotations

import importlib.util
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.pdf_extract import (
    DEFAULT_MAX_CHARS,
    PdfExtractionError,
    PdfPage,
    PdfPageStream,
    _is_rtl_hint,
)

#: The optional Python packages the ``[ocr]`` extra installs (import names).
OCR_DEPENDENCIES = ("pypdfium2", "pytesseract", "PIL")

#: How those packages are named for installation (``pip install``).
OCR_PIP_PACKAGES = ("pypdfium2", "pytesseract", "Pillow")

#: The optional extra, named the same way the reader names it.
OCR_EXTRA = "fayd-al-ruh[ocr]"

#: Maintained OCR engine and the system packages that provide it and its data.
OCR_ENGINE_NAME = "Tesseract"
OCR_SYSTEM_PACKAGES: dict[str, str] = {
    "binary": "tesseract-ocr",
    "eng": "tesseract-ocr-eng",
    "ara": "tesseract-ocr-ara",
    "deu": "tesseract-ocr-deu",
    "fra": "tesseract-ocr-fra",
}

#: Default rasterisation resolution (dots per inch). 200 keeps a typical A4 page
#: near 1650x2340 px -- enough for Tesseract without a huge bitmap.
DEFAULT_DPI = 200


class OcrError(ValueError):
    """Base class for OCR problems (unavailable, unsupported language, ...)."""


class OcrUnavailableError(OcrError):
    """The OCR Python packages or the Tesseract engine are not installed."""


class OcrLanguageError(OcrError):
    """None of the requested OCR languages is installed."""


class OcrLimitError(OcrError):
    """One page exceeds a configured OCR resource limit (recorded, not fatal)."""


@dataclass(frozen=True)
class OcrConfig:
    """The knobs for the optional OCR path. OCR is off unless ``enabled`` is True.

    Every limit has a conservative default so a caller has to opt *in* to
    generous ones, and each is validated on construction so a nonsensical config
    fails loudly rather than silently doing nothing.
    """

    enabled: bool = False
    """The master switch. ``False`` (the default) disables OCR entirely."""

    languages: tuple[str, ...] = ("eng",)
    """Requested Tesseract language codes, e.g. ``("ara", "eng")``."""

    dpi: int = DEFAULT_DPI
    """Rasterisation resolution."""

    batch_size: int = 1
    """How many pages are rasterised/recognised before progress is flushed."""

    max_pages: int = 0
    """Cap on pages OCR'd per document; ``0`` means no cap."""

    max_pixels_per_page: int = 20_000_000
    """Refuse a page whose raster would exceed this many pixels (memory guard)."""

    max_page_dimension: int = 20_000
    """Refuse a page whose raster width *or* height would exceed this many pixels."""

    time_budget_seconds: float = 0.0
    """Stop OCR after this many seconds; ``0`` means no time budget."""

    max_temp_bytes: int = 32 * 1024 * 1024
    """Largest raster file written to temp disk for one page."""

    min_chars_per_page: int = 16
    """A page with fewer than this many characters is treated as needing OCR."""

    retries: int = 1
    """Per-page retries for a transient engine failure (``0`` disables retry)."""

    tesseract_cmd: str = ""
    """Override the Tesseract binary path (else found on ``PATH``)."""

    tesseract_config: str = ""
    """Extra Tesseract options, e.g. ``"--psm 3"``."""

    def __post_init__(self) -> None:
        if not self.languages:
            raise ValueError("languages cannot be empty.")
        for language in self.languages:
            if not language.strip():
                raise ValueError("languages cannot contain an empty code.")
        if self.dpi < 30:
            raise ValueError("dpi must be at least 30.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_pages < 0:
            raise ValueError("max_pages cannot be negative.")
        if self.max_pixels_per_page < 1:
            raise ValueError("max_pixels_per_page must be at least 1.")
        if self.max_page_dimension < 1:
            raise ValueError("max_page_dimension must be at least 1.")
        if self.time_budget_seconds < 0:
            raise ValueError("time_budget_seconds cannot be negative.")
        if self.max_temp_bytes < 1024:
            raise ValueError("max_temp_bytes must be at least 1 KiB.")
        if self.min_chars_per_page < 0:
            raise ValueError("min_chars_per_page cannot be negative.")
        if self.retries < 0:
            raise ValueError("retries cannot be negative.")

    def with_languages(self, languages: Iterable[str]) -> "OcrConfig":
        """A copy with a different language set (used after availability check)."""
        from dataclasses import replace

        return replace(self, languages=tuple(dict.fromkeys(languages)))


@dataclass(frozen=True)
class OcrPage:
    """One page's OCR text, with its 1-based number and the language used."""

    number: int
    text: str
    language: str = ""

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("OcrPage number is 1-based and must be >= 1.")


@dataclass(frozen=True)
class OcrFailure:
    """One page that could not be OCR'd, and why (kept separate from successes)."""

    number: int
    reason: str


@dataclass(frozen=True)
class OcrPdfResult:
    """The pages of a PDF after native extraction plus optional OCR, with provenance.

    ``pages`` holds every page's final text (native where it existed, OCR where it
    was needed). ``ocr_pages`` names the pages whose text came from OCR, so a
    caller can label or debug OCR output without mistaking it for native text.
    ``failures`` records OCR problems per page without aborting the file.
    """

    pages: tuple[PdfPage, ...]
    page_count: int
    title: str
    ocr_pages: tuple[int, ...] = ()
    failures: tuple[OcrFailure, ...] = ()
    skipped_pages: tuple[int, ...] = ()
    """Pages left to OCR but not attempted (a cap or the time budget was hit)."""

    languages_used: tuple[str, ...] = ()
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def native_pages(self) -> tuple[int, ...]:
        return tuple(
            page.number for page in self.pages if page.text.strip() and page.number not in self.ocr_pages
        )

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)


# --------------------------------------------------------------------------- #
# Low-text detection
# --------------------------------------------------------------------------- #


def page_needs_ocr(text: str, config: OcrConfig) -> bool:
    """Whether a page's native text is too thin and should be OCR'd.

    A scanned page yields no text at all; a page with only a printed page number
    yields a handful of characters. Both fall below ``min_chars_per_page``. A page
    with real text is never flagged, which is what keeps OCR from replacing valid
    text with a lower-quality guess.
    """
    return len(text.strip()) < config.min_chars_per_page


# --------------------------------------------------------------------------- #
# Engine protocol and the maintained Tesseract engine
# --------------------------------------------------------------------------- #


@runtime_checkable
class PageOcrEngine(Protocol):
    """Recognises the text of one page of a PDF.

    A concrete engine (the default :class:`TesseractOcrEngine`, or a test double)
    returns the page's text, or raises :class:`OcrError` for a page it cannot read.
    """

    def available_languages(self) -> tuple[str, ...]:
        """The language codes the engine can actually use."""

    def recognize_page(
        self,
        path: Path,
        number: int,
        *,
        languages: tuple[str, ...],
        config: OcrConfig,
    ) -> str:
        """Return the recognised text of 1-based page ``number`` of ``path``."""


def ocr_available() -> bool:
    """Whether the OCR Python packages *and* the Tesseract engine are present."""
    if any(importlib.util.find_spec(name) is None for name in OCR_DEPENDENCIES):
        return False
    try:
        import pytesseract  # type: ignore[import-not-found]

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def _require_dependencies() -> tuple[Any, Any]:
    """Import ``pypdfium2`` and ``pytesseract`` or raise an actionable error."""
    try:
        import pypdfium2  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OcrUnavailableError(
            f"OCR needs the optional packages {', '.join(OCR_PIP_PACKAGES)}. Install them with "
            f"`pip install {OCR_EXTRA}`. The {OCR_ENGINE_NAME} engine and its language data "
            f"({OCR_SYSTEM_PACKAGES['binary']}, {OCR_SYSTEM_PACKAGES['eng']}, "
            f"{OCR_SYSTEM_PACKAGES['ara']}) are separate system packages."
        ) from error
    return pypdfium2, pytesseract


def check_raster_limits(width: int, height: int, config: OcrConfig, number: int) -> None:
    """Refuse a page whose raster would exceed the configured memory guards.

    Split out from the engine so the limit logic is testable without the optional
    rasteriser installed. Raises :class:`OcrLimitError` (an explicit refusal, never
    a silent downscale) when the width/height or the pixel count is too large.
    """
    if width > config.max_page_dimension or height > config.max_page_dimension:
        raise OcrLimitError(
            f"page {number} raster {width}x{height} px exceeds the "
            f"{config.max_page_dimension} px dimension limit"
        )
    if width * height > config.max_pixels_per_page:
        raise OcrLimitError(
            f"page {number} raster of {width * height} px exceeds the "
            f"{config.max_pixels_per_page} px limit"
        )


class TesseractOcrEngine:
    """The default OCR engine: rasterise with pypdfium2, read with Tesseract.

    Imported lazily inside each call so the base install never imports the
    optional packages. Only one page's bitmap is live at a time and the temp file
    handed to Tesseract is measured against :attr:`OcrConfig.max_temp_bytes` and
    removed immediately, so OCR's disk and memory use stay bounded.
    """

    def __init__(self, *, tesseract_cmd: str = "", tesseract_config: str = "") -> None:
        self.tesseract_cmd = tesseract_cmd
        self.tesseract_config = tesseract_config

    def _pytesseract(self) -> Any:
        _, pytesseract = _require_dependencies()
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        return pytesseract

    def available_languages(self) -> tuple[str, ...]:
        pytesseract = self._pytesseract()
        try:
            langs = pytesseract.get_languages(config=self.tesseract_config)
        except Exception as error:  # engine installed but unusable
            raise OcrUnavailableError(
                f"Could not query {OCR_ENGINE_NAME} languages: {error}. Is the "
                f"{OCR_SYSTEM_PACKAGES['binary']} package installed?"
            ) from error
        return tuple(str(lang) for lang in langs)

    def recognize_page(
        self,
        path: Path,
        number: int,
        *,
        languages: tuple[str, ...],
        config: OcrConfig,
    ) -> str:
        pypdfium2, pytesseract = _require_dependencies()
        scale = config.dpi / 72.0
        document = pypdfium2.PdfDocument(str(path))
        temp_path: str | None = None
        try:
            page = document[number - 1]
            width_pt, height_pt = page.get_size()
            width = int(width_pt * scale)
            height = int(height_pt * scale)
            check_raster_limits(width, height, config, number)
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            try:
                temp_path = _write_bounded_png(image, config.max_temp_bytes, number)
            finally:
                image.close()
            lang = "+".join(languages)
            return pytesseract.image_to_string(
                temp_path, lang=lang, config=config.tesseract_config
            ) or ""
        finally:
            document.close()
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)


def _write_bounded_png(image: Any, max_bytes: int, number: int) -> str:
    """Save ``image`` as a PNG temp file only if it fits the byte budget.

    The raster is serialised to memory first so its size is known *before* any
    disk is touched; an over-budget page raises :class:`OcrLimitError` instead of
    filling the temp directory.
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    size = buffer.tell()
    if size > max_bytes:
        raise OcrLimitError(
            f"page {number} raster is {size} bytes, over the {max_bytes}-byte temp limit"
        )
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        handle.write(buffer.getvalue())
        return handle.name
    finally:
        handle.close()


# --------------------------------------------------------------------------- #
# Orchestration: native extraction + per-page OCR, bounded and honest
# --------------------------------------------------------------------------- #


def resolve_languages(engine: PageOcrEngine, requested: Sequence[str]) -> tuple[str, ...]:
    """The requested languages the engine actually has, or raise if none remain.

    A requested language that is missing degrades to the ones present; only when
    *none* are available is this an error, so a missing Arabic model never stops
    an English scan from being read.
    """
    available = set(engine.available_languages())
    usable = tuple(lang for lang in requested if lang in available)
    if usable:
        return usable
    missing = ", ".join(requested)
    install = ", ".join(
        OCR_SYSTEM_PACKAGES.get(lang, f"tesseract-ocr-{lang}") for lang in requested
    )
    raise OcrLanguageError(
        f"None of the requested OCR languages ({missing}) is installed. "
        f"Install the language data (e.g. `apt install {install}`) and try again."
    )


def usable_languages(
    requested: Sequence[str], *, tesseract_cmd: str = "", tesseract_config: str = ""
) -> tuple[str, ...]:
    """Which of ``requested`` are installed, using the default Tesseract engine."""
    engine = TesseractOcrEngine(tesseract_cmd=tesseract_cmd, tesseract_config=tesseract_config)
    return resolve_languages(engine, requested)


def _usable_languages(engine: PageOcrEngine, config: OcrConfig) -> tuple[str, ...]:
    """The requested languages the engine actually has, or raise if none remain."""
    return resolve_languages(engine, config.languages)


def _native_pages(stream: PdfPageStream) -> list[PdfPage]:
    """All pages with text where available, tolerating a fully scanned document."""
    try:
        return list(stream)
    except PdfExtractionError as error:
        if "no extractable text" not in str(error):
            raise
        # A fully scanned PDF: no page yielded text, but the count is known.
        return [PdfPage(number=n, text="") for n in range(1, stream.page_count + 1)]


def extract_pdf_pages(
    path: str | Path,
    *,
    config: OcrConfig | None = None,
    engine: PageOcrEngine | None = None,
    password: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> OcrPdfResult:
    """Extract a PDF's pages natively and OCR only the low-text ones when enabled.

    Native extraction runs first (through :mod:`core.pdf_extract`). When
    ``config.enabled`` is False this returns exactly the native pages with no OCR
    and no OCR dependency import. When it is True, each page whose native text is
    below ``min_chars_per_page`` is OCR'd in bounded batches; a page that already
    has real text is never sent to OCR and is never overwritten.

    Per-page OCR failures are collected in :attr:`OcrPdfResult.failures` and never
    abort the document; a page is retried ``config.retries`` times for a transient
    error before it is recorded as failed.
    """
    source = Path(path).expanduser()
    stream = PdfPageStream(source, password=password, max_chars=max_chars)
    pages = _native_pages(stream)
    result_pages: dict[int, PdfPage] = {page.number: page for page in pages}
    ocr_pages: list[int] = []
    failures: list[OcrFailure] = []
    skipped: list[int] = []
    languages_used: tuple[str, ...] = ()

    if config is not None and config.enabled:
        targets = [page.number for page in pages if page_needs_ocr(page.text, config)]
        if config.max_pages:
            skipped.extend(targets[config.max_pages :])
            targets = targets[: config.max_pages]
        if targets:
            engine = engine or TesseractOcrEngine(
                tesseract_cmd=config.tesseract_cmd, tesseract_config=config.tesseract_config
            )
            languages_used = _usable_languages(engine, config)
            effective = config.with_languages(languages_used) if len(
                languages_used
            ) != len(config.languages) else config
            started = time.monotonic()
            for index, batch in enumerate(_batched(targets, effective.batch_size)):
                if index and _budget_exhausted(started, effective):
                    skipped.extend(batch)
                    continue
                for number in batch:
                    text = _ocr_one(engine, source, number, effective, failures)
                    if text is None:
                        continue
                    native = result_pages[number].text
                    merged = f"{native}\n{text}".strip() if native.strip() else text
                    result_pages[number] = PdfPage(
                        number=number,
                        text=merged,
                        char_count=len(merged),
                        rtl_hint=_is_rtl_hint(merged),
                    )
                    ocr_pages.append(number)

    ordered = tuple(result_pages[number] for number in sorted(result_pages))
    return OcrPdfResult(
        pages=ordered,
        page_count=stream.page_count,
        title=stream.title,
        ocr_pages=tuple(ocr_pages),
        failures=tuple(failures),
        skipped_pages=tuple(dict.fromkeys(skipped)),
        languages_used=languages_used,
        truncated=stream.truncated,
        metadata=stream.metadata,
    )


def _ocr_one(
    engine: PageOcrEngine,
    path: Path,
    number: int,
    config: OcrConfig,
    failures: list[OcrFailure],
) -> str | None:
    """OCR one page, retrying a transient failure; record and return ``None`` on give-up."""
    attempt = 0
    while True:
        try:
            text = engine.recognize_page(
                path, number, languages=config.languages, config=config
            )
            return text
        except OcrLimitError as error:
            # A resource-limit refusal is not transient: do not retry it.
            failures.append(OcrFailure(number=number, reason=str(error)))
            return None
        except OcrError as error:
            if attempt >= config.retries:
                failures.append(OcrFailure(number=number, reason=str(error)))
                return None
            attempt += 1
        except Exception as error:  # a broken engine must not sink the document
            if attempt >= config.retries:
                failures.append(
                    OcrFailure(number=number, reason=f"{type(error).__name__}: {error}")
                )
                return None
            attempt += 1


def _batched(items: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _budget_exhausted(started: float, config: OcrConfig) -> bool:
    if not config.time_budget_seconds:
        return False
    return (time.monotonic() - started) >= config.time_budget_seconds


__all__ = [
    "DEFAULT_DPI",
    "OCR_DEPENDENCIES",
    "OCR_ENGINE_NAME",
    "OCR_EXTRA",
    "OCR_PIP_PACKAGES",
    "OCR_SYSTEM_PACKAGES",
    "OcrConfig",
    "OcrError",
    "OcrFailure",
    "OcrLanguageError",
    "OcrLimitError",
    "OcrPage",
    "OcrPdfResult",
    "OcrUnavailableError",
    "PageOcrEngine",
    "TesseractOcrEngine",
    "check_raster_limits",
    "extract_pdf_pages",
    "ocr_available",
    "page_needs_ocr",
    "resolve_languages",
    "usable_languages",
]
