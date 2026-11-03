"""The OCR engine for images and scanned pages (sections 4 and 5).

OCR is the bridge from a *picture* to text: a scanned page, a photographed
document, a screenshot. It is **optional** -- the Tesseract engine and its Python
packages are a separate extra -- and the engine here reports honestly when it is
missing rather than failing a run.

It reuses the project's existing OCR machinery rather than writing a second one:

* :class:`core.pdf_ocr.TesseractOcrEngine` recognises a *PDF page* (rasterised
  with ``pypdfium2``);
* the same module's raster guards (:func:`core.pdf_ocr.check_raster_limits`) bound
  a page's pixel size before anything is allocated;
* :func:`core.pdf_ocr.ocr_available` / :func:`core.pdf_ocr.missing_ocr_languages`
  answer the availability and language questions.

What this module adds is **image OCR** -- recognising a plain image file, which the
PDF engine does not do -- plus the normalized :class:`~perception.schema.PerceptionResult`
with OCR-specific provenance.

Honesty rules, matching the rest of the project:

* **OCR text is never ground truth.** The result's ``content_type`` is ``OCR`` and
  its confidence is bounded below 1.0, so a downstream layer never treats it as a
  native text layer or a verified fact.
* **A confidence is recorded only where the engine reports one.** Tesseract's
  per-word confidence is averaged when available; otherwise the confidence is the
  engine-level default, never invented.
* **Preprocessing is minimal and documented.** Greyscale conversion and an
  orientation probe are the only preprocessing, both optional and off unless the
  caller asks, because over-processing a clean image can hurt recognition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from perception.detector import file_digest
from perception.limits import CancelToken, PerceptionLimits, PerceptionTimeout, cancelled
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

#: The default OCR languages: English + Arabic, matching the project's own default
#: and the target user (an Arabic-first petroleum-engineering mind).
DEFAULT_LANGUAGES: tuple[str, ...] = ("eng", "ara")

#: A per-page/per-image confidence the engine-level reading is assigned when the
#: engine does not report per-word scores. Deliberately below 1.0: OCR is good,
#: never certain.
DEFAULT_OCR_CONFIDENCE = 0.75


class OcrPerceptionError(ValueError):
    """An image could not be OCR'd (missing engine, unreadable image)."""


def ocr_available() -> bool:
    """Whether OCR is usable here (the packages *and* the Tesseract engine)."""
    from core.pdf_ocr import ocr_available as _available

    return _available()


def ocr_install_hint() -> str:
    """The real next step when OCR is missing (one command the app can run)."""
    from core.pdf_ocr import missing_ocr_python_packages, tesseract_available

    if missing_ocr_python_packages():
        return "Install the OCR packages: python -m cli deps --install ocr"
    if not tesseract_available():
        return (
            "Install the Tesseract engine and its language data: "
            "python -m cli deps --install ocr"
        )
    return "OCR is not usable in this build."


def _languages(requested: tuple[str, ...]) -> tuple[str, ...]:
    """The requested languages Tesseract really has; falls back to what is present.

    A missing Arabic model never stops an English scan from being read -- the
    project's own ``resolve_languages`` rule, reused -- and when *none* are
    available the error names the install step.
    """
    from core.pdf_ocr import OcrLanguageError, usable_languages

    try:
        return usable_languages(requested)
    except OcrLanguageError:
        raise OcrPerceptionError(
            f"None of the requested OCR languages ({', '.join(requested)}) is installed. "
            "python -m cli deps --install ocr"
        ) from None
    except Exception:
        return requested


def perceive_image_ocr(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    preprocess: bool = False,
    token: CancelToken | None = None,
    region: str = "",
) -> PerceptionResult:
    """OCR a plain image file into a normalized perception result.

    ``preprocess=True`` converts the image to greyscale before recognition (a
    documented, reversible step that sometimes helps a photographed page and
    sometimes hurts a clean screenshot, so it is off by default). ``region`` names
    a sub-rectangle (``"x,y,w,h"``) when the caller wants only part of the image.

    Returns a ``FAILED`` result (never raises) when OCR is not installed or the
    image cannot be opened, so a caller has one consistent thing to inspect.
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    if not source.exists() or not source.is_file():
        return _failed(source, "the file does not exist.")

    if not ocr_available():
        return _failed(
            source,
            "OCR is not installed, so an image's text cannot be read. " + ocr_install_hint(),
        )

    try:
        usable = _languages(languages)
    except OcrPerceptionError as error:
        return _failed(source, str(error))

    if cancelled(token):
        return _failed(source, "cancelled before OCR started.")

    deadline = settings.deadline()
    try:
        text, confidence = _recognise_image(source, usable, preprocess=preprocess, region=region)
    except OcrPerceptionError as error:
        return _failed(source, str(error))
    except Exception as error:  # the engine itself failed on this image
        return _failed(source, f"OCR failed on this image: {error}")
    try:
        settings.check_timeout(deadline)
    except PerceptionTimeout as error:
        return _failed(source, str(error))

    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass

    text = text or ""
    has_text = bool(text.strip())
    status = PerceptionStatus.OK if has_text else PerceptionStatus.EMPTY
    warnings: list[PerceptionWarning] = []
    if not has_text:
        warnings.append(PerceptionWarning("OCR found no readable text in this image"))
    from core.voices import detect_language

    language = detect_language(text) if has_text else ""
    metadata = extract_metadata(source)
    return PerceptionResult(
        source_type=SourceType.IMAGE,
        source_reference=str(source),
        extraction_method=ExtractionMethod.OCR,
        content=text.strip(),
        content_type=ContentType.OCR,
        language=language,
        segments=(
            Segment(
                index=1,
                text=text.strip(),
                label="image",
                content_type=ContentType.OCR,
                confidence=confidence,
                region=region,
            ),
        ),
        confidence=confidence,
        status=status,
        warnings=tuple(warnings),
        provenance={
            "source_digest": digest,
            "ocr_languages": list(usable),
            "preprocess": preprocess,
        },
        metadata=metadata,
    )


def _recognise_image(
    source: Path,
    languages: tuple[str, ...],
    *,
    preprocess: bool,
    region: str,
) -> tuple[str, float]:
    """Run Tesseract on an image file; return its text and a confidence signal."""
    import pytesseract  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(source) as image:
        working: Any = image
        if region:
            working = _crop(image, region)
        if preprocess:
            working = working.convert("L")
        lang = "+".join(languages)
        text = pytesseract.image_to_string(working, lang=lang) or ""
        confidence = _image_confidence(pytesseract, working, lang)
    return text, confidence


def _crop(image: Any, region: str) -> Any:
    """Crop an image to an ``"x,y,w,h"`` region, or return it unchanged if malformed."""
    try:
        parts = [int(value) for value in region.split(",")]
        if len(parts) != 4:
            return image
        x, y, width, height = parts
        return image.crop((x, y, x + width, y + height))
    except (ValueError, TypeError):
        return image


def _image_confidence(pytesseract: Any, image: Any, lang: str) -> float:
    """Tesseract's own mean word confidence, or the honest engine default.

    The per-word confidence is averaged when the engine reports it; a mean of
    ``-1`` (Tesseract's "no confidence available" sentinel) falls back to the
    engine-level default rather than being presented as a real score.
    """
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        scores = [int(value) for value in data.get("conf", []) if str(value).lstrip("-").isdigit()]
        scores = [score for score in scores if score >= 0]
        if scores:
            mean = sum(scores) / len(scores) / 100.0
            return round(min(mean, 0.95), 4)
    except Exception:
        pass
    return DEFAULT_OCR_CONFIDENCE


def _failed(source: Path, reason: str) -> PerceptionResult:
    return PerceptionResult(
        source_type=SourceType.IMAGE,
        source_reference=str(source),
        extraction_method=ExtractionMethod.OCR,
        status=PerceptionStatus.FAILED,
        errors=(reason,),
    )


def ocr_engine_version() -> str:
    """The Tesseract version string, or ``""`` when it cannot be read."""
    try:
        import pytesseract  # type: ignore[import-not-found]

        return str(pytesseract.get_tesseract_version())
    except Exception:
        return ""


def language_report() -> dict[str, bool]:
    """Which of the default languages are installed, for an honest status line."""
    from core.pdf_ocr import missing_ocr_languages

    missing = set(missing_ocr_languages(DEFAULT_LANGUAGES))
    return {language: language not in missing for language in DEFAULT_LANGUAGES}


__all__ = [
    "DEFAULT_LANGUAGES",
    "DEFAULT_OCR_CONFIDENCE",
    "OcrPerceptionError",
    "language_report",
    "ocr_available",
    "ocr_engine_version",
    "ocr_install_hint",
    "perceive_image_ocr",
]
