"""The image perception interface (section 5).

An image can be read two ways, and the project keeps them strictly apart:

* **Deterministic perception** -- the image's own measurable features (size,
  orientation, brightness) via :mod:`core.vision`'s dependency-free
  :class:`core.vision.ImageBuffer` / :func:`core.vision.extract_features`, and its
  *text* via OCR (see :mod:`perception.ocr`). These are facts about the picture.
* **Model interpretation** -- a description a local vision model generates
  (Doc's Studio's Qwen3-VL). This is a *model's reading*, not ground truth, so it
  is labelled ``content_type=MODEL`` and its confidence is bounded well below 1.0.

:func:`perceive_image` is the deterministic entry point: it always returns the
image's measurable features and, when OCR is installed, its text. The model
description is a *separate* call, :func:`describe_image_with_model`, so a caller
never accidentally stores a hallucinated description as if it were read text.

Nothing here loads a model unless the caller explicitly asks for a description,
and nothing requires a cloud vision API.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from perception.detector import file_digest
from perception.limits import PerceptionLimits
from perception.metadata import extract_metadata
from perception.ocr import ocr_available, perceive_image_ocr
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionResult,
    PerceptionStatus,
    PerceptionWarning,
    Segment,
    SourceType,
)

#: The confidence assigned to a model-generated description. Deliberately low and
#: never 1.0: a description is an interpretation, and the project's rule is that a
#: generated description is not ground truth.
MODEL_DESCRIPTION_CONFIDENCE = 0.4


def image_features(path: str | Path) -> dict[str, object]:
    """The image's measurable features via :mod:`core.vision`, or ``{}`` when unreadable.

    Reads a PPM image with the project's own dependency-free decoder. A PNG/JPEG
    is not decoded here (the project's ``ImageBuffer`` is PPM-only on purpose, to
    stay dependency-free); its *dimensions* are still read by
    :mod:`perception.metadata` from the header, and OCR can read it through
    Pillow when the OCR extra is present.
    """
    source = Path(path)
    if source.suffix.lower() != ".ppm":
        return {}
    try:
        from core.vision import ImageBuffer, extract_features

        buffer = ImageBuffer.read_ppm(source)
        features = extract_features(buffer)
        return {
            "width": features.width,
            "height": features.height,
            "aspect_ratio": features.aspect_ratio,
            "brightness": features.brightness,
            "saturation": features.saturation,
            "edge_density": features.edge_density,
            "is_landscape": features.is_landscape,
            "is_portrait": features.is_portrait,
        }
    except Exception:
        return {}


def perceive_image(
    path: str | Path,
    *,
    limits: PerceptionLimits | None = None,
    languages: tuple[str, ...] = ("eng", "ara"),
    ocr: bool | None = None,
) -> PerceptionResult:
    """Perceive a plain image: its measurable features, plus its text when OCR runs.

    ``ocr`` selects the OCR policy (``True``/``False``/``None`` = auto when
    installed). The result's ``content_type`` is ``OCR`` when text was read (so it
    is never mistaken for a native text layer), and ``NONE`` when the image carries
    only features. A model *description* is never produced here -- see
    :func:`describe_image_with_model` -- so this result is always deterministic.
    """
    source = Path(path).expanduser()
    settings = limits or PerceptionLimits()
    if not source.exists() or not source.is_file():
        return PerceptionResult(
            source_type=SourceType.IMAGE,
            source_reference=str(source),
            extraction_method=ExtractionMethod.IMAGE,
            status=PerceptionStatus.FAILED,
            errors=("the file does not exist.",),
        )

    features = image_features(source)
    metadata = extract_metadata(source)
    digest = ""
    try:
        digest = file_digest(source)
    except OSError:
        pass

    use_ocr = ocr if ocr is not None else ocr_available()
    segments: list[Segment] = []
    warnings: list[PerceptionWarning] = []
    content = ""
    content_type = ContentType.NONE
    confidence = 0.9

    if use_ocr and ocr_available():
        ocr_result = perceive_image_ocr(
            source, limits=settings, languages=languages
        )
        if ocr_result.ok and ocr_result.content.strip():
            content = ocr_result.content
            content_type = ContentType.OCR
            confidence = ocr_result.confidence
            segments.append(
                Segment(
                    index=1,
                    text=content,
                    label="image text (OCR)",
                    content_type=ContentType.OCR,
                    confidence=ocr_result.confidence,
                )
            )
        elif ocr_result.status is PerceptionStatus.FAILED:
            warnings.extend(ocr_result.warnings)
            return replace(
                ocr_result,
                metadata={**metadata, **features},
                provenance={**ocr_result.provenance, "source_digest": digest},
            )
        else:
            warnings.append(PerceptionWarning("OCR found no readable text in this image."))
    elif ocr is True and not ocr_available():
        from perception.ocr import ocr_install_hint

        warnings.append(
            PerceptionWarning("OCR was requested but is not installed. " + ocr_install_hint())
        )

    from core.voices import detect_language

    language = detect_language(content) if content.strip() else ""
    has_anything = bool(content.strip()) or bool(features) or bool(metadata)
    status = PerceptionStatus.OK if has_anything else PerceptionStatus.EMPTY
    if content.strip() and warnings:
        status = PerceptionStatus.PARTIAL

    return PerceptionResult(
        source_type=SourceType.IMAGE,
        source_reference=str(source),
        extraction_method=ExtractionMethod.IMAGE if not content else ExtractionMethod.OCR,
        content=content,
        content_type=content_type,
        language=language,
        segments=tuple(segments),
        confidence=confidence if content.strip() else 0.5,
        status=status,
        warnings=tuple(warnings),
        provenance={"source_digest": digest},
        metadata={**metadata, **features},
    )


def describe_image_with_model(
    path: str | Path,
    *,
    question: str = "Describe this image in one or two sentences.",
    limits: PerceptionLimits | None = None,
    runner: object | None = None,
) -> PerceptionResult:
    """Ask the local vision model to describe an image (a *model reading*, not truth).

    The result's ``content_type`` is ``MODEL`` and its confidence is
    :data:`MODEL_DESCRIPTION_CONFIDENCE`, so a downstream layer never stores a
    description as a verified fact. When no vision model is installed the result is
    ``FAILED`` with the real next step, never a fabricated description.

    ``runner`` is the injection seam: a test passes its own runner so the whole
    path is exercised with no model and no weights.
    """
    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        return PerceptionResult(
            source_type=SourceType.IMAGE,
            source_reference=str(source),
            extraction_method=ExtractionMethod.MODEL,
            content_type=ContentType.MODEL,
            status=PerceptionStatus.FAILED,
            errors=("the file does not exist.",),
        )

    from core.docs_model import docs_model_status

    status = docs_model_status()
    if not status.available and runner is None:
        return PerceptionResult(
            source_type=SourceType.IMAGE,
            source_reference=str(source),
            extraction_method=ExtractionMethod.MODEL,
            content_type=ContentType.MODEL,
            status=PerceptionStatus.FAILED,
            errors=("no local vision model is installed, so the image cannot be described.",),
            warnings=(),
            metadata={"hint": status.hint or "python -m cli docs download --approve"},
        )

    try:
        description = _run_vision(source, question, runner=runner)
    except Exception as error:
        return PerceptionResult(
            source_type=SourceType.IMAGE,
            source_reference=str(source),
            extraction_method=ExtractionMethod.MODEL,
            content_type=ContentType.MODEL,
            status=PerceptionStatus.FAILED,
            errors=(f"the vision model could not describe the image: {error}",),
        )

    from core.voices import detect_language

    description = description.strip()
    return PerceptionResult(
        source_type=SourceType.IMAGE,
        source_reference=str(source),
        extraction_method=ExtractionMethod.MODEL,
        content=description,
        content_type=ContentType.MODEL,
        language=detect_language(description) if description else "",
        segments=(
            Segment(
                index=1,
                text=description,
                label="model description",
                content_type=ContentType.MODEL,
                confidence=MODEL_DESCRIPTION_CONFIDENCE,
            ),
        ),
        confidence=MODEL_DESCRIPTION_CONFIDENCE,
        status=PerceptionStatus.OK if description else PerceptionStatus.EMPTY,
        model_name=status.model,
        model_version=status.backend,
        provenance={"model": status.model, "backend": status.backend},
        metadata={"question": question, "description": True},
    )


def _run_vision(source: Path, question: str, *, runner: object | None) -> str:
    """Run the local vision model over one image, returning its description text."""
    from core.docs_model import LlamaCppDocsClient

    client = LlamaCppDocsClient(runner=runner)  # type: ignore[arg-type]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _file_uri(source)}},
            ],
        }
    ]
    return client.chat(messages, temperature=0.2)


def _file_uri(path: Path) -> str:
    """A ``file://`` URI that ``urllib`` opens on every platform (incl. Windows)."""
    from core.docs_studio import file_uri

    return file_uri(path)


__all__ = [
    "MODEL_DESCRIPTION_CONFIDENCE",
    "describe_image_with_model",
    "image_features",
    "perceive_image",
]
