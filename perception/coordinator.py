"""The perception coordinator (section 3).

The coordinator is the one public entry point: give it a path (or a bare string)
and it routes the input, runs the right pipeline, validates the result, attaches
provenance, and returns a :class:`perception.schema.PerceptionResult`. Every
surface uses it, so the CLI, a window, and the cognitive bridge perceive an input
identically.

It is deliberately a **coordinator**, not a second implementation: each pipeline is
a module it calls (:mod:`perception.documents`, :mod:`perception.pdf`,
:mod:`perception.ocr`, :mod:`perception.images`, :mod:`perception.audio`), each of
which in turn reuses the project's own readers and engines.

Two convenience shapes round it out:

* :func:`perceive` -- perceive one path with the default coordinator.
* :meth:`PerceptionCoordinator.perceive_text` -- normalize a bare string into the
  same result shape, so a pasted sentence and a file share one path into the
  cognitive bridge.

Nothing here loads a model unless the caller asks for a model description, and
nothing reaches the network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from perception.detector import validate_integrity
from perception.limits import CancelToken, PerceptionLimits
from perception.provenance import PerceptionProvenanceManager
from perception.router import Pipeline, route_path, route_text
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionResult,
    PerceptionStatus,
    SourceType,
)
from perception.validator import ValidationVerdict, validate_result


class PerceptionCoordinator:
    """Routes inputs to pipelines and returns validated, provenance-carrying results.

    ``limits`` bound every run; ``provenance`` is the ledger wrapper (a read-only
    session can pass ``file_into_ledger=False`` through :meth:`perceive`); ``token``
    is the cooperative cancellation signal, checked between pages/records.
    """

    def __init__(
        self,
        *,
        limits: PerceptionLimits | None = None,
        provenance: PerceptionProvenanceManager | None = None,
        token: CancelToken | None = None,
    ) -> None:
        self.limits = limits or PerceptionLimits()
        self.provenance = provenance or PerceptionProvenanceManager()
        self.token = token

    # -- the public entry points ------------------------------------------- #

    def perceive(
        self,
        path: str | Path,
        *,
        ocr: bool | None = None,
        languages: tuple[str, ...] = ("eng", "ara"),
        normalize: bool = False,
        describe_image: bool = False,
        transcribe: bool = True,
        file_into_ledger: bool = True,
    ) -> PerceptionResult:
        """Perceive one path, routing it to the right pipeline.

        ``ocr`` forces or refuses OCR; ``languages`` selects OCR/ASR languages;
        ``normalize`` applies the deterministic text pass; ``describe_image`` asks
        the local vision model to describe an image (a *model reading*, labelled as
        such); ``transcribe`` runs local speech-to-text on audio (``False`` reads
        metadata only). The result is validated and its provenance attached unless
        ``file_into_ledger`` is false.
        """
        source = Path(path).expanduser()
        route = route_path(source, ocr=ocr)
        if route.pipeline is Pipeline.REFUSED:
            return PerceptionResult(
                source_type=SourceType.UNKNOWN,
                source_reference=str(source),
                extraction_method=ExtractionMethod.NONE,
                status=PerceptionStatus.UNSUPPORTED,
                errors=(route.reason,),
            )

        integrity = validate_integrity(source, max_bytes=self.limits.max_source_bytes)
        if not integrity.ok:
            return PerceptionResult(
                source_type=SourceType.UNKNOWN,
                source_reference=str(source),
                extraction_method=ExtractionMethod.NONE,
                status=PerceptionStatus.FAILED,
                errors=(integrity.reason,),
            )

        result = self._run(route, source, ocr=ocr, languages=languages, normalize=normalize,
                           describe_image=describe_image, transcribe=transcribe)
        return self._finalise(
            result, file_into_ledger=file_into_ledger, source_digest=integrity.digest
        )

    def perceive_text(
        self,
        text: str,
        *,
        source_reference: str = "input",
        normalize: bool = False,
        file_into_ledger: bool = False,
    ) -> PerceptionResult:
        """Normalize a bare string into a perception result (the document path).

        A pasted sentence has no file, so its provenance is the ``source_reference``
        (``"input"`` by default, or a session id). This is what lets the cognitive
        bridge treat text and a file identically.
        """
        route = route_text(text)
        if route.pipeline is Pipeline.REFUSED:
            return PerceptionResult(
                source_type=SourceType.TEXT,
                source_reference=source_reference,
                extraction_method=ExtractionMethod.NONE,
                status=PerceptionStatus.EMPTY,
                errors=(route.reason,),
            )
        content = text
        if normalize:
            from perception.normalize import normalize_content

            content, report = normalize_content(text)
            language = report.language
        else:
            from core.voices import detect_language

            language = detect_language(text)
        result = PerceptionResult(
            source_type=SourceType.TEXT,
            source_reference=source_reference,
            extraction_method=ExtractionMethod.READER,
            content=content.strip(),
            content_type=ContentType.NATIVE,
            language=language,
            confidence=1.0,
            status=PerceptionStatus.OK if content.strip() else PerceptionStatus.EMPTY,
            provenance={"source_digest": _text_digest(text)},
        )
        return self._finalise(result, file_into_ledger=file_into_ledger)

    def validate(self, result: PerceptionResult) -> ValidationVerdict:
        """Validate a result (delegates to the one validator)."""
        return validate_result(result)

    # -- internals --------------------------------------------------------- #

    def _run(
        self,
        route,
        source: Path,
        *,
        ocr: bool | None,
        languages: tuple[str, ...],
        normalize: bool,
        describe_image: bool,
        transcribe: bool,
    ) -> PerceptionResult:
        if route.pipeline is Pipeline.DOCUMENT:
            from perception.documents import perceive_document

            return perceive_document(
                source,
                limits=self.limits,
                normalize=normalize,
                ocr=ocr,
                languages=languages,
                token=self.token,
            )
        if route.pipeline is Pipeline.IMAGE:
            from perception.images import describe_image_with_model, perceive_image

            result = perceive_image(
                source, limits=self.limits, languages=languages, ocr=ocr
            )
            if describe_image and result.ok:
                description = describe_image_with_model(source, limits=self.limits)
                if description.ok:
                    result = _merge_description(result, description)
            return result
        if route.pipeline is Pipeline.AUDIO:
            from perception.audio import perceive_audio

            return perceive_audio(
                source,
                limits=self.limits,
                transcribe=transcribe,
                recognizer=None,
                token=self.token,
            )
        raise AssertionError("unreachable route")  # pragma: no cover

    def _finalise(
        self,
        result: PerceptionResult,
        *,
        file_into_ledger: bool,
        source_digest: str = "",
    ) -> PerceptionResult:
        if source_digest and "source_digest" not in result.provenance:
            result = replace(
                result, provenance={**result.provenance, "source_digest": source_digest}
            )
        try:
            result = self.provenance.attach(result, file_into_ledger=file_into_ledger)
        except Exception:  # a provenance filing failure must not lose the result
            pass
        return result


def _merge_description(result: PerceptionResult, description: PerceptionResult) -> PerceptionResult:
    """Attach a model description to an image result as a *separate* segment.

    The description is added as its own segment with ``content_type=MODEL`` so the
    deterministic content and the model's interpretation are never conflated: a
    consumer can see exactly which part of the result is a model's reading.
    """
    from perception.schema import Segment

    model_segment = Segment(
        index=len(result.segments) + 1,
        text=description.content,
        label="model description",
        content_type=ContentType.MODEL,
        confidence=description.confidence,
    )
    return replace(
        result,
        segments=(*result.segments, model_segment),
        model_name=description.model_name or result.model_name,
        model_version=description.model_version or result.model_version,
        provenance={**result.provenance, "described_by": description.model_name},
    )


def _text_digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_coordinator() -> PerceptionCoordinator:
    """A coordinator with the default limits and a fresh provenance manager."""
    return PerceptionCoordinator()


def perceive(path: str | Path, **options: object) -> PerceptionResult:
    """Perceive one path with a default coordinator (the convenience entry point)."""
    return PerceptionCoordinator().perceive(path, **options)  # type: ignore[arg-type]


__all__ = [
    "PerceptionCoordinator",
    "default_coordinator",
    "perceive",
]
