"""The one normalized shape a perception result takes (Phase 7, section 9).

Every perception component -- a document reader, the PDF engine, OCR, the image
interface, the audio pipeline, speech-to-text -- produces a
:class:`PerceptionResult`. One shape for all of them means a consumer (the
Cognitive Core bridge, Memory, Learning, a window, a CLI) never has to branch on
which reader ran, and a result can be logged, validated, and replayed uniformly.

Two honesty rules are built into the shape itself:

* **Extracted content is kept separate from conclusions.** ``content`` is what was
  *perceived* (a page's text, a transcript, an OCR reading); it is never a claim
  the system inferred. A downstream layer decides what, if anything, the content
  supports. ``content_type`` records *how* it was obtained -- ``ocr``, ``asr``, a
  model ``description`` -- so a generated description is never mistaken for
  ground truth.
* **Nothing is born verified.** A result carries a ``confidence`` and a
  ``status``; ``confidence`` is a quality signal about the *extraction*, never a
  statement of truth, and no result is ever treated as a verified fact.

The record is immutable and dependency-free (standard library only), so it can be
built, serialized, and tested without any perception engine installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class SourceType(str, Enum):
    """What kind of media a perception result came from."""

    TEXT = "text"
    DOCUMENT = "document"
    PDF = "pdf"
    WORD = "word"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    SCANNED = "scanned"
    AUDIO = "audio"
    SPEECH = "speech"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """How the content was obtained -- the provenance of the *text itself*.

    ``NATIVE`` is a document's own text layer; ``OCR`` is read from a picture;
    ``ASR`` is transcribed from audio; ``MODEL`` is a description a language or
    vision model generated. The last is explicitly **not** ground truth: a
    description is a model's interpretation, and the bridge stores it as such.
    """

    NATIVE = "native"
    OCR = "ocr"
    ASR = "asr"
    MODEL = "model"
    STRUCTURED = "structured"
    METADATA = "metadata"
    NONE = "none"


class ExtractionMethod(str, Enum):
    """The engine or reader that produced the content."""

    READER = "reader"
    PDF_TEXT = "pdf-text"
    PDF_OCR = "pdf-ocr"
    DOCX = "docx"
    OCR = "ocr"
    IMAGE = "image"
    AUDIO = "audio"
    SPEECH = "speech"
    MODEL = "model"
    METADATA = "metadata"
    NONE = "none"


class PerceptionStatus(str, Enum):
    """The processing outcome, so a caller can tell success from a partial read."""

    OK = "ok"
    PARTIAL = "partial"
    """Some content was read and some was skipped -- a page that failed, a
    language that was not installed. The result is usable but incomplete, and the
    warnings say what was missed. Nothing is silently dropped."""

    EMPTY = "empty"
    """The source was read but produced no usable content (a blank page, silence)."""

    FAILED = "failed"
    """The source could not be read at all. ``errors`` says why."""

    UNSUPPORTED = "unsupported"
    """The source is a recognised format this build cannot process (a legacy
    ``.doc``), or a media type no component claims. ``errors`` names the next step."""


class PerceptionError(ValueError):
    """A perception operation could not produce a result.

    Raised by a component when it cannot even build a failed :class:`PerceptionResult`
    (a missing source path, a nonsensical limit). A component that *reads* a file
    and finds it unreadable returns a ``FAILED`` result instead, so the caller has
    one consistent thing to inspect.
    """


@dataclass(frozen=True)
class PerceptionWarning:
    """A non-fatal problem recorded during perception.

    A skipped page, a missing OCR language, a truncated read: each is reported
    rather than hidden, and carries where it happened so a reader can find it.
    """

    message: str
    where: str = ""
    """The page number, segment, or region the warning refers to (a string so it
    can name a page, a timestamp, or a field)."""

    def describe(self) -> str:
        return f"{self.where}: {self.message}" if self.where else self.message


@dataclass(frozen=True)
class Segment:
    """A page, region, or time span of the source, with its own content.

    Page boundaries are preserved where the format has them (a PDF page, a Word
    block, an audio time range), so a citation can point at "page 4" or
    "00:01:20" rather than at a whole file.
    """

    index: int
    """The 1-based position of this segment in the source."""

    text: str = ""
    label: str = ""
    """A readable label: ``"page 4"``, ``"00:01:20"``, ``"table row 3"``."""

    content_type: ContentType = ContentType.NATIVE
    start_seconds: float | None = None
    """For an audio/speech segment, its start time; ``None`` otherwise."""

    end_seconds: float | None = None
    confidence: float | None = None
    """A per-segment quality signal (an OCR confidence, a word score) when the
    engine reports one; ``None`` when it does not."""

    region: str = ""
    """A rectangle or named region for an image, e.g. ``"0,0,400,200"``."""

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("Segment index is 1-based and must be >= 1.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Segment confidence must be between 0 and 1.")

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    def describe(self) -> str:
        name = self.label or f"segment {self.index}"
        return f"{name} ({self.content_type.value}): {len(self.text)} chars"


@dataclass(frozen=True)
class PerceptionResult:
    """One normalized perception output: what was read, from where, and how.

    The fields mirror the task's schema section exactly: an id, the source type
    and reference, the extraction method, the extracted content and its type, the
    language, a timestamp, page/region/segment information, confidence, status,
    errors and warnings, model information, and provenance.

    ``provenance`` is a plain mapping so the record stays dependency-free; the
    :mod:`perception.provenance` manager fills it with the source path, digest,
    engine name and version, and any parent artifact ids. ``metadata`` carries
    text-free facts about the source (page count, title, duration).
    """

    result_id: str = field(default_factory=lambda: f"PER-{uuid4().hex[:8]}")
    source_type: SourceType = SourceType.UNKNOWN
    source_reference: str = ""
    """A path, URL, session id, or microphone name -- whatever names the source."""

    extraction_method: ExtractionMethod = ExtractionMethod.NONE
    content: str = ""
    content_type: ContentType = ContentType.NONE
    language: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    segments: tuple[Segment, ...] = ()
    confidence: float = 0.0
    """A quality signal about the extraction in ``[0, 1]``; never a truth claim."""

    status: PerceptionStatus = PerceptionStatus.OK
    errors: tuple[str, ...] = ()
    warnings: tuple[PerceptionWarning, ...] = ()
    model_name: str = ""
    model_version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Perception confidence must be between 0 and 1.")
        if (
            self.status is PerceptionStatus.OK
            and not self.content.strip()
            and not self.segments
            and not self.metadata
        ):
            # An "ok" result with nothing in it is a lie; make the caller say
            # EMPTY instead. A result is "ok" only when it produced *something* --
            # text, segments, or metadata (an image's measurable features, an audio
            # file's duration) -- so every consumer can rely on that.
            raise ValueError(
                "A perception result with status OK must carry content, segments, or "
                "metadata; use PerceptionStatus.EMPTY for a source that read cleanly "
                "but was blank."
            )

    @property
    def ok(self) -> bool:
        return self.status in (PerceptionStatus.OK, PerceptionStatus.PARTIAL)

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip()) or any(segment.has_text for segment in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def page_numbers(self) -> tuple[int, ...]:
        """The 1-based indices of segments that carry text (page numbers for a PDF)."""
        return tuple(segment.index for segment in self.segments if segment.has_text)

    def segment(self, index: int) -> Segment | None:
        for item in self.segments:
            if item.index == index:
                return item
        return None

    def summary(self) -> str:
        """A short, honest one-line description of the result."""
        bits = [self.source_type.value, self.extraction_method.value, self.status.value]
        if self.content.strip():
            bits.append(f"{self.word_count} words")
        if self.segments:
            bits.append(f"{len(self.segments)} segment(s)")
        if self.language:
            bits.append(self.language)
        return f"{self.result_id} — {', '.join(bits)}"

    def describe(self) -> str:
        """A multi-line report: the summary, then any warnings and errors."""
        lines = [self.summary()]
        if self.source_reference:
            lines.append(f"  source: {self.source_reference}")
        for warning in self.warnings:
            lines.append(f"  ! {warning.describe()}")
        for error in self.errors:
            lines.append(f"  x {error}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        """A JSON-ready mapping (the dataclass is already plain data)."""
        return {
            "result_id": self.result_id,
            "source_type": self.source_type.value,
            "source_reference": self.source_reference,
            "extraction_method": self.extraction_method.value,
            "content": self.content,
            "content_type": self.content_type.value,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "segments": [
                {
                    "index": segment.index,
                    "text": segment.text,
                    "label": segment.label,
                    "content_type": segment.content_type.value,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "confidence": segment.confidence,
                    "region": segment.region,
                }
                for segment in self.segments
            ],
            "confidence": self.confidence,
            "status": self.status.value,
            "errors": list(self.errors),
            "warnings": [
                {"message": warning.message, "where": warning.where} for warning in self.warnings
            ],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ContentType",
    "ExtractionMethod",
    "PerceptionError",
    "PerceptionResult",
    "PerceptionStatus",
    "PerceptionWarning",
    "Segment",
    "SourceType",
]
