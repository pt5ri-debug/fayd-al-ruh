"""File type detection and integrity validation (sections 3 and 4).

The detector answers two questions before anything expensive runs:

1. **What is this?** -- :func:`detect_kind` maps a path to a :class:`SourceKind`
   (text, Word, PDF, image, audio, or unknown) by its *suffix*, and
   :func:`detect_file` adds the cheap magic-byte sniff that catches a renamed file
   (a ``.txt`` that is really a PDF). Nothing is guessed: an unrecognised suffix is
   ``UNKNOWN`` and an unsupported one is reported, never silently routed.
2. **Is it readable?** -- :func:`validate_integrity` checks that the file exists,
   is a file, is non-empty, and is within the size limit, returning a
   :class:`IntegrityReport` that names the real problem rather than raising a bare
   ``OSError`` somewhere downstream.

The suffix lists are the one place the supported formats are declared, and they
are *derived* from the readers the project already has
(:data:`core.docs_studio.SUPPORTED_SUFFIXES`) plus the audio suffixes the speech
layer knows, so the detector can never disagree with a reader about what it
accepts.

Nothing here opens a file for anything but a bounded header read; no content is
executed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: The audio suffixes the speech layer can be handed (see
#: ``core.audiocpp_speech.VOICE_CLIP_SUFFIXES`` plus the common WAV family). A
#: ``.wav`` is read directly; another format needs a converter (ffmpeg/sox), which
#: the audio pipeline reports honestly when it is absent.
AUDIO_SUFFIXES: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)

#: Image suffixes. Kept in step with Doc's Studio's own image list so the two
#: cannot disagree about what a "plain image" is.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: Formats recognised but deliberately unsupported, refused with a clear next step.
UNSUPPORTED_SUFFIXES: tuple[str, ...] = (".doc", ".rtf")

#: The first bytes of a PDF file, used to catch a renamed/mislabelled document.
_PDF_MAGIC = b"%PDF"

#: PNG magic (the raster the project's own renderers write).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The ZIP local-file header, which is what an Office Open XML file (``.docx``)
#: begins with. A ``.docx`` is a ZIP holding ``word/document.xml``.
_ZIP_MAGIC = b"PK\x03\x04"

#: A RIFF header; ``WAVE`` follows at offset 8.
_RIFF_MAGIC = b"RIFF"

#: A JPEG SOI marker.
_JPEG_MAGIC = b"\xff\xd8\xff"


class SourceKind(str, Enum):
    """The coarse family a source belongs to, used to pick a pipeline."""

    TEXT = "text"
    WORD = "word"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedFile:
    """What the detector concluded about a path."""

    path: Path
    suffix: str
    kind: SourceKind
    media_type: str = ""
    """A best-effort media type (``"application/pdf"``, ``"image/png"``, ...)."""

    sniffed: str = ""
    """The format the magic bytes suggest, when they were read and recognised."""

    def describe(self) -> str:
        parts = [f"{self.path.name}: {self.kind.value}"]
        if self.media_type:
            parts.append(self.media_type)
        if self.sniffed and self.sniffed != self.kind.value:
            parts.append(f"(bytes look like {self.sniffed})")
        return " ".join(parts)


@dataclass(frozen=True)
class IntegrityReport:
    """Whether a source can be read, and why not when it cannot."""

    path: Path
    ok: bool
    reason: str = ""
    size_bytes: int = 0
    digest: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        if self.ok:
            bits = [f"{self.path.name} readable ({self.size_bytes} bytes)"]
            bits.extend(self.warnings)
            return "; ".join(bits)
        return f"{self.path.name} not readable: {self.reason}"


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def _media_type(kind: SourceKind, suffix: str) -> str:
    """A best-effort media type for a kind/suffix pair (informational only)."""
    if kind is SourceKind.PDF:
        return "application/pdf"
    if kind is SourceKind.WORD:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if kind is SourceKind.TEXT:
        if suffix in (".csv", ".tsv"):
            return "text/csv"
        if suffix in (".json", ".jsonl", ".ndjson"):
            return "application/json"
        return "text/plain"
    if kind is SourceKind.IMAGE:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(suffix, "image/*")
    if kind is SourceKind.AUDIO:
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".opus": "audio/opus",
            ".wma": "audio/x-ms-wma",
        }.get(suffix, "audio/*")
    return ""


def detect_kind(path: str | Path) -> SourceKind:
    """The family a path belongs to, by suffix. Unknown suffixes are ``UNKNOWN``.

    The document suffixes are taken from the readers the project already has
    (:data:`core.docs_studio.SUPPORTED_SUFFIXES`), so this can never claim to read
    a format no reader accepts.
    """
    source = Path(path)
    suffix = _suffix(source)
    if suffix in UNSUPPORTED_SUFFIXES:
        return SourceKind.UNSUPPORTED
    if suffix in AUDIO_SUFFIXES:
        return SourceKind.AUDIO
    if suffix in IMAGE_SUFFIXES:
        return SourceKind.IMAGE
    from core.docs_studio import SUPPORTED_SUFFIXES

    if suffix not in SUPPORTED_SUFFIXES:
        return SourceKind.UNKNOWN
    if suffix == ".pdf":
        return SourceKind.PDF
    if suffix in (".docx", ".docm"):
        return SourceKind.WORD
    return SourceKind.TEXT


def _sniff(head: bytes) -> str:
    """Name the format the leading bytes suggest, or ``""`` when unrecognised."""
    if head.startswith(_PDF_MAGIC):
        return SourceKind.PDF.value
    if head.startswith(_PNG_MAGIC):
        return SourceKind.IMAGE.value
    if head.startswith(_JPEG_MAGIC):
        return SourceKind.IMAGE.value
    if head.startswith(_ZIP_MAGIC):
        return SourceKind.WORD.value
    if head.startswith(_RIFF_MAGIC) and len(head) >= 12 and head[8:12] == b"WAVE":
        return SourceKind.AUDIO.value
    return ""


def detect_file(path: str | Path, *, sniff: bool = True, head_bytes: int = 16) -> DetectedFile:
    """Detect a file's kind, media type, and (optionally) what its bytes look like.

    A bounded header read only -- at most ``head_bytes`` -- so a huge file is never
    opened whole just to classify it.
    """
    source = Path(path)
    suffix = _suffix(source)
    kind = detect_kind(source)
    sniffed = ""
    if sniff and source.is_file():
        try:
            with source.open("rb") as handle:
                sniffed = _sniff(handle.read(head_bytes))
        except OSError:
            sniffed = ""
    return DetectedFile(
        path=source,
        suffix=suffix,
        kind=kind,
        media_type=_media_type(kind, suffix),
        sniffed=sniffed,
    )


def file_digest(path: Path, *, algorithm: str = "sha256", chunk_bytes: int = 1 << 20) -> str:
    """A content digest of ``path``, streamed so a large file is never held whole."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_integrity(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    digest: bool = True,
) -> IntegrityReport:
    """Check that ``path`` exists, is a non-empty file, and fits the byte budget.

    Returns a report rather than raising, so a caller can turn a failure into a
    ``FAILED`` perception result with a readable reason. ``max_bytes`` (when given)
    refuses a source that is too large up front instead of reading it and then
    rejecting it. A digest is computed by default so a result can record what
    exact bytes it read.
    """
    source = Path(path)
    if not source.exists():
        return IntegrityReport(source, False, "the file does not exist.")
    if not source.is_file():
        return IntegrityReport(source, False, "the path is not a regular file.")
    try:
        size = source.stat().st_size
    except OSError as error:
        return IntegrityReport(source, False, f"the file could not be inspected: {error}")
    if size == 0:
        return IntegrityReport(source, False, "the file is empty.", size_bytes=0)
    if max_bytes is not None and size > max_bytes:
        return IntegrityReport(
            source,
            False,
            f"the file is {size} bytes, over the {max_bytes}-byte limit.",
            size_bytes=size,
        )
    checksum = ""
    warnings: list[str] = []
    if digest:
        try:
            checksum = file_digest(source)
        except OSError as error:
            warnings.append(f"the digest could not be computed: {error}")
    return IntegrityReport(source, True, size_bytes=size, digest=checksum, warnings=tuple(warnings))


__all__ = [
    "AUDIO_SUFFIXES",
    "IMAGE_SUFFIXES",
    "UNSUPPORTED_SUFFIXES",
    "DetectedFile",
    "IntegrityReport",
    "SourceKind",
    "detect_file",
    "detect_kind",
    "file_digest",
    "validate_integrity",
]
