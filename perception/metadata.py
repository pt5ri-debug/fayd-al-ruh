"""Metadata extraction: text-free facts about a source (section 3).

Metadata is what a result records *about* a source without quoting it: a PDF's
title and page count, a Word document's block count, an image's pixel size, an
audio file's duration and sample rate. Keeping it apart from the content is what
lets a result be logged, indexed, and cited without leaking the text.

Every reader is optional and total: a missing ``pypdf``/``pypdfium2`` (or a
corrupt header) yields an empty mapping rather than an exception, so a metadata
probe never fails an ingest that would otherwise succeed. Nothing here decodes a
whole file: a PDF header is read for the page count, an image header for its
dimensions, and an audio WAV header for its frame count.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

from perception.detector import SourceKind, detect_kind

#: The keys a metadata mapping may carry, documented so a reader and a writer
#: cannot drift on a literal. Extra keys are allowed; these are the standard ones.
KEY_PAGE_COUNT = "page_count"
KEY_TITLE = "title"
KEY_AUTHOR = "author"
KEY_WORD_COUNT = "word_count"
KEY_BLOCK_COUNT = "block_count"
KEY_WIDTH = "width"
KEY_HEIGHT = "height"
KEY_DURATION_SECONDS = "duration_seconds"
KEY_SAMPLE_RATE = "sample_rate"
KEY_CHANNELS = "channels"
KEY_FORMAT = "format"


def _pdf_metadata(path: Path) -> dict[str, Any]:
    """Title, author, and page count from a PDF, or ``{}`` when unavailable."""
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        reader = pypdf.PdfReader(str(path))
        metadata: dict[str, Any] = {}
        info = getattr(reader, "metadata", None)
        if info is not None:
            title = getattr(info, "title", None)
            author = getattr(info, "author", None)
            if title:
                metadata[KEY_TITLE] = str(title)
            if author:
                metadata[KEY_AUTHOR] = str(author)
        try:
            metadata[KEY_PAGE_COUNT] = len(reader.pages)
        except Exception:  # a corrupt page tree: report what we have
            pass
        return metadata
    except Exception:
        return {}


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Width and height from a PNG's IHDR chunk, without decoding the image."""
    try:
        with path.open("rb") as handle:
            head = handle.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def _gif_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            head = handle.read(10)
    except OSError:
        return None
    if len(head) < 10 or head[:3] not in (b"GIF",):
        return None
    width, height = struct.unpack("<HH", head[6:10])
    return int(width), int(height)


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Scan JPEG segment markers for the SOF frame header (no full decode)."""
    try:
        with path.open("rb") as handle:
            data = handle.read(64 * 1024)
    except OSError:
        return None
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        # SOF0..SOF15 (except DHT/DAC/RST markers) carry the frame size.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if index + 3 >= len(data):
            break
        segment_length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        index += 2 + segment_length
    return None


def _image_metadata(path: Path) -> dict[str, Any]:
    """Pixel dimensions of a PNG/JPEG/GIF image, or ``{}`` when unknown."""
    dimensions = _png_dimensions(path) or _jpeg_dimensions(path) or _gif_dimensions(path)
    if dimensions is None:
        return {}
    width, height = dimensions
    return {KEY_WIDTH: width, KEY_HEIGHT: height}


def wav_metadata(path: Path) -> dict[str, Any]:
    """Duration, sample rate, and channel count of a WAV, or ``{}`` when unreadable."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
    except (wave.Error, OSError, EOFError):
        return {}
    duration = frames / rate if rate else 0.0
    return {
        KEY_DURATION_SECONDS: round(duration, 3),
        KEY_SAMPLE_RATE: rate,
        KEY_CHANNELS: channels,
    }


def _audio_metadata(path: Path) -> dict[str, Any]:
    """WAV metadata when the file is a real WAV, else an honest empty mapping.

    A non-WAV (an MP3) needs a decoder to measure; rather than guess a duration,
    this returns ``{}`` and the audio pipeline reports the real next step (install
    a converter) when it tries to transcribe.
    """
    return wav_metadata(path)


def _word_metadata(path: Path) -> dict[str, Any]:
    """Block and character counts from a ``.docx``, reusing the project's reader."""
    try:
        from core.docx_extract import extract_docx

        document = extract_docx(path)
        text = "\n".join(block.text for block in document.blocks)
        return {
            KEY_BLOCK_COUNT: len(document.blocks),
            KEY_WORD_COUNT: len(text.split()),
        }
    except Exception:
        return {}


def extract_metadata(path: str | Path) -> dict[str, Any]:
    """The metadata a source carries, as a text-free mapping.

    Dispatches by the detector's kind and is total: any failure yields ``{}`` for
    the fields that could not be read rather than raising. The ``format`` key is
    always set when the kind is known, so a caller can tell what was probed.
    """
    source = Path(path)
    kind = detect_kind(source)
    metadata: dict[str, Any] = {}
    if kind is SourceKind.PDF:
        metadata.update(_pdf_metadata(source))
    elif kind is SourceKind.WORD:
        metadata.update(_word_metadata(source))
    elif kind is SourceKind.IMAGE:
        metadata.update(_image_metadata(source))
    elif kind is SourceKind.AUDIO:
        metadata.update(_audio_metadata(source))
    if kind is not SourceKind.UNKNOWN:
        metadata[KEY_FORMAT] = kind.value
    return metadata


__all__ = [
    "KEY_AUTHOR",
    "KEY_BLOCK_COUNT",
    "KEY_CHANNELS",
    "KEY_DURATION_SECONDS",
    "KEY_FORMAT",
    "KEY_HEIGHT",
    "KEY_PAGE_COUNT",
    "KEY_SAMPLE_RATE",
    "KEY_TITLE",
    "KEY_WIDTH",
    "KEY_WORD_COUNT",
    "extract_metadata",
    "wav_metadata",
]
