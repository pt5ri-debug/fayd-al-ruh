"""The input router (section 3).

The router is the one place that decides *which pipeline* a raw input goes to. It
turns a :class:`perception.detector.DetectedFile` (or a bare path) into a routing
:class:`Route`, and the coordinator runs the pipeline the route names. Keeping the
decision in one place means every surface -- the CLI, a window, the cognitive
bridge -- routes an input identically, and adding a format is a one-line change
here rather than a change in every caller.

Routing rules, all deterministic:

* a **document** (text, Word, PDF) goes to the document pipeline;
* an **image** goes to the image interface (features + optional OCR), and an image
  the user *knows* is a scan can be routed to OCR explicitly;
* **audio** goes to the audio pipeline (metadata + optional transcription);
* an **unsupported** format is routed to a refusal with the real next step;
* an **unknown** format is refused honestly rather than guessed at.

A caller may also route *text* or an *audio clip* directly (a pasted sentence, a
microphone capture) -- those have no file, so the router builds an in-memory route.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from perception.detector import DetectedFile, SourceKind, detect_file


class Pipeline(str, Enum):
    """The pipeline a route selects."""

    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    REFUSED = "refused"


@dataclass(frozen=True)
class Route:
    """Which pipeline an input goes to, and why."""

    pipeline: Pipeline
    detected: DetectedFile | None = None
    reason: str = ""
    explicit_ocr: bool = False
    """True when the caller asked for OCR on an image regardless of detection."""

    def describe(self) -> str:
        if self.pipeline is Pipeline.REFUSED:
            return f"refused: {self.reason}"
        target = self.detected.describe() if self.detected else "(in-memory input)"
        return f"{self.pipeline.value} <- {target}"


def route_path(path: str | Path, *, ocr: bool | None = None) -> Route:
    """Choose a pipeline for a filesystem path.

    ``ocr`` forces OCR on an image (``True``) or refuses it (``False``); ``None``
    leaves the decision to the pipeline's own auto rule. The route is refused with
    a readable reason for an unsupported or unknown format -- the router never
    guesses at a format it does not recognise.
    """
    detected = detect_file(path)
    kind = detected.kind
    if kind is SourceKind.UNSUPPORTED:
        return Route(
            Pipeline.REFUSED,
            detected,
            f"{detected.suffix} is a legacy format; save it as .docx or .pdf.",
        )
    if kind is SourceKind.UNKNOWN:
        return Route(
            Pipeline.REFUSED,
            detected,
            f"{detected.path.name!r} is not a format the perception pipeline reads.",
        )
    if kind is SourceKind.AUDIO:
        return Route(Pipeline.AUDIO, detected)
    if kind is SourceKind.IMAGE:
        return Route(Pipeline.IMAGE, detected, explicit_ocr=bool(ocr))
    return Route(Pipeline.DOCUMENT, detected)


def route_text(text: str) -> Route:
    """Route a bare string (a pasted sentence, a note) to the document pipeline."""
    if not text.strip():
        return Route(Pipeline.REFUSED, None, "the text is empty.")
    return Route(Pipeline.DOCUMENT, None, reason="in-memory text")


__all__ = [
    "Pipeline",
    "Route",
    "route_path",
    "route_text",
]
