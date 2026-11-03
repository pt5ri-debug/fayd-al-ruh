"""Local multimodal perception for Fayd Al-Ruh (Phase 7).

Perception turns a *raw input* -- a document, a page image, a recording, a
spoken sentence -- into structured, traceable information the Cognitive Core,
Memory, Learning, and Reasoning systems can consume. It is deliberately built as
a thin, modular layer over the readers, engines, and stores the project already
has, never a second implementation of them:

* the document/PDF/Word readers are :mod:`core.document_reader`,
  :mod:`core.pdf_extract`, :mod:`core.pdf_ocr`, :mod:`core.docx_extract`, and
  :mod:`core.docs_studio` (reused, not rewritten);
* the speech engines are :mod:`core.speech`, :mod:`core.qwen_speech`,
  :mod:`core.reading`, and the project's own voice catalog;
* the image path reuses :mod:`core.vision` and Doc's Studio's raster guards;
* the model layer reuses the Model Manager (:mod:`core.model_manager`) and its
  resource/registry modules.

The package is **optional-import safe**: importing it loads no model, needs no
network, and pulls no third-party package that the base install does not already
have. A perception engine that is not installed reports itself honestly through
:class:`perception.validator.PerceptionStatus` rather than failing a run.

The pieces, each with a stable API and replaceable independently:

* :mod:`perception.schema` -- the one normalized result shape.
* :mod:`perception.limits` -- resource limits, cancellation, and temp cleanup.
* :mod:`perception.detector` -- file type detection and integrity validation.
* :mod:`perception.documents` -- the document processing pipeline.
* :mod:`perception.pdf` -- the PDF extraction engine (text / scanned / mixed).
* :mod:`perception.ocr` -- the OCR engine.
* :mod:`perception.images` -- the image perception interface.
* :mod:`perception.audio` -- the audio processing pipeline.
* :mod:`perception.speech` -- the STT and TTS interfaces.
* :mod:`perception.metadata` -- metadata extraction.
* :mod:`perception.normalize` -- content normalization.
* :mod:`perception.validator` -- the perception result validator.
* :mod:`perception.provenance` -- the perception provenance manager.
* :mod:`perception.models` -- local-model-manager integration.
* :mod:`perception.router` -- the input router.
* :mod:`perception.coordinator` -- the perception coordinator.

:mod:`core.perception_bridge` is the one place a perceived artifact becomes
`Evidence`/`Document` for the answer path, so perception and reasoning share a
single provenance path.
"""

from __future__ import annotations

from perception.coordinator import (
    PerceptionCoordinator,
    default_coordinator,
    perceive,
)
from perception.detector import (
    DetectedFile,
    SourceKind,
    detect_file,
    detect_kind,
    file_digest,
    validate_integrity,
)
from perception.limits import PerceptionLimits, PerceptionTimeout, cancelled
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionError,
    PerceptionResult,
    PerceptionStatus,
    PerceptionWarning,
    Segment,
    SourceType,
)

__all__ = [
    "ContentType",
    "DetectedFile",
    "ExtractionMethod",
    "PerceptionCoordinator",
    "PerceptionError",
    "PerceptionLimits",
    "PerceptionResult",
    "PerceptionStatus",
    "PerceptionTimeout",
    "PerceptionWarning",
    "Segment",
    "SourceKind",
    "SourceType",
    "cancelled",
    "default_coordinator",
    "detect_file",
    "detect_kind",
    "file_digest",
    "perceive",
    "validate_integrity",
]
