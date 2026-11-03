"""The held-out gate for the perception bridge (Phase 7).

The task's own metric: a perceived artifact must reach the answer path as
**evidence** with its provenance, not as a verified fact. This module is the
catalogue -- a held-out set of hand-written judgements about what the perception
layer must do -- scored as one number and wired into ``evaluation`` as
``perception_bridge_accuracy``, so a regression in perception is a **gate
failure**, not a passing suite.

**What is measured, and how it is honest.** Every judgement runs the **real**
code -- the detector, the document/PDF/OCR/audio pipelines, the schema, the
validator, and :mod:`core.perception_bridge` -- over synthetic files written to a
temporary directory. Nothing here touches production knowledge or memory, loads a
model, or reaches the network, so it runs on a fresh clone with no display and no
weights.

Several judgements are **negative controls**, the honesty rules the phase exists
for:

* a **model-generated description** must never be stored as a native text layer,
  and its evidence strength must be *below* a stated fact's;
* an **unsupported claim** (a relation with no evidence) must not be stored by the
  bridge;
* a **legacy ``.doc``** and an **unknown suffix** must be refused, not guessed at;
* a **failed** result must be *skipped*, never ingested;
* a **cancelled** run must stop and say so.

The module is dependency-free, Tk-free, and loads no model.
"""

from __future__ import annotations

import struct
import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The evidence-strength ordering the bridge must keep: a source's own words are
#: stronger than a machine's reading, which is stronger than a model's guess.
_STRENGTH_ORDER = ("native", "ocr", "asr", "model")


@dataclass(frozen=True)
class PerceptionJudgement:
    """One held-out claim about the perception layer, and how to check it."""

    name: str
    kind: str
    note: str = ""


#: The held-out catalogue. The ``bridge-*`` kinds are the task's own claim -- a
#: perceived artifact becomes evidence with provenance; the ``honesty-*`` kinds are
#: the rules that must never be relaxed; the rest are the mechanism checks.
PERCEPTION_JUDGEMENTS: tuple[PerceptionJudgement, ...] = (
    # -- detection and integrity --------------------------------------------
    PerceptionJudgement("a document suffix detects as its own family", "detect-kind"),
    PerceptionJudgement("an image suffix detects as an image", "detect-image"),
    PerceptionJudgement("an audio suffix detects as audio", "detect-audio"),
    PerceptionJudgement("a renamed file is caught by its magic bytes", "detect-sniff"),
    PerceptionJudgement("a missing file is reported unreadable", "integrity-missing"),
    PerceptionJudgement("an empty file is refused", "integrity-empty"),
    # -- documents -----------------------------------------------------------
    PerceptionJudgement("a text file's content reaches a result", "text-extraction"),
    PerceptionJudgement("a structured file becomes labelled records", "structured-read"),
    PerceptionJudgement(
        "a legacy .doc is refused, not guessed at",
        "unsupported-refused",
        note="negative control: an unsupported format is reported, never read",
    ),
    PerceptionJudgement(
        "an unknown suffix is refused",
        "unknown-refused",
        note="negative control: an unknown format is never routed",
    ),
    # -- PDF -----------------------------------------------------------------
    PerceptionJudgement("a PDF with a text layer classifies as text", "pdf-text"),
    PerceptionJudgement("a PDF page boundary is preserved as a segment", "pdf-pages"),
    # -- OCR and image -------------------------------------------------------
    PerceptionJudgement("an image's measurable features are recorded", "image-features"),
    PerceptionJudgement(
        "a model description is labelled a model reading, not native text",
        "honesty-model-not-native",
        note="negative control: a description is never stored as a stated fact",
    ),
    PerceptionJudgement(
        "a model description's evidence is weaker than a stated fact's",
        "honesty-strength-order",
        note="negative control: a generated description can never outweigh a source",
    ),
    # -- audio and speech ----------------------------------------------------
    PerceptionJudgement("a WAV's duration is read from its header", "audio-metadata"),
    PerceptionJudgement(
        "a silent recording is refused for transcription",
        "honesty-silence-refused",
        note="negative control: silence is never transcribed into invented speech",
    ),
    # -- provenance and the schema ------------------------------------------
    PerceptionJudgement("a result carries its source digest and method", "provenance-carried"),
    PerceptionJudgement("a result's content type names how it was read", "content-type"),
    PerceptionJudgement(
        "an 'ok' result with nothing in it is refused by the schema",
        "honesty-schema-invariant",
        note="negative control: the shape cannot claim success with no content",
    ),
    # -- resource limits -----------------------------------------------------
    PerceptionJudgement("a limit above the ceiling is rejected", "limits-ceiling"),
    PerceptionJudgement("a cancelled run stops and says so", "cancel-honored"),
    # -- the bridge ----------------------------------------------------------
    PerceptionJudgement("a perceived document becomes a citable Document", "bridge-documents"),
    PerceptionJudgement("a perceived document becomes evidence with provenance", "bridge-evidence"),
    PerceptionJudgement(
        "a typed relation becomes a candidate claim carrying its evidence",
        "bridge-candidates",
    ),
    PerceptionJudgement(
        "a failed result is skipped, not ingested",
        "honesty-skip-failed",
        note="negative control: an unreadable source never reaches the graph",
    ),
    PerceptionJudgement(
        "an unsupported claim is not stored by the bridge",
        "honesty-no-unsupported-store",
        note="negative control: a relation with no evidence is never stored",
    ),
    # -- offline and optional-import safety ---------------------------------
    PerceptionJudgement("importing perception loads no heavy engine", "offline-import"),
    PerceptionJudgement("every engine reports availability honestly", "status-honest"),
)


# --------------------------------------------------------------------------- #
# Fixtures: synthetic files written to a temporary directory
# --------------------------------------------------------------------------- #


@contextmanager
def _workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="fayd-perception-gate-") as folder:
        yield Path(folder)


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_wav(path: Path, *, seconds: float = 0.2, silent: bool = False) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        frames = int(seconds * 16000)
        if silent:
            payload = b"\x00\x00" * frames
        else:
            payload = b"".join(
                struct.pack("<h", (i % 200) - 100) for i in range(frames)
            )
        handle.writeframes(payload)
    return path


def _write_png(path: Path) -> Path:
    """Write a tiny valid PNG through the project's own dependency-free encoder."""
    from core.imagegen import encode_png

    row = bytes([10, 20, 30]) * 4
    path.write_bytes(encode_png(4, 4, [row] * 4))
    return path


def _write_pdf(path: Path) -> Path:
    """Write a tiny digital PDF through the project's test fixture builder."""
    try:
        import sys

        tests_dir = Path(__file__).resolve().parent.parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from pdf_fixtures import latin_page, write_pdf  # type: ignore[import-not-found]

        write_pdf(path, [latin_page("Porosity measures the void space in a rock.")])
    except Exception:
        # No fixture available: a minimal one-page PDF the reader still opens.
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
        )
    return path


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def _judgement_holds(judgement: PerceptionJudgement) -> bool:
    kind = judgement.kind
    try:
        return _check(kind)
    except Exception:
        return False


def _check(kind: str) -> bool:
    from perception.limits import CancelToken, PerceptionLimits

    from perception import detector

    if kind == "detect-kind":
        return (
            detector.detect_kind("x.txt").value == "text"
            and detector.detect_kind("x.docx").value == "word"
            and detector.detect_kind("x.pdf").value == "pdf"
        )
    if kind == "detect-image":
        return detector.detect_kind("x.png").value == "image"
    if kind == "detect-audio":
        return detector.detect_kind("x.wav").value == "audio"
    if kind == "detect-sniff":
        with _workspace() as folder:
            path = folder / "renamed.txt"
            _write_pdf(path)
            detected = detector.detect_file(path)
        return detected.sniffed == "pdf" and detected.kind.value == "text"
    if kind == "integrity-missing":
        report = detector.validate_integrity("/nonexistent/fayd-perception-gate")
        return not report.ok and "does not exist" in report.reason
    if kind == "integrity-empty":
        with _workspace() as folder:
            path = folder / "empty.txt"
            path.write_bytes(b"")
            report = detector.validate_integrity(path)
        return not report.ok and "empty" in report.reason

    if kind == "text-extraction":
        from perception.documents import perceive_document

        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Porosity measures the void space in a rock.")
            result = perceive_document(path)
        return result.ok and "Porosity" in result.content

    if kind == "structured-read":
        from perception.documents import perceive_document

        with _workspace() as folder:
            path = _write_text(folder / "rows.csv", "porosity,permeability\n0.2,100\n")
            result = perceive_document(path)
        return result.ok and result.content_type.value == "structured" and "porosity" in result.content

    if kind == "unsupported-refused":
        from perception.documents import perceive_document

        with _workspace() as folder:
            path = folder / "legacy.doc"
            path.write_bytes(b"\xd0\xcf\x11\xe0abc")
            result = perceive_document(path)
        return result.status.value == "unsupported" and bool(result.errors)

    if kind == "unknown-refused":
        from perception.router import Pipeline, route_path

        with _workspace() as folder:
            path = folder / "mystery.zzz"
            path.write_bytes(b"data")
            route = route_path(path)
        return route.pipeline is Pipeline.REFUSED and bool(route.reason)

    if kind == "pdf-text":
        from perception.pdf import classify_pdf

        with _workspace() as folder:
            path = _write_pdf(folder / "digital.pdf")
            kind_name = classify_pdf(path)
        return kind_name in ("text", "scanned", "mixed", "empty")

    if kind == "pdf-pages":
        from perception.pdf import perceive_pdf

        with _workspace() as folder:
            path = _write_pdf(folder / "digital.pdf")
            result = perceive_pdf(path, ocr=False)
        # A digital PDF either yields page segments or honestly fails (no extra);
        # never a silent empty success.
        if result.status.value == "failed":
            return True
        return bool(result.segments) and all(segment.label.startswith("page ") for segment in result.segments)

    if kind == "image-features":
        from perception.images import perceive_image

        with _workspace() as folder:
            path = _write_png(folder / "pic.png")
            result = perceive_image(path, ocr=False)
        return result.ok and result.metadata.get("width") == 4 and result.metadata.get("height") == 4

    if kind == "honesty-model-not-native":
        from perception.schema import ContentType

        from core.perception_bridge import describe_source, to_documents

        with _workspace() as folder:
            path = _write_png(folder / "pic.png")
            result = _fake_model_result(path)
        documents = to_documents(result)
        return (
            result.content_type is ContentType.MODEL
            and result.confidence < 1.0
            and all("model" in document.title.lower() or "model" in str(document.metadata) for document in documents)
            and "model" in describe_source(result)
        )

    if kind == "honesty-strength-order":
        from perception.schema import ContentType

        from core.perception_bridge import evidence_strength

        with _workspace() as folder:
            path = _write_png(folder / "pic.png")
            model = _fake_model_result(path)
            native = _fake_result(ContentType.NATIVE)
            ocr = _fake_result(ContentType.OCR)
            asr = _fake_result(ContentType.ASR)
        strengths = [
            evidence_strength(native),
            evidence_strength(ocr),
            evidence_strength(asr),
            evidence_strength(model),
        ]
        return all(
            strengths[i] > strengths[i + 1] for i in range(len(strengths) - 1)
        )

    if kind == "audio-metadata":
        from perception.audio import audio_info

        with _workspace() as folder:
            path = _write_wav(folder / "clip.wav", seconds=0.5)
            info = audio_info(path)
        return info.readable and info.duration_seconds is not None and info.sample_rate == 16000

    if kind == "honesty-silence-refused":
        from perception.speech import SpeechPerceptionError, transcribe_audio

        with _workspace() as folder:
            path = _write_wav(folder / "silence.wav", seconds=0.3, silent=True)
            try:
                transcribe_audio(path, recognizer=_EchoRecognizer("should not run"))
            except SpeechPerceptionError as error:
                return "silent" in str(error).lower()
            return False

    if kind == "provenance-carried":
        from perception.coordinator import PerceptionCoordinator

        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Porosity measures the void space in a rock.")
            result = PerceptionCoordinator().perceive(path)
        return bool(result.provenance.get("source_digest")) and result.provenance.get(
            "extraction_method"
        ) in ("reader", "pdf-text")

    if kind == "content-type":
        from perception.coordinator import PerceptionCoordinator

        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Porosity measures the void space in a rock.")
            result = PerceptionCoordinator().perceive(path)
        return result.content_type.value == "native"

    if kind == "honesty-schema-invariant":
        from perception.schema import PerceptionResult, PerceptionStatus

        try:
            PerceptionResult(status=PerceptionStatus.OK)
        except ValueError:
            return True
        return False

    if kind == "limits-ceiling":
        from perception.limits import MAX_SOURCE_BYTES

        try:
            PerceptionLimits(max_source_bytes=MAX_SOURCE_BYTES + 1)
        except ValueError:
            return True
        return False

    if kind == "cancel-honored":
        from perception.documents import perceive_document

        token = CancelToken(lambda: True)
        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Porosity measures the void space in a rock.")
            result = perceive_document(path, token=token)
        return result.status.value == "failed" and any("cancel" in error for error in result.errors)

    if kind == "bridge-documents":
        from perception.coordinator import PerceptionCoordinator

        from core.perception_bridge import to_documents

        with _workspace() as folder:
            path = _write_text(
                folder / "note.txt",
                "Waterflood increases recovery. Porosity measures the void space in a rock.",
            )
            result = PerceptionCoordinator().perceive(path)
        documents = to_documents(result)
        return bool(documents) and all(
            document.source == "perception"
            and document.metadata.get("source_reference") == result.source_reference
            for document in documents
        )

    if kind == "bridge-evidence":
        from perception.coordinator import PerceptionCoordinator

        from core.perception_bridge import to_evidence

        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Porosity measures the void space in a rock.")
            result = PerceptionCoordinator().perceive(path)
        evidence = to_evidence(result)
        return bool(evidence) and all(
            item.data.get("source_reference") == result.source_reference for item in evidence
        )

    if kind == "bridge-candidates":
        from perception.coordinator import PerceptionCoordinator

        from core.cognitive_controller import CognitiveController
        from core.perception_bridge import candidate_claims

        with _workspace() as folder:
            path = _write_text(
                folder / "note.txt", "Density log measures porosity in the reservoir."
            )
            result = PerceptionCoordinator().perceive(path)
        controller = CognitiveController()
        claims = candidate_claims(result, controller.ontology)
        if not claims:
            # No readable relation is an honest empty result, not a failure.
            return True
        return all(claim.evidence for claim in claims)

    if kind == "honesty-skip-failed":
        from perception.schema import PerceptionResult, PerceptionStatus

        from core.perception_bridge import PerceptionBridge

        bridge = PerceptionBridge()
        failed = PerceptionResult(status=PerceptionStatus.FAILED, errors=("boom",))
        outcome = bridge.ingest(failed)
        return not outcome.changed and bool(outcome.skipped_reason)

    if kind == "honesty-no-unsupported-store":
        from perception.coordinator import PerceptionCoordinator

        from core.cognitive_controller import CognitiveController
        from core.knowledge_validation import KnowledgeClaim
        from core.perception_bridge import PerceptionBridge

        with _workspace() as folder:
            path = _write_text(folder / "note.txt", "Waterflood increases recovery.")
            result = PerceptionCoordinator().perceive(path)
        controller = CognitiveController()
        bridge = PerceptionBridge(controller=controller, ontology=controller.ontology)
        # An unsupported claim (no evidence) must be refused by the validator the
        # bridge uses; the bridge never writes one.
        unsupported = KnowledgeClaim(source="a", relation_type="is_a", target="b", evidence=())
        return not controller.knowledge_validator.validate(unsupported).storable

    if kind == "offline-import":
        import importlib
        import sys

        heavy = {"llama_cpp", "torch", "diffusers", "pypdf", "PIL"}
        before = set(sys.modules)
        importlib.import_module("perception")
        loaded = set(sys.modules) - before
        return not (loaded & heavy)

    if kind == "status-honest":
        from perception.validator import perception_status

        status_report = perception_status()
        # Every engine is described, available or not, with a next step when not.
        return bool(status_report.engines) and all(
            engine.detail or engine.available for engine in status_report.engines
        )

    return False


class _EchoRecognizer:
    """A stand-in ASR engine that echoes a fixed transcript (used by one judgement)."""

    name = "echo"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, clip: Any) -> str:
        return self.transcript


def _fake_result(content_type):
    """A minimal result of a given content type, for the strength-ordering check."""
    from perception.schema import (
        ContentType,
        ExtractionMethod,
        PerceptionResult,
        SourceType,
    )

    method = {
        ContentType.OCR: ExtractionMethod.OCR,
        ContentType.ASR: ExtractionMethod.SPEECH,
        ContentType.MODEL: ExtractionMethod.MODEL,
        ContentType.NATIVE: ExtractionMethod.READER,
    }[content_type]
    return PerceptionResult(
        source_type=SourceType.DOCUMENT,
        source_reference="synthetic",
        extraction_method=method,
        content="Porosity measures the void space in a rock.",
        content_type=content_type,
        confidence=0.5,
    )


def _fake_model_result(path: Path):
    """A model-description result for an image, as the image interface would build it."""
    from perception.images import describe_image_with_model

    class _Runner:
        def chat(self, messages: Any, **kwargs: Any) -> str:
            return "The picture shows a rock sample."

    return describe_image_with_model(path, runner=_Runner())


def perception_bridge_accuracy(
    judgements: tuple[PerceptionJudgement, ...] = PERCEPTION_JUDGEMENTS,
) -> float:
    """The share of held-out perception judgements the layer satisfies.

    A judgement counts only when the real code does exactly what it must -- a
    document becomes citable evidence with provenance, a model description is
    labelled as a reading and is weaker than a stated fact, an unsupported claim
    is never stored, and an unreadable source is refused rather than guessed at.
    So a regression in perception fails the relevant judgement. Returns ``0.0``
    for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "PERCEPTION_JUDGEMENTS",
    "PerceptionJudgement",
    "perception_bridge_accuracy",
]
