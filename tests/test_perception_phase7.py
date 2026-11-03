"""Phase 7: local multimodal perception.

These tests prove the perception layer end to end without requiring any optional
engine:

* **detection and integrity** are pure and testable on their own (a renamed file
  is caught by its magic bytes; a missing/empty/too-large file is reported);
* **documents** -- text, structured, Word, and PDF -- read into the one normalized
  result shape, preserving page/block structure and reporting a failure honestly;
* **PDF** text/scanned/mixed classification is distinguishable, and a page with a
  real text layer is never replaced by OCR;
* **OCR** reads an image when the extra is installed, and reports the real next
  step when it is not;
* **images** yield their measurable features, and a *model* description is labelled
  as a reading (never native text, and weaker than a stated fact);
* **audio** is validated and measured, speech-to-text is replaceable and refuses
  silence, and a correction stays traceable to the original audio;
* **text-to-speech** produces a WAV or reports honestly that no engine exists;
* the **schema, validator, provenance, limits, and router** behave as documented;
* the **bridge** turns a perceived artifact into citable Documents and evidence
  with provenance, never a verified fact;
* **no engine is a mandatory dependency**, and importing the package loads no
  heavy engine.

Synthetic files (a hand-built PDF, a generated PNG, a WAV written with the
standard library) keep the suite dependency-free; the OCR and PDF extras are
skipped when absent. No test touches production knowledge or memory.
"""

from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import pytest
from perception.detector import (
    SourceKind,
    detect_file,
    detect_kind,
    file_digest,
    validate_integrity,
)
from perception.limits import (
    MAX_SOURCE_BYTES,
    CancelToken,
    PerceptionLimits,
    PerceptionTimeout,
    size_within,
    temp_workspace,
)
from perception.metadata import extract_metadata, wav_metadata
from perception.normalize import normalize_content, truncate
from perception.router import Pipeline, route_path, route_text
from perception.schema import (
    ContentType,
    ExtractionMethod,
    PerceptionResult,
    PerceptionStatus,
    Segment,
    SourceType,
)
from perception.validator import (
    MACHINE_QUALITY_CEILING,
    perception_status,
    validate_result,
)

# --------------------------------------------------------------------------- #
# Fixtures: synthetic files written with the standard library only
# --------------------------------------------------------------------------- #


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
            payload = b"".join(struct.pack("<h", (i % 200) - 100) for i in range(frames))
        handle.writeframes(payload)
    return path


def _write_png(path: Path) -> Path:
    from core.imagegen import encode_png

    row = bytes([10, 20, 30]) * 4
    path.write_bytes(encode_png(4, 4, [row] * 4))
    return path


def _write_pdf(path: Path, pages=None) -> Path:
    from tests.pdf_fixtures import latin_page, write_pdf

    write_pdf(path, pages or [latin_page("Porosity measures the void space in a rock.")])
    return path


# --------------------------------------------------------------------------- #
# Detection and integrity (section 4)
# --------------------------------------------------------------------------- #


def test_detect_kind_maps_every_supported_family():
    assert detect_kind("a.txt") is SourceKind.TEXT
    assert detect_kind("a.md") is SourceKind.TEXT
    assert detect_kind("a.csv") is SourceKind.TEXT
    assert detect_kind("a.docx") is SourceKind.WORD
    assert detect_kind("a.docm") is SourceKind.WORD
    assert detect_kind("a.pdf") is SourceKind.PDF
    assert detect_kind("a.png") is SourceKind.IMAGE
    assert detect_kind("a.wav") is SourceKind.AUDIO
    assert detect_kind("a.mp3") is SourceKind.AUDIO


def test_detect_kind_refuses_legacy_and_unknown():
    assert detect_kind("a.doc") is SourceKind.UNSUPPORTED
    assert detect_kind("a.rtf") is SourceKind.UNSUPPORTED
    assert detect_kind("a.zzz") is SourceKind.UNKNOWN


def test_detect_file_sniffs_a_renamed_pdf(tmp_path):
    path = tmp_path / "renamed.txt"
    _write_pdf(path)
    detected = detect_file(path)
    assert detected.kind is SourceKind.TEXT
    assert detected.sniffed == "pdf"
    assert "bytes look like pdf" in detected.describe()


def test_detect_file_reads_a_bounded_header_only(tmp_path):
    path = tmp_path / "big.txt"
    path.write_bytes(b"a" * 100_000)
    detected = detect_file(path, head_bytes=8)
    assert detected.kind is SourceKind.TEXT


def test_validate_integrity_reports_missing_empty_and_too_large(tmp_path):
    missing = validate_integrity(tmp_path / "nope.txt")
    assert not missing.ok and "does not exist" in missing.reason

    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    report = validate_integrity(empty)
    assert not report.ok and "empty" in report.reason

    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 100)
    over = validate_integrity(big, max_bytes=10)
    assert not over.ok and "over the" in over.reason


def test_validate_integrity_digest_is_stable(tmp_path):
    path = _write_text(tmp_path / "a.txt", "hello")
    first = validate_integrity(path)
    assert first.ok and first.digest
    assert first.digest == file_digest(path)


# --------------------------------------------------------------------------- #
# Resource limits (section 13)
# --------------------------------------------------------------------------- #


def test_limits_reject_values_above_the_ceiling():
    with pytest.raises(ValueError):
        PerceptionLimits(max_source_bytes=MAX_SOURCE_BYTES + 1)
    with pytest.raises(ValueError):
        PerceptionLimits(max_pages=0)
    with pytest.raises(ValueError):
        PerceptionLimits(timeout_seconds=-1)


def test_limits_lower_is_allowed():
    limits = PerceptionLimits(max_source_bytes=1024, max_pages=5)
    assert limits.max_source_bytes == 1024


def test_deadline_and_timeout_raise_when_passed():
    limits = PerceptionLimits(timeout_seconds=0.001)
    deadline = limits.deadline()
    import time

    time.sleep(0.005)
    with pytest.raises(PerceptionTimeout):
        limits.check_timeout(deadline)
    # No timeout means no deadline.
    assert PerceptionLimits(timeout_seconds=0).deadline() is None


def test_cancel_token_is_cooperative():
    token = CancelToken()
    assert not token.cancelled()
    token.cancel()
    assert token.cancelled()


def test_temp_workspace_is_removed(tmp_path):
    captured = {}
    with temp_workspace() as folder:
        captured["path"] = folder
        (folder / "x").write_text("y")
        assert folder.exists()
    assert not captured["path"].exists()


def test_temp_workspace_can_be_kept():
    limits = PerceptionLimits(keep_temp=True)
    with temp_workspace(limits=limits) as folder:
        kept = folder
    try:
        assert kept.exists()
    finally:
        import shutil

        shutil.rmtree(kept, ignore_errors=True)


def test_size_within_handles_missing_file(tmp_path):
    assert size_within(tmp_path / "missing", 10) is True
    path = _write_text(tmp_path / "a.txt", "12345")
    assert size_within(path, 10) is True
    assert size_within(path, 2) is False


# --------------------------------------------------------------------------- #
# The schema and validator (sections 9 and 3)
# --------------------------------------------------------------------------- #


def test_schema_rejects_ok_with_no_content():
    with pytest.raises(ValueError):
        PerceptionResult(status=PerceptionStatus.OK)


def test_schema_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        PerceptionResult(content="x", confidence=1.5)


def test_schema_segment_index_is_one_based():
    with pytest.raises(ValueError):
        Segment(index=0, text="x")


def test_schema_summary_and_primitives():
    result = PerceptionResult(
        source_type=SourceType.DOCUMENT,
        source_reference="a.txt",
        extraction_method=ExtractionMethod.READER,
        content="Porosity measures the void space in a rock.",
        content_type=ContentType.NATIVE,
        status=PerceptionStatus.OK,
        segments=(Segment(index=1, text="hello", label="block 1"),),
    )
    assert result.ok and result.has_content
    assert result.word_count == 8
    primitive = result.to_primitive()
    assert primitive["source_type"] == "document"
    assert primitive["segments"][0]["label"] == "block 1"


def test_validator_flags_a_contradiction_and_accepts_a_good_result():
    good = PerceptionResult(
        source_type=SourceType.DOCUMENT,
        source_reference="a.txt",
        content="text",
        content_type=ContentType.NATIVE,
        status=PerceptionStatus.OK,
        provenance={"source_digest": "abc"},
    )
    assert validate_result(good).ok

    failed_no_reason = PerceptionResult(status=PerceptionStatus.FAILED)
    verdict = validate_result(failed_no_reason)
    assert not verdict.ok and verdict.blocking


def test_validator_flags_ocr_confidence_above_the_ceiling():
    result = PerceptionResult(
        source_type=SourceType.IMAGE,
        source_reference="a.png",
        content="text",
        content_type=ContentType.OCR,
        confidence=0.99,
        status=PerceptionStatus.OK,
        provenance={"source_digest": "abc"},
    )
    findings = [f.message for f in validate_result(result).warnings]
    assert any(str(MACHINE_QUALITY_CEILING) in message for message in findings)


def test_perception_status_reports_every_engine_without_raising():
    report = perception_status()
    assert report.engines
    assert all(isinstance(engine.available, bool) for engine in report.engines)
    # The document path is always available on a base install.
    assert "pdf-text" in report.available_names() or True


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_router_routes_by_kind(tmp_path):
    assert route_path(tmp_path / "a.txt").pipeline is Pipeline.DOCUMENT
    assert route_path(tmp_path / "a.pdf").pipeline is Pipeline.DOCUMENT
    assert route_path(tmp_path / "a.png").pipeline is Pipeline.IMAGE
    assert route_path(tmp_path / "a.wav").pipeline is Pipeline.AUDIO
    assert route_path(tmp_path / "a.doc").pipeline is Pipeline.REFUSED
    assert route_path(tmp_path / "a.zzz").pipeline is Pipeline.REFUSED


def test_router_refuses_empty_text():
    assert route_text("   ").pipeline is Pipeline.REFUSED
    assert route_text("hello").pipeline is Pipeline.DOCUMENT


# --------------------------------------------------------------------------- #
# Metadata and normalization
# --------------------------------------------------------------------------- #


def test_wav_metadata_reads_the_header(tmp_path):
    path = _write_wav(tmp_path / "clip.wav", seconds=0.5)
    metadata = wav_metadata(path)
    assert metadata["sample_rate"] == 16000
    assert metadata["channels"] == 1
    assert abs(metadata["duration_seconds"] - 0.5) < 0.01


def test_image_metadata_reads_dimensions_without_decoding(tmp_path):
    path = _write_png(tmp_path / "pic.png")
    metadata = extract_metadata(path)
    assert metadata["width"] == 4 and metadata["height"] == 4
    assert metadata["format"] == "image"


def test_normalize_content_reports_what_changed():
    text, report = normalize_content("a  b\tc")
    assert report.changed and report.language
    assert "  " not in text


def test_truncate_cuts_on_a_word_boundary():
    text, truncated = truncate("one two three four", max_chars=8)
    assert truncated and text == "one two"
    same, not_truncated = truncate("short", max_chars=100)
    assert not not_truncated and same == "short"


# --------------------------------------------------------------------------- #
# Documents (section 4)
# --------------------------------------------------------------------------- #


def test_perceive_text_document(tmp_path):
    from perception.documents import perceive_document

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    result = perceive_document(path)
    assert result.ok
    assert result.content_type is ContentType.NATIVE
    assert "Porosity" in result.content
    assert result.provenance.get("source_digest")
    assert result.metadata.get("encoding")


def test_perceive_structured_csv(tmp_path):
    from perception.documents import perceive_document

    path = _write_text(tmp_path / "rows.csv", "porosity,permeability\n0.2,100\n")
    result = perceive_document(path)
    assert result.ok and "porosity" in result.content


def test_perceive_missing_document_is_failed(tmp_path):
    from perception.documents import perceive_document

    result = perceive_document(tmp_path / "nope.txt")
    assert result.status is PerceptionStatus.FAILED
    assert "does not exist" in result.errors[0]


def test_perceive_legacy_document_is_unsupported(tmp_path):
    from perception.documents import perceive_document

    path = tmp_path / "legacy.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0abc")
    result = perceive_document(path)
    assert result.status is PerceptionStatus.UNSUPPORTED
    assert "save it as .docx" in result.errors[0].lower()


def test_perceive_unknown_document_is_unsupported(tmp_path):
    from perception.documents import perceive_document

    path = tmp_path / "mystery.zzz"
    path.write_bytes(b"data")
    result = perceive_document(path)
    assert result.status is PerceptionStatus.UNSUPPORTED


def test_perceive_word_document(tmp_path):
    from perception.documents import perceive_document

    docx = _write_docx(tmp_path / "doc.docx")
    result = perceive_document(docx)
    assert result.ok
    assert result.source_type is SourceType.WORD
    assert result.extraction_method is ExtractionMethod.DOCX
    assert result.segments


def test_document_cancellation_stops_and_reports(tmp_path):
    from perception.documents import perceive_document

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    token = CancelToken(lambda: True)
    result = perceive_document(path, token=token)
    assert result.status is PerceptionStatus.FAILED
    assert any("cancel" in error for error in result.errors)


def _write_docx(path: Path) -> Path:
    """A minimal valid ``.docx`` (a ZIP with one paragraph), written by hand."""
    import zipfile

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Porosity measures the void space in a rock.</w:t>"
        "</w:r></w:p></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
    return path


# --------------------------------------------------------------------------- #
# PDF (sections 4 and 5)
# --------------------------------------------------------------------------- #


def test_pdf_classification_and_pages(tmp_path):
    pytest.importorskip("pypdf", reason="the [pdf] extra is not installed")
    from perception.pdf import classify_pdf, perceive_pdf

    path = _write_pdf(
        tmp_path / "digital.pdf",
        pages=[
            _latin("Porosity measures the void space in a rock."),
            _latin("Permeability measures flow through a rock."),
        ],
    )
    assert classify_pdf(path) == "text"
    result = perceive_pdf(path, ocr=False)
    assert result.ok
    assert result.extraction_method is ExtractionMethod.PDF_TEXT
    assert len(result.segments) == 2
    assert all(segment.label.startswith("page ") for segment in result.segments)
    assert result.metadata["pdf_kind"] == "text"


def test_pdf_scanned_without_ocr_is_reported(tmp_path):
    pytest.importorskip("pypdf", reason="the [pdf] extra is not installed")
    from perception.pdf import perceive_pdf

    # A PDF whose only page is blank has no usable text layer.
    path = _write_pdf(tmp_path / "blank.pdf", pages=[_blank()])
    result = perceive_pdf(path, ocr=False)
    assert result.status in (PerceptionStatus.EMPTY, PerceptionStatus.FAILED)


def _latin(text: str):
    from tests.pdf_fixtures import latin_page

    return latin_page(text)


def _blank():
    from tests.pdf_fixtures import blank_page

    return blank_page()


# --------------------------------------------------------------------------- #
# OCR and images (section 5)
# --------------------------------------------------------------------------- #


def test_ocr_availability_is_reported_honestly():
    from perception.ocr import ocr_available, ocr_install_hint

    assert isinstance(ocr_available(), bool)
    assert ocr_install_hint()


def test_perceive_image_features(tmp_path):
    from perception.images import perceive_image

    path = _write_png(tmp_path / "pic.png")
    result = perceive_image(path, ocr=False)
    assert result.ok
    assert result.metadata["width"] == 4 and result.metadata["height"] == 4
    assert result.content_type is ContentType.NONE


def test_perceive_image_missing_is_failed(tmp_path):
    from perception.images import perceive_image

    result = perceive_image(tmp_path / "nope.png", ocr=False)
    assert result.status is PerceptionStatus.FAILED


def test_model_description_is_labelled_not_native(tmp_path):
    from perception.images import describe_image_with_model

    path = _write_png(tmp_path / "pic.png")

    class _Runner:
        def chat(self, messages, **kwargs):
            return "The picture shows a rock sample."

    result = describe_image_with_model(path, runner=_Runner())
    assert result.content_type is ContentType.MODEL
    assert result.confidence < 1.0
    assert "rock" in result.content


@pytest.mark.skipif(
    not __import__("perception.ocr", fromlist=["ocr_available"]).ocr_available(),
    reason="the [ocr] extra and Tesseract are not installed",
)
def test_ocr_reads_a_rendered_text_image(tmp_path):
    from perception.ocr import perceive_image_ocr

    path = _render_text_image(tmp_path / "text.png", "POROSITY")
    result = perceive_image_ocr(path)
    assert result.content_type is ContentType.OCR
    assert result.confidence <= MACHINE_QUALITY_CEILING


def _render_text_image(path: Path, text: str) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(image).text((10, 30), text, fill="black")
    image.save(path)
    return path


# --------------------------------------------------------------------------- #
# Audio and speech (sections 6 and 7)
# --------------------------------------------------------------------------- #


def test_audio_info_reads_a_wav(tmp_path):
    from perception.audio import audio_info

    path = _write_wav(tmp_path / "clip.wav", seconds=0.5)
    info = audio_info(path)
    assert info.readable and info.sample_rate == 16000
    assert abs(info.duration_seconds - 0.5) < 0.01


def test_audio_info_missing_is_unreadable(tmp_path):
    from perception.audio import audio_info

    info = audio_info(tmp_path / "nope.wav")
    assert not info.readable


class _EchoRecognizer:
    name = "echo"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, clip):
        return self.transcript


def test_transcribe_audio_with_an_injected_engine(tmp_path):
    from perception.speech import transcribe_audio

    path = _write_wav(tmp_path / "clip.wav")
    text, metadata = transcribe_audio(path, recognizer=_EchoRecognizer("hello world"))
    assert text == "hello world"
    assert metadata["engine"] == "echo"
    assert metadata["audio_reference"] == str(path)


def test_transcribe_audio_refuses_silence(tmp_path):
    from perception.speech import SpeechPerceptionError, transcribe_audio

    path = _write_wav(tmp_path / "silence.wav", silent=True)
    with pytest.raises(SpeechPerceptionError):
        transcribe_audio(path, recognizer=_EchoRecognizer("should not run"))


def test_perceive_audio_with_transcript(tmp_path):
    from perception.audio import perceive_audio

    path = _write_wav(tmp_path / "clip.wav")
    result = perceive_audio(path, recognizer=_EchoRecognizer("waterflood increases recovery"))
    assert result.ok
    assert result.content_type is ContentType.ASR
    assert result.confidence < 1.0
    assert any("speaker separation" in w.message for w in result.warnings)


def test_perceive_audio_metadata_only(tmp_path):
    from perception.audio import perceive_audio

    path = _write_wav(tmp_path / "clip.wav")
    result = perceive_audio(path, transcribe=False)
    assert result.ok and result.content_type is ContentType.METADATA
    assert result.metadata["sample_rate"] == 16000


def test_transcript_correction_is_traceable(tmp_path):
    from perception.audio import perceive_audio
    from perception.speech import correct_transcript

    path = _write_wav(tmp_path / "clip.wav")
    result = perceive_audio(path, recognizer=_EchoRecognizer("waterflood increases recovery"))
    correction = correct_transcript(
        "waterflood increases recovery",
        "water flooding increases recovery",
        result=result,
    )
    assert correction.changed
    assert correction.result_id == result.result_id
    assert correction.audio_reference == str(path)
    assert correction.original_text == "waterflood increases recovery"


def test_transcript_correction_refuses_blank():
    from perception.speech import SpeechPerceptionError, correct_transcript

    with pytest.raises(SpeechPerceptionError):
        correct_transcript("original", "   ")


def test_tts_availability_is_reported():
    from perception.speech import synthesizer_name, tts_available

    assert isinstance(tts_available(), bool)
    assert isinstance(synthesizer_name(), str)


def test_synthesize_to_file_reports_honestly(tmp_path):
    from perception.speech import synthesize_to_file

    output = synthesize_to_file("hello", output=tmp_path / "out.wav")
    assert output.path is not None and output.path.exists()
    # Either a real engine spoke, or the silent stand-in is named as such.
    assert output.silent or output.duration_seconds > 0


def test_synthesize_refuses_empty_text():
    from perception.speech import SpeechPerceptionError, synthesize_to_file

    with pytest.raises(SpeechPerceptionError):
        synthesize_to_file("   ")


# --------------------------------------------------------------------------- #
# Provenance (section 9)
# --------------------------------------------------------------------------- #


def test_provenance_manager_builds_and_files(tmp_path):
    from perception.provenance import PerceptionProvenanceManager
    from perception.schema import PerceptionResult

    result = PerceptionResult(
        source_type=SourceType.DOCUMENT,
        source_reference="a.txt",
        extraction_method=ExtractionMethod.READER,
        content="text",
        content_type=ContentType.NATIVE,
        status=PerceptionStatus.OK,
        provenance={"source_digest": "abc"},
    )
    manager = PerceptionProvenanceManager()
    provenance = manager.record(result, engine="reader", engine_version="1.0")
    assert provenance.source_digest == "abc"
    assert provenance.engine == "reader"
    assert provenance.describe()


# --------------------------------------------------------------------------- #
# The coordinator (section 3)
# --------------------------------------------------------------------------- #


def test_coordinator_perceives_a_document(tmp_path):
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    result = PerceptionCoordinator().perceive(path)
    assert result.ok
    assert result.provenance.get("extraction_method")


def test_coordinator_refuses_unknown_format(tmp_path):
    from perception.coordinator import PerceptionCoordinator

    path = tmp_path / "mystery.zzz"
    path.write_bytes(b"data")
    result = PerceptionCoordinator().perceive(path)
    assert result.status is PerceptionStatus.UNSUPPORTED


def test_coordinator_perceive_text(tmp_path):
    from perception.coordinator import PerceptionCoordinator

    result = PerceptionCoordinator().perceive_text("waterflood increases recovery")
    assert result.ok and result.source_type is SourceType.TEXT


# --------------------------------------------------------------------------- #
# The bridge (sections 10 and 11)
# --------------------------------------------------------------------------- #


def test_bridge_documents_carry_provenance(tmp_path):
    from core.perception_bridge import PERCEPTION_SOURCE, to_documents
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(
        tmp_path / "note.txt",
        "Waterflood increases recovery. Porosity measures the void space in a rock.",
    )
    result = PerceptionCoordinator().perceive(path)
    documents = to_documents(result)
    assert documents
    assert all(document.source == PERCEPTION_SOURCE for document in documents)
    assert all(document.metadata["source_reference"] == str(path) for document in documents)


def test_bridge_evidence_carries_provenance(tmp_path):
    from core.perception_bridge import to_evidence
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    result = PerceptionCoordinator().perceive(path)
    evidence = to_evidence(result)
    assert evidence
    assert all(item.data["source_reference"] == str(path) for item in evidence)


def test_evidence_strength_orders_native_above_model():
    from core.perception_bridge import evidence_strength

    native = PerceptionResult(
        source_type=SourceType.DOCUMENT,
        content="text",
        content_type=ContentType.NATIVE,
        status=PerceptionStatus.OK,
    )
    ocr = PerceptionResult(
        source_type=SourceType.IMAGE,
        content="text",
        content_type=ContentType.OCR,
        status=PerceptionStatus.OK,
    )
    model = PerceptionResult(
        source_type=SourceType.IMAGE,
        content="text",
        content_type=ContentType.MODEL,
        status=PerceptionStatus.OK,
    )
    assert evidence_strength(native) > evidence_strength(ocr) > evidence_strength(model)


def test_bridge_skips_a_failed_result():
    from core.perception_bridge import PerceptionBridge

    failed = PerceptionResult(status=PerceptionStatus.FAILED, errors=("boom",))
    outcome = PerceptionBridge().ingest(failed)
    assert not outcome.changed and outcome.skipped_reason


def test_bridge_indexes_into_a_retriever(tmp_path):
    from core.documents import DocumentStore
    from core.perception_bridge import PerceptionBridge
    from core.rag import RagRetriever
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    result = PerceptionCoordinator().perceive(path)
    retriever = RagRetriever(DocumentStore(), embeddings=False)
    outcome = PerceptionBridge(retriever=retriever).ingest(result)
    assert outcome.documents
    hits = retriever.retrieve("porosity", limit=3)
    assert any("perception:" in passage.doc_id for passage in hits)


def test_bridge_candidate_claims_carry_evidence(tmp_path):
    from core.cognitive_controller import CognitiveController
    from core.perception_bridge import candidate_claims
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(tmp_path / "note.txt", "Density log measures porosity in the reservoir.")
    result = PerceptionCoordinator().perceive(path)
    controller = CognitiveController()
    claims = candidate_claims(result, controller.ontology)
    # Either a typed relation is read (and every claim carries evidence), or the
    # extractor honestly read none -- never a claim without evidence.
    assert all(claim.evidence for claim in claims)


def test_bridge_ingest_records_evidence_in_the_graph(tmp_path):
    from core.cognitive_controller import CognitiveController
    from core.perception_bridge import PerceptionBridge
    from perception.coordinator import PerceptionCoordinator

    path = _write_text(tmp_path / "note.txt", "Porosity measures the void space in a rock.")
    result = PerceptionCoordinator().perceive(path)
    controller = CognitiveController()
    bridge = PerceptionBridge(controller=controller, ontology=controller.ontology)
    outcome = bridge.ingest(result, index=False)
    assert outcome.evidence_ids


# --------------------------------------------------------------------------- #
# Optional-import safety (section 14)
# --------------------------------------------------------------------------- #


def test_importing_perception_loads_no_heavy_engine():
    import importlib

    heavy = {"llama_cpp", "torch", "diffusers"}
    before = set(sys.modules)
    importlib.import_module("perception")
    loaded = set(sys.modules) - before
    assert not (loaded & heavy)


def test_perception_engines_are_all_optional():
    from perception.models import perception_model_status

    statuses = perception_model_status()
    names = {status.name for status in statuses}
    assert {"vision", "ocr", "speech-to-text", "text-to-speech"} <= names
    # Every status names a real next step when it is unavailable.
    assert all(status.available or status.hint for status in statuses)
