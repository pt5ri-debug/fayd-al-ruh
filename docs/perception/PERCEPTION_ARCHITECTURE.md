# Phase 7 — Local Multimodal Perception

**Status:** implemented, gated, and wired.
**Metric:** `perception_bridge_accuracy` (held-out, 29 judgements, 1.0000).
**Baseline:** `evaluation/baseline.json` `overall` 0.9900 (was 0.9898).

## 1. Goal

Route perception — documents, images, audio, and speech — into the **same evidence
pipeline as text**, so perception and reasoning share one provenance path. A
perceived artifact (a page's text, a scanned page's OCR reading, an image's
description, an audio transcript) becomes **evidence** and a **citable document**
carrying its provenance, never a verified fact.

## 2. What was already there (reused, not rebuilt)

Phase 7 **extends** the existing architecture rather than replacing it. Every
reader, engine, and store already existed; this phase adds the one normalized
shape, the coordinator, the router, the limits, and the bridge over them.

| Existing module | What Phase 7 reuses |
| --- | --- |
| `core/document_reader.py` | text/Markdown/CSV/JSON/Word/PDF reading, safe encoding detection, streaming |
| `core/pdf_extract.py` | the per-page digital-PDF text layer (`pypdf`) |
| `core/pdf_ocr.py` | the OCR path, the raster guards, `usable_languages`/`ocr_available`, Tesseract |
| `core/docx_extract.py` | Word `.docx`/`.docm` paragraphs and table rows |
| `core/docs_studio.py` | `SUPPORTED_SUFFIXES`, `file_uri`, the scanned-PDF render path |
| `core/docs_model.py` | the local Qwen3-VL vision model (Doc's Studio's) |
| `core.vision` | the dependency-free `ImageBuffer`/`extract_features` |
| `core.speech` | `AudioClip`, `load_recognizer`, `load_synthesizer`, `system_voices` |
| `core.qwen_speech`, `core.piper_speech`, `core.audiocpp_speech` | the optional STT/TTS engines |
| `core.text_normalize`, `core.voices` | the deterministic text pass and the one language check |
| `core.provenance` | `ProvenanceLedger`/`ProvenanceRecord` (the audit trail) |
| `core.evidence`, `core.evidence_manager` | `Evidence`, `EvidenceManager`, `AcquisitionMethod` |
| `core.documents`, `core.rag`, `core.knowledge_base` | `Document`, `RagRetriever`, `chunk_text` |
| `core.relation_extraction` | the typed-relation reader (a perceived document teaches *how*) |
| `core.knowledge_validation`, `core.cognitive_controller` | `KnowledgeClaim`, validation/integration, memory |
| `core.model_manager`, `core.model_resources` | the model registry, status, and resource readings |
| `core.optional_deps` | the `[pdf]`/`[ocr]`/`[docs]` extras and their install commands |
| `evaluation/` | the metric/benchmark/baseline gate |

## 3. New modules

### The `perception/` package (the perception layer)

| Module | Responsibility |
| --- | --- |
| `perception/schema.py` | **The one normalized result shape**: `PerceptionResult`, `Segment`, `SourceType`, `ContentType`, `ExtractionMethod`, `PerceptionStatus`, `PerceptionWarning` |
| `perception/limits.py` | `PerceptionLimits` (validated, bounded), `CancelToken`, `temp_workspace`, `PerceptionTimeout` |
| `perception/detector.py` | File type detection (suffix + magic-byte sniff) and integrity validation |
| `perception/router.py` | The input router: a path or text → a `Route`/`Pipeline` |
| `perception/documents.py` | The document pipeline (text/structured/Word/PDF dispatch) |
| `perception/pdf.py` | The PDF engine: `classify_pdf`, `perceive_pdf` (text/scanned/mixed) |
| `perception/ocr.py` | The OCR engine for images: `perceive_image_ocr`, availability, install hint |
| `perception/images.py` | The image interface: `perceive_image` (features) and `describe_image_with_model` |
| `perception/audio.py` | The audio pipeline: `audio_info`, `perceive_audio`, `to_wav` |
| `perception/speech.py` | The STT/TTS interfaces: `transcribe_audio`, `synthesize_to_file`, `correct_transcript` |
| `perception/metadata.py` | Text-free metadata extraction (page count, dimensions, duration) |
| `perception/normalize.py` | Content normalization (reuses `core.text_normalize`) + `truncate` |
| `perception/validator.py` | The result validator + `perception_status()` (engine availability) |
| `perception/provenance.py` | The provenance manager over `core.provenance` |
| `perception/models.py` | Local-model-manager integration + `can_load_vision_model` |
| `perception/coordinator.py` | **The one public entry point**: route → run → validate → attach provenance |

### `core/perception_bridge.py` (the bridge into evidence)

The module the whole phase exists for: a `PerceptionResult` becomes the **same**
kinds of artifact the text path uses.

* `to_documents(result)` → citable `core.documents.Document`s (chunked with the
  project's own `chunk_text`, each carrying source/method/digest in metadata).
* `to_evidence(result)` → `core.evidence.Evidence` items, one per content-bearing
  segment, each carrying its provenance in `data`.
* `candidate_claims(result, ontology)` → `KnowledgeClaim` candidates from the
  project's own `RelationExtractor`, each carrying the result's evidence.
* `PerceptionBridge.ingest(result)` → submits to every wired target (retriever,
  evidence manager, controller, memory) and returns a `BridgeOutcome`.
* `evidence_strength(result)` → the ordering that makes the honesty rule explicit
  (native > OCR > ASR > model).

### `core/perception_gate.py` (the held-out gate)

`PERCEPTION_JUDGEMENTS` (29 hand-written judgements with negative controls) and
`perception_bridge_accuracy()`, wired into `evaluation` as
`perception_bridge_accuracy`. It runs the **real** detector, pipelines, schema,
validator, and bridge over synthetic files in a temporary directory — no model, no
network, no production data.

### `cli/perception.py` (the headless twin)

`status` / `detect` / `read` / `classify` / `ocr` / `describe` / `audio` / `speak` /
`ingest` / `ingest-text`, driving the *same* coordinator and bridge. Registered in
`cli/__main__.py` as `perception` (alias `perceive`) and mirrored in `cli/menu.py`.

## 4. The pipeline (section 10)

```
raw input
   │
   ▼
detector ──► router ──► pipeline (documents | pdf | ocr/images | audio/speech)
   │                          │
   │                          ▼
   │                    PerceptionResult   (schema, provenance, confidence)
   │                          │
   │                          ▼
   │                    validator           (shape invariants + honesty findings)
   │                          │
   │                          ▼
   └────────────────► core.perception_bridge
                              │
              ┌───────────────┼────────────────┬───────────────┐
              ▼               ▼                ▼               ▼
         Documents       Evidence      Candidate claims     Memory
        (retrievable)   (provenance)   (validated first)  (opt-in)
```

## 5. Honesty rules (structural, not advisory)

1. **Extracted content is not trusted knowledge.** A perceived sentence becomes
   *evidence* (which a later layer may weigh) and *candidate* claims (which must
   pass validation). Nothing is written to the graph merely because a document
   said it.
2. **A model description is not ground truth.** Content typed `MODEL` becomes an
   evidence item that says so, with a confidence bounded well below 1.0 and a
   strength below a native text layer's.
3. **OCR/STT output is never verified.** Their confidence is a bounded quality
   signal (≤ `MACHINE_QUALITY_CEILING`, 0.95), and the caller may correct a
   transcript (a correction is a *new* record pointing at the original audio).
4. **Nothing is silently dropped.** A skipped page, a capped file, a missing OCR
   language, an unreadable source: each is a `PerceptionWarning` or an error on
   the result, and the status is `PARTIAL`/`FAILED`/`UNSUPPORTED` accordingly.
5. **No engine is mandatory.** A base install reads documents; OCR, vision, and
   speech each report their own real next step.

## 6. Offline and optional-import safety

Importing `perception` loads no model and no third-party package the base install
does not already have (pinned by a test). The heavy paths are lazy:

* `pypdf` only inside `perception/metadata.py`/`perception/pdf.py`;
* `pytesseract`/`PIL` only inside `perception/ocr.py`'s recognition call;
* `llama_cpp` only inside `perception/images.py`'s model-description call.

Every install-time download (the OCR engine and its language data, the vision
model, the speech models) is separate from runtime; nothing at runtime reaches the
network.

## 7. Testing

`tests/test_perception_phase7.py` (64 tests) plus `core/perception_gate.py`'s 29
held-out judgements. See `PERCEPTION_TESTS.md` for the coverage map.

## 8. See also

* `DOCUMENT_PROCESSING.md`, `PDF_AND_OCR.md`, `IMAGE_PERCEPTION.md`
* `AUDIO_PROCESSING.md`, `SPEECH_TO_TEXT.md`, `TEXT_TO_SPEECH.md`
* `PERCEPTION_SCHEMA.md`, `PERCEPTION_API.md`, `LOCAL_MODEL_REQUIREMENTS.md`
* `PERCEPTION_TESTS.md`
