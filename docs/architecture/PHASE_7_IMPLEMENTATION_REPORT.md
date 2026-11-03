# Phase 7 — Local Multimodal Perception

**Status:** implemented, gated, and wired.
**Metric:** `perception_bridge_accuracy` (held-out, 29 judgements, 1.0000).
**Baseline:** `evaluation/baseline.json` `overall` 0.9900 (was 0.9898).
**Branch:** `feature/phase-7-multimodal-perception`.

## 1. Goal

Route perception — documents, images, audio, and speech — into the **same
evidence pipeline as text**, so perception and reasoning share one provenance
path. A perceived artifact (a page's text, a scanned page's OCR reading, an
image's description, an audio transcript) becomes **evidence** and a **citable
document** carrying its provenance, never a verified fact.

## 2. Pre-implementation review (section 2)

1. **Phases 1–6 accepted.** `evaluation` runs green on `main` before this phase
   (`overall` 0.9898), so the baseline is a real reference.
2. **Architecture read.** `docs/architecture/IMPLEMENTATION_ROADMAP.md` (Phase 7
   goal + required modifications), `CURRENT_ARCHITECTURE.md`,
   `TARGET_ARCHITECTURE.md`, and the Phase 1/2/3/6 reports.
3. **Existing perception modules inspected** (see the table in section 3).
4. **Local Model Manager inspected.** `core/model_manager.py`, `model_registry`,
   `model_files`, `model_settings`, `model_runtime`, `model_resources`.
5. **Dependencies and reusable pipelines identified** — every reader, engine, and
   store already existed; Phase 7 adds the normalized shape and the bridge.
6. **Windows 11 compatibility.** The chosen libraries are cross-platform: `pypdf`,
   `pypdfium2`, `pytesseract`/Pillow (OCR), `python-pptx`, `llama-cpp-python`
   (vision GGUF), and the project's own `urllib`/`wave`/`sqlite3` paths. The
   `file://` page URI is built with `Path.as_uri()` so a Windows drive path works
   (the Phase 18c fix, reused).
7. **Hardware measured.** Intel/AMD 4-core x86_64, 16 GB RAM, no usable CUDA GPU.
   The CPU-first rule is respected: no perception model is mandatory, and the
   vision model is a *sequential, optional* load (see section 12).
8. **Checkpoint created.** `ba36403` "checkpoint: baseline before Phase 7 (all
   gates green)".
9. **Compatibility problems reported, not assumed.** The vision model and a real
   STT/TTS engine are **not installed on this machine**; each reports its own
   real next step rather than failing (section 9).

## 3. What was already there (reused, not rebuilt)

Phase 7 **extends** the existing architecture rather than replacing it.

| Existing module | What Phase 7 reuses |
| --- | --- |
| `core/document_reader.py` | text/Markdown/CSV/JSON/Word/PDF reading, safe encoding detection, streaming |
| `core/pdf_extract.py` | the per-page digital-PDF text layer (`pypdf`) |
| `core/pdf_ocr.py` | the OCR path, the raster guards, `usable_languages`/`ocr_available`, Tesseract |
| `core/docx_extract.py` | Word `.docx`/`.docm` paragraphs and table rows |
| `core/docs_studio.py` | `SUPPORTED_SUFFIXES`, `file_uri`, the scanned-PDF render path |
| `core/docs_model.py` | the local Qwen3-VL vision model (Doc's Studio's) |
| `core/vision.py` | the dependency-free `ImageBuffer`/`extract_features` |
| `core/speech.py` | `AudioClip`, `load_recognizer`, `load_synthesizer`, `system_voices` |
| `core/qwen_speech.py`, `core/piper_speech.py`, `core/audiocpp_speech.py` | the optional STT/TTS engines |
| `core/text_normalize.py`, `core/voices.py` | the deterministic text pass and the one language check |
| `core/provenance.py` | `ProvenanceLedger`/`ProvenanceRecord` (the audit trail) |
| `core/evidence.py`, `core/evidence_manager.py` | `Evidence`, `EvidenceManager`, `AcquisitionMethod` |
| `core/documents.py`, `core/rag.py`, `core/knowledge_base.py` | `Document`, `RagRetriever`, `chunk_text` |
| `core/relation_extraction.py` | the typed-relation reader (a perceived document teaches *how*) |
| `core/knowledge_validation.py`, `core/cognitive_controller.py` | `KnowledgeClaim`, validation/integration, memory |
| `core/model_manager.py`, `core/model_resources.py` | the model registry, status, and resource readings |
| `core/optional_deps.py` | the `[pdf]`/`[ocr]`/`[docs]` extras and their install commands |
| `evaluation/` | the metric/benchmark/baseline gate |

## 4. New and modified modules

### The `perception/` package (16 modules — the perception layer)

| Module | Responsibility |
| --- | --- |
| `perception/schema.py` | **The one normalized result shape**: `PerceptionResult`, `Segment`, `SourceType`, `ContentType`, `ExtractionMethod`, `PerceptionStatus`, `PerceptionWarning` |
| `perception/limits.py` | `PerceptionLimits` (validated, bounded), `CancelToken`, `temp_workspace`, `PerceptionTimeout` |
| `perception/detector.py` | File type detection (suffix + magic-byte sniff) and integrity validation |
| `perception/router.py` | The input router: a path or text → a `Route`/pipeline |
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

`status` / `detect` / `read` / `classify` / `ocr` / `describe` / `audio` / `speak`
/ `ingest` / `ingest-text`, driving the *same* coordinator and bridge. Registered
in `cli/__main__.py` as `perception` (alias `perceive`) and mirrored in
`cli/menu.py`.

### Modified

| File | Change |
| --- | --- |
| `core/evidence_manager.py` | `AcquisitionMethod` gains `OCR`, `SPEECH`, `MODEL` — a machine's reading of a picture or a voice is kept distinct from a document's own text layer |
| `pyproject.toml` | `perception*` added to setuptools package discovery and to mypy's `files` |
| `cli/__main__.py`, `cli/menu.py` | the `perception`/`perceive` command and five menu entries |
| `evaluation/benchmark.py`, `runner.py`, `baseline.json` | the new metric |
| `tests/test_portable_cli.py` | the out-of-range menu value is derived from the live menu, so a new menu entry never breaks it |

## 5. The pipeline (sections 3, 10)

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

## 6. Document processing (section 4)

Supported locally: `.txt`, `.md`, `.csv`/`.tsv`, `.json`/`.jsonl`, `.docx`/`.docm`,
digital `.pdf`, scanned `.pdf` (OCR), and images (OCR/features). Detection is by
suffix **and** magic-byte sniff (a renamed PDF is still read as a PDF). Integrity
is validated (missing, empty, too-large). Large files stream; the character total
is capped by `PerceptionLimits`. Page boundaries are preserved as `Segment`s;
headings are kept where the reader reports them. Encoding problems raise through
`core.text_encoding` (never silently corrupt). A legacy `.doc`/`.rtf` is
**unsupported** with a "save as .docx" hint, not scrambled. Nothing is silently
dropped: a skipped/capped/failed page is a `PerceptionWarning`, and the status is
`PARTIAL`/`FAILED`/`UNSUPPORTED`.

PDFs are classified `text` / `scanned` / `mixed`; **OCR runs only on low-text
pages**, and a page with a real text layer is never replaced.

## 7. OCR and image perception (section 5)

`perceive_image_ocr` reads common image formats through the project's existing
Tesseract path (`pypdfium2` rasterisation, `pytesseract` recognition), preserving
the source reference and a bounded confidence. The four content provenances are
kept apart: **native text** (`READER`/`PDF_TEXT`), **OCR-derived** (`OCR`),
**model-generated description** (`MODEL`), and — for anything a later layer
*asserts* — the graph's own verified facts. A model description is **never** native
text and its `evidence_strength` is the lowest. No cloud vision API is used;
`describe_image_with_model` needs the local `[docs]` vision GGUF and reports its
next step when absent.

## 8. Audio perception and STT/TTS (sections 6–8)

* `audio_info` reads WAV metadata from the header (duration, channels, rate) with
  no decode; `perceive_audio` validates and, unless refused, transcribes.
* `transcribe_audio` drives the project's own `load_recognizer` (Qwen3-ASR →
  `whisper` → the transcript stand-in); an injected engine is the test seam. A
  silent clip is refused honestly. Transcript timestamps are carried where the
  engine reports them. **No online transcription service** is used.
* `synthesize_to_file` drives the project's own `load_synthesizer` and writes a
  WAV; it is optional and never affects reasoning or knowledge storage.
* **Section 7 transcript correction:** `correct_transcript(original, corrected,
  …)` returns a `TranscriptCorrection` — a **new** record pointing back at the
  original audio (its result id and reference), never a mutation. A blank
  correction is refused. The corrected text is honestly labelled user-corrected.

Speaker *identity* recognition is **not** claimed; the module states this rather
than implying a capability it does not have.

## 9. Local model management and measured availability (section 12)

All perception models go through the existing Model Manager and their own
engines; nothing is loaded at import and no two large models load at once. On
**this machine** (verified, not assumed):

| Engine | State here | Real next step reported |
| --- | --- | --- |
| PDF text (`pypdf`) | available | — |
| OCR (`tesseract` 5.5.0, `eng`+`ara`) | available | — |
| Vision model (Qwen3-VL) | **not installed** | `python -m pip install -r requirements-docs.txt` + `python -m cli docs download --approve` |
| Speech-to-text | **no real engine** (stand-in only) | `python -m cli qwen-asr download --approve`, or install `whisper` |
| Text-to-speech | **no engine** | `python -m cli deps --install speech`, or install Piper |

`perceive` never fails because an engine is missing: the result is `UNSUPPORTED`/
`FAILED` with the reason and the next step.

## 10. Perception result schema (section 9)

`PerceptionResult` carries: a `result_id`; `source_type`; `source_reference`; the
`extraction_method`; the `content`; the `content_type`; the `language`; a
timestamp; `Segment`s (page/region/time) where applicable; `confidence`; the
`status`; `warnings`/`errors`; `model_name`/`model_version`; and a `provenance`
map. Extracted content is kept separate from anything inferred: the bridge emits
*candidate* claims, which must pass the existing validation before storage.

## 11. Cognitive Core and memory integration (sections 10–11)

The workflow is exactly the nine steps: ingest → extract → normalize → preserve
provenance → validate quality → identify candidate entities/relations → submit for
validation → store accepted information through the existing APIs → preserve
rejected/uncertain content. `PerceptionBridge.ingest` writes `Evidence` and indexes
`Document`s for retrieval; `candidate_claims` feeds the existing
`KnowledgeValidator`/`KnowledgeIntegrator`, so nothing bypasses evidence,
confidence, contradiction, or provenance. Memory capture is **opt-in** and honours
the personality isolation policy. Extracted content is **never** used for training
automatically — eligibility remains Phase 6's dataset governance.

## 12. Resource management (section 13)

CPU-first, bounded, cancellable, time-limited:

* `PerceptionLimits` validates and caps `max_bytes`, `max_pages`, `max_chars`,
  `max_seconds`, `timeout_seconds`; a value above a ceiling is rejected.
* `CancelToken` is cooperative and polled between pages/files; a cancelled run
  reports `FAILED` with "cancelled".
* `temp_workspace` removes its files on exit (and can be kept for debugging).
* Large documents stream; the character total is capped; the vision model is a
  *sequential, optional* load (~3–4 GB Q4), never simultaneous with a chat model.
* Measured here: importing `perception` + the bridge costs **~45 ms** and pulls
  **no** heavy engine (no `llama_cpp`/`torch`/`diffusers`/`pypdf`/`PIL`), peak RSS
  ~19 MB. The optional GPU paths are **not** assumed; this machine has no usable
  CUDA device and every path falls back to CPU.

## 13. Offline verification (section 14)

Verified end to end with the network unused (the only `urllib` reference is a
`file://` URI for the vision path):

| Capability | Result here |
| --- | --- |
| Document extraction (text/CSV) | ✅ `ok` / `reader` / `structured` |
| PDF processing | ✅ `ok` / `pdf-text`, page boundaries preserved |
| OCR | ✅ `ok` / `ocr` on a rendered image |
| Image processing | ✅ features extracted |
| Audio metadata | ✅ `ok` / `metadata`, duration read |
| Speech-to-text | ⚠️ engine not installed; honest `UNSUPPORTED` + next step |
| Text-to-speech | ⚠️ engine not installed; honest refusal + next step |
| Perception result storage | ✅ provenance attached |
| Cognitive Core integration | ✅ bridge writes evidence/documents |

Install-time downloads (the OCR engine + language data, the vision model, the
speech models) are **separate** from runtime; nothing at runtime reaches the
network.

## 14. Testing (section 15)

`tests/test_perception_phase7.py` — **64 tests**, all passing, covering file type
detection, text/CSV/Word extraction, PDF page handling, scanned-PDF classification,
OCR output, image processing, audio validation, STT (injected engine), transcript
correction, TTS availability, provenance, extraction errors, large-file/limit
handling, cancellation, model availability reporting, offline import safety, and
the bridge/Cognitive-Core/memory integration. Tests use synthetic files in a
temporary directory and **never** touch production knowledge or memory. The
held-out `core/perception_gate.py` adds 29 judgements with negative controls.

## 15. Documentation (section 16)

`docs/perception/` — `PERCEPTION_ARCHITECTURE.md`, `DOCUMENT_PROCESSING.md`,
`PDF_AND_OCR.md`, `IMAGE_PERCEPTION.md`, `AUDIO_PROCESSING.md`,
`SPEECH_TO_TEXT.md`, `TEXT_TO_SPEECH.md`, `PERCEPTION_SCHEMA.md`,
`PERCEPTION_API.md`, `LOCAL_MODEL_REQUIREMENTS.md`, `PERCEPTION_TESTS.md`. Each
documents the **actual** supported formats, installed models, and verified
capabilities.

## 16. Known limitations (honest)

* The **vision model** and a **real STT/TTS engine** are not installed on this
  machine; their paths are implemented and gated but were exercised only through
  the availability/refusal paths here, not with weights.
* **Speaker separation/identification** is not implemented and is not claimed.
* Arabic OCR quality depends on the installed Tesseract language data (present
  here: `ara`).
* Image orientation/preprocessing is basic (the OCR engine's own default); no
  deskew model is bundled.
* The PDF engine reads the **text layer** and renders pages for OCR; it does not
  reconstruct complex layout (columns/tables) beyond what the reader reports.

## 17. Acceptance checklist (section 18)

| # | Criterion | Status |
| --- | --- | --- |
| 1 | The perception architecture is modular | ✅ 16 replaceable modules, each a stable API |
| 2 | Supported documents processed locally | ✅ text/MD/CSV/JSON/Word/PDF |
| 3 | PDF text vs scanned workflows distinguishable | ✅ `classify_pdf` → text/scanned/mixed |
| 4 | OCR works for supported formats | ✅ verified on a rendered image |
| 5 | Image processing integrated where feasible | ✅ features + optional local vision model |
| 6 | Local STT/TTS work where compatible, else documented | ⚠️ engines absent; documented with next steps |
| 7 | Provenance preserved | ✅ on every result, document, and evidence item |
| 8 | Integrates with Cognitive Core and Memory | ✅ via `core/perception_bridge.py` |
| 9 | Resource limits respected | ✅ bounded, cancellable, time-limited |
| 10 | Offline runtime verified | ✅ (section 13) |
| 11 | Tests pass; unsupported capabilities documented | ✅ 64 tests + 29 judgements; limitations named |
| 12 | Previous phases remain stable | ✅ pytest 6244 passed / 293 skipped; evaluation PASS |

## 18. Gates

* `python -m pytest -q` — **6244 passed, 293 skipped** (skips are display/optional
  extras, unchanged from baseline).
* `python -m ruff check .` — clean.
* `python -m mypy` — clean (973 files).
* `python -m evaluation` — **PASS**, `perception_bridge_accuracy` 1.0000,
  `overall` 0.9900 (was 0.9898).

## 19. Implemented vs. planned/unsupported

**Implemented and verified here:** document/PDF/OCR/image/audio pipelines, the
normalized schema, limits/cancellation, the validator, provenance, the bridge into
evidence/documents, candidate claims, transcript correction, the CLI, the
held-out gate, and the docs.

**Implemented but not exercised with weights here** (engines absent): the local
vision model description, a real STT transcript, and real TTS audio — each
reachable, each reporting its own real next step.

**Not implemented (and not claimed):** speaker identification, cloud vision,
model training on extracted content, and any mandatory multimodal dependency.

## 20. Stop

Phase 7 is complete. Phase 8 is **not** started; awaiting the user's review and
explicit approval.
