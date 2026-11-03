# Perception Tests

Phase 7 is verified by two layers: a pytest suite over the real modules, and a
held-out evaluation gate. Neither requires an optional engine, loads a model, or
touches production knowledge or memory.

## The pytest suite — `tests/test_perception_phase7.py` (64 tests)

Synthetic files (a hand-built PDF from `tests/pdf_fixtures`, a generated PNG, a
WAV written with the standard library, a hand-built `.docx`) keep the suite
dependency-free; the PDF and OCR extras are skipped when absent.

| Area | What is proven |
| --- | --- |
| **File type detection** | every supported suffix maps to its family; `.doc`/`.rtf` are `UNSUPPORTED`, unknown is `UNKNOWN`; a renamed file is caught by its magic bytes; only a bounded header is read |
| **Integrity** | missing / empty / too-large files are reported with a reason; the digest is stable |
| **Resource limits** | a value above the ceiling is rejected; lowering is allowed; a passed deadline raises; the cancel token is cooperative; the temp workspace is always removed (and can be kept) |
| **Schema** | an `OK` with no content is refused; an out-of-range confidence is refused; a segment index is 1-based; `summary`/`to_primitive` are correct |
| **Validator** | a contradiction is blocking; a good result passes; an OCR confidence above the ceiling is a warning; `perception_status()` reports every engine without raising |
| **Router** | routes by kind; refuses legacy/unknown/empty |
| **Metadata & normalization** | a WAV header is read; image dimensions are read without decoding; normalization reports what changed; truncation cuts on a word boundary |
| **Documents** | text, structured CSV, Word; a missing file is `FAILED`; legacy/unknown are `UNSUPPORTED`; cancellation stops and reports |
| **PDF** | a digital PDF classifies as `text` and preserves page segments; a blank PDF is `EMPTY`/`FAILED` |
| **OCR & images** | availability is reported; features are recorded; a model description is labelled `MODEL` (not native); OCR reads a rendered text image (skipped without the extra) |
| **Audio & speech** | a WAV's metadata; a missing file is unreadable; transcription with an injected engine; silence is refused; `perceive_audio` transcript and metadata-only; transcript correction is traceable and refuses blank; TTS availability and honest output |
| **Provenance** | the manager builds and files a record |
| **Coordinator** | perceives a document, refuses an unknown format, perceives bare text |
| **Bridge** | documents/evidence carry provenance; evidence strength orders native > OCR > model; a failed result is skipped; a retriever indexes the documents; candidate claims carry evidence; ingest records evidence in the graph |
| **Optional-import safety** | importing `perception` loads no heavy engine; every perception engine is optional and reports a next step |

Run it:

```bash
python -m pytest tests/test_perception_phase7.py -q
```

## The held-out gate — `core/perception_gate.py` (29 judgements)

`perception_bridge_accuracy()` runs the **real** detector, pipelines, schema,
validator, and bridge over synthetic files in a temporary directory, scoring a
held-out catalogue (`PERCEPTION_JUDGEMENTS`) wired into `evaluation` as the metric
`perception_bridge_accuracy`.

The catalogue includes **negative controls**, the honesty rules the phase exists
for:

* a model-generated description is labelled `MODEL`, is **not** native text, and
  its evidence strength is **below** a stated fact's;
* an unsupported claim (no evidence) is never stored by the bridge;
* a legacy `.doc` and an unknown suffix are refused, not guessed at;
* a **failed** result is **skipped**, never ingested;
* an `OK` result with nothing in it is refused by the schema;
* a limit above the ceiling is rejected;
* a cancelled run stops and says so;
* importing `perception` loads no heavy engine.

Run the gate:

```bash
python -m evaluation          # prints perception_bridge_accuracy 1.0000
python -m pytest tests/test_perception_phase7.py -q
```

## What the tests do *not* do

* They never write to the project's knowledge base, memory, or learning stores.
* They never require a model, a network, or a display.
* They never modify model weights.
* An optional engine that is absent is **skipped** (PDF/OCR) or exercised through
  an injected seam (STT/TTS/vision), never mocked in a way that hides a real path.
