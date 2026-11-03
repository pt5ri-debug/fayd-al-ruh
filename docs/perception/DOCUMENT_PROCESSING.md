# Document Processing

`perception.documents` and `perception.detector` read the text-bearing formats the
project supports into the one normalized result shape. It is a **thin dispatcher**
over the readers that already exist — it reimplements none of them.

## Supported formats

| Format | Suffixes | Reader reused | Structure preserved |
| --- | --- | --- | --- |
| Plain text | `.txt`, `.md`, `.markdown` | `core.document_reader.read_document` | line/paragraph blocks |
| Structured | `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson` | `ingest` `FileConnector` (streaming) | labelled records |
| Word | `.docx`, `.docm` | `core.document_reader` → `core.docx_extract` | paragraphs + table rows |
| PDF | `.pdf` | `perception.pdf` → `core.pdf_extract`/`core.pdf_ocr` | page segments |
| Legacy | `.doc`, `.rtf` | *refused* | — |

`.doc`/`.rtf` are **refused** with the next step ("save it as .docx (Word) or
.pdf"), never scrambled into replacement characters.

## Detection (`perception/detector.py`)

* `detect_kind(path)` maps a suffix to a `SourceKind`
  (`TEXT`/`WORD`/`PDF`/`IMAGE`/`AUDIO`/`UNSUPPORTED`/`UNKNOWN`). The document
  suffixes come from `core.docs_studio.SUPPORTED_SUFFIXES`, so the detector can
  never claim to read a format no reader accepts.
* `detect_file(path, sniff=True)` adds a bounded **magic-byte sniff** (≤ 16 bytes)
  that catches a renamed file — a `.txt` that is really a PDF is reported
  (`"bytes look like pdf"`) while still routed by its suffix.
* `validate_integrity(path, max_bytes=...)` reports *why* a file cannot be read:
  missing, not a regular file, empty, or over the byte budget. It computes a
  streamed SHA-256 digest by default (never holding the file whole).

## Encoding handling

Text and Markdown are read with `core.document_reader`'s **safe encoding
detection** (BOM → strict UTF-8 → a short list of legacy encodings). A
mis-decoded file **raises** rather than filling with `U+FFFD`, so a decode failure
is explicit and never silently corrupts a large amount of text.

## Large files and incremental reads

* A large text/Markdown file is streamed line by line up to
  `core.document_reader.MAX_TEXT_CHARS`; `DocumentText.truncated` reports a capped
  read.
* Perception adds its own `PerceptionLimits.max_content_chars` cap and reports the
  truncation as a `PerceptionWarning` (`"content capped at the character limit"`).
* The router validates the file's size against `max_source_bytes` **before** the
  read, so a too-large source is refused up front rather than loaded and rejected.

## Failure reporting

Every failure is a result, not an exception:

| Situation | Status | Where the reason is |
| --- | --- | --- |
| Missing / unreadable file | `FAILED` | `errors[0]` |
| Legacy `.doc`/`.rtf` | `UNSUPPORTED` | `errors[0]` names the next step |
| Unknown suffix | `UNSUPPORTED` | `errors[0]` |
| Blank document | `EMPTY` | — |
| Capped / skipped content | `PARTIAL` | `warnings` |

A page or block that could not be read is **named in the warnings**, never
dropped. `PerceptionResult.page_numbers` and the segment labels (`"page 4"`,
`"paragraph 3"`) keep the boundaries a citation needs.

## Source references and provenance

Each result carries `source_reference` (the path), a `source_digest` in
`provenance`, the `extraction_method`, and the `content_type` — so an import can
record exactly which bytes were read and how.

## No execution

A document is parsed into text and nothing else. A macro, an embedded script, or
an instruction inside the text is **inert**; nothing in the pipeline executes it.
