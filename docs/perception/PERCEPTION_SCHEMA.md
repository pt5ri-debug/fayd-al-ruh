# The Perception Result Schema

Every perception component produces one `perception.schema.PerceptionResult`. One
shape for all of them means a consumer (the bridge, Memory, Learning, a window, the
CLI) never has to branch on which reader ran.

## Fields (section 9)

| Field | Meaning |
| --- | --- |
| `result_id` | the perception id (`PER-xxxxxxxx`), unique per result |
| `source_type` | `SourceType` — `text`/`document`/`pdf`/`word`/`spreadsheet`/`image`/`scanned`/`audio`/`speech`/`unknown` |
| `source_reference` | the path, URL, session id, or microphone name |
| `extraction_method` | `ExtractionMethod` — `reader`/`pdf-text`/`pdf-ocr`/`docx`/`ocr`/`image`/`audio`/`speech`/`model`/`metadata`/`none` |
| `content` | the extracted text (what was *perceived*) |
| `content_type` | `ContentType` — `native`/`ocr`/`asr`/`model`/`structured`/`metadata`/`none` |
| `language` | the detected language (`ar`/`en`) |
| `created_at` | a UTC timestamp |
| `segments` | `Segment`s — page/region/segment boundaries with their own text |
| `confidence` | a quality signal in `[0, 1]`, **never** a truth claim |
| `status` | `PerceptionStatus` — `ok`/`partial`/`empty`/`failed`/`unsupported` |
| `errors` | the reasons a read failed |
| `warnings` | non-fatal problems, each with a `where` |
| `model_name`, `model_version` | the engine/model that produced the content |
| `provenance` | source digest, extraction method, content type, engine, parent ids |
| `metadata` | text-free facts (page count, dimensions, duration) |

## `Segment`

A page, region, or time span with its own content: `index` (1-based), `text`,
`label` (`"page 4"`, `"00:01:20"`, `"table row 3"`), `content_type`,
`start_seconds`/`end_seconds` (audio), `confidence` (an OCR/word score when the
engine reports one), and `region` (an image rectangle).

## Invariants (enforced in `__post_init__`)

* `confidence` must be in `[0, 1]`.
* A `Segment`'s index is 1-based; its confidence is in `[0, 1]`.
* **An `OK` result must carry content, segments, or metadata.** An "ok" result
  with nothing in it is a lie — the caller must say `EMPTY` instead. This is why
  an image with only measurable features is `OK` (it has metadata) while a blank
  page is `EMPTY`.

## Extracted content is kept apart from conclusions

`content` is what was *perceived*; it is never a claim the system inferred. A
downstream layer decides what, if anything, the content supports. `content_type`
records *how* it was obtained, so a generated description (`model`) is never
mistaken for ground truth.

## Nothing is born verified

`confidence` is a quality signal about the extraction, and `status` says how
complete it was. No result is ever treated as a verified fact; the bridge stores a
perceived sentence as *evidence* and *candidate* claims that must pass validation.

## Helpers

* `result.ok` — `status in (OK, PARTIAL)`.
* `result.has_content` — content or a text-bearing segment.
* `result.word_count`, `result.page_numbers`, `result.segment(index)`.
* `result.summary()` / `result.describe()` — a one-line / multi-line report.
* `result.to_primitive()` — a JSON-ready mapping (the dataclass is plain data).

## The validator

`perception.validator.validate_result(result)` returns a `ValidationVerdict` with
findings, distinguishing a **blocking** problem (an internal contradiction — an
`OK` with no content, a `FAILED` with no error) from a **warning** (a suspicious
value — a `1.0` confidence on OCR, content with no provenance). The bridge refuses
to publish a result with a blocking finding.

## Provenance

`perception.provenance.PerceptionProvenanceManager` wraps the project's own
`core.provenance.ProvenanceLedger`: `record()`/`attach()` build a readable
`PerceptionProvenance` (source, digest, method, content type, engine, parent ids)
**and** file a `ProvenanceRecord` at the `OBSERVATION` stage, so the lineage is
both readable on the result and auditable in the ledger.
