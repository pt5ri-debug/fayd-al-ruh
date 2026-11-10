# Workplace — Phase 11: Cumulative Institutional Knowledge

The user's most-emphasised requirement is that the institutional knowledge base be
**cumulative**: every completed project, every approved report, and every
documented technical experience should add value to the institution's knowledge,
while the **source** and the **confidentiality** of each piece are preserved.

The Workplace already *held* the pieces — a knowledge registry
(`knowledge_records`), an assistant that cites it, and the finished work itself
(projects, reports, documents, findings, inspections, tests). What was missing was
the *step that never happens*: turning a finished piece of work into a knowledge
record. Phase 11 is that step, and only that step.

## 1. The rule this phase follows (§11): a view, not a second store

The brief's §11 is explicit — *do not create duplicate document ingestion engines,
duplicate RAG systems, duplicate model managers, or parallel knowledge databases
without a documented architectural justification*. Phase 11 therefore **adds no
table, no second index, and no parallel corpus**. A captured record is an ordinary
`knowledge_records` row in the table the Workplace assistant already reads; the
capture is a *view* over the finished work that produces them.

## 2. `core/workplace/knowledge_capture.py`

`KnowledgeCapture(repository, *, domain=None, projects=None)` reads a source
record, decides whether it is **finished**, and — only then — writes a
`KnowledgeRecord` through the audited domain service.

### 2.1 Which sources yield knowledge

| Source kind | Captured when | Knowledge kind |
|---|---|---|
| `project` | status is `completed` or `closed` | `technical_note` |
| `report` | status is `approved` or `issued` | `technical_note` |
| `document` | it carries an approval or an issue date | `technical_note` |
| `finding` | status is `closed` | `lesson` |
| `inspection` | status is `completed` | `best_practice` |
| `test` | it carries a result | `technical_note` |

A source that is **not** finished yields **nothing** — a project in progress, a
draft report, an open NCR — so the capture can never record a half-finished
experience as knowledge. This is the negative control the gate pins.

### 2.2 Provenance and confidentiality are preserved

Every captured record carries:

- `source = "workplace:<source_kind>"` and `reference = <source id>` — the
  **provenance** the user asked for, so a reader can always trace a knowledge
  record back to the exact project, report, or document it came from;
- the **source's own confidentiality** — a `restricted` document yields a
  `restricted` knowledge record, so the assistant's clearance filter
  (`build_workplace_documents`) keeps withholding it from a reader who may not see
  the source. Confidentiality is not widened by capturing.

### 2.3 Idempotent

The `(source, reference)` pair is the **dedup key**. A second capture of the same
source is reported *skipped*, never written again — so re-running a capture (on a
schedule, or after adding one more project) can never double the registry. The key
is computed over *all* knowledge records, including deactivated ones, so a
soft-deleted capture is not silently re-created.

### 2.4 Honest about failure

- an **unknown** source id raises `WorkplaceKnowledgeError` — never an invented
  record;
- a source that cannot be read is recorded in the report's `errors` and never
  aborts a batch;
- a batch that finds nothing to do reports `"Nothing to capture."` rather than a
  false success.

### 2.5 The API

```python
capture.candidate(source_kind, source_id)          # the knowledge a source yields, or None
capture.candidates(source_kind=None)               # every finished source
capture.capture(source_kind, source_id)            # write one (raises when not finished)
capture.capture_all(source_kind=None)              # every finished source (idempotent)
capture.capture_project(project_id)                # a project and its own finished work
capture.knowledge(project_id=None)                 # the captured records, newest first
capture.summary()                                  # counts by source kind
capture.describe()                                 # an honest one-line status
```

`CaptureReport` carries what a run did: `captured`, `skipped`, `refused`,
`errors`, `count`, `describe()`, and `as_dict()`.

## 3. Wiring

- **Service** (`gui/workplace.WorkplaceService`): `knowledge_capture`,
  `knowledge_candidates`, `capture_knowledge`, `capture_all_knowledge`,
  `capture_project_knowledge`, `captured_knowledge`, `knowledge_capture_status`,
  `knowledge_source_kinds`.
- **CLI** (`python -m cli workplace …`): `knowledge-sources`,
  `knowledge-candidates [--kind]`, `knowledge-status`, `knowledge [--project]`,
  `capture-knowledge <kind> <id>`, `capture-all-knowledge [--kind]`,
  `capture-project-knowledge <project>`. All accept `--json`. A refusal exits `4`.
- **Menu** (`cli/menu.py`): four entries (status, candidates, capture-all, list).
- **Window** (`gui/workplace_dialog.py`): a **Cumulative knowledge** sidebar
  section — an honest status line and four buttons (Show what can be captured,
  Capture everything finished, Capture the selected project's work, List captured
  knowledge).

## 4. Held-out gate

`core/workplace/phase11_gate.py` runs twelve hand-written judgements against the
**real** capture engine and repository, each with its own negative control:

- a completed project / approved report / closed NCR is captured;
- an in-progress project / draft report / open NCR yields nothing (negative
  controls);
- a second capture is skipped, never duplicated;
- the record preserves its source and inherits the source's confidentiality;
- an unknown source id is refused;
- the capture writes to the existing `knowledge_records` table (no second store);
- `capture_project` gathers a project's own finished work.

`workplace_phase11_accuracy` is wired into `evaluation` as a metric that enters
`overall`. It is `0.0` with no judgements rather than a faked pass.

## 5. Cumulative knowledge, end to end

The chain the user asked for now closes:

1. work is done and recorded in the Workplace (a project, a report, a finding);
2. the work reaches a **finished** state (the project completed, the report
   approved, the NCR closed);
3. `capture-all-knowledge` turns that finished work into knowledge records,
   carrying their provenance and confidentiality;
4. `corpus` rebuilds the assistant's corpus, and the new knowledge becomes
   **citable** — the Workplace assistant can now answer from what the institution
   learned.

Each new project, approved document, and documented experience therefore *adds* to
the institutional knowledge, and the source and confidentiality of every piece are
kept.

## 6. Tests

`tests/test_workplace_phase11.py` (32 tests) pins the engine (each source kind,
the eligibility rules and their negatives, provenance, inherited confidentiality,
idempotence, the unknown-id refusal, bulk and per-project capture, the summary and
status), the cumulative wiring (a captured record enters the assistant corpus), the
gate (green, zero with no judgements, the negative controls present), and the CLI.
`tests/test_workplace_window.py` gained three window tests for the new sidebar
block.
