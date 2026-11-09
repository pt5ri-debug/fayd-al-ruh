# Workplace Module — Phase 1: Repository Inspection & Architecture Assessment

**Project:** Fayd al-Ruh (فيض الروح) — EPRI Workplace Enterprise Module
**Institution:** Egyptian Petroleum Research Institute (EPRI) — Technical Support &
Technology Center (TSTC) *(بترول - صناعة)*
**Arabic name:** معهد بحوث البترول المصري – مركز الدعم الفني والتكنولوجي (بترول - صناعة)
**Phase:** 1 of 10 — inspection and design only. No implementation yet.
**Status:** awaiting architecture approval.

---

## 0. Scope and honesty rules for this module

This module is a **digital workplace** (مكتب رقمي ذكي) for the Technical Support &
Technology Center inside Fayd al-Ruh. It is *not* another chatbot and *not* a
static information page. It must let the user manage technical work, projects,
staff, documents, and client relations, and provide a specialised assistant that
knows how the institute and the centre work.

Three non-negotiable rules govern everything below, taken directly from the
brief and from the project's own conventions (see `AGENTS.md`):

1. **No parallel knowledge system.** The project already has a document store,
   a sharded JSONL knowledge base, an FTS5 index, a RAG retriever, ingestion
   pipelines, and model managers. The Workplace **reuses and extends** these. It
   does not build a second ingestion engine, a second RAG, a second model
   manager, or a parallel knowledge database.
2. **Nothing invented about the institution.** Publicly documented EPRI service
   centres and research areas may be seeded. The ten TSTC divisions are
   **proposed, configurable defaults**, not a claim of an officially approved
   structure. The architecture document says this plainly wherever it matters.
3. **AI output is never an approved engineering decision.** A recommendation,
   a drafted report, or a proposed proposal is a *draft artifact* that a human
   must review. The system records this distinction structurally (see §12).

---

## 1. What already exists and must be reused (repository inspection)

### 1.1 The main Chat Composer and how modes are added

- **`gui/chat_composer.py`** (1170 lines) is the single shared input panel used
  by the main window and every dialog. Public surface relevant here:
  - `ChatComposer(parent, theme=, typography=, placeholder=, on_submit=,
    with_attach=, with_status=, variable=, min_lines=, max_lines=, with_model_bar=)`
  - `add_mode_selector(icon=, label=, options=[(text, callable)], tip=)` → a
    compact pill with a real menu. **This is the exact mechanism the brief
    asks for**: the "Workplace" button must be a mode pill beside "Code Studio".
  - `add_mic(...)`, `add_tool(widget)`, `add_web_toggle(...)`, `model_bar()`.
- **`gui/app.py`** wires the pills at `_composer.add_mode_selector(...)`:
  `_code_mode_selector` (⌘ **Code Studio**), `_training_mode_selector`
  (🎓 Training), `_chat_mode_selector` (💬 model). Each pill opens a dialog
  through a `_open_*` method, e.g. `_open_code_studio()` builds a service and a
  `CodeStudioDialog`, keeps one window instance (`self._code_studio_dialog`),
  re-raises it if already open, and clears the reference on close.
- **Integration point:** add a fourth pill, `icon="🏢"`, `label="Workplace"`,
  immediately after the Code Studio pill, calling a new `_open_workplace()`.

### 1.2 The dialog/window pattern every window follows

Every dialog is built the same way and the Workplace window must follow it:

- `gui/dialog_base.py` — `ThemedDialog` mixin: a dialog calls `_themed_build()`
  instead of `self._build()` and moves post-build work to `_after_build()`, so a
  theme switch or a font change restyles it in place (`retheme`, `retype`).
- `gui/window_controls.py` — `install_window_controls(window, header, ...)`
  gives the title bar minimize/maximize/close; `fit_geometry(...)` sizes it to
  the work area.
- `gui/sidebar.py` — `SidebarColumn(...)` + `SidebarSection(...)` are the one
  collapsible left column (persisted open/closed state per `scope`).
- `gui/components.py` — the shared component library (`Button`/`button`,
  `Section`, `DialogHeader`, `StatusLine`, `EmptyState`, `SearchInput`,
  `SettingsRow`, `WorkerPump`, `FormValidator`, `DataTable`, `Notification…`).
  `tests/test_ui_components_guard.py` fails if a dialog grows its own
  `_button`/`_section` — the Workplace window must use these.
- `gui/rich_text.py` (`insert_markdown`) — Markdown painting.
- Worker-thread + `queue` drained on the UI thread is the mandatory pattern for
  long work (Tk is not thread-safe).
- Reference implementations to imitate: `gui/code_studio_dialog.py` (1788
  lines), `gui/presentation_dialog.py`, `gui/docs_dialog.py`,
  `gui/scene3d_dialog.py`.

### 1.3 The Tk-free service pattern

Every feature pairs a **Tk-free service** (the engine, testable headless) with a
**Tk window** and often a **CLI twin**. Examples: `gui/code_studio.py`,
`gui/presentation.py`, `gui/scene3d.py`, `gui/model_manager.py`,
`gui/training_center.py`. The CLI twin drives the *same* service
(`cli/studio.py`, `cli/presentation.py`, `cli/models.py`). The Workplace will do
the same: `gui/workplace.py` (service) + `gui/workplace_dialog.py` (window) +
`cli/workplace.py` (CLI).

### 1.4 Storage and persistence already available (reuse, do not duplicate)

- **`storage/sqlite_store.py`** — `SqliteStore`: namespaced JSON-value store on
  stdlib `sqlite3`.
- **`storage/json_store.py`** — `JsonStore`: namespaced, atomic JSON documents
  (percent-encoded keys; Windows-safe; legacy fallback).
- **`core/memory_db.py`** — the strongest precedent: a **versioned, transactional
  SQLite database** with `SCHEMA_VERSION`, `migrate()` (numbered steps, refuses a
  newer schema), bounded paged reads, corrupt-row tolerance, and `backup`/
  `restore` via SQLite's online-backup API. **The Workplace relational database
  will be modelled on this file** — same versioning discipline, same backup
  discipline — but as its own module with its own schema (see §5).
- **`core/paths.py`** — the one place writable locations are resolved
  (`data_dir()`, `FAYD_DATA_DIR`-aware, git-ignored `data/`). New Workplace
  folders will be added here (`workplace_dir()`, `workplace_db_path()`,
  `workplace_documents_dir()`).
- **`core/learning_isolation.py`** — the one access-guard pattern for per-owner
  stores; the Workplace's role-based access will follow the same "guard before
  resolving a path" discipline.

### 1.5 Knowledge, documents, retrieval (reuse, do not duplicate)

- **`core/documents.py`** — `Document`, `DocumentStore`, `build_document_store`,
  `build_knowledge_documents`.
- **`core/knowledge_base.py`** — `KnowledgeBaseStore` (sharded JSONL),
  `chunk_text`, `build_knowledge_base`.
- **`core/knowledge_index.py`** — `KnowledgeTextIndex`, the SQLite **FTS5**
  index over the shards (ms queries).
- **`core/rag.py`** — `RagRetriever` (lexical + semantic + ontology + bulk) and
  `BulkCorpusRetriever`; `RetrievedPassage` with citations.
- **`core/assistant.py`** — `GroundedAssistant`, `build_default_assistant(...)`
  (accepts `documents=`, `bulk=`, `learned=`). **The Workplace assistant will be
  a `GroundedAssistant` over a Workplace-scoped `DocumentStore`**, optionally
  fused with the project's shared corpus — one RAG engine, one citation path.
- **`ingest/`** — the streaming ingestion pipeline (`IngestionPipeline`,
  `KnowledgeImporter` in `gui/import_center.py`, `SqlImportService`,
  `ingest.discovery`). The Workplace document intake reuses this; it does not
  write a second reader.
- **`core/document_reader.py` / `core/pdf_extract.py` / `core/docx_extract.py` /
  `core/pdf_ocr.py`** — the format readers (PDF, DOCX, TXT, CSV, JSON, OCR).
  Already supports the formats the brief lists. Reused as-is.
- **`core/relation_extraction.py`** — typed-relation reader; the Workplace's
  "learn a technical fact from a document" path reuses it.
- **`core/character_knowledge.py`** — the precedent for a **per-owner corpus**
  (`character_knowledge_dir`, retrieval, `reference_block`, FTS index, held-out
  gate). The Workplace corpus is the institutional analogue and will mirror this
  design.

### 1.6 Model management and assistant construction (reuse)

- **`core/language_models.py`** — the unified model catalog; **`ai/mouths.py`** —
  the `(mouth, model key) -> assistant` factory; **`core/model_manager.py`** —
  discover/register/load/test; **`ai/characters.py`** — `CharacterMouth`
  (persona + memory + learning + knowledge-corpus injection, transparent
  wrapper). The Workplace assistant profiles (11 of them) are **personas over
  this existing machinery**, not a new model stack.
- **`core/character_config.py` / `core/character_registry.py`** — the precedent
  for a **structured, editable, persisted, validated** configuration with an
  APPLY that snapshots and rolls back. The Workplace's configurable divisions
  and the assistant profiles use the same discipline.

### 1.7 CLI wiring

- **`cli/menu.py`** — `run_*` functions + `MENU`; **`cli/__main__.py`** — the
  `_COMMANDS` dict mapping command → `run_*`. Adding `run_workplace` and a
  `"workplace"` entry is all that is required.
- **`gui/actions.py`** — sidebar `ACTIONS`/`ACTION_GROUPS`; a new action +
  group (or an entry in an existing group) exposes the Workplace from the
  sidebar too. The composer pill is the primary entry, per the brief.

### 1.8 Evaluation gate (the project's hard rule)

`AGENTS.md` requires `python -m evaluation` to print **PASS** with no regression,
and every new capability to carry a **held-out gate** with negative controls, a
`MetricSpec` in `evaluation/benchmark.py`, a value in `evaluation/runner.py`, and
an entry in `evaluation/baseline.json`. The Workplace will ship one
(`workplace_accuracy`) with negative controls (see §11).

---

## 2. Proposed Workplace module structure

One Tk-free domain package plus the standard three surfaces. No second knowledge
system anywhere.

```
core/workplace/
    __init__.py            # public exports; no model, no Tk, no network at import
    model.py               # the relational data model (dataclasses) + identifiers
    schema.py              # SQL DDL, SCHEMA_VERSION, numbered migrations
    database.py            # WorkplaceDatabase over stdlib sqlite3 (mirrors memory_db)
    repository.py          # typed CRUD over the schema (the only SQL in the package)
    institution.py         # EPRI + TSTC hierarchy: institutes, units, divisions
    taxonomy.py            # the professional knowledge taxonomy (§4 of the brief)
    seed.py                # seeded defaults (documented EPRI facts + proposed TSTC
                           #   divisions + taxonomy), all editable, all marked "proposed"
    projects.py            # projects, contracts, assignments, deliverables
    personnel.py           # personnel, qualifications, competency matrices
    standards.py           # technical library: standards, procedures, specs, guidance
    documents.py           # Workplace document registry over core.documents/ingest
    reports.py             # report + proposal + correspondence workflows
    inspections.py         # inspections, tests, findings, NCR/CAR
    dashboard.py           # dashboard aggregation over real records (§5 of brief)
    knowledge.py           # the Workplace corpus (per-owner, FTS, reuse of core.rag)
    assistant.py           # the 11 assistant profiles over the existing assistant
    security.py            # roles, permissions, confidentiality, audit
    audit.py               # append-only Workplace audit trail
    gate.py                # held-out gate: workplace_accuracy (+ negative controls)
    paths.py               # (thin) Workplace path helpers delegating to core.paths

gui/workplace.py           # WorkplaceService (Tk-free) — the one engine both
                           #   the window and the CLI drive
gui/workplace_dialog.py    # the Workplace window (ThemedDialog + SidebarColumn)
cli/workplace.py           # the headless twin (python -m cli workplace …)

docs/workplace/
    WORKPLACE_PHASE1_ARCHITECTURE.md   # this document
    WORKPLACE_DATA_MODEL.md            # the relational model, per table
    WORKPLACE_UI_NAVIGATION.md         # navigation map + RTL/LTR + themes
    WORKPLACE_IMPLEMENTATION_REPORT.md # updated at each later phase
```

**Why a `core/workplace/` package and not loose modules?** The module is large
(≈20 concerns). The project already uses packages for exactly this
(`core/agents/`, `core/planning/`, `core/execution/`, `cognition/`, `ingest/`,
`perception/`, `companion/`, `hakeem/`). A package keeps the surface tidy and
lets each file stay small and testable.

**Why not put it in `core/` as flat modules?** `core/` already has ~500 files;
a bounded subpackage is the established pattern for a domain layer this size.

---

## 3. Integration plan (how Workplace plugs into the existing application)

| Concern | Existing component reused | Workplace's role |
|---|---|---|
| Entry point | `gui/chat_composer.add_mode_selector` | add a **Workplace** pill beside **Code Studio** |
| Window shell | `ThemedDialog`, `SidebarColumn`, `gui/components`, `window_controls` | build `gui/workplace_dialog.py` |
| Service | the `gui/<feature>.py` Tk-free-service pattern | `gui/workplace.py` |
| CLI | `cli/menu.run_*` + `cli/__main__._COMMANDS` | `cli/workplace.py`, `run_workplace` |
| Sidebar | `gui/actions.py` | one visible action + group |
| Documents | `core/documents`, `ingest/`, `gui/import_center` | a Workplace *registry* over them |
| Retrieval | `core/rag.RagRetriever`, `core/knowledge_index` | a Workplace-scoped `DocumentStore` |
| Assistant | `core/assistant.build_default_assistant`, `ai/characters` | 11 profiles as personas |
| Models | `core/language_models`, `ai/mouths`, `core/model_manager` | choose the profile's model |
| Persistence | `core/memory_db` pattern, `storage/sqlite_store` | `WorkplaceDatabase` (own schema) |
| Paths | `core/paths` | new git-ignored Workplace folders |
| Evaluation | `evaluation/benchmark|runner|baseline` | `workplace_accuracy` gate |

**Navigation between chat / Code Studio / Workplace.** All three are *dialogs*
owned by the main window (`gui/app.py` keeps one instance each and re-raises it),
so the main chat session and its state are preserved by construction. The
Workplace window is opened from the composer pill (and the sidebar), exactly as
the Code Studio is; no navigation framework is added.

---

## 4. Database design (summary; full detail in `WORKPLACE_DATA_MODEL.md`)

**Engine:** stdlib `sqlite3`, one file at `data/workplace/workplace.db`
(git-ignored, `FAYD_DATA_DIR`-aware), modelled on `core/memory_db.py`:
transactional writes, `SCHEMA_VERSION` + numbered migrations, a newer schema
refused, bounded reads, corrupt-row tolerance, and `backup`/`restore`.

**Core tables (stable string ids, foreign keys, audit columns):**

- `institutions`, `organisational_units` (self-referential parent → tree),
  `divisions` (the configurable TSTC divisions).
- `disciplines`, `taxonomy_nodes` (the expandable knowledge taxonomy).
- `personnel`, `qualifications`, `personnel_qualifications`, `competencies`.
- `clients`, `contracts`, `projects`, `project_personnel` (assignments),
  `project_milestones`, `project_documents`.
- `standards` (with `source_kind`: official / internal procedure / project spec /
  guidance), `procedures`.
- `inspections`, `tests`, `findings` (incl. NCR/CAR), `approvals`.
- `documents` (registry: hash, revision, confidentiality, path, source),
  `document_revisions`.
- `reports`, `proposals`, `correspondence`, `technical_queries`.
- `knowledge_records` (institutional knowledge, provenance-carrying).
- `users`, `roles`, `role_permissions`, `project_access`, `audit_events`.

**Invariants the brief demands:**

- *Deleting or modifying an organisational unit never destroys historical
  project records.* → projects reference units/divisions by **stable id**;
  deactivation is a **soft** flag (`active=0`), never a row delete; a `RESTRICT`
  foreign-key policy plus an explicit "deactivate" operation replaces deletion.
- *Stable identifiers, foreign keys, audit trails, document hashes, revision
  tracking, migration support.* → every table carries `created_at`/`updated_at`;
  documents carry `content_hash` + `revision`; `audit_events` is append-only;
  migrations are numbered.

---

## 5. UI / navigation design (summary; full detail in `WORKPLACE_UI_NAVIGATION.md`)

- **Shell:** a `Toplevel` dialog like the Code Studio: title bar (window
  controls), a **collapsible institutional sidebar** (`SidebarColumn`) and a
  centred work area. Fully RTL-aware (`gui/layout_direction.py`; `side_for`,
  `default_alignment`) and theme-aware (the eight `gui/themes` palettes).
- **Sidebar sections (collapsible, `SidebarColumn.add`):**
  1. **Dashboard** (KPIs and widgets from real records).
  2. **Institution** (EPRI → units → divisions; add/edit/deactivate divisions).
  3. **Projects** (list, open, milestones, personnel, documents).
  4. **Personnel** (people, qualifications, competency matrix).
  5. **Technical Library** (standards, procedures, specs, guidance).
  6. **Documents** (registry, revisions, import, search).
  7. **Inspections & Tests** (plans, results, findings, NCR/CAR).
  8. **Reports & Proposals** (draft → review → approve workflows).
  9. **Knowledge** (the Workplace corpus: build index, search, learn).
  10. **Assistant** (the 11 profiles; ask; draft; review).
  11. **Settings** (divisions, taxonomy, roles, confidentiality, language).
- **Work area:** tabs/pages driven by the sidebar selection; a **dashboard** of
  widgets (active/completed projects, pending assignments, upcoming inspections,
  outstanding comments, milestones, personnel assignments, reports awaiting
  approval, contractual deadlines, KPIs) and an **advanced search** box.
- **Assistant panel:** a right-hand or bottom panel with a profile selector, a
  question box, and a citation-carrying answer area (the existing
  `ChatComposer` + `gui/rich_text`).
- **RTL/LTR:** one direction preference (reuse `gui/layout_direction`), Arabic
  and English labels (the project's `label`/`label_ar` convention).

---

## 6. Existing components to reuse (explicit list)

Storage/persistence: `storage/sqlite_store.SqliteStore`, `storage/json_store.JsonStore`,
`core/memory_db` (pattern), `core/paths`.

Knowledge/documents/retrieval: `core/documents`, `core/knowledge_base`,
`core/knowledge_index`, `core/rag` (`RagRetriever`, `BulkCorpusRetriever`),
`core/document_reader`, `core/pdf_extract`, `core/docx_extract`, `core/pdf_ocr`,
`core/relation_extraction`, `core/character_knowledge` (pattern),
`ingest/`, `gui/import_center.KnowledgeImporter`, `gui/sql_import.SqlImportService`.

Assistant/models: `core/assistant.build_default_assistant`, `core/grounding`,
`core/answer_*` bridges, `ai/characters.CharacterMouth`, `ai/mouths`,
`core/language_models`, `core/model_manager`, `core/character_config` +
`core/character_registry` (pattern), `core/best_effort`.

UI: `gui/chat_composer`, `gui/dialog_base.ThemedDialog`, `gui/window_controls`,
`gui/sidebar`, `gui/components`, `gui/rich_text`, `gui/themes`,
`gui/typography`, `gui/layout_direction`, `gui/actions`, `gui/service.ChatService`.

CLI: `cli/menu`, `cli/__main__`.

Evaluation: `evaluation/benchmark|runner|baseline`.

---

## 7. New files and components required (explicit list)

Backend: the `core/workplace/` package in §2 (≈17 modules).
UI: `gui/workplace.py`, `gui/workplace_dialog.py`.
CLI: `cli/workplace.py`.
Wiring edits (small, additive): `gui/chat_composer.py` (no change needed — the
pill uses the existing `add_mode_selector`), `gui/app.py` (one pill + one
`_open_workplace`), `gui/actions.py` (one action/group), `cli/menu.py` +
`cli/__main__.py` (one `run_workplace` + one `_COMMANDS` entry),
`core/paths.py` (Workplace folders), `evaluation/benchmark.py` +
`evaluation/runner.py` + `evaluation/baseline.json` (the gate),
`core/version.py`/`pyproject.toml`/`README` (version bump at the end).
Docs: `docs/workplace/*` and an `AGENTS.md` section.
Tests: `tests/test_workplace_*.py` (model/db/migrations/repository, taxonomy,
projects, personnel, standards, documents, inspections, reports, knowledge,
assistant, security/audit, gate, service, CLI, window).

---

## 8. Security and governance design (summary)

- **RBAC:** `roles` + `role_permissions` + `project_access`; every service
  method checks the caller's role *before* touching the database, following
  `core/learning_isolation`'s "guard before resolving a path" discipline.
- **Confidentiality classifications:** each document/knowledge record carries a
  classification; a role may only read up to its clearance.
- **Audit trail:** `audit_events` is append-only (every create/update/approve/
  delete/deactivate and every *refused* access).
- **Approval workflows:** reports, proposals, and correspondence move through
  explicit states (`draft → under_review → approved/rejected`), and the record's
  `approved_by`/`approved_at` are set only by an explicit approval action.
- **AI never auto-approves:** every AI-produced artifact is stored as a *draft*
  with `origin="ai"` and cannot reach `approved` without a human action. This is
  a structural property of the schema, not a convention.
- **Read-only approved records:** an approved record is immutable except through
  a new revision.

---

## 9. Risks and compatibility issues

1. **Scale of the module.** Ten phases is a lot of surface. Mitigation: each
   phase is independently testable, additive, and committed on its own branch
   (per the project's "work on a feature branch" rule), with the four hard gates
   green at every step.
2. **A second database.** Justified and documented (the brief explicitly asks
   for a relational model with foreign keys and migrations). It is *not* a
   second knowledge base: documents still live in the existing store and are
   only *registered* here. The architecture doc records this justification, as
   the brief requires.
3. **Copyrighted standards.** The Technical Library stores **metadata only**
   (number, title, edition, revision, date, source, applicability, status) and
   never reproduces a copyrighted standard. Seeded entries are metadata for
   well-known standards, clearly marked as references, not the standards' text.
4. **Institutional facts.** Only publicly documented EPRI information is seeded;
   the TSTC divisions are marked **proposed/configurable**. A visible banner and
   a `proposed=True` flag make this explicit in the UI and the data.
5. **Tk thread-safety.** Long work (indexing, model calls, imports) must use the
   worker-thread + queue pattern; the window tests run under `xvfb-run` in CI.
6. **Evaluation regression.** A new metric entering `overall` must be 1.0 on
   the shipped wiring, so the gate must be built to pass on real code, with
   negative controls that *lower* it when a guard is removed (proven to
   discriminate, like the project's other gates).
7. **Version drift.** The version lives in three places; the final phase bumps
   all three together (`tests/test_version_consistency.py`).
8. **Arabic/RTL correctness.** Labels and content must be bilingual and RTL-safe
   from the first phase, not retrofitted.

---

## 10. Delivery approach per phase (agreed workflow)

For every phase, per the brief and the user's instruction:

1. Inspect the existing implementation before changing files.
2. Avoid unnecessary refactoring; preserve all existing functionality.
3. Implement with automated tests.
4. Verify database migrations.
5. Verify Arabic and English interfaces.
6. Write a clear implementation report.
7. Commit in logical, reviewable units.

Then, as instructed: **push → open PR → merge automatically** after each phase,
with the four hard gates (`pytest`, `ruff`, `mypy`, `evaluation`) green.

---

## 11. The held-out gate (`workplace_accuracy`) — planned shape

A catalogue of hand-written judgements run against the **real** code, with
**negative controls**, so a regression fails the build:

- a division can be added, renamed, reorganised, and deactivated **without source
  changes**; a deactivated division's historical project records survive;
- deleting/modifying a unit never destroys a project;
- a standard's `source_kind` distinguishes official / internal / project /
  guidance;
- a project's personnel selection honours a required qualification;
- a document revision is append-only and hash-checked;
- an AI-drafted report cannot reach `approved` without a human action;
- a cross-role read of a confidential record is refused and audited;
- a Workplace assistant answer carries citations from the Workplace corpus;
- the migration from v0 → v1 (and later) is applied and reversible in test;
- **negative controls:** an unknown role is refused; a confidential record is
  unreadable to a low-clearance role; an unapproved draft is not exportable.

---

## 12. Awaiting approval

This document is the Phase 1 deliverable. Per the brief (§16), implementation
begins with the phases in order, each followed by push → PR → merge.

**Next step:** Phase 2 — Workplace navigation and dedicated window (the composer
pill, the service, the window shell, and the CLI twin), built on the components
listed in §6 and following the pattern in §1.2/§1.3.
