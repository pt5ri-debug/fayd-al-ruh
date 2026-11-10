# Workplace — Phase 7: the assistant and RAG integration

Phases 2-6 built the Workplace's own records: the institution and its configurable
divisions, the professional taxonomy, projects and personnel, the technical library,
and the document registry. Phase 7 gives the Workplace its **specialised assistant**
(the brief's §10) — and it does so the way the brief's §11 demands: by **reusing**
the systems the project already has, never by building a parallel knowledge system.

## 1. What the brief asked for (§10, §11)

The brief asks for an assistant connected to the Workplace knowledge base that can
review documents, evaluate procedures against requirements, identify gaps, prepare
proposals and correspondence, compare standard revisions, and retrieve historical
project experience — with **eleven selectable profiles**. And §11 is explicit: *do
not create duplicate document ingestion engines, duplicate RAG systems, duplicate
model managers, or parallel knowledge databases.*

Phase 7 therefore adds **one module** — `core/workplace/assistant.py` — that is a
*thin layer* over:

* `core.documents` — the project's citable `Document`/`DocumentStore`;
* `core.rag.RagRetriever` — the project's lexical + semantic + ontology-guided +
  optional bulk retrieval;
* `core.assistant.GroundedAssistant` — the project's grounded, cited answer path,
  with its extractive fallback and its groundedness gate.

Nothing is re-implemented: the Workplace assistant *builds documents from the
Workplace's own records* and hands them to the project's existing assistant.

## 2. The corpus is built from the records, never copied

`build_workplace_documents(repository, *, documents=, clearance=, ...)` turns every
Workplace record into one citable `Document` — a **view** over a row, so exactly one
copy of the facts exists (in the Workplace database):

| record | becomes a document of kind |
| --- | --- |
| `Institution` | `institution` |
| `OrganisationalUnit` | `unit` |
| `Division` (+ its `DivisionDetail` profile) | `division` |
| `Standard` | `standard` |
| `Client` | `client` |
| `Project` | `project` |
| `Contract` | `contract` |
| `Person` (+ their `Qualification`s) | `person` |
| `Procedure` | `procedure` |
| `Inspection` | `inspection` |
| `Test` | `test` |
| `Finding` | `finding` |
| `Report` | `report` |
| `Correspondence` | `correspondence` |
| `KnowledgeRecord` | `knowledge` |
| `Assignment` | `assignment` |
| a registered `WorkplaceDocument` | `document` (its text, read through the project's own reader) |

**Cumulative by construction.** Every build reads the records afresh, so a project
created, a document registered, or a finding recorded since the last build becomes
citable the next time — the institutional knowledge grows with the work. Each
document keeps its own provenance (`source="workplace:<kind>"`, the originating
`record_id` in its `doc_id`, and the owning `project_id`/`division_id` in its
metadata).

**Honest by construction.** A document classified above the caller's `clearance`
(`DEFAULT_CLEARANCE = ("public", "internal")`) is **withheld and counted**, never
silently included; a registered document with no readable text is **skipped and
counted** (`CorpusReport.withheld` / `.unreadable`). `CorpusReport.describe()`
reports the real counts.

## 3. Controlled reuse of the shared knowledge (the brief's §11)

The Workplace's own records are joined to the **project's own corpus** through
`core.documents.build_document_store(extra=...)`, so a Workplace question can also
reach the project's petroleum knowledge (the disciplines, courses, and corpus), and
the Workplace's own records rank beside it. This is the brief's "separate
institutional and project metadata while allowing controlled reuse of shared
knowledge" — one retriever, one index, one answer path.

## 4. The eleven profiles (the brief's §10)

`PROFILES` ships the eleven assistants the brief names. Each `WorkplaceProfile`
carries its key, its English and Arabic name, its specialization, the extra system
instruction, and the divisions/reference bodies it focuses on. A profile's
`system_instruction(base)` lays its own instruction **over** the project's base
instruction, so the honesty rules underneath are never replaced — only narrowed.

| key | assistant |
| --- | --- |
| `petroleum` | Petroleum Engineering Assistant |
| `welding` | Welding Engineering Assistant |
| `coating` | Coating and Corrosion Assistant |
| `civil` | Civil and Concrete Engineering Assistant |
| `electrical` | Electrical Engineering Assistant |
| `qa-qc` | QA/QC and Inspection Assistant |
| `surveillance` | Technical Surveillance Assistant |
| `project-management` | Project Management Assistant |
| `tender` | Tender and Proposal Assistant |
| `contract` | Contract and Correspondence Assistant |
| `research` | Research and Laboratory Assistant |

`profile_for(key)` maps an unknown key to the default (`petroleum`) — never a guess.

## 5. The assistant never decides (the brief's §13)

Every answer carries `DISCLAIMER`:

> This is AI-generated technical guidance for the Workplace. It is not an approved
> engineering decision, a signed inspection result, or an official EPRI/TSTC
> commitment; a qualified engineer must review and approve it.

`WorkplaceAssistant.answer_text(...)` appends it; the CLI prints it; the window
shows it in the transcript. With **no model** wired the assistant still answers —
extractively, from the cited records — because the project's `GroundedAssistant`
falls back to the extractive path.

## 6. Wiring

* `gui.workplace.WorkplaceService` gains `assistant`, `assistant_profiles`,
  `ask_workplace`, `refresh_assistant`, and `assistant_status`, so the window and
  the CLI share **one** assistant over **one** database.
* The Workplace **window** gains an **Assistant** sidebar section: the corpus status,
  the eleven-profile picker, a question box, *Ask the assistant*, and *Rebuild the
  corpus*. The answer is shown in the transcript under the profile's name, with the
  disclaimer.
* The **CLI** (`python -m cli workplace`) gains `profiles`, `ask`, `corpus`, and
  `assistant-status` (all `--json`-capable, shared flags before or after the
  subcommand), mirrored as three menu entries.

## 7. Tests

`tests/test_workplace_phase7.py` (19 tests) exercises the eleven profiles, the
profile fallback and instruction layering, the corpus build (from the records, its
cumulative growth, the shared-knowledge reuse, a registered document becoming
citable, a confidential document withheld, the `CorpusReport` counts), the assistant
answering with citations, the disclaimer, the no-model path, the profile narrowing,
the CLI (profiles, corpus, ask, status, flags before/after), and the honesty rule
that an unanswerable question never cites a record that does not exist.
`tests/test_workplace_window.py` gains the window's assistant block (status line,
eleven-profile picker, ask → answer in the transcript with the disclaimer).

Gates: ruff clean, mypy clean (1114 files), evaluation PASS (`overall` 0.9909
unchanged), the Workplace suite 175 passed / 18 skipped.
