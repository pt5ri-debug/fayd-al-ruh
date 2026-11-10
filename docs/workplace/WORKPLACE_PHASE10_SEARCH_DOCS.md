# Workplace — Phase 10: Search, optimization, and documentation (§14, §15)

This phase completes the EPRI/TSTC Workplace module. It delivers the brief's
**advanced search** (§14), the **optimization** the module needed (the project
listing's N+1 reads removed), and the **documentation** the brief's §15 asks every
phase to carry — including the *reuse map* §11 and §16 require.

It adds one module, `core/workplace/search.py`, one held-out gate,
`core/workplace/phase10_gate.py`, and wires both into the service, the CLI, the
menu, and the window. It adds **no** new table, **no** second index, **no** second
document engine, and **no** second RAG — the search is a *view* over the repository
every other Workplace surface already reads (the brief's §11 rule).

## 1. Advanced search (§14)

`search_workplace(repository, query, *, kinds=None, limit=20, min_score=0.0)`
searches every Workplace record kind for a word or phrase and returns ranked hits:

- **one query, every kind** — a term is found in projects, personnel, standards,
  procedures, documents, findings, contracts, correspondence, and the rest, so a
  reader asks *"where does API 1104 appear?"* once instead of opening twenty lists;
- **a hit names what matched** — `SearchHit.matched` lists the fields that carried
  the term, so nothing is fabricated and the reason a row surfaced is visible;
- **the project's own tokenizer** — scoring uses `core.tokenize.content_terms`, so
  an Arabic query and an English one are compared in the same normalised space the
  retriever uses (Phase 23's shared space);
- **a coverage score, not a claim** — `score` is the share of the query's content
  terms a row's own text carries (`0 < score <= 1`); the strongest match ranks
  first, with a stable tie-break on kind, title, and id so the same query always
  returns the same order.

**Honest by construction.** A blank query returns **nothing** (not every row); a
query naming nothing returns no hit; a kind the caller cannot read is skipped; and
the acting user's **clearance** is applied to the hits, so a restricted record is
never surfaced to a reader who may not see it (the same policy Phase 9 installed,
and the same default-clearance fallback the assistant's corpus filter uses, so a
fresh Workplace is searchable before its first user is added).

The engine reads a *declared* field set per kind (`_SEARCH_FIELDS`) and falls back
to a record's own string fields, so a newly added record is searchable without
editing the table.

### Surfaces

- **Service** — `WorkplaceService.search(query, *, kinds, limit, min_score)`,
  `searchable_kinds()`, and `record(kind, record_id)` (the last opens a hit's own
  record).
- **Window** — a new **Advanced search** sidebar section: a query box, a kind
  selector (*all* or one kind), a ranked result list, and **Open the record**
  (which opens a project through its resolved view, or any other record through its
  own fields).
- **CLI** — `python -m cli workplace search "<query>" [--kind K]… [--limit N]
  [--min-score F] [--json]` and `… search-kinds`, plus two menu entries.

## 2. Optimization: the project listing no longer reads a table per project

`core/workplace/projects.py`'s `ProjectManager.rows()` used to count each
project's findings, reports, documents, and members with a query **per project**
(the classic N+1). It now reads each supporting table **once** for the whole
listing and folds the counts in memory, so the listing's cost is independent of the
project count. The counts are unchanged — `rows()` still reports the same real
figures the dashboard shows.

The held-out gate pins both halves: the counts are **correct** *and* each
supporting table is read **exactly once** regardless of how many projects exist, so
the N+1 cannot silently return.

## 3. The held-out gate (`workplace_phase10_accuracy`)

`core/workplace/phase10_gate.py` is a catalogue of nine hand-written judgements,
run against the **real** search engine and repository, each with a negative
control, so a regression fails the evaluation instead of passing silently:

| judgement | what it proves |
| --- | --- |
| `finds-across-kinds` | one query reaches every kind that carries it |
| `blank-returns-nothing` | a blank query returns no rows *(negative control)* |
| `kind-narrows` | a kind filter returns only that kind |
| `strongest-first` | the row carrying the whole query ranks first |
| `unknown-query-empty` | a query naming nothing yields no hit *(negative control)* |
| `no-second-index` | the search adds no table to the schema |
| `clearance-withholds` | a classified record is withheld from an uncleared reader |
| `listing-counts-correct` | the listing's per-project counts are right |
| `listing-is-bounded` | the listing reads each table once (no N+1) |

`workplace_phase10_accuracy()` is the share the shipped behaviour satisfies
(`1.0`), and `0.0` when handed no judgements. It is a metric in
`evaluation/benchmark.py` and `evaluation/runner.py`, and it enters `overall`.

## 4. Documentation and the reuse map (the brief's §11, §16)

The Workplace is built **on** Fayd al-Ruh's existing systems, never beside them:

| the brief asks for | what the Workplace reuses |
| --- | --- |
| document ingestion | `core.document_reader`, `core.docx_extract`, `core.pdf_extract`, `core.pdf_ocr`, and `ingest/` (via the Import Center) |
| knowledge storage | `core.knowledge_base.KnowledgeBaseStore` (sharded JSONL), its `checkpoint.json`, and the SQLite FTS5 index (`core.knowledge_index`) |
| retrieval | `core.rag.RagRetriever` (lexical + semantic + ontology + bulk) and the project's learned `core.embeddings` space |
| the assistant | `core.assistant` / the project's own model manager, reached through `build_default_assistant` |
| the window layer | `gui.sidebar.SidebarColumn`, `gui.components`, `gui.themed` dialog base, `gui.chat_composer`, `gui.rich_text`, and `gui.window_controls` |
| the CLI | `cli.menu` actions and the shared argparse conventions |
| the schema | one SQLite file per data root, with a `SCHEMA_VERSION` and additive migrations (`core/workplace/schema.py`) |

No duplicate ingestion engine, duplicate RAG, duplicate model manager, or parallel
knowledge database is created. The Workplace adds **domain** tables and a **domain
service**, and it reads the shared systems where they already are.

## 5. Phase index

| phase | deliverable | doc |
| --- | --- | --- |
| 1 | repository inspection and architecture | `WORKPLACE_PHASE1_ARCHITECTURE.md` |
| 2 | Workplace navigation and dedicated window | `WORKPLACE_PHASE2_WINDOW.md` |
| 3 | institutional hierarchy and configurable divisions | `WORKPLACE_PHASE3_HIERARCHY.md` |
| 4 | Workplace database and migrations | `WORKPLACE_PHASE4_DATA.md` |
| 5 | project and personnel management | `WORKPLACE_PHASE5_PROJECTS_PERSONNEL.md` |
| 6 | technical library and document management | `WORKPLACE_PHASE6_LIBRARY_DOCUMENTS.md` |
| 7 | Workplace AI assistant and RAG integration | `WORKPLACE_PHASE7_ASSISTANT.md` |
| 8 | technical reporting and proposal workflows | `WORKPLACE_PHASE8_REPORTING.md` |
| 9 | security, permissions, and audit trails | `core/workplace/security.py`, `core/workplace/security_gate.py` |
| 10 | testing, optimization, and documentation | this file |

## 6. Verification

- `python -m pytest tests/test_workplace_phase10.py` — 19 tests over the search
  engine, the listing fix, the gate, and the CLI.
- `python -m evaluation` — PASS, with `workplace_phase10_accuracy` at `1.0000`.
- `python -m ruff check .` and `python -m mypy` — clean.

## 7. Honesty statement

An AI-written Workplace output is **never** an approved engineering decision or an
official EPRI commitment (the brief's §13, enforced by `is_ai_actor`). A search hit
is a real row with the field that matched named; a blank query returns nothing; the
score is a coverage figure, never a claim that a row answers the question. Nothing
here fabricates a record, a standard, or a figure.
