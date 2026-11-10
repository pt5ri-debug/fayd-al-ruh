# Workplace — Phase 6: the technical library and document management

Phase 4 gave the Workplace its relational records — including a `standards` table
and a `workplace_documents` table — and an audited domain service over them. Phase 6
adds the two *views* the brief asks for on top of those records, without a second
store, a second RAG, or a second document engine:

* the **technical library** (the brief's §7) — the standards and references a
  project works to, with their issuing bodies, revisions, and the four reference
  kinds kept apart; and
* the **document registry** (the brief's §9) — a project's own documents, with their
  revision chains, integrity checks, and text read through the project's *existing*
  reader.

Everything is additive: no table changes, no migration, and no existing behaviour is
altered. Writes route through the *same* audited domain service Phase 4 introduced,
so adding a reference or registering a document is recorded in the audit trail
exactly like any other Workplace change.

## 1. The technical library (the brief's §7)

`core/workplace/library.py` adds `TechnicalLibrary`, a view over the repository the
service already holds.

* `TechnicalLibrary.standards(body=, reference_kind=, query=)` lists the library,
  optionally narrowed by issuing body, by reference kind, or by a free-text query
  (matched as whole phrases against the number, title, applicability, and body).
* `TechnicalLibrary.standard(record_id)`, `.revisions(body=, number=)`,
  `.current(body=, number=)`, and `.compare_revisions(...)` give the revision story:
  revisions are ordered by publication date, the current one is the newest, and
  `compare_revisions` reports exactly which metadata fields changed between two of
  them (`RevisionComparison.changes()` → `(field, before, after)` rows).
* `TechnicalLibrary.bodies()` lists every issuing body with its readable title and
  how many references the library holds of it — the shipped catalogue is included, so
  a body the library holds nothing of is still offered, honestly with `0`.
* `TechnicalLibrary.counts_by_kind()` reports the four reference kinds.

**The four reference kinds are never confused.** `REFERENCE_KINDS` names them — an
*official standard*, an *internal procedure*, a *project specification*, and
*technical guidance* — and `reference_kind_label` renders each in English and Arabic.

**Honest by construction.** The library stores a standard's *metadata* (number,
title, edition, revision, publication date, source, applicability), never the
standard's text: it does **not** reproduce a copyrighted standard, and it never
fabricates a requirement. `source` records where the copy came from, so a reader
knows the provenance.

## 2. The document registry (the brief's §9)

`core/workplace/documents.py` adds `DocumentManager`, again a view over the same
repository. It is deliberately a **registry**, not a document store:

* `DocumentManager.register_file(source, ...)` copies the file into the Workplace's
  own folder (under the git-ignored `data/workplace/documents`), **content-addressed
  by its `blake2b` hash** so the same bytes are stored once, and records a metadata
  row pointing at it.
* `DocumentManager.register_text(text, title=, ...)` registers a document whose text
  is supplied directly (written to a `.txt` file, so every document has a file the
  project's reader can read and no text lives in a second place).
* `DocumentManager.documents(project_id=, doc_type=, query=)`, `.document(id)`, and
  `.history(id)` list and resolve the registry and walk a document's revision chain.
* `DocumentManager.read_text(id)` reads a document through the project's **existing**
  `core.document_reader.read_document` (Word, PDF, text, CSV/JSON, and the optional
  OCR path) — the Workplace never re-implements a reader.
* `DocumentManager.chunks(id)` chunks that text with the project's **existing**
  `core.knowledge_base.chunk_text`, ready for the project's own retrieval.
* `DocumentManager.verify(id)` re-hashes the stored file and reports whether it still
  matches the recorded hash — an integrity check, not an assumption.
* `DocumentManager.find_by_hash(digest)` reports a duplicate: registering a file
  whose content the registry already holds can be seen rather than silently accepted.

**A revision is a new row.** `revision` + `supersedes_id` chain a document's
revisions; `record_revision(id, ...)` records a new row pointing back at the one it
supersedes, so the original is never mutated and `history()` can walk the chain —
the same "a revision is a new record" discipline the learning records use.

**Every document type is labelled.** `DOCUMENT_TYPES` covers the proposal,
inspection report, technical report, method statement, ITP, welding documentation,
coating procedure, PQT, NCR/CAR, correspondence, minutes, contract, and other, and
`document_type_label` renders each in English and Arabic.

## 3. Wiring

`gui/workplace.py`'s `WorkplaceService` exposes both views as thin delegations
(`library`, `document_manager`, and the `library_*`, `standard_*`, `document_*`
methods), so the window and the CLI share one implementation:

* the **window** (`gui/workplace_dialog.py`) gains a **Technical library** sidebar
  section — the standards list with the four-kind counts above it, Show the
  revisions…, Compare revisions…, and Add a reference…; and the document list with
  Register a file…, Read the document, Verify the file, and Revision history…;
* the **CLI** (`cli/workplace.py`) gains `library`, `library-bodies`,
  `library-kinds`, `standard-revisions`, `compare-standards`, `add-standard`,
  `documents`, `document`, `document-history`, `verify-document`, `document-text`,
  and `register-document`, each with `--json`.

Three entries are added to the CLI menu (*the technical library of standards and
references*, *the project document registry*) beside the existing Workplace entries.

## 4. Tests

`tests/test_workplace_phase6.py` (29 tests) pins the reference kinds and the body
catalogue, the library's filters, the four-kind distinction, the revision ordering
and comparison (and the honest "nothing to compare" case), the read-only library
without a domain, the document types, the file/text registration, the reader and
chunker reuse, the filters, the revision chain, the integrity verification (match,
mismatch, and a missing file), the duplicate check, the audit row, and the CLI.

`tests/test_workplace_window.py` gains three window tests (the library list, the
revision comparison, and the document registry with verification), skipped with no
display like the rest of the window suite.
