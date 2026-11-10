# Workplace — Phase 5: project and personnel management

Phase 4 gave the Workplace its relational records (projects, people, contracts,
inspections, tests, documents, reports, correspondence, assignments, findings,
approvals, knowledge records) and an audited domain service over them. Phase 5 adds
the two *views* the brief asks for on top of those records, without a second store:

* the **project view** (the brief's §6) — one project assembled with every reference
  resolved and every record that belongs to it gathered; and
* the **personnel directory** (the brief's §8) — the people, their qualifications
  and project history, a competency matrix, and an explainable project-specific
  selection.

Everything is additive and read-mostly: no table changes, no migration, and no
existing behaviour is altered. The write helpers route through the *same* audited
domain service Phase 4 introduced, so a personnel change is recorded in the audit
trail exactly like any other Workplace change.

## 1. The project view (the brief's §6)

`core/workplace/projects.py` adds `ProjectManager`, a read-only view over the
repository the service already holds (it never opens a second database).

* `ProjectManager.resolve(project_id)` returns a **`ProjectSummary`**: the project
  with its client, contractor, consultant, division, and manager resolved to the
  records they point at (or `None` when a reference does not resolve — never a
  fabricated name), plus its contracts, inspections, tests, documents, reports,
  correspondence, assignments, findings, approvals, and knowledge records.
* `ProjectManager.rows(status=, division_id=)` returns light **`ProjectRow`s** for a
  listing, each carrying the project's own real counts (open findings, reports
  awaiting approval, documents, assignments).
* `ProjectManager.documents(project_id, doc_type=)` is the project's own document
  repository, optionally narrowed to one document type.
* `ProjectManager.project_by_number(number)` finds a project by its number.

Three honest sub-views are computed from the project's own rows, so a surface never
displays a made-up figure: `ProjectSummary.open_findings` (an *answered* NCR is
still open), `ProjectSummary.reports_awaiting_approval` (a `draft`/`submitted`
report), and `ProjectSummary.planned_inspections`. `ProjectSummary.counts()` is the
one place those figures are computed.

**Approvals resolve to the project's records too.** An approval names the record it
approved (`subject_id`/`subject_kind`), so `ProjectSummary.approvals` is the
project's own approvals *plus* the ones recorded against its reports, documents, and
findings — the whole approval history, not only the project row's.

An unknown project id yields `None` rather than a guess.

## 2. The personnel directory (the brief's §8)

`core/workplace/personnel.py` adds `PersonnelDirectory` over the same repository,
with the audited domain service supplied for its writes.

* **Directory** — `people(division_id=, discipline=, available_only=)` lists the
  staff; `profile(person_id)` returns a **`PersonProfile`** (the person, their
  qualifications, and the projects they are assigned to); `disciplines()` lists the
  distinct disciplines the records actually carry.
* **Competency matrix** — `competency_matrix(division_id=)` returns a
  **`CompetencyMatrix`**: the columns are the qualification titles at least one
  person holds and each cell is filled when that person holds that qualification.
  `coverage()` counts how many people hold each qualification, so a coverage gap is
  visible rather than assumed. Nothing is invented: a person with no qualification
  is a row of blanks, which is itself the honest picture.
* **Project-specific selection** — `select(skills=, qualifications=,
  division_id=, limit=, available_only=)` ranks the people who can cover the task.
  A person's *holdings* are the words their own record carries — their role, their
  discipline, and the titles of their qualifications — matched case-insensitively as
  whole phrases. The score is the share of the required items the person holds
  (`matched / required`), so a full match is `1.0`; the ranking is by score, then by
  name, so it is deterministic. Each **`Candidate`** names what the person **holds**
  and what they are **missing**, so the choice is explainable and a gap is never
  hidden. With nothing required, everyone scores `1.0` (equally suitable).

Writes (`add_person`, `add_qualification`, `assign`) go through
`WorkplaceDomain`, so they are audited exactly like any other change. A directory
built without a domain service is read-only and refuses a write with a clear error.

## 3. The service, the window, and the CLI

`gui/workplace.WorkplaceService` gains `projects` and `personnel` properties plus
thin delegations — `project_rows`, `project_view`, `project_documents`,
`project_assignments`, `assign_person`, `personnel_people`, `person_view`,
`competency_matrix`, `select_personnel` — so the window and the CLI drive one
engine and can never disagree.

* **The window** (`gui/workplace_dialog.py`) gains a **Projects & personnel**
  sidebar section: the project list (each row its number, title, and status), a
  *Show the project…* action that draws the full view in the transcript, the
  personnel list, and *Show the profile…* / *Select personnel…* / *Competency
  matrix* actions. It is built with the same `SidebarColumn`, `_button`, and
  `_add_message` the rest of the window uses.
* **The CLI** (`cli/workplace.py`) gains `projects`, `project <id|number>`,
  `people`, `person <id>`, `matrix`, and `select --skill … --qualification …`,
  each with `--json`. Exit code `3` (`EXIT_UNAVAILABLE`) is returned for an unknown
  project, matching the CLI's existing convention.

## 4. What was deliberately *not* built

The brief's §11 rule is respected: nothing here re-implements document ingestion,
retrieval, or a knowledge store. `WorkplaceDocument` and `KnowledgeRecord` remain
**registry rows** pointing at the project's own reader and knowledge base; the
project view reads those rows and resolves their metadata, it does not copy a
document. Phase 7 (the Workplace AI assistant and RAG integration) is where the
existing `RagRetriever` will be wired to this corpus — Phase 5 only assembles the
records it will read.

## 5. Tests

`tests/test_workplace_phase5.py` (25 tests) exercises the real service, the real
database, and the real repository with no display: the project view's reference
resolution, its own-record gathering, the open-findings / awaiting-approval counts,
the approval history, the row filters, the document filter, the audited assignment,
the personnel directory's filters, the profile, the competency matrix (and its
empty case), the selection's ranking, determinism, and availability filter, the
read-only refusal, and every CLI command (plain and `--json`, including the
unavailable exit code).

`tests/test_workplace_window.py` gains two window tests (skipped with no display):
the Projects & personnel section lists a real project and person and draws the
selected project's view, and the competency matrix is drawn in the transcript.

`ruff` is clean, `mypy` reports no issues, and `python -m evaluation` is unchanged.
