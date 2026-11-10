# Workplace — Phase 8: Technical reporting and proposal workflows

This phase delivers §9 (technical documentation management) and §10 (the reporting
half of the Workplace assistant) of the EPRI/TSTC brief, and enforces the §13 rule
that **an AI recommendation is never an approved engineering decision**.

It adds one module, `core/workplace/reporting.py`, and wires it into the service,
the CLI, the menu, and the window. It builds on the rows Phase 4 gave every report,
document, and approval, and on the project view Phase 5 assembled; it does **not**
add a second document store, a second RAG, or a second knowledge database (the
brief's §11 rule).

## What it does

### 1. Drafting from the project's own records

`ReportingWorkflow.draft(project_id, kind=…, use_assistant=…)` assembles a report
from the records the Workplace already holds:

- the project's overview (client, contractor, consultant, division, manager,
  location, status);
- the scope of work (the project's and the division's);
- the applicable standards (the ITP references, the test standards, and the
  division's own standard profile items);
- the inspections and their results;
- the tests and measurements, with their values and units;
- the findings and nonconformities, with their severity and status;
- the assigned personnel;
- the reference documents (the project's own registry);
- the deliverables;
- the contractual basis (the project's contracts).

The report kinds are `technical_report`, `inspection_report`, `progress_report`,
`management_summary`, `method_statement`, `proposal`, and `correspondence`
(`REPORT_KINDS`), and each kind has its own ordered section list.

**Nothing is invented.** A section with nothing to say reports `(nothing on file)`
rather than padding itself, and the counts (open findings, reports awaiting
approval) are the same honest figures the dashboard shows.

With `use_assistant=True` each section is filled with the Workplace assistant's
**cited** answer and marked `ai=True`; the draft carries the assistant's
`DISCLAIMER` and its citations. An assistant failure never breaks a draft: the
section keeps its deterministic lines. With no assistant the draft is assembled
deterministically and is byte-for-byte reproducible.

### 2. The report lifecycle

A report moves through a small, closed state machine (`REPORT_TRANSITIONS`):

```
draft ──► submitted ──► approved ──► issued ──► superseded
  ▲            │            
  └──── rejected ◄────────┘
```

`can_transition(current, target)` is the one rule. The important invariant: a
report can reach **`issued` only through `approved`**, so an issued deliverable
always carries a human approval behind it.

### 3. Approval is a person's decision, never the AI's

`ReportingWorkflow.approve(report_id, approved_by=…)`:

- refuses a blank `approved_by` (`An approval must name the person who approved it`);
- records the approval as a **new** `Approval` row through the audited domain
  service (a re-approval never overwrites the first);
- moves the report to `approved`.

`issue(report_id)` is refused unless the report is `approved`, so the §13 rule is
structural: the AI cannot issue a deliverable on its own.

### 4. The deliverable reuses the Phase-6 registry

`issue(...)` registers the report's body as a document through the Workplace's
**own** `DocumentManager` (`register_text`), so the deliverable lives in the same
registry every other Workplace document lives in — no second store.

## Wiring

- **Service** (`gui/workplace.py`): `reporting` (lazy), `report_kinds`,
  `report_statuses`, `draft_report`, `save_report_draft`, `create_report`,
  `project_reports`, `report_record`, `submit_report`, `reject_report`,
  `approve_report`, `issue_report`, `supersede_report`, `report_approvals`,
  `project_deliverables`, `report_can_transition`.
- **CLI** (`cli/workplace.py`): `report-kinds`, `report-statuses`,
  `draft-report`, `reports`, `report`, `submit-report`, `reject-report`,
  `approve-report`, `issue-report`, `supersede-report`, `report-approvals`,
  `deliverables` (all `--json`-capable).
- **Menu** (`cli/menu.py`): three new entries.
- **Window** (`gui/workplace_dialog.py`): a *Reports & proposals* sidebar section
  with a report list and the five actions (draft, show, submit, approve, issue);
  the list refreshes when the selected project changes.

## Honesty and governance

- Drafting reads; creating, submitting, approving, rejecting, and issuing write
  through the **same** audited domain service every other Workplace write uses.
- The AI's contribution is always marked (`ai=True`, `[AI advice — not an approved
  decision]`), and an AI section is never presented as an approval.
- A read-only workflow (no domain) refuses every write rather than pretending.
- An unknown project, report, or kind is refused with the real reason.

## Tests

`tests/test_workplace_phase8.py` (33 tests) exercises the vocabulary, the draft
(every section from the real records, the empty-section honesty, the determinism,
the assistant advice path and its failure fallback), the lifecycle (every legal
and illegal transition, the approval row, the blank-approver refusal, the
re-issue-after-rejection), issuing (the registered deliverable, the skip-file
option, the deliverables list, the supersession), and the CLI (kinds, statuses,
draft-and-save, the issue refusal, the approvals, the show).

`tests/test_workplace_window.py` gains two window tests (the reports block drafts
and lists; issuing is refused until a person approves).
