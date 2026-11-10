# Workplace — Phase 4: the domain data architecture

Phase 2 gave the Workplace its shape (the window, the institution, the ten
divisions) and Phase 3 made the hierarchy and the taxonomy configurable. Phase 4
adds the **work itself**: the relational records the brief's §12 lists, the
append-only audit trail the security phase needs, the dashboard figures the brief's
§5 asks for, and a typed, audited domain service over all of it.

Everything is additive: the institutional hierarchy, the divisional profiles, and
the taxonomy are unchanged, and no existing table is dropped or rewritten.

## 1. The domain records (the brief's §12)

The brief names twenty entities. Eight already existed from Phase 2/3 (institutions,
organisational units, professional divisions, disciplines -- via the taxonomy --
personnel-as-divisional-detail, standards-as-divisional-detail, procedures-as-
divisional-detail, documents-as-divisional-detail). Phase 4 gives the *work* its own
first-class records, in one frozen dataclass each (`core/workplace/records.py`):

| table                 | record              | holds                                                        |
|-----------------------|---------------------|--------------------------------------------------------------|
| `people`              | `Person`            | engineers, chemists, geologists, inspectors, specialists      |
| `qualifications`      | `Qualification`     | certifications and their validity                            |
| `clients`             | `Client`            | clients, contractors, consultants                            |
| `projects`            | `Project`           | number, title, client/contractor/consultant, scope, standards, status |
| `contracts`           | `Contract`          | contract number, value, dates, terms                         |
| `standards`           | `Standard`          | API/ASME/ASTM/ISO/AWS/AMPP/DNV/EN/IEC + internal procedures  |
| `procedures`          | `Procedure`         | method statements and work instructions                      |
| `inspections`         | `Inspection`        | planned/witness inspections                                  |
| `tests`               | `Test`              | individual tests, linked to an inspection                    |
| `workplace_documents` | `WorkplaceDocument` | a **registry** row: path, hash, revision, supersedes         |
| `reports`             | `Report`            | technical reports and their approval state                   |
| `correspondence`      | `Correspondence`    | letters, submittals, technical queries                       |
| `assignments`         | `Assignment`        | a person's role on a project                                 |
| `findings`            | `Finding`           | NCR / CAR / observation / comment                            |
| `approvals`           | `Approval`          | an approval record (a **new** row per decision)              |
| `knowledge_records`   | `KnowledgeRecord`   | a **registry** row into the project's own knowledge store    |
| `audit_entries`       | `AuditEntry`        | the append-only audit trail                                  |

Every record carries the same base fields: a stable `record_id`
(`PRJ-…`, `PER-…`, `STD-…`), `created_at`/`updated_at`, the `active` soft-delete
flag, and a free `metadata` map for fields a later phase adds without a migration.
The ids carry a readable prefix per entity (`DOMAIN_PREFIXES`), so a bare id names
its own kind.

### Two honesty rules the schema enforces

* **A document record is a registry row, never a copy.** `WorkplaceDocument` holds a
  `path`, a `content_hash`, a `revision`, and a `supersedes_id` -- *no document
  text*. The text itself is ingested into the project's existing knowledge base
  (`core.knowledge_base` / `core.knowledge_index` / `core.documents`), which is why
  there is no second document store and no second RAG.
* **A knowledge record is a registry row, never a second corpus.**
  `KnowledgeRecord` points at a `source` and a `reference`; the knowledge lives in
  the same stores the rest of the project reads.

## 2. Migration 3

`schema.py` adds a v2 → v3 migration that creates the sixteen domain tables
(the schema is now **version 3**). The migration is additive: it never touches an
existing row, so a Phase-2 or Phase-3 database upgrades in place with nothing lost.
A database from a *newer* build is still refused up front, and a v1 database
migrates all the way through v2 and v3 (pinned by a test).

The domain tables are declared by a small generic builder, so a new record type is
one dataclass plus one table spec -- no hand-written SQL per entity.

## 3. The audited domain service (`core/workplace/domain.py`)

`WorkplaceDomain` is the typed service over the repository. It offers each entity a
small API -- create / read / list / update / deactivate -- and **appends an audit
row on every write**, so the trail is complete from the first record.

* An edit is a **new value**: `update` applies `dataclasses.replace`, bumps
  `updated_at`, and writes the result. The frozen record is never mutated, and the
  audit row names the fields that changed (`before` / `after`).
* `deactivate` is the **soft delete**: the record's `active` flag is cleared and the
  fact is audited. Nothing is destroyed, so a project that references a deactivated
  division still resolves it -- the brief's "deleting a unit must not destroy
  history" rule, pinned by a test.
* A create validates its foreign keys: a project's client, a qualification's
  person, an assignment's person and project must exist, or the operation is
  refused with a clear reason.
* `approve` records a **new** approval row per decision, so a re-approval never
  overwrites the first.

The service is Tk-free, model-free, and network-free, so the window, the CLI, and
the tests drive the *same* service.

## 4. The dashboard figures (the brief's §5)

`WorkplaceService.dashboard()` returns a `WorkplaceDashboard` whose every figure is
a **real count read from the database at that moment** -- active/completed/total
projects, planned inspections, outstanding findings, reports awaiting approval,
assignments, registered documents, personnel, clients, standards in the library,
knowledge records, and divisions. A figure with no meaning yet is honestly `0`;
nothing is fabricated. The rows are bilingual (`as_rows(language="ar")`).

## 5. The surfaces

* **CLI** (`python -m cli workplace`): `dashboard`, `entities`, `add <entity>
  key=value …`, `list <entity>`, `show <entity> <id>`, `update <entity> <id>
  key=value …`, `deactivate <entity> <id>`, and `audit [--for <id>] [--limit N]`.
  Exit codes follow the project's convention: `0` success, `3` an unknown record,
  `4` a refused operation. New menu entries expose the dashboard, the entity list,
  and the audit trail.
* **Window** (`gui/workplace_dialog.py`): the Dashboard section now shows the new
  domain figures, and a new **Projects & records** section lists an entity's
  records, adds one from a form, and shows the audit trail. The section reuses the
  existing `SidebarColumn`, button, and worker-thread patterns, so it reads as part
  of the same page.

## 6. Tests

`tests/test_workplace_phase4.py` (29 tests) pins the records module, the domain
service (create/read/list/update/deactivate, the foreign-key checks, the revision
chain, the soft delete, the append-only audit), the dashboard figures, the v2 → v3
migration, the newer-database refusal, and the CLI. The window tests in
`tests/test_workplace_window.py` add the dashboard figures, the records list, and
the audit trail (they skip with no display).

## 7. What is deliberately *not* here

* **No second knowledge system.** Documents and knowledge are registry rows; the
  text lives in the project's existing stores (brief §11).
* **No invented institutional data.** Only publicly documented EPRI information is
  seeded; the TSTC divisions remain proposed, configurable defaults.
* **No automatic AI decisions.** The approval records and the audit trail exist so
  the security phase (Phase 9) can build on them; nothing here treats an
  AI-generated recommendation as an approved engineering decision.
