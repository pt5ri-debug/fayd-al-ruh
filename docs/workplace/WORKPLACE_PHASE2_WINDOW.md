# Workplace — Phase 2: navigation, dedicated window, institution & divisions

Phase 2 of the EPRI/TSTC Workplace module gives the module its **shape**: the
`Workplace` control in the chat composer, a dedicated window, the institutional
hierarchy, and the ten *configurable* divisions — all built on the project's own
conventions and reusing its existing components. No second knowledge base, no
second document engine, no second assistant: the Workplace is a **domain module**
over the services Fayd al-Ruh already has.

## 1. What Phase 2 delivers

| Requirement (§2, §3, §4, §5) | Delivered in Phase 2 |
| --- | --- |
| A **Workplace** control beside **Code Studio** in the composer | ✅ `gui/app.py` — a second mode pill (`🏢 Workplace`) right after `⌘ Code Studio`, opening the dedicated window |
| A dedicated Workplace window | ✅ `gui/workplace_dialog.py` (`WorkplaceDialog`), built like every other window |
| Preserve the main chat session | ✅ the window is a separate `Toplevel`; the chat session is untouched |
| Navigate between chat, Code Studio, and Workplace | ✅ the composer pills and the sidebar entries all open their own window |
| RTL Arabic / LTR English labels | ✅ every label carries an English and an Arabic form; the sidebar is the shared `SidebarColumn` |
| Institutional hierarchy — **EPRI** (official facts only) | ✅ `core/workplace/seed.py` seeds EPRI with `official=True` and a `source` note |
| Institutional hierarchy — **TSTC** and ten **configurable** divisions | ✅ seeded `proposed=True`, add/rename/move/deactivate with no source change |
| The professional knowledge **taxonomy** (§4) | ✅ `core/workplace/taxonomy.py` — 8 areas, 52 disciplines, bilingual |
| A dashboard of **real** figures (§5) | ✅ the window's dashboard reads live counts from the database |

## 2. Reused components (no parallel systems)

Phase 2 deliberately reuses the project's existing machinery rather than
duplicating it:

* **Window shell** — `gui.dialog_base.ThemedDialog` (live re-theme/re-type),
  `gui.components.DialogHeader` (title band + window controls + spinner),
  `gui.components.button`, `gui.sidebar.SidebarColumn` (the collapsible
  institutional sidebar), `gui.app.RoundedBox`, `gui.clipboard`, and
  `gui.typography`.
* **Paths** — `core.paths` gained `workplace_dir` / `workplace_db_path` /
  `workplace_documents_dir` / `workplace_backups_dir`, all under the git-ignored
  `data/workplace`, exactly like every other module's data folder.
* **CLI plumbing** — `cli/menu.run_workplace`, `cli/__main__` dispatch, and the
  menu entries, following the Code Studio CLI's "flags before or after the
  subcommand" rule.
* **Sidebar wiring** — a visible `workplace` action in the *Knowledge & Data*
  group (`gui/actions.py`), opened by `FaydGUI._open_workplace`.

Later phases will reuse the project's own document ingestion
(`core.knowledge_base`, `ingest/`), FTS index (`core.knowledge_index`), retrieval
(`core.rag`), and assistant construction (`core.assistant`) for the Workplace
knowledge base and the EPRI assistant — **not** new engines.

## 3. New files

| File | Purpose |
| --- | --- |
| `core/workplace/__init__.py` | The package's public surface. |
| `core/workplace/model.py` | Frozen records (`Institution`, `OrganisationalUnit`, `Division`) + stable id minting. |
| `core/workplace/schema.py` | Declarative table specs, the version, and the migration list. |
| `core/workplace/database.py` | The SQLite store: migrate, upsert, list, backup/restore, integrity. |
| `core/workplace/repository.py` | Typed access over the store. |
| `core/workplace/taxonomy.py` | The professional knowledge taxonomy (declarative). |
| `core/workplace/seed.py` | The EPRI/TSTC defaults, with the honesty rules. |
| `gui/workplace.py` | The Tk-free `WorkplaceService` the window and CLI share. |
| `gui/workplace_dialog.py` | The Workplace window. |
| `cli/workplace.py` | The headless CLI twin. |
| `tests/test_workplace_core.py` | Model, database, migrations, repository, seed, taxonomy. |
| `tests/test_workplace_service.py` | The service and the CLI. |
| `tests/test_workplace_window.py` | The real Tk window (skips with no display). |

## 4. Data design (Phase 4 foundation, delivered early)

The relational store is standard-library `sqlite3` only, with:

* **Stable identifiers** — `INST-…`, `UNIT-…`, `DIV-…`, minted once and never reused.
* **Foreign keys**, enforced (`PRAGMA foreign_keys = ON`), with `ON DELETE RESTRICT`
  so a unit cannot be deleted while divisions reference it.
* **Soft delete** — deactivating a division sets `active = 0`; the row survives,
  so historical project records that reference it stay intact (§12).
* **Audit columns** — `created_at` / `updated_at` on every row.
* **Schema version + migrations** — a database from a newer build is refused up
  front rather than read with a schema this build does not understand.
* **Backup/restore** — through SQLite's own online-backup API, integrity-checked
  before a restore is applied, with the previous file preserved.

## 5. Honesty rules (§3, §7, §13)

* EPRI's record is marked **official**; its internal tree is **not** invented.
* TSTC and its ten divisions ship as **proposed, configurable defaults** — the
  window says so from the first moment, and every record carries a `source` note.
* Nothing here is a claim of an officially approved structure.

## 6. Verification

* `python -m pytest tests/test_workplace_core.py tests/test_workplace_service.py` — green.
* `python -m pytest tests/test_workplace_window.py` — green under a display
  (skips with no display, like every other window test).
* `python -m ruff check .` — clean.
* `python -m mypy` — clean.
* `python -m cli workplace status|hierarchy|divisions|taxonomy` — verified.

## 7. What is deferred (by design)

Per §16, Phase 2 is the navigation/window/hierarchy phase. Projects, personnel,
the technical library, document management, the Workplace assistant, reporting,
and the security/permissions layer are the later phases (5–10), each of which
will reuse the project's existing services as §11 requires.
