# Romance & Companion Studio — Phase 1 Report (Foundation and Database)

Phase 1 delivers the storage foundation every later phase stands on: the record
model, the derived schema, the versioned transactional database, the audited
domain service, retention/backup/export, and the held-out gate. **No application
behaviour elsewhere changed** — the work is additive (`core/romance/`, one
`core/paths.py` addition, one metric in `evaluation`) and the full suite is green.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/model.py` | The frozen record dataclasses (17 tables) + `RecordBase`, `utc_now`, `make_id`, `RETENTION_*` |
| `core/romance/schema.py` | The schema **derived** from the dataclass annotations; `TableSpec`/`ColumnSpec`; `SCHEMA_VERSION`, `MIGRATIONS`, `TABLES`, `RECORDS`, `build_schema_sql` |
| `core/romance/database.py` | `RomanceDatabase`: versioned SQLite store, transactions, bounded reads, corrupt-row tolerance, backup/restore, `RomanceLoadReport`, `RomanceDatabaseError` |
| `core/romance/repository.py` | `RomanceRepository`: typed CRUD over one table, append-only audit |
| `core/romance/domain.py` | `RomanceDomain`: audited create/update/deactivate/count |
| `core/romance/storage.py` | `CompanionStorageService`: paths, retention, export, backup/restore, `ExportBundle`/`DeleteReport`/`BackupReport`/`RestoreReport` |
| `core/romance/gate.py` | The held-out `companion_studio_accuracy` gate (22 judgements) |
| `core/paths.py` (edit) | `romance_dir()`, `romance_db_path()`, `romance_backups_dir()` |
| `tests/test_romance_storage.py` | 29 tests over the whole foundation |

## Design decisions (and why)

- **Derive the schema from the annotations.** A field's SQL type is read from its
  Python annotation, so there is one place a column's shape lives and the record
  and its table cannot drift. `tuple` → a JSON array, `dict` → a JSON object,
  `X | None` → a nullable column, `bool` → a 0/1 `INTEGER`.
- **Stable string ids, never reused.** A record's identity survives a
  deactivation, so a foreign key that points at it still resolves.
- **Soft delete, not delete.** A record is deactivated (`active=0`); the default
  read hides it. Only an explicit retention choice (`purge`) removes rows, and it
  removes them through the same transaction machinery.
- **An edit is a new value.** `domain.update` writes a fresh `updated_at` and a
  new row value; the caller's original object is untouched. Every write is
  recorded in the append-only `companion_audit_entries`.
- **Fail closed.** A database from a *newer* build is refused rather than read
  with a schema this build does not understand; a backup that fails its integrity
  check is refused and the good database is preserved (as `<name>.pre-restore`).

## Verification

- `python -m pytest tests/test_romance_storage.py -q` — **29 passed**.
- `python -m pytest -q` (full suite) — **7559 passed, 477 skipped** (the skips are
  the pre-existing display/optional-extra skips; the count rose by the new tests).
- `python -m ruff check .` — All checks passed.
- `python -m mypy` — Success, 1156 source files.
- `python -m evaluation` — **PASS**, `companion_studio_accuracy 1.0000`,
  `overall 0.9917 → 0.9918` (no regression).

## Honest limitations / deferred to later phases

- The **records** exist for every domain object, but the *services* that use them
  (character manager, relationship engine, memory manager, model router, voice
  adapter, safety manager, orchestrator) are Phase 2+; Phase 1 ships storage only.
- No window or CLI yet: `gui/romance.py`, `gui/romance_dialog.py`, and
  `cli/romance.py` arrive with the phases that have something to show.
- The personality dials and relationship dimensions are **stored** but their
  update rules (what a conversation does to them) are the Relationship Engine
  (Phase 5) and the Memory System (Phase 6).
- Encryption of the local database is a Phase 9 (privacy) item; the file is
  local-only and git-ignored today.
