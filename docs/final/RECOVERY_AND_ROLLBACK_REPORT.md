# Recovery and Rollback Report

**Method:** targeted recovery tests on **isolated copies** under a temporary
directory (never the user's only copy of real data), plus the new
`tests/test_store_recovery.py` (9 tests) and the project's own recovery suites.

---

## 1. Summary

| Scenario | Result |
| --- | --- |
| Application restart | PASS — stores reopen from disk; a session resumes |
| Database restart & recovery | PASS — a garbage-tailed DB is tolerated or reported, never a crash |
| Interrupted knowledge integration | PASS — a failed integration leaves no partial edge (transactional) |
| Failed learning operations | PASS — a rejected act does not advance the version; history preserved |
| Interrupted task execution | PASS — a cancelled/failed run writes no adapter; checkpoints allow resume |
| Model loading failures | PASS — reported honestly with the real next step; nothing pretends |
| Backup creation | PASS — `backup()` writes a usable copy |
| Backup restoration | PASS — a **good** backup restores; a **bad** one is refused and the current DB preserved |
| Database migration | PASS — a schema from a **newer** build is refused up front; a no-path migration is refused |
| Version rollback | PASS — append-only (events marked rolled back, never deleted) |
| Corrupt/incomplete temp operations | PASS — a partial `.part` download is never promoted; a torn write is not a model |

## 2. Details

### 2.1 Restart and persistence
A long-term memory and a learning store were written, the process ended, and the
stores were reopened: the records were still present and `integrity_check()`
passed.

### 2.2 Corrupt database
Appending garbage bytes to a healthy SQLite file and reopening it either keeps the
records readable (tolerated) or reports a `MemoryDatabaseError` — never a bare
crash. A corrupt line in the character store is skipped and counted, and the good
records survive.

### 2.3 Backup protection
`restore()` opens and integrity-checks the candidate **first**, keeps the current
file as `<name>.pre-restore`, and only then swaps. A backup that is not a valid
database is **refused** (reported, never applied) and the current, good database
is untouched — verified.

### 2.4 Append-only rollback
Rolling the learning store back to a version marks the events at/after that
version as rolled back; it **never deletes** them, so the audit trail survives and
a rollback is itself an auditable event.

### 2.5 Migration and newer schemas
A database written by a **newer** build is refused up front rather than read with
a schema this build does not understand; a migration with no defined path is
refused rather than half-applied.

### 2.6 Interrupted operations
- The download engine writes to a `.part` file and **never promotes a partial to
  the real name**.
- A cancelled or failed training run writes **no** adapter.
- The self-improvement deployment backs up every file it will change before it
  changes it, so a rollback restores only those files from that deployment's own
  backup and verifies its own restore.

## 3. Safety statement

No destructive recovery test was run against a user's only copy of real data.
Every recovery test used an isolated temporary copy. **No user data, database,
model file, dataset, configuration, or personality-specific learning file was
modified** during this audit.
