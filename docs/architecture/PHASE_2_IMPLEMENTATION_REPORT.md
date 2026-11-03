# Fayd Al-Ruh — Phase 2 Implementation Report

**Version:** 2.3.0 · **Branch:** `feature/phase-2-memory` · **Date:** 2026-11-03

Phase 2 — **Long-Term Memory Architecture** — makes the project's memory
*durable*, *structured*, *temporally aware*, and *consolidating* across sessions,
without changing a single reasoning result. Every module is **additive**: nothing
existing was deleted, renamed, or weakened, and the durable memory is wired in
*beneath* the layers already there.

This document records what was actually built and measured, so Phase 3 can build
on it without re-deriving it.

---

## 1. What was added

| Module | Kind | Purpose |
| --- | --- | --- |
| `core/memory_record.py` | **new** | The one record contract: `MemoryRecord` carries content, **provenance** (source + reference), **confidence**, **verification status**, **temporal validity** (`valid_from`/`valid_until`), access counts, supersession links, and links to the Cognitive Core (entities, relations, evidence). A correction is a *new* record via `supersede()`, never a mutation. |
| `core/memory_db.py` | **new** | The durable SQLite store over the standard library: **transactional** writes, a **schema version** with a migration path (a newer database is refused), **bounded** paged reads, **corrupt-row tolerance** (skipped and counted, never raised), **backup/restore** through SQLite's own online-backup API with a validated swap, and a term index for keyword search. |
| `core/long_term_memory.py` | **new** | The four-category engine: `WorkingMemoryStore` (bounded, LRU) plus the durable episodic/semantic/procedural records over one database. The full lifecycle — capture → normalise → assign a category → link → persist → index → retrieve → consolidate → archive — with **archive, never delete**. |
| `core/memory_retrieval.py` | **new** | One retrieval interface with an **explainable** ranking: six separate signals (relevance, semantic, recency, frequency, importance, source quality) and a documented weighted sum. Retrieval similarity is **never** reported as factual confidence. No feedback loop; no deletion of a rarely used memory. |
| `core/memory_management.py` | **new** | Inspect, search, correct, supersede, verify, archive, find duplicates, export, back up, and restore — plus `PersonalityMemoryRegistry`, which gives each personality its **own** memory and enforces the project's isolation policy structurally. |
| `core/memory_gate.py` | **new** | The held-out gate (`memory_architecture_accuracy`) over the whole Phase-2 architecture, with negative controls. |

### Extended

- **`core/memory_consolidator.py`** — `consolidate(..., durable=...)` now also
  writes a promoted working item into the durable store as an episodic record, so
  a consolidation survives the process that made it. With no `durable` argument
  the behaviour is exactly as before.
- **`core/memory_retriever.py`** — `retrieve(..., durable=...)` reads the durable
  store's **episodic** records as a fourth source beside the in-process lexical
  and semantic ones. With no `durable` argument the behaviour is exactly as
  before.
- **`core/event_bus.py`** — three new lifecycle events: `memory.stored`,
  `memory.consolidated`, `memory.archived`.
- **`core/cognitive_controller.py`** — accepts an optional
  `long_term_memory=LongTermMemory(...)`; when present it exposes `remember`,
  `recall`, `consolidate_memory`, and `long_term_memory_status`, and publishes the
  memory events. A controller built without one behaves exactly as before.
- **`core/paths.py`** — `memory_root_dir()` (`data/memory`, git-ignored).
- **`evaluation/benchmark.py` / `runner.py` / `baseline.json`** — the new metric
  `memory_architecture_accuracy` enters `overall`.

---

## 2. The four memory categories

| Category | Runtime | Durable | Where |
| --- | --- | --- | --- |
| **Working** | `WorkingMemoryStore` (bounded, LRU) | no (runtime state by design) | in-process |
| **Episodic** | `LongTermMemory` | **yes** | `data/memory/<owner>/memory.sqlite3` |
| **Semantic** | `LongTermMemory` | **yes** | same store, *references* the Cognitive Core graph |
| **Procedural** | `LongTermMemory` | **yes** | same store |

**Semantic memory is linked to the Cognitive Core, never a competing graph.** A
semantic record carries `entities`, `relations`, and `evidence_ids` that
*reference* the Phase-1 knowledge graph; the graph stays canonical. This satisfies
the task's "avoid duplicating existing functionality".

---

## 3. The lifecycle, in code

```
capture  →  normalise (whitespace, content-addressed id)
         →  assign a category (working / episodic / semantic / procedural)
         →  link (entities, relations, evidence, task refs, tags)
         →  persist (one transaction)  →  index (term table)
         →  retrieve (bounded page + term index + optional semantic space)
         →  consolidate (working → episodic, above an importance threshold)
         →  archive (kept, excluded from default reads)  →  expire (valid_until)
```

Nothing is deleted to resolve a conflict. A correction writes a **new** record
and marks the old one `superseded` with a pointer back, so both are kept and the
history is auditable.

---

## 4. Honesty rules, each enforced in code and pinned by a test

1. **A fresh record is never verified.** `capture` always writes
   `verification='unverified'`; only an explicit `MemoryManager.verify` reaches a
   trusted status.
2. **Retrieval similarity is never confidence.** `MemoryHit.score` (a retrieval
   number) and `MemoryHit.confidence` (the record's own) are separate fields; the
   gate's `similarity-is-not-confidence` judgement fails if they ever converge.
3. **Archive, never delete.** `archive`/`expire` set a flag; the record stays and
   is readable with `include_archived=True`. The only destructive operation,
   `reset`, is named as such and requires `confirmed=True`.
4. **A bad backup cannot destroy a good database.** `restore` opens and
   integrity-checks the candidate first, and keeps the current file as
   `<name>.pre-restore`.
5. **A corrupt row is skipped, not raised.** A row whose payload will not parse is
   counted in a `MemoryLoadReport`, never fatal.
6. **A failed batch rolls back.** `save_many` is one transaction.
7. **No feedback loop.** A retrieval bumps a record's access count *once* (the
   bounded frequency signal); nothing else reinforces anything.
8. **Isolation is structural.** `PersonalityMemoryRegistry.read_memory` consults
   the policy **before** it resolves a path, so Natasha's memory is never even
   opened by another reader.

---

## 5. Isolation and the seven personalities

Each personality has its **own** database under `data/memory/<owner>/`. The
registry reuses `core.learning_isolation`, so the rules are the project's own:

- a personality reads its own memory;
- the main model may read a **shareable** personality's memory (Jarvis, Ultron,
  Vision, Banner, Hakeem, Cloud);
- **Natasha's memory is private** — refused to the main model and to every other
  personality;
- a cross-personality read is refused before any path is resolved.

Reading *shared semantic knowledge* is the Phase-1 Cognitive Core, not this store,
so a private experience is never transferred by reading shared knowledge.

---

## 6. Measured results

| Item | Value |
| --- | --- |
| New modules | 6 |
| Extended modules | 6 |
| New tests (`tests/test_long_term_memory.py`) | 58 |
| Held-out gate `memory_architecture_accuracy` | **1.0000** |
| Gate judgements | 12 (with negative controls) |
| `overall` (evaluation) | 0.9880 → **0.9882** |
| `ruff` | All checks passed |
| `mypy` | no issues in 899 source files |

The gate's twelve judgements: persistence across stores, correction keeps
history, archive-not-delete, backup/restore, **bad-backup-refused**, corrupt-row
tolerance, transaction rollback, temporal filtering, explainable ranking,
consolidation promotes-and-skips, **personality isolation**, and
**similarity-is-not-confidence**. The three bolded ones are the negative controls.

---

## 7. Acceptance criteria (from the roadmap) — met

| Criterion | Status |
| --- | --- |
| An episode stored in one process is retrievable in the next | ✅ `test_backup_and_restore_round_trip`, gate `persistence-across-stores` |
| A contradiction is reported, not silently merged | ✅ correction keeps both records (`test_correct_keeps_history`) |
| A corrupt store never crashes a session | ✅ `test_database_skips_a_corrupt_row`, gate `corrupt-row-tolerated` |
| `evaluation` green | ✅ PASS, overall 0.9882, no regression |

---

## 8. What Phase 3 builds on

- The durable store (`LongTermMemory`) is the sink for the learning pipeline's
  records and the cognition layer's patterns and review schedules.
- The event bus already carries `memory.stored`/`memory.consolidated`, so Phase 3
  can observe consolidation without being called.
- The isolation registry already gives each personality a memory; Phase 5 extends
  the same owner vocabulary to datasets, training history, and permissions.
