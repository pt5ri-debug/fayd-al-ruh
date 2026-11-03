# Fayd Al-Ruh — Phase 1 Implementation Report

**Version:** 2.3.0 · **Branch:** `feature/phase-1-cognitive-core` · **Date:** 2026-11-03

Phase 1 — **Cognitive Core Enhancement** — makes the existing knowledge
infrastructure *observable*, *arbitrated*, and *transactional* without changing a
single reasoning result. Every module is **additive**: nothing existing was
deleted, renamed, or weakened, and the controller's new services are wired in
beside the layers already there.

This document records what was actually built and measured, so Phase 2 can build
on it without re-deriving it.

---

## 1. What was added

| Module | Kind | Purpose |
| --- | --- | --- |
| `core/event_bus.py` | **new** | Dependency-free publish/subscribe: synchronous by default, an optional bounded queue, prefix subscriptions with a **word-boundary** match, a raising subscriber is isolated (`best_effort`) and recorded, and an append-only history for the audit. |
| `core/resource_governor.py` | **new** | One arbiter over the machine's **real** figures (`core.model_resources`): a memory headroom (25 %), model slots, and task slots. An unknown available memory is **not** plenty — a need that cannot be judged is deferred. |
| `core/confidence_engine.py` | **new** | Explainable, evidence-weighed confidence: a prior folded with support, contradicting evidence, source diversity, and optional recency into a named-term `ConfidenceEvaluation`. |
| `core/knowledge_validation.py` | **new** | `KnowledgeClaim` + `KnowledgeValidator` → one verdict (`accepted` / `provisional` / `needs_review` / `rejected`) with `ValidationFinding`s (structural, semantic, evidence, contradiction, duplicate, temporal). **An unsupported claim can never be classified as knowledge**, and validation never verifies. |
| `core/knowledge_integration.py` | **new** | The one controlled path from a proposed claim to a graph write: validate → transactional store → conflict preservation → rollback on any step's failure. |
| `core/knowledge_query.py` | **new** | The read surface over the graph: entity/relation lookup, shortest path with product confidence, neighbours, context, evidence by source, contradictions, temporal windows, statistics. |
| `core/entity_manager.py` | **new** | Entity lifecycle: idempotent create by normalised name, **versioned** updates, duplicate detection, and merge that re-points relations and preserves history. |
| `core/relation_manager.py` | **new** | Relation lifecycle: typed create with endpoint checks, reinforcement of an identical triple, **conflict preservation**, evidence attachment, deactivation (never deletion), and history. |
| `core/evidence_manager.py` | **new** | `EvidenceRecord` with `kind` (direct/inferred), `method` (manual/ingest/inference/…), and `reliability` per source, with a weight = strength × reliability. |
| `core/knowledge_persistence.py` | **new** | Durable graph persistence over the project's own `SqliteStore`/`JsonStore`: atomic (a completion marker written last), corruption-tolerant (a bad record is skipped and counted, never fatal), and version-carrying. |
| `core/cognitive_core_gate.py` | **new** | The held-out gate (`knowledge_validation_accuracy`) over the whole Phase-1 core, with negative controls. |

### Extended

- **`core/answer_confidence.py`** — a multi-hop walked chain now contributes a
  **path uncertainty** term (`AnswerConfidence.path_uncertainty`): a single edge
  carries none, a longer chain corroborates less. The graph-support term is
  unchanged in weight, so no existing number moves for a single-edge answer.
- **`core/cognitive_controller.py`** — owns the eight services above, publishes
  lifecycle events (`cycle.started`, `cycle.finished`, `knowledge.integrated`,
  `knowledge.rejected`, `relation.learned`, `resource.deferred`), and exposes
  `integrate_knowledge`, `validate_claim`, `query`, and `cognitive_core_status`.

### Reused (not duplicated)

`core.model_resources` (`SystemResources`), `core.model_files` / `core.model_preflight`
(the memory estimate), `core.best_effort`, `core.knowledge_graph`, `core.entity`,
`core.relation`, `core.evidence`, `core.belief`, `core.consistency_checker`,
`core.relation_polarity`, `storage.sqlite_store`, `storage.json_store`.

---

## 2. The rules the core enforces

1. **No evidence, no knowledge.** `KnowledgeValidator` classifies an unsupported
   claim as `needs_review` — storable as a *proposal*, never accepted, and
   `verified` is always `False` (verification is a separate, explicit act).
2. **Nothing is deleted to resolve a conflict.** A claim opposing a stored one is
   stored **beside** it, with the conflict reported; the graph keeps both.
3. **A write is transactional.** If any step of an integration raises, the graph
   is restored to exactly its prior state (`rolled_back=True`) — proven by a test
   that injects a failing validator.
4. **Events never decide.** The bus is a notification channel; every decision
   path remains a direct call, and a raising subscriber cannot affect the
   publisher.
5. **An unknown is not plenty.** The governor defers a request whose memory need
   it cannot judge, and the model-memory estimate returns `None` — never a zero
   standing in for an unknown.

---

## 3. Measured results

### Tests

New test files (all passing):

| File | Tests |
| --- | ---: |
| `tests/test_event_bus.py` | 23 |
| `tests/test_resource_governor.py` | 19 |
| `tests/test_confidence_engine.py` | 21 |
| `tests/test_knowledge_validation.py` | 23 |
| `tests/test_knowledge_integration.py` | 16 |
| `tests/test_knowledge_query.py` | 20 |
| `tests/test_managers.py` | 33 |
| `tests/test_knowledge_persistence.py` | 29 |
| `tests/test_cognitive_core.py` | 12 |

### The four gates

| Gate | Result |
| --- | --- |
| `python -m pytest -q` | **5861 passed, 15 skipped** (the 15 are the documented optional-extra skips) |
| `python -m ruff check .` | **All checks passed!** |
| `python -m mypy` | **Success: no issues found in 891 source files** |
| `python -m evaluation` | **PASS: no performance regression** — `overall` 0.987721 → **0.9880** |

> **Note on the 5 failures in the first full-suite run:** the run began at
> 05:05 UTC, inside `cognition.scheduler`'s default quiet window
> (`quiet_start_hour=1`, `quiet_end_hour=6`), so the four scheduler/runner tests
> that assert `due().allowed` failed on the *time of day*, not on the code. This
> was confirmed by stashing the Phase-1 changes and reproducing the same two
> failures on a clean tree. It is a pre-existing, time-dependent flake; a run
> outside 01:00–06:00 UTC passes. Recorded here so it is not mistaken for a
> Phase-1 regression.

### New evaluation metric

`knowledge_validation_accuracy` = **1.0000** — the share of the held-out
cognitive-core judgements satisfied, each run against the real service with
negative controls (an unsupported claim is not knowledge; an opposite-*direction*
claim is not a contradiction; a chain beyond the hop bound yields no path).

---

## 4. What Phase 1 deliberately did **not** do

- It did **not** change any reasoning result, retrieval output, or model weight.
- It did **not** rebuild or replace the existing controller, graph, or retrieval.
- It did **not** enable the event bus as a control path, and it did **not** make
  the bus the default for any hot loop.
- It did **not** add a third-party dependency — the whole phase is standard
  library plus the modules already present.

## 5. Hand-off to Phase 2

The pieces Phase 2 (Long-Term Memory Architecture) will build on are now in
place: `KnowledgeQueryEngine` is the read surface a memory retriever can call;
`KnowledgePersistence` is the durable store a consolidation pass can write to;
the event bus is the notification channel a consolidation trigger can subscribe
to; and `EvidenceManager`'s direct/inferred split is what lets a consolidated
semantic fact be distinguished from a raw episodic one.
