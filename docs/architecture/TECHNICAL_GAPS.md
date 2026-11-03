# Fayd Al-Ruh — Technical Gaps (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

Each gap states what is missing, the **measured** evidence, the impact, and
whether it needs a **new component** or an **extension** of an existing one. The
phases that close them are in `IMPLEMENTATION_ROADMAP.md`.

Priority: **P0** blocks the target architecture · **P1** high value · **P2**
improvement · **P3** nice-to-have.

---

## 1. The neural path

### G-1 · The in-project language model is functionally unusable — **P0**

**Evidence (executed):**

```
checkpoint  ml/llm/checkpoints/fayd-llm.npz   (6.2 MB)
parameters  842,624     (manifest: n_layer=4, n_embd=128, block_size=128)
tokenizer   byte  ({"kind":"byte","merges":[]})
perplexity  test 4.2654 on 13,121 tokens
```

Live answers from `ai.mind.MindAssistant`:

```
Q: what is porosity?
A: The saturation forecast the reservoir simulation belongs to reservoir engineering.

Q: what does permeability measure?
A: Reservoir rock properties reservoir to the reservoir to properties.

Q: how does waterflood increase production?
A: Expion, then recovery factor, and flood the flooding flow path.
```

The output is **not coherent English**. The symbolic/retrieval path is
excellent; the *generative* path is the weak point.

**Impact:** the "language model as a component" cannot yet carry a conversation
or draft prose. Every useful answer today comes from retrieval + reasoning.

**Fix:** **extend** `ml/llm/` — (a) train and use the **BPE tokenizer** already
implemented in `ml/llm/bpe.py` (the checkpoint uses the byte tokenizer, the
weakest option); (b) raise capacity to a real tier; (c) train far longer on a
balanced corpus. **Do not replace** the model — the plumbing (train/save/resume/
evaluate/schedule) is correct and tested.

### G-2 · No parameter-efficient fine-tuning (LoRA / PEFT) — **P0**

**Evidence:** a repository-wide AST/grep search for `lora`, `peft`, `qlora`
finds **zero** occurrences in `core/`, `ml/`, `cognition/`. The only weight
training is the in-project numpy transformer (`ml/llm/transformer.py`) and the
staged trainer (`cognition/training.py`).

**Impact:** the task requires "gradual model adaptation using LoRA and other
parameter-efficient methods". Today the system cannot adapt a **real** GGUF
model at all — only its own tiny transformer.

**Fix:** **NEW** `core/finetune.py` + `ml/lora/`, behind the **existing**
`cognition/versions.ModelRegistry` gate (base immutable → candidate → evaluate →
accept/rollback). Reuse `ml/dataset.py`, `ml/trainer.py`, `core/model_files.py`.

### G-3 · No local embedding model — **P2**

**Evidence:** `core/semantic_memory.hashed_embedding` is a hash; the replacement
`core/embeddings.py` is a **learned distributional space** (PPMI + truncated SVD,
128-d) — real, but trained on the project's own small corpus.

**Impact:** semantic recall is limited by corpus size; no transfer from a
pretrained embedding model.

**Fix:** **extend** `core/embeddings.py` with an optional local GGUF embedding
model behind the same interface; keep the learned space as the fallback.

---

## 2. Orchestration

### G-4 · No event bus / publish-subscribe — **P0**

**Evidence:** grep for `EventBus`, `event_bus`, `publish(`, `subscribe(`,
`EventEmitter` across `core/`, `cognition/`, `ai/`, `gui/` returns **nothing**.

**Impact:** subsystems are wired by direct calls. A monitor (needed for
self-improvement), a UI live view, and a training hook all have to be bolted on
individually. The target's "event-driven communication" and the automated
self-improvement loop both need this.

**Fix:** **NEW** `core/event_bus.py` — dependency-free, synchronous + optional
queue mode, bounded, with an audit-friendly record. Publish lifecycle events from
the controller, memory, model manager, and training.

### G-5 · No unified task scheduler / resource governor — **P1**

**Evidence:** `cognition/scheduler.py` schedules **thought cycles** only;
`cognition/runner.py` checks machine load for **those** cycles only;
`core/model_resources.py` reads the machine but arbitrates nothing. There is no
component that decides "may I load this model now, given the background mind and
the user's activity".

**Impact:** a model load and a background thought cycle can compete for 16 GB.

**Fix:** **EXTEND** — a thin `core/resource_governor.py` over
`model_resources` + `scheduler`, plus a general bounded task queue. Do not
rewrite the scheduler.

---

## 3. Memory

### G-6 · Working and episodic memory are in-process only — **P0** ✅ CLOSED (Phase 2)

**Evidence:** `core/working_memory.WorkingMemory` and
`core/episodic_memory.EpisodicMemory` hold plain dicts; `CognitiveMemory` stores
by id in memory. Only `core/knowledge_carry.py` persists anything across sessions
(learned **relations**), and `core/knowledge_store.py` can snapshot graph+memory
**when explicitly asked**. The controller does not persist its memory by default.

**Impact:** the task requires durable long-term memory (episodic + semantic +
procedural). Today an experience is lost at process exit unless a caller
snapshots.

**Fix (delivered):** `core/memory_db.py` (durable SQLite, schema-versioned,
transactional, corrupt-row-tolerant, backup/restore) beneath
`core/long_term_memory.py` (the four categories + the full lifecycle),
`core/memory_retrieval.py` (explainable ranking), `core/memory_management.py`
(management + per-personality isolation), and `core/memory_gate.py` (the held-out
gate). `memory_consolidator` writes into the durable store and `memory_retriever`
reads from it; the controller accepts an optional `long_term_memory=`.
See `docs/architecture/PHASE_2_IMPLEMENTATION_REPORT.md`.

### G-7 · No conflict resolution across the memory kinds — **P1** ✅ CLOSED (Phase 2)

**Evidence:** `core/conflict_resolver.py` exists for beliefs;
`memory_consolidator.py` promotes working→episodic without a cross-kind conflict
rule.

**Impact:** a new episode can silently contradict a stored semantic fact.

**Fix (delivered):** a correction is a **new** record via
`MemoryRecord.supersede()` / `LongTermMemory.correct()` -- the old record is kept
and marked `superseded` with a pointer to its replacement, so a contradiction is
*reported*, never silently merged. `MemoryManager.duplicates()` surfaces duplicate
groups for review without merging them, and `provenance()` walks the supersession
chain. (Belief-level contradiction detection remains in `conflict_resolver`, which
Phase 1's `KnowledgeValidator` already uses.)

---

## 4. Offline guarantees

### G-8 · Offline operation is not enforced or tested — **P0**

**Evidence:** web search is **off by default** (Phase 53,
`ai/jarvis.build_session`), and every download needs `approved=True` — good. But
there is **no test** that asserts the base install never opens a socket, and
`requests` is a **base** dependency (so it is importable everywhere).

**Impact:** a future change could start reaching the network and nothing would
catch it; the task explicitly requires fully local operation.

**Fix:** **NEW** `tests/test_offline.py` — run the answer path, a memory write, a
training smoke step, and a personality exchange with network calls monkeypatched
to raise, and assert none is attempted. Plus a documented `FAYD_OFFLINE=1`
hard-off switch that makes any network attempt raise.

---

## 5. Self-improvement

### G-9 · The self-improvement loop is human-gated only — **P0**

**Evidence:** `core/self_improvement.SelfImprovement.propose` returns an
`ImprovementPlan(..., requires_human_review=True)`; `core/architecture_learning`
produces **text recommendations**; `core/code_studio.py` can edit code in an
isolated worktree and run checks — but **nothing connects them** into a
propose→test→measure→accept/reject loop.

**Impact:** the target's "controlled self-improvement" is only half built: the
*pieces* exist (isolated worktree, evaluation, version rollback) but not the
*coordinator*.

**Fix:** **NEW** `core/self_improvement_loop.py` that orchestrates the existing
pieces, gated on explicit human approval at the merge step. Reuse
`code_studio`, `architecture_learning`, `evaluation`, `versions`.

### G-10 · `self_model` name collision — **P3**

**Evidence:** `core/self_model.py:SelfModel` is a **declared identity** (a domain
frame); `cognition/metacognition.py:SelfModel` is the **learned** self-model
(strategy payoffs). Two different things share the name.

**Impact:** reader confusion; import aliasing.

**Fix:** **EXTEND/RENAME** — keep both behaviours, disambiguate the names
(e.g. `DomainIdentity` vs `MetacognitiveSelfModel`). Low risk, high clarity.

---

## 6. Reasoning

### G-11 · Uncertainty is not propagated through the composed answer — **P1**

**Evidence:** `core/uncertainty_propagator.py` and `answer_confidence.py` both
exist; the answer confidence folds graph support and conflicts, but a *multi-hop*
chain's propagated uncertainty is not folded into the final figure end to end.

**Impact:** a long weak chain can read as confident as a direct edge.

**Fix:** **EXTEND** `answer_confidence` to take the propagated uncertainty of the
chain (`relation_paths` already returns product confidence — wire it in).

### G-12 · Tool selection is declared, not chosen — **P1**

**Evidence:** `core/tool_agent.py` — "a plan is *declared* by the caller or by a
planner function, not invented by the model".

**Impact:** the agent cannot choose tools for an open task; only replay a
supplied plan.

**Fix:** **EXTEND** — a bounded chooser over the sandbox allow-list (the model
proposes, the sandbox disposes). Authority is unchanged.

---

## 7. Planning / execution

### G-13 · Goals are not persisted — **P2**

**Evidence:** `core/goal.py:Goal` is an in-memory dataclass; nothing stores the
goal set across sessions.

**Impact:** an open goal is lost on restart.

**Fix:** **EXTEND** — persist goals through `JsonStore`, alongside the existing
snapshot mechanism.

---

## 8. Personalities

### G-14 · No per-personality training dataset / training history — **P1**

**Evidence:** `core/character_learning.py` records facts and exchanges and
`character_learning_sentences()` renders them as training text; but there is no
**dataset version**, no **training history**, and no **model version** per
character. `cognition/versions.ModelRegistry` tracks the *project* model only.

**Impact:** the task requires each personality to have "its own learning
datasets, training history". Today the data exists; the *record* does not.

**Fix:** **EXTEND** `cognition/versions.ModelRegistry` to be owner-scoped, and
`character_learning` to expose a dataset snapshot with a version id.

### G-15 · No explicit per-personality permissions — **P2**

**Evidence:** learning isolation is structural (`core/learning_isolation.py`),
but a character's *allowed capabilities* (may it search the web? may it run
tools? may it write files?) are not declared per character and enforced.

**Impact:** the task requires "permissions" per personality.

**Fix:** **EXTEND** `core/character_config.py` with a capability set, enforced in
`ai/characters.py`'s wrapper and the mouth builders.

---

## 9. Testing and data integrity

### G-16 · No offline, DB-recovery, or backup/restore test — **P0**

**Evidence:** `ls tests | grep -i 'offline\|backup\|recover'` → none. Snapshot
tests exist (`test_snapshot.py`, `test_snapshot_migrations.py`) but no test
corrupts a store and asserts recovery.

**Impact:** data-integrity guarantees are asserted, not proven.

**Fix:** **NEW** tests: `test_offline.py`, `test_store_recovery.py`
(corrupt a JSON/SQLite file → the reader tolerates or reports), and a backup/
restore round-trip for the knowledge base and character stores.

### G-17 · No unified migration framework — **P2**

**Evidence:** `core/snapshot_migrations.py` migrates **snapshots** only; other
stores (`JsonStore`, `SqliteStore`, character learning, cognition stores) have
ad-hoc or no migration.

**Impact:** a schema change risks unreadable user data.

**Fix:** **EXTEND** `snapshot_migrations` into a small registry usable by every
store, with the same ordered, side-effect-free rule.

---

## 10. Trained-artifact provenance

### G-18 · The committed model was trained at an older revision — **P1**

**Evidence (measured):**

```
HEAD                       9c0b8f5   ("Merge pull request #246 … feature/learner-asks-teacher")
manifest git_revision      b615191   (ml/artifacts/release_manifest.json)
manifest trained_at         2026-10-08T18:04:33Z
checkpoint mtime            2026-11-02 (checked out, not retrained)
```

The committed LLM checkpoint and the release manifest were produced at revision
`b615191`, **many commits before the current HEAD** `9c0b8f5`. The corpus the
manifest describes (`kb_max_chunks: 15000`, `tokenizer: byte`,
`ontology_concepts: 495`) therefore reflects an older tree, and nothing fails if
the corpus changes without the model being regenerated.

**Impact:** the "trained and ready" release claim is only true for the older
revision; a reader cannot tell how stale the weights are, and the in-project
model's already-weak output (G-1) may be worse than a fresh training would give.

**Fix:** **EXTEND** `ml/release.py` — record the HEAD revision at training time
and add a check (in `python -m cli selfcheck` or `evaluation`) that warns when
the manifest revision differs from HEAD. Do not retrain in Phase 0 (rule: do not
modify model weights).

---

## 11. Gap summary

| ID | Gap | Priority | New or extend |
| --- | --- | --- | --- |
| G-1 | In-project LLM unusable | P0 | extend `ml/llm` |
| G-2 | No LoRA/PEFT | P0 | new `core/finetune.py` + `ml/lora/` |
| G-4 | No event bus | P0 | new `core/event_bus.py` |
| G-6 | Memory not durable | P0 | ✅ closed (Phase 2: `memory_db`+`long_term_memory`) |
| G-8 | Offline not enforced/tested | P0 | new test + hard-off switch |
| G-9 | Self-improvement not automated | P0 | new `core/self_improvement_loop.py` |
| G-16 | No offline/recovery/backup tests | P0 | new tests |
| G-5 | No resource governor/scheduler | P1 | extend `model_resources`+`scheduler` |
| G-7 | No cross-kind memory conflict rule | P1 | ✅ closed (Phase 2: supersession keeps history) |
| G-11 | Uncertainty not propagated to answer | P1 | extend `answer_confidence` |
| G-12 | Tool selection declared, not chosen | P1 | extend `tool_agent` |
| G-14 | No per-personality training records | P1 | extend `versions`+`character_learning` |
| G-3 | No neural embedding model | P2 | extend `embeddings` |
| G-13 | Goals not persisted | P2 | extend `goal`+storage |
| G-15 | No per-personality permissions | P2 | extend `character_config` |
| G-17 | No unified migration framework | P2 | extend `snapshot_migrations` |
| G-18 | Model trained at an older revision | P1 | extend `ml/release` |
| G-10 | `self_model` name collision | P3 | rename |

**No gap requires rebuilding the project.** Every P0 is either a new small
coordinator over existing pieces, or an extension of a module that is already
tested.
