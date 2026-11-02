# Fayd Al-Ruh — Implementation Roadmap (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

Nine incremental phases. Each **preserves every existing working capability**,
reuses the modules already present, and states what is new. Effort is in
**engineer-days (ED)** for one developer familiar with the codebase; hardware
notes target the stated development machine (Windows 11, i7, 16 GB RAM,
Radeon R7 M360 2 GB / Intel HD 520 1 GB) and are re-checked in
`RESOURCE_ASSESSMENT.md`.

Dependency rule: **Phase 0 must be complete before Phase 1** (this document is
part of that completion). Phases 1–9 are ordered so each builds on the previous,
but 2 and 4 can proceed in parallel with 1 once the event bus (Phase 1) lands.

---

## Phase 1 — Cognitive Core Enhancement

**Goal:** make the core observable and arbitrated — the event bus, a resource
governor, a unified knowledge-validation service, and honest multi-hop
uncertainty — without changing any reasoning result.

**Required modifications**

- **NEW** `core/event_bus.py`: dependency-free publish/subscribe. Synchronous by
  default; an optional bounded queue mode for background work; a subscriber may
  never raise into the publisher (`best_effort`); an append-only record for the
  audit.
- **NEW** `core/resource_governor.py`: one arbiter over `core/model_resources`
  and `cognition/scheduler`. Answers "may I load this model / run this cycle
  now?" from real figures with an honest `None`.
- **NEW** `core/knowledge_validation.py`: one callable that runs
  `acceptance_policy` + `consistency_checker` + `contradiction_detector` on a
  proposed write and returns one verdict.
- **EXTEND** `core/answer_confidence.py`: fold the propagated uncertainty of a
  multi-hop chain (`relation_paths` product confidence) into the final figure.
- **EXTEND** `core/cognitive_controller.py`: publish lifecycle events
  (`cycle_started`, `cycle_finished`, `belief_revised`, `relation_learned`).

**Existing modules reused:** `model_resources`, `scheduler`, `runner`,
`acceptance_policy`, `consistency_checker`, `contradiction_detector`,
`relation_paths`, `answer_confidence`, `best_effort`, `audit_log`.

**New modules:** 3 (`event_bus`, `resource_governor`, `knowledge_validation`).

**Dependencies:** none (first phase).

**Technical risks:** an event bus can hide control flow. Mitigation: events are
*notifications*, never a path that a decision depends on; the decision path stays
direct calls. A synchronous bus can slow a hot loop — mitigation: the bus is
opt-in per event, bounded, and measured.

**Hardware:** no change (CPU-only, negligible RAM).

**Testing:** unit tests for the bus (ordering, a raising subscriber is isolated,
bounded), the governor (defers on low memory, honest `None`), the validation
service (a contradiction is caught), and the confidence fold (a weak chain lowers
the number, a strong one raises it). Add a held-out metric
`knowledge_validation_accuracy` with negative controls.

**Acceptance criteria:** ruff/mypy/pytest/evaluation all green; the evaluation
`overall` does not fall; a controller write that contradicts stored knowledge is
refused by the validation service; no reasoning result changes.

**Effort:** 6–8 ED.

---

## Phase 2 — Long-Term Memory Architecture

**Goal:** make episodic and semantic memory durable, consolidating, and
conflict-checked across sessions — the four-memory model the task requires.

**Required modifications**

- **NEW** `core/durable_memory.py`: a store over `storage/sqlite_store.py` (and a
  JSON fallback) holding episodic and semantic items with provenance, importance,
  access counts, and timestamps. Reads are streaming and bounded; a corrupt row
  is skipped and counted, never a crash.
- **EXTEND** `core/memory_consolidator.py`: write working→episodic promotions into
  the durable store.
- **EXTEND** `core/memory_retriever.py`: read from the durable store as a fourth
  source beside lexical/semantic/associative.
- **EXTEND** `core/conflict_resolver.py`: a cross-kind rule (a new episode that
  contradicts a stored semantic fact is flagged, not silently merged).
- **EXTEND** `core/knowledge_store.py`: snapshot/restore the durable memory with
  the graph.
- **EXTEND** `core/procedural_memory.py`: persist learned procedures.

**Existing modules reused:** `working_memory`, `episodic_memory`,
`semantic_memory`, `procedural_memory`, `memory`, `memory_consolidator`,
`memory_decay`, `retention_policy`, `memory_retriever`, `knowledge_carry`
(the proven pattern), `sqlite_store`, `json_store`.

**New modules:** 1.

**Dependencies:** Phase 1 (the event bus lets consolidation publish
`memory_stored`).

**Technical risks:** unbounded growth. Mitigation: `retention_policy` +
`memory_decay` already exist; the durable store enforces a cap and prunes by
decay. SQLite on a network drive is unsupported — document local-only.

**Hardware:** SQLite adds a few MB; negligible.

**Testing:** durability across a simulated restart; consolidation writes and
reads back; a contradiction is flagged; a corrupt store is tolerated; retention
bounds growth. New held-out metric `durable_memory_accuracy`.

**Acceptance criteria:** an episode stored in one process is retrievable in the
next; a contradiction is reported; a corrupt store never crashes a session;
`evaluation` green.

**Effort:** 8–10 ED.

---

## Phase 3 — Cognitive and Experience-Based Learning

**Goal:** unify the two symbolic learning paths behind one governed pipeline and
make consolidation/review durable, **without** touching model weights (that is
Phase 6).

**Required modifications**

- **EXTEND** `core/learning_pipeline.py`: one entry point from a response to a
  stored record (already exists) — add the *experience* path from
  `cognition/learning_system.py` so cognitive and experience learning share the
  provenance record.
- **EXTEND** `cognition/consolidation.py`: persist patterns and review schedules
  into the durable stores from Phase 2.
- **EXTEND** `core/relation_extraction.py` + `ontology_learner`: unchanged
  behaviour; wire the harvest through the Phase 1 validation service.
- **EXTEND** `cognition/study.py`: keep the measured benefit report; add a
  durable record of each study session.

**Existing modules reused:** `learning_pipeline`, `learning_records`,
`learning_isolation`, `continuous_learning`, `knowledge_update`,
`ontology_learner`, `relation_extraction`, `cognition/{learning_system,
consolidation,curriculum,study,reliability}`.

**New modules:** 0 (extensions only).

**Dependencies:** Phase 2 (durable stores).

**Technical risks:** over-generalisation. Mitigation: `cognition/reliability`
already refuses an unverified statement as a fact, and the `development_accuracy`
gate has negative controls.

**Hardware:** none.

**Testing:** a learned statement is provisional; a second identical write does
not duplicate; a consolidation rule needs real support; isolation holds (Natasha
private). Gates already exist (`development_accuracy`,
`knowledge_carry_accuracy`); add `experience_learning_accuracy`.

**Acceptance criteria:** cognitive and experience learning write one provenance
record; nothing auto-verifies; `evaluation` green.

**Effort:** 6–8 ED.

---

## Phase 4 — Reasoning and Inference Engine

**Goal:** close the reasoning gaps (multi-hop uncertainty, bounded tool choice)
and expose a single reasoning API, without changing what the graph entails.

**Required modifications**

- **EXTEND** `core/answer_confidence.py` (already in Phase 1) — verify the fold
  end to end.
- **EXTEND** `core/tool_agent.py`: a **bounded chooser** — the model proposes,
  the sandbox disposes; the allow-list is unchanged.
- **NEW** `core/reasoning_api.py`: one facade over the existing engines
  (deduce/induce/abduce, causal, analogy, temporal, plan, decision, belief) that
  returns a uniform `ReasoningResult` with its trace, so callers stop reaching
  into 20 modules.
- **EXTEND** `core/reasoning_report.py`: render the new uncertainty term.

**Existing modules reused:** the whole reasoning layer (69 modules), `tools`,
`tool_agent`, `permissions`, `reasoning_report`, `answer_*`.

**New modules:** 1 (facade; no logic duplicated).

**Dependencies:** Phase 1 (validation + events).

**Technical risks:** a facade can become a god-object. Mitigation: it delegates,
it holds no state, and it adds no inference of its own.

**Hardware:** none.

**Testing:** the facade returns what the engines return (no divergence);
the chooser never selects a tool outside the allow-list; uncertainty lowers a
weak chain's confidence. Gates: `answer_path_accuracy`,
`reasoning_trace_accuracy`, plus a new `tool_selection_accuracy`.

**Acceptance criteria:** a caller can ask the facade for a plan/decision/verdict
and get the same object as calling the engine directly; the sandbox is unchanged.

**Effort:** 6–8 ED.

---

## Phase 5 — Seven Independent Personalities

**Goal:** give each personality its own dataset record, training history, and
permissions, on top of the isolation and memory already present.

**Required modifications**

- **EXTEND** `cognition/versions.py`: make `ModelRegistry` **owner-scoped** so a
  character has its own version history.
- **EXTEND** `core/character_learning.py`: expose a **dataset snapshot** with a
  version id and a manifest (counts, dates, source) for training.
- **EXTEND** `core/character_config.py`: a **capability set** per character
  (may search, may run tools, may write files), enforced in `ai/characters.py`
  and the mouth builders.
- **EXTEND** `gui/character_dialog.py` + `cli/characters.py`: surface the
  dataset, history, and permissions.

**Existing modules reused:** `character_learning`, `character_config`,
`character_registry`, `character_engine`, `character_identity`,
`character_isolation_gate`, `learning_isolation`, `knowledge_sharing`,
`mouth_profile`, `ai/characters.py`.

**New modules:** 0 (extensions only).

**Dependencies:** Phase 3 (the dataset is the learning output).

**Technical risks:** widening a character's authority. Mitigation: the
capability set is deny-by-default and enforced in the wrapper; the isolation gate
already has negative controls.

**Hardware:** none.

**Testing:** a character's dataset is versioned and its history recorded; a
capability not granted is refused; Natasha stays private. Gate:
`character_identity_accuracy`, `character_isolation_accuracy`, plus
`character_permissions_accuracy`.

**Acceptance criteria:** each of the seven (and the project model) has a dataset
record, a training history, and an enforced permission set; isolation unchanged.

**Effort:** 8–10 ED.

---

## Phase 6 — Controlled Model Weight Training

**Goal:** the largest gap — parameter-efficient fine-tuning (LoRA/QLoRA) of a
local model, behind the existing version/rollback gate, plus the repair of the
in-project LLM.

**Required modifications**

- **NEW** `ml/lora/` (`dataset.py`, `trainer.py`, `adapter.py`): LoRA/QLoRA over
  a local model. Reuse `ml/dataset.py` for splits and `ml/trainer.py`'s metrics.
- **NEW** `core/finetune.py`: the governed entry point — validate the dataset,
  train a candidate, evaluate, hand it to `cognition/versions` for
  accept/rollback, register the adapter in `core/language_models`.
- **EXTEND** `core/model_runtime.py`: a runtime adapter that loads a base model
  plus a LoRA adapter (llama.cpp supports LoRA; a `transformers` path is the
  fallback for a CPU-only machine).
- **EXTEND** `ml/llm/`: train and use the **BPE tokenizer** (`ml/llm/bpe.py` is
  already implemented but unused by the checkpoint); raise capacity to a real
  tier; train longer on the balanced corpus (`corpus_balance()` already measures
  the mix).
- **EXTEND** `core/model_preflight.py`: estimate the training memory need too.
- **EXTEND** `ml/release.py`: record the LoRA adapter and the tokenizer kind in
  the manifest.

**Existing modules reused:** `cognition/training.py` (staged, resumable),
`cognition/versions.py` (the gate), `ml/dataset.py`, `ml/trainer.py`,
`ml/llm/{bpe,tokenizer,transformer,schedule,state,evaluate}`,
`core/model_files.py`, `core/model_registry.py`, `core/optional_deps.py`.

**New modules:** 4.

**Dependencies:** Phase 5 (per-owner datasets) for personality adapters; Phase 1
(events) for progress.

**Technical risks**

- **No usable GPU.** The stated GPUs (Radeon R7 M360, Intel HD 520) do **not**
  support modern CUDA/ROCm acceleration. Mitigation: train on CPU with a **small
  adapter** and a **small base model** (1.7B–4B, quantised), or export the
  dataset and train on a stronger machine. The gate makes a slow run safe.
- **`transformers`/`torch` size** (~2 GB). Mitigation: an optional extra; the
  base install is unaffected.
- **Catastrophic forgetting.** Mitigation: the base is immutable and every
  candidate is evaluated before acceptance (`should_accept`).

**Hardware:** CPU LoRA on a 4B quantised model is feasible but slow (hours);
on 1.7B it is practical. 16 GB RAM fits a 4B Q4 model + a small adapter.
**Sequential only** — never train while a model is loaded for chat.

**Testing:** a tiny LoRA run on a synthetic dataset changes the adapter only
(base file hash unchanged); a regressing candidate is rejected and rolled back;
the manifest records the adapter. New held-out metric
`finetune_gate_accuracy`.

**Acceptance criteria:** a base model file is never modified; a rejected run
leaves the active version untouched; the in-project LLM produces coherent
English on a held-out prompt set (a measurable improvement over G-1's baseline).

**Effort:** 15–20 ED.

---

## Phase 7 — Local Multimodal Perception

**Goal:** route perception (documents, images, audio) into the same evidence
pipeline as text, so perception and reasoning share one provenance path.

**Required modifications**

- **NEW** `core/perception_bridge.py`: turn a perceived artifact (a page image's
  transcription, an audio transcript, an image description) into
  `Evidence`/`Document` with provenance, and index it for retrieval.
- **EXTEND** `core/docs_studio.py`, `core/qwen_speech.py`, `core/vision.py`: emit
  through the bridge.
- **EXTEND** `core/document_reader.py`: one dispatch for every format, reusing
  what exists.
- **EXTEND** `ingest/pipeline.py`: route ingested media through the bridge.

**Existing modules reused:** `docs_model`, `docs_studio`, `vision`,
`qwen_speech`, `speech`, `pdf_extract`, `pdf_ocr`, `docx_extract`,
`document_reader`, `imagegen`, `imagegen_diffusion`, `imageedit`,
`evidence_processor`, `ingest/*`.

**New modules:** 1.

**Dependencies:** Phase 2 (durable memory to store the evidence) and Phase 4
(the reasoning facade to consume it).

**Technical risks:** a vision model hallucinating text. Mitigation: a
transcription is stored as **perceived evidence** with its source and a
confidence, never as a verified fact; the groundedness gate still applies.

**Hardware:** Qwen3-VL 4B Q4 needs ~3–4 GB; run it **sequentially** with any
chat model.

**Testing:** a scanned page's transcription becomes citable evidence with
provenance; an image description is evidence, not a fact; OCR-free and OCR paths
both work. New metric `perception_bridge_accuracy`.

**Acceptance criteria:** a question about an attached scanned page is answered
with citations to the transcription; nothing perceived is stored as verified.

**Effort:** 10–12 ED.

---

## Phase 8 — Autonomous Planning and Task Execution

**Goal:** persist goals, decompose and execute tasks under the existing sandbox,
record the experience, and evaluate the result — the full H block of the target.

**Required modifications**

- **EXTEND** `core/goal.py` + storage: persist goals and their status.
- **EXTEND** `core/planner.py`: keep BFS + `order_steps`; add plan persistence.
- **EXTEND** `core/tool_agent.py`: run a chosen plan through the sandbox, record
  each step as an experience (Phase 3).
- **EXTEND** `cognition/runner.py` + the Phase 1 task queue: schedule a task,
  not only a thought cycle.
- **EXTEND** `core/plan_evaluator.py`: evaluate the outcome and feed the
  experience store.

**Existing modules reused:** `goal`, `plan`, `planner`, `plan_evaluator`,
`action*`, `state*`, `tools`, `tool_agent`, `sandbox_*`, `progress`,
`cognition/{runner,scheduler,learning_system}`.

**New modules:** 0 (extensions only).

**Dependencies:** Phases 1, 3, 4.

**Technical risks:** unbounded execution. Mitigation: the sandbox step budget
and the resource governor already bound it; a plan cannot exceed the allow-list.

**Hardware:** none beyond a model for choice (Phase 4).

**Testing:** a goal survives a restart; a task plan executes within its budget;
a step outside the allow-list is refused; the experience is recorded. New metric
`task_execution_accuracy`.

**Acceptance criteria:** an open goal is persisted, planned, executed in the
sandbox, evaluated, and recorded as experience; authority unchanged.

**Effort:** 8–10 ED.

---

## Phase 9 — Controlled Self-Improvement

**Goal:** the automated, **human-gated** loop: monitor → detect → propose →
isolated test → measure → accept/reject → rollback.

**Required modifications**

- **NEW** `core/self_improvement_loop.py`: orchestrates the existing pieces.
  Each stage reuses a proven component:
  - monitor → `core/error_monitor.py` + the Phase 1 event bus;
  - detect → `core/architecture_learning.py` + `cognition/knowledge_depth`;
  - propose → a change plan (a task description for the Code Studio);
  - isolated test → `core/code_studio.py` (git worktree) + `run_checks`;
  - measure → `evaluation/` against the baseline;
  - accept/reject → `cognition/versions.should_accept`;
  - rollback → `cognition/versions.rollback`.
- **EXTEND** `core/self_improvement.py`: keep `requires_human_review=True` — the
  merge step is **always** a human decision.
- **EXTEND** `gui/cognition_dialog.py` + `cli/cognition.py`: surface a proposal,
  its isolated test result, its measured delta, and the accept/reject action.

**Existing modules reused:** `code_studio`, `code_agent`, `architecture_learning`,
`self_improvement`, `error_monitor`, `evaluation`, `versions`, `progress`.

**New modules:** 1.

**Dependencies:** all prior phases (it consumes their telemetry).

**Technical risks:** an unsafe auto-change. Mitigation: the loop **never** merges
without explicit human approval; it works in a throwaway git worktree; the
evaluation gate and the version rollback are the backstops.

**Hardware:** none (the loop orchestrates; the model work is Phase 6).

**Testing:** a proposal that regresses the evaluation is rejected and the
worktree discarded; a proposal that passes is offered for approval, never merged
automatically; a rejected run leaves the checkout untouched. New metric
`self_improvement_loop_accuracy` with negative controls (a regressing change must
be rejected; a change that touches a forbidden file must be refused).

**Acceptance criteria:** the loop runs end to end on a trivial proposal in an
isolated worktree, measures the delta, and stops at the human approval step;
the running checkout and stored data are never at risk.

**Effort:** 12–15 ED.

---

## Roadmap summary

| Phase | Name | New modules | Effort (ED) | Depends on |
| --- | --- | ---: | ---: | --- |
| 1 | Cognitive Core Enhancement | 3 | 6–8 | — |
| 2 | Long-Term Memory Architecture | 1 | 8–10 | 1 |
| 3 | Cognitive & Experience Learning | 0 | 6–8 | 2 |
| 4 | Reasoning & Inference Engine | 1 | 6–8 | 1 |
| 5 | Seven Independent Personalities | 0 | 8–10 | 3 |
| 6 | Controlled Model Weight Training | 4 | 15–20 | 5 (and 1) |
| 7 | Local Multimodal Perception | 1 | 10–12 | 2, 4 |
| 8 | Autonomous Planning & Execution | 0 | 8–10 | 1, 3, 4 |
| 9 | Controlled Self-Improvement | 1 | 12–15 | all |
| | **Total** | **11** | **79–101** | |

## Preserving existing capabilities (mandatory rules, restated)

1. **Do not rebuild.** Every phase extends.
2. **Do not delete** a working module without documented justification.
3. **Do not duplicate.** The 11 new modules are coordinators/facades; none
   re-implements logic.
4. **No destructive migrations.** Every store migration is additive and
   reversible (`snapshot_migrations` pattern).
5. **Do not modify model weights** in Phase 0 (and only through the gate after).
6. **Do not change personality learning datasets** in Phase 0.
7. **Do not weaken security controls** — the permission policy, approval gate,
   audit log, and isolation are invariants.
8. **Windows 11 compatibility** preserved (Windows-safe paths/encodings).
9. **GUI functionality** preserved — new surfaces are additive.
10. **Backward compatibility** wherever possible.

## Recommended first task for Phase 1

**Implement `core/event_bus.py` and publish the controller's lifecycle events
through it.**

Why first: it is the smallest, lowest-risk change (a new dependency-free module,
no behaviour change), it is a hard dependency of the self-improvement monitor
(Phase 9) and the resource governor (Phase 1), and it establishes the
observability pattern every later phase uses. It can be landed and verified in
isolation, keeping the four gates green.
