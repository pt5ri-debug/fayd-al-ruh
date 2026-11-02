# Fayd Al-Ruh — Target Architecture (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

This document defines the **target** Cognitive Intelligence System and maps it
onto what already exists. The rule throughout is *extend, do not replace*: every
target block names the existing module(s) it builds on, or states that a new
component is required and why. The gap analysis and effort estimates are in
`TECHNICAL_GAPS.md` and `IMPLEMENTATION_ROADMAP.md`.

---

## 1. Design principles

1. **Local-first, offline-capable.** No cloud service or external API is
   required for normal operation. Network use is an explicit, governed opt-in.
2. **Model-independent.** The cognitive core (graph, memory, reasoning) never
   depends on a specific language model. Models are pluggable components behind
   `core/model_interface.CognitiveModel` and `core/model_runtime` adapters.
3. **Evidence-aware.** Every claim carries its evidence, confidence, source,
   and status; contradictions are preserved and reported, never hidden.
4. **Honest by construction.** An estimate is never presented as a measurement;
   a missing capability is reported with the real next step; a derivation is
   labelled as derived.
5. **Human-gated authority.** No component may widen its own authority. External
   actions pass capability + human-approval + audit gates.
6. **Incremental and reversible.** Every change is bounded, tested against a
   held-out gate, and rollback-safe.
7. **Windows 11 compatible.** All paths, encodings, and processes are
   Windows-safe (already a project invariant).
8. **Scalable.** The design must accept more RAM/GPU later without redesign.

---

## 2. Target layering

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INTERFACES        gui/  ·  cli/  ·  (future) local HTTP API             │
├──────────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION     ✦ Event bus + task scheduler + resource arbiter (NEW) │
│                    cognition/runner.py, cognition/scheduler.py (extend)  │
├──────────────────────────────────────────────────────────────────────────┤
│  CHARACTERS        seven personalities + project model                   │
│                    ai/characters.py, core/character_*.py (extend)        │
├──────────────────────────────────────────────────────────────────────────┤
│  COGNITION         reasoning · planning · decision · belief · analogy ·  │
│                    temporal · causal · self-model · metacognition        │
│                    core/* + cognition/* (extend)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  MEMORY            working · episodic · semantic · procedural            │
│                    + durable episodic/semantic store (NEW/extend)        │
├──────────────────────────────────────────────────────────────────────────┤
│  KNOWLEDGE         ontology · graph · evidence · beliefs · provenance    │
│                    core/ontology.py, knowledge_graph.py (extend)         │
├──────────────────────────────────────────────────────────────────────────┤
│  PERCEPTION        documents · PDF/OCR · images · audio · speech         │
│                    core/{document_reader,pdf_*,vision,docs_model,speech} │
├──────────────────────────────────────────────────────────────────────────┤
│  LEARNING          cognitive · experience · controlled weights (LoRA)   │
│                    core/learning_*.py · cognition/training.py · ml/      │
├──────────────────────────────────────────────────────────────────────────┤
│  MODELS            registry · runtime adapters · selection · preflight   │
│                    core/model_*.py, language_models.py (extend)          │
├──────────────────────────────────────────────────────────────────────────┤
│  GOVERNANCE        permissions · approval · audit · sandbox · isolation  │
│                    core/{permissions,human_approval,audit_log,           │
│                    sandbox_runtime,learning_isolation}.py (extend)       │
├──────────────────────────────────────────────────────────────────────────┤
│  STORAGE           JSON · SQLite · FTS5 · graph · snapshots              │
│                    storage/*, core/knowledge_index.py (extend)           │
└──────────────────────────────────────────────────────────────────────────┘
```

The **new** boxes are deliberately few. The project is already ~80% of the
target; the work is in four areas: orchestration, durable memory, controlled
weight training (LoRA), and the automated self-improvement loop.

---

## 3. Target components, mapped

### A. Core infrastructure

| Target | Build on | New work |
| --- | --- | --- |
| Local runtime management | `core/model_runtime.py`, `model_manager.py` | none |
| **Event-driven communication** | — | **NEW** `core/event_bus.py` (dependency-free, synchronous + queue modes). Subsystems publish lifecycle events (`thought_cycle_done`, `memory_stored`, `model_loaded`). Enables decoupling and the self-improvement monitor. |
| **Resource management** | `core/model_resources.py`, `cognition/scheduler.py` | **EXTEND** into one `core/resource_governor.py` that arbitrates RAM/CPU across model loads and background cycles |
| **Task scheduling** | `cognition/scheduler.py` | **EXTEND** into a general bounded task queue (priority + budget), reusing the existing quiet-window/budget rules |
| Database management | `storage/*`, `core/knowledge_index.py` | **EXTEND** a thin migration registry over `core/snapshot_migrations.py` |
| Security & permissions | `core/permissions.py`, `human_approval.py`, `audit_log.py`, `sandbox_runtime.py` | none (already strong) |
| Monitoring & logging | `core/{audit_log,model_log,error_monitor,progress}.py` | **EXTEND** a unified health/metrics snapshot (reads the above; adds nothing authoritative) |

### B. Local model management

Already strong. Target additions: an **embedding-model slot** (a local GGUF
embedding model behind `core/embeddings.py`'s existing interface, with the
learned distributional space as the fallback), and LoRA adapter registration in
the catalog (see F).

### C. Perception and understanding

Already strong. Target additions: a **multimodal ingestion bridge** that routes
a perceived artifact (page image, audio transcript) into the same evidence
pipeline as text, so perception and reasoning share one provenance path. Reuse
`core/docs_studio.py`, `core/qwen_speech.py`, `core/evidence_processor.py`.

### D. Independent cognitive core

Already implemented and rich. Target additions: a **knowledge-validation
service** that runs `acceptance_policy` + `consistency_checker` +
`contradiction_detector` as one callable gate on every write (today each is
callable separately), and a **provenance query** over `core/provenance.py`.

### E. Long-term memory (the second-largest gap)

| Target memory | Today | Target work |
| --- | --- | --- |
| Working | ✅ in-process | keep |
| Episodic | ✅ in-process | **NEW** durable episodic store (JSONL/SQLite) reusing `storage/sqlite_store.py` |
| Semantic | 🟡 hash + learned space | **EXTEND** a durable semantic store keyed by the learned space |
| Procedural | ✅ in-process | **EXTEND** persistence for learned procedures |
| Consolidation | ✅ | wire consolidation → durable stores |
| Retrieval | ✅ | extend `memory_retriever` to read durable stores |
| Cross-session | 🟡 (`knowledge_carry` for relations) | **EXTEND** to memory items |

### F. Learning architecture (the largest gap)

| Target | Today | Target work |
| --- | --- | --- |
| Cognitive knowledge learning | ✅ | keep |
| Experience-based learning | ✅ | keep |
| **Controlled weight training (LoRA/PEFT)** | ❌ | **NEW** `core/finetune.py` + `ml/lora/` — LoRA/QLoRA over a local GGUF/transformers model, behind the existing `cognition/versions.ModelRegistry` gate (base immutable, candidate, evaluate, accept/rollback) |
| Dataset validation | ✅ | keep |
| Versioning / backup / rollback | ✅ | keep |

The LoRA work must reuse, not duplicate, `cognition/training.py` (staged,
resumable) and `cognition/versions.py` (the accept/rollback gate). LoRA only
adds a *parameter-efficient trainer* behind the same gate.

### G. Reasoning and intelligence

Already complete. Target addition: **uncertainty propagation through multi-step
reasoning** is present (`uncertainty_propagator.py`) — extend it to the composed
answer confidence so a chain's confidence is honest end to end.

### H. Planning and execution

Mostly complete. Target additions: **model-driven tool selection** (today a plan
is declared, not chosen) — a bounded chooser over the sandbox allow-list, so a
proposal still cannot widen authority; and **goal persistence** across sessions.

### I. Seven independent personalities

Already implemented with structural isolation. Target additions: **per-personality
training datasets and training history** (today a character has memory and
config; it has no dataset/version record), and **per-personality permissions**
(a character's allowed capabilities made explicit and enforced).

### J. Controlled self-improvement

Target: an automated but **human-gated** loop:

```
monitor → detect → propose → isolated test (code_studio worktree)
        → measure (evaluation) → accept | reject → rollback
```

Build on `core/code_studio.py` (isolated worktrees), `core/architecture_learning.py`
(recommendations), `cognition/versions.py` (accept/rollback), and `evaluation/`
(the measurement). The new piece is a **coordinator** (`core/self_improvement_loop.py`)
plus the event bus. Nothing merges without explicit human approval.

---

## 4. Offline-first target

| Layer | Offline status today | Target |
| --- | --- | --- |
| Language models | ✅ local GGUF/Ollama; cloud opt-in | keep; add a runtime check that the base path never opens a socket |
| Embeddings | ✅ learned in-project | keep; optional local embedding model |
| Databases | ✅ SQLite/JSON | keep |
| Retrieval | ✅ local corpus | keep |
| Memory | 🟡 partly durable | make durable |
| OCR | ✅ local Tesseract (extra) | keep |
| ASR/TTS | ✅ local engines | keep |
| Image processing | ✅ local (heuristic + local diffusion/VLM) | keep |
| Learning/training | ✅ local numpy; LoRA to add | keep; LoRA runs locally |
| Personalities | ✅ local | keep |
| Task execution | ✅ local sandbox | keep |

**Target guarantee:** a *base install* (only `numpy` + `requests` + `pypdf` +
`python-pptx`) must complete a full answer, a memory write, a training smoke
run, and a personality exchange **with the network disabled**. This becomes a
test (see `TEST_BASELINE.md`).

---

## 5. What must NOT change

- The knowledge graph, ontology, and reasoning layer (extend only).
- The groundedness gate and its fail-closed behaviour.
- The deny-by-default permission policy, approval gate, and audit log.
- Learning isolation (Natasha private; the other six shareable).
- Windows 11 compatibility and the GUI's existing functionality.
- Existing model weights and personality learning datasets (Phase 0 rule).
