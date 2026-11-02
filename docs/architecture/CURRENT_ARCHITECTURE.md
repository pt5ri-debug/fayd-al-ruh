# Fayd Al-Ruh — Current Architecture (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02 · **Method:** direct source
inspection and execution. Every figure in this document was measured from the
checkout at commit `9c0b8f5`; none is copied from the README alone.

This document describes **what exists today**. It deliberately separates
*implemented* from *partial* from *absent*. The target design is in
`TARGET_ARCHITECTURE.md`; the gap list is in `TECHNICAL_GAPS.md`.

---

## 1. What the system is

Fayd Al-Ruh is an **evidence-aware symbolic cognitive architecture** framed as a
petroleum-engineering mind. Its centre of gravity is a **symbolic knowledge
graph** (ontology + entities + relations + evidence + beliefs) with a rich
reasoning layer, wrapped in a retrieval-augmented answer path, a Tk desktop
front end, a headless CLI twin of every window, and seven character "mouths"
plus the project's own small trained language model.

The project is **not** a large language model application. Language models are
components it can use; the cognitive core is the graph and the reasoning layer.

---

## 2. Measured size and shape

| Area | Modules (`.py`) | Lines | Role |
| --- | ---: | ---: | --- |
| `core/` | 303 | 82,356 | Ontology, reasoning, memory, beliefs, evidence, causal, planning, decision, analogy, speech, models, characters |
| `gui/` | 48 | 35,810 | Tk desktop window and 16 dialogs |
| `ai/` | 29 | 8,524 | Mouths (Jarvis/Ultron/Vision/Banner/Cloud/Natasha/Hakeem), session wiring, model mouths |
| `cli/` | 25 | 11,409 | Headless twin of every window; 69 subcommands; 99-item numbered menu |
| `cognition/` | 29 | 9,692 | Autonomous cognitive development system (proactive mind, learning system, metacognition, consolidation, curriculum, study) |
| `companion/` | 18 | 6,871 | Natasha — self-contained companion package (own SQLite DB) |
| `ingest/` | 19 | 5,153 | Local-first ingestion (files, SQL, web; OCR path) |
| `ml/` | 27 | 6,421 | Q-policy, petroleum tasks, in-project transformer LLM, release trainer |
| `hakeem/` | 7 | 2,119 | Hakeem — scholar package (own model folder + DB) |
| `storage/` | 4 | 154 | `JsonStore`, `SqliteStore`, `GraphDatabase` |
| `evaluation/` | 7 | 2,393 | Regression gate: 55 metrics + baseline |
| `tests/` | 378 | 79,537 | Test suite |
| **Total** | **897** | **250,570** | |

Non-test source is therefore ≈ **519 modules / ≈ 171,000 lines**; the test
suite is ≈ **46%** of total lines — a high, healthy ratio.

Largest modules: `gui/app.py` (4,279), `core/speech.py` (2,439),
`core/presentation.py` (2,113), `core/audiocpp_speech.py` (1,969),
`gui/presentation.py` (1,944).

---

## 3. Layered architecture as built

```
┌────────────────────────────────────────────────────────────────────────┐
│  FRONT ENDS        gui/ (Tk, 16 dialogs)   cli/ (69 commands, 99 menu)  │
├────────────────────────────────────────────────────────────────────────┤
│  MOUTHS            ai/  Jarvis · Ultron · Vision · Banner · Cloud ·     │
│                    Natasha · Hakeem · project model (mind)              │
│                    ai/characters.py wraps each with personality+memory  │
├────────────────────────────────────────────────────────────────────────┤
│  ANSWER PATH       core/assistant.py (GroundedAssistant)                │
│                    core/rag.py (lexical + learned-semantic + ontology)  │
│                    core/grounding.py (groundedness gate, fails closed)  │
│                    core/answer_lead.py · reasoning_report.py            │
├────────────────────────────────────────────────────────────────────────┤
│  REASONING BRIDGES core/answer_{plan,decision,belief,analogy,temporal}.py│
│                    core/relation_{facts,paths,connections,inference,    │
│                    consistency}.py · core/causal_paths.py               │
│                    core/answer_confidence.py                            │
├────────────────────────────────────────────────────────────────────────┤
│  COGNITIVE CORE    core/cognitive_controller.py  (hub, 70 methods)      │
│                    core/ontology.py (495 concepts, 474 relations)       │
│                    core/knowledge_graph.py (entity/relation/evidence/   │
│                    belief, referential integrity)                       │
│                    core/{deductive,inductive,abductive}_reasoner.py     │
│                    core/{planner,decision_engine,belief_revision}.py    │
├────────────────────────────────────────────────────────────────────────┤
│  MEMORY            core/{working,episodic,semantic,procedural}_memory.py│
│                    core/memory.py (CognitiveMemory) · memory_consolidator│
├────────────────────────────────────────────────────────────────────────┤
│  LEARNING          core/ontology_learner.py (bounded) · relation_extract│
│                    ml/ (Q-policy, transformer LLM) · cognition/training │
├────────────────────────────────────────────────────────────────────────┤
│  MODEL LAYER       core/model_manager.py (the one entry point)          │
│                    core/model_runtime.py (llama.cpp/Ollama/in-process/  │
│                    cloud adapters) · core/language_models.py (catalog)  │
├────────────────────────────────────────────────────────────────────────┤
│  GOVERNANCE        core/permissions.py (deny-by-default)                │
│                    core/human_approval.py · core/audit_log.py           │
│                    core/sandbox_runtime.py (3 gates)                    │
├────────────────────────────────────────────────────────────────────────┤
│  PERSISTENCE       storage/{json_store,sqlite_store,graph_database}.py  │
│                    core/{persistence,knowledge_store,snapshot}.py       │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 The cognitive controller is the hub

`core/cognitive_controller.py:CognitiveController` owns a `KnowledgeGraph`, a
`CognitiveMemory`, a `CognitiveState`, a `StateHistory`, a `CausalGraph`, an
`Ontology`, and a `ProvenanceLedger`. It exposes **70 methods** covering
reasoning, discovery, inquiry, belief revision, consistency, planning,
decision, introspection, analogy, and relation extraction/learning. Its
`__init__` seeds the ontology from three declarative sources and applies the
Arabic glossary:

```python
seed_ontology(self.ontology)          # 17 disciplines, concepts + relations
seed_curriculum(self.ontology)        # 48 courses
seed_petroleum_corpus(self.ontology)  # 48 corpus entries
apply_arabic_aliases(self.ontology)   # 495/495 Arabic aliases
self.causal_graph = causal_graph_from_ontology(self.ontology)  # 111 links
```

**Measured:** 495 concepts, 474 relations, 180 distinct relation types,
111 directed causal links, 17 causal relation types, Arabic alias coverage
495/495 (100%), curriculum coverage 1.0, corpus coverage 1.0.

### 3.2 The answer path

`core/assistant.py:GroundedAssistant.ask` is the one user-facing answer path:

1. retrieve passages (`core/rag.RagRetriever` — lexical 0.7 / learned-semantic
   0.1 / ontology 0.2, plus a BM25 bulk corpus at weight 0.6);
2. optionally let a model write the prose, then run `core/grounding.check_grounding`
   (term coverage ≥ 0.3, per-sentence support ≥ 0.5, causal-direction check);
   **fail closed** → fall back to the deterministic extractive answer;
3. attach reasoning as *data*: `relation_facts`, `inferred`, `connections`,
   `causal_paths`, `relation_conflicts`, `plan`, `decision`, `belief`,
   `analogy`, `temporal`, `reasoning` trace, `confidence_report`;
4. build a **lead** (`core/answer_lead.py`) and open the answer body with it
   when the question is procedural/decisional/evaluative/analogical/temporal.

**Measured behaviour (executed):**

- `"what is porosity?"` → cited prose from the corpus (3 citations).
- `"كيف أقيس المسامية؟"` (Arabic procedural) → procedure lead, Arabic body,
  Arabic citations. The Arabic path works end to end.
- `"how do I measure porosity on a core sample?"` → `procedure` lead + cited body.

This is the strongest part of the system.

### 3.3 The model layer

`core/model_manager.py:ModelManager` is the single entry point over a registry,
file inspection, settings, runtimes, and a log. `core/model_runtime.py` declares
adapters for `llama.cpp`, `ollama`, `fayd-inprocess`, and `cloud`.
`core/language_models.py` is a unified catalog de-duplicated by identity.
Task-based selection (`core/model_selection.py`), memory estimation and preflight
(`core/model_preflight.py`), and honest resource reporting
(`core/model_resources.py`) all exist.

---

## 4. Subsystem-by-subsystem status

Legend: **✅ implemented** · **🟡 partial** · **❌ absent**

### A. Core infrastructure

| Capability | Status | Evidence |
| --- | --- | --- |
| Local runtime management | ✅ | `core/model_runtime.py` (4 adapters), `core/model_manager.py` |
| Event-driven communication | ❌ | no event bus / pub-sub anywhere (grep for `EventBus`/`publish`/`subscribe` finds nothing) |
| Resource management | 🟡 | `core/model_resources.py`, `cognition/scheduler.py`, `cognition/runner.py` (LIGHT/BALANCED/INTENSIVE) — but no global arbiter across subsystems |
| Task scheduling | 🟡 | `cognition/scheduler.py` schedules *thought cycles* only; no general task queue |
| Database management | 🟡 | `storage/sqlite_store.py`, `core/knowledge_index.py` (FTS5), `storage/graph_database.py` (in-memory); no unified DB layer or migrations framework beyond `core/snapshot_migrations.py` |
| Security & permissions | ✅ | `core/permissions.py`, `core/human_approval.py`, `core/audit_log.py`, `core/sandbox_runtime.py` |
| Monitoring & logging | 🟡 | `core/audit_log.py`, `core/model_log.py`, `core/error_monitor.py`, `core/progress.py` — no unified metrics/health surface |

### B. Local model management

| Capability | Status | Evidence |
| --- | --- | --- |
| Multiple local models | ✅ | catalog: Ultron, Vision, Banner, Natasha, Hakeem, shared, presentation, coder, trained |
| GGUF + llama.cpp | ✅ | `core/model_runtime.LlamaCppAdapter`, lazy import |
| Ollama integration | ✅ | `core/model_runtime.OllamaAdapter`, `core/coder_model.OllamaCoderClient` |
| Load / unload | ✅ | `ModelManager.load`/`unload`, `active_models()` |
| Resource estimation | ✅ | `core/model_preflight.estimate_memory` (band + unknowns), `preflight` |
| Embedding models | 🟡 | no neural embedding model; a **learned distributional space** (`core/embeddings.py`, PPMI + SVD, 128-d) is trained in-project |
| Specialised models | ✅ | coder (Ollama/GGUF), petroleum (OGAI), presentation (Qwen3), docs (Qwen3-VL) |

### C. Perception and understanding

| Capability | Status | Evidence |
| --- | --- | --- |
| NLP | ✅ | tokenizer, relation extraction, composition, Arabic normalisation |
| Document understanding | ✅ | `core/document_reader.py`, `core/docx_extract.py` |
| PDF + OCR | ✅ | `core/pdf_extract.py`, `core/pdf_ocr.py` (Tesseract, optional extra) |
| Image understanding | ✅ | `core/vision.py` (heuristic), `core/docs_model.py` (Qwen3-VL vision-language, local) |
| Audio processing | ✅ | `core/speech.py` (ASR/TTS contracts) |
| Speech recognition | ✅ | system + `core/qwen_speech.QwenAsrRecognizer` (Qwen3-ASR, Arabic) |
| Text-to-speech | ✅ | system (SAPI/OneCore/espeak/say), Piper, Qwen3-TTS, VoiceTut/audio.cpp |
| Multimodal | 🟡 | Qwen3-VL reads page images; no general multimodal fusion into the graph |

### D. Independent cognitive core

| Capability | Status | Evidence |
| --- | --- | --- |
| Entity management | ✅ | `core/entity.py` |
| Relationship management | ✅ | `core/relation.py`, `core/ontology.py` |
| Evidence management | ✅ | `core/evidence.py`, `evidence_processor/evaluator/weighting/trace.py` |
| Knowledge graph | ✅ | `core/knowledge_graph.py` (referential integrity) |
| Knowledge validation | ✅ | `core/acceptance_policy.py`, `core/consistency_checker.py` |
| Contradiction detection | ✅ | `core/contradiction_detector.py`, `relation_polarity.py`, `relation_consistency.py` |
| Confidence evaluation | ✅ | `core/confidence_model.py`, `uncertainty.py`, `answer_confidence.py` |
| Knowledge provenance | ✅ | `core/provenance.py`, `core/learning_records.py` |

### E. Long-term memory

| Memory | Status | Evidence |
| --- | --- | --- |
| Working | ✅ | `core/working_memory.py` (capacity 7, LRU) |
| Episodic | ✅ | `core/episodic_memory.py` (time-stamped, importance) |
| Semantic | 🟡 | `core/semantic_memory.py` (hash index) + `core/embeddings.py` (learned space); no persistent unified semantic store |
| Procedural | ✅ | `core/procedural_memory.py` (declarative only — never executes) |
| Consolidation | ✅ | `core/memory_consolidator.py`, `memory_decay.py`, `retention_policy.py` |
| Retrieval | ✅ | `core/memory_retriever.py` (fuses lexical + semantic + associative) |
| Persistence | 🟡 | `core/knowledge_store.py` snapshots graph+memory; `core/knowledge_carry.py` persists *learned relations* across sessions; **working/episodic memory are in-process only** |

### F. Learning architecture

| Mechanism | Status | Evidence |
| --- | --- | --- |
| Cognitive knowledge learning | ✅ | `core/ontology_learner.py` (bounded), `relation_extraction.py`, `knowledge_update.py`, `learning_pipeline.py` |
| Experience-based learning | ✅ | `core/continuous_learning.py`, `cognition/learning_system.py`, `cognition/consolidation.py` |
| Controlled weight training | 🟡 | `ml/` + `cognition/training.py` + `cognition/versions.py` (base immutable, candidate, evaluate, accept/rollback) — but **only the in-project numpy transformer**, and **no LoRA/PEFT** |
| Dataset validation | ✅ | `core/acceptance_policy.py`, `cognition/reliability.py` |
| Evaluation / versioning / backup / rollback | ✅ | `cognition/versions.ModelRegistry` (accept only on no-regression; rollback; base kept) |
| Parameter-efficient fine-tuning (LoRA) | ❌ | **no LoRA/PEFT/QLoRA anywhere** (grep confirms) |

### G. Reasoning and intelligence

| Capability | Status | Evidence |
| --- | --- | --- |
| Problem analysis | ✅ | `core/inquiry.py`, `information_gap.py`, `question_generator.py` |
| Hypothesis generation | ✅ | `core/hypothesis*.py`, `cognition/engines/ideas.py` |
| Evidence-based inference | ✅ | `core/inference*.py`, `ontology_reasoner.py`, `relation_inference.py` |
| Multi-step reasoning | ✅ | `relation_paths.py`, `relation_connections.py`, `causal_paths.py` |
| Logical consistency | ✅ | `consistency_checker.py`, `belief_conflict.py` |
| Result verification | ✅ | `core/grounding.py`, `evaluation_suite.py`, `plan_evaluator.py` |
| Confidence estimation | ✅ | `answer_confidence.py`, `uncertainty_propagator.py` |
| Decision support | ✅ | `decision_engine.py`, `utility_model.py`, `risk_model.py`, `answer_decision.py` |

### H. Planning and execution

| Capability | Status | Evidence |
| --- | --- | --- |
| Goal management | ✅ | `core/goal.py` |
| Task decomposition | ✅ | `core/plan.py`, `planner.order_steps` (topological) |
| Planning | ✅ | `core/planner.py` (BFS search + dependency order) |
| Tool selection | 🟡 | `core/tool_agent.ToolPlan` — plan is *declared*, not chosen by the model |
| Controlled execution | ✅ | `core/tools.ToolSandbox` (deny-by-default, root confinement) |
| Progress monitoring | ✅ | `core/progress.py` |
| Result evaluation | ✅ | `core/plan_evaluator.py`, `experiment_runner.py` |
| Experience recording | ✅ | `cognition/learning_system.py` (episodic store) |

### I. Seven personalities

| Character | Package | Own config | Own memory | Own model | Own voice |
| --- | --- | --- | --- | --- | --- |
| Jarvis | `ai/characters.py` | ✅ | ✅ | retrieval (no model) | ✅ |
| Ultron | `core/ultron.py` | ✅ | ✅ | ✅ | ✅ |
| Vision | `core/vision_model.py` | ✅ | ✅ | ✅ | ✅ |
| Banner | `core/banner.py` | ✅ | ✅ | ✅ | ✅ |
| Natasha | `companion/` | ✅ | ✅ | ✅ (`data/natasha`) | ✅ |
| Hakeem | `hakeem/` | ✅ | ✅ | ✅ (`data/hakeem`) | ✅ |
| Cloud | `core/cloud_model.py` | ✅ | ✅ | remote | ✅ |
| (project model) | `ml/llm/` | ✅ | ✅ (`fayd_al_ruh`) | ✅ | — |

Isolation is enforced structurally: `core/learning_isolation.py` resolves the
directory **before** any read/write, Natasha is the one `PRIVATE_OWNER`, and
`core/knowledge_sharing.py` refuses her before it touches a path.

### J. Controlled self-improvement

| Capability | Status | Evidence |
| --- | --- | --- |
| Monitor performance | ✅ | `core/error_monitor.py`, `evaluation/`, `cognition/evaluation.py` |
| Detect errors/weaknesses | ✅ | `core/self_improvement.py`, `core/architecture_learning.py` |
| Identify missing capabilities | ✅ | `core/information_gap.py`, `cognition/knowledge_depth.weak_spots()` |
| Propose code improvements | 🟡 | `core/architecture_learning.py` proposes *recommendations* (text) only |
| Isolated tests | ✅ | `core/code_studio.py` (git worktree per change) + `run_checks` |
| Measure improvements | ✅ | `cognition/versions.should_accept`, `evaluation` baseline |
| Detect regressions | ✅ | `evaluation/` (55 metrics vs `baseline.json`) |
| Reject / restore | ✅ | `cognition/versions.rollback`, `update_baseline` |
| **Automated** end-to-end self-modification | ❌ | the loop is **human-gated**: `SelfImprovement.propose` returns a plan with `requires_human_review=True`; nothing edits code automatically |

---

## 5. The autonomous cognition subsystem (`cognition/`)

A proactive mind that runs on the project's **own** graph (never a second
graph/database). `cognition/proactive_mind.py` composes eight engines into one
bounded thought cycle; `cognition/scheduler.py` decides *when* it may run;
`cognition/runner.py` drives it in LIGHT/BALANCED/INTENSIVE modes and defers when
the machine is busy. `cognition/study.py` asks the question bank, reads the
grounded assistant's cited answers, and **measures** the benefit with a
`CognitiveReport` before/after. `cognition/training.py` + `versions.py` provide
staged, rollback-safe weight training. `cognition/knowledge_depth.py` measures
graph reach honestly (isolated concepts, Arabic coverage, weak spots).

Nothing here writes into a character's learning files or into the belief graph.

---

## 6. Persistence and data layout

- Source checkout writes learned data to a git-ignored `data/` beside the code;
  `FAYD_DATA_DIR` overrides; a frozen build writes next to the executable
  (`core/paths.py`).
- `storage/json_store.JsonStore` — namespaced JSON documents, atomic
  (`tmp` + `replace`), percent-encoded keys (Windows-safe), legacy-path tolerant.
- `storage/sqlite_store.SqliteStore` — namespaced JSON values in SQLite.
- `core/knowledge_index.py` — contentless SQLite **FTS5** index over the bulk
  corpus (~81k chunks indexed in ~5 s; query in ms).
- Snapshots: `core/snapshot.py` + `snapshot_migrations.py` (schema v2, ordered
  side-effect-free migrations).

---

## 7. Governance and safety as built

- **Deny-by-default permission policy** (`core/permissions.py`): external tools
  are refused unless explicitly granted.
- **Human approval gate** (`core/human_approval.py`): a human decision with
  expiry, required for external actions.
- **Append-only audit log** (`core/audit_log.py`): every governed decision.
- **Three-gate runtime** (`core/sandbox_runtime.GovernedRuntime`): capability →
  approval → audit.
- **Tool sandbox** (`core/tools.py`): allow-list, root confinement, read/write
  separation.
- **Learning isolation** (`core/learning_isolation.py`): structural, per-owner.
- **Model download gates**: every fetch requires `approved=True`.

---

## 8. Front ends

- **GUI** (`gui/app.py`, 4,279 lines): OpenHands-styled Tk chat window, 8
  appearance themes, Word-like font/paragraph settings, collapsible sidebar,
  model-switch row (7 mouths + 4 speech-engine studios), 16 dialogs
  (Code Studio, Presentation Studio + 2 siblings, Doc's Studio, Import Center,
  SQL import, Scene Studio, image gen/edit, per-mouth chat, voice studios,
  Characters, Cognition, Model Manager). 53 actions in `gui/actions.py`.
- **CLI** (`cli/`): 69 commands, a 99-item numbered menu, the headless twin of
  every window, UTF-8 console reconfiguration for Arabic on Windows.

---

## 9. What is genuinely strong

1. **The grounded answer path** — retrieval + citations + a real groundedness
   gate that fails closed, plus a user-visible reasoning trace.
2. **Reasoning-to-answer bridges** — plan, decision, belief, analogy, temporal,
   causal, connections, inference, confidence: all reach the answer as data and
   each has a held-out gate with negative controls.
3. **Arabic support** — 100% alias coverage, Arabic prose generation, Arabic
   TTS number/unit preparation, Arabic answer bodies.
4. **Governance** — deny-by-default permissions, human approval, append-only
   audit, structural learning isolation.
5. **Engineering discipline** — ruff clean, mypy clean on 872 files, 55-metric
   evaluation gate with a baseline, high test ratio.
6. **Model-agnostic by construction** — the cognitive core never depends on one
   model; a catalog + adapters + fallback ladders.

## 10. What is weak or missing (summary — detail in `TECHNICAL_GAPS.md`)

1. **The in-project language model is functionally unusable** — 842k parameters,
   a byte tokenizer, and garbled output (measured).
2. **No LoRA / PEFT** — "controlled model weight training" cannot yet adapt a
   real GGUF model; only the numpy transformer is trainable.
3. **No event bus** and no unified scheduler/resource arbiter.
4. **Memory is not fully durable** — working/episodic memory are in-process.
5. **No offline-operation test** and no enforcement that the base install never
   reaches the network (it is only *off by default*).
6. **Self-improvement is human-gated only** — no automated propose→test→measure
   →accept/reject loop.
7. **No DB recovery/backup test**; no unified migration framework.
