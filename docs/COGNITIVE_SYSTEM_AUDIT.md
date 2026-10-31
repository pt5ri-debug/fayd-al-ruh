# Fayd Al-Ruh — Cognitive System Audit

**Version:** 2.3.0 · **Branch:** `feature/autonomous-cognitive-development` ·
**Audit date:** 2026-10-31 · **Method:** direct source inspection and execution
(no claim in this document is taken from the README alone).

This is the entry audit for the **Fayd Al-Ruh Autonomous Cognitive Development
System** task. It records what already exists, what is reused, what is missing,
and the environment the new system must run within.

---

## 1. Project structure

| Area | Files | Purpose |
| --- | --- | --- |
| `core/` | 297 modules | The cognitive architecture: ontology, reasoning, memory, beliefs, evidence, learning, causal, planning, decision, analogy, speech, models, characters |
| `ml/` | 13 modules + `ml/llm/` (12) | Learning: the Q-policy, the petroleum tasks, and the in-project transformer language model |
| `gui/` | 42 modules | The Tk desktop window and its dialogs |
| `cli/` | 23 modules | The headless twin of every window |
| `ai/` | mouths + session wiring | Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, Cloud, the model mouth |
| `storage/` | `json_store`, `sqlite_store`, `graph_database` | Persistence |
| `evaluation/` | `runner`, `benchmark`, `baseline.json` | The regression gate (50 metrics) |
| `ingest/` | connectors + pipeline | Local-first document ingestion |
| `companion/`, `hakeem/` | self-contained character packages | Natasha and Hakeem |
| `tests/` | 327 files | The test suite (5,115 passing, 247 skipped at audit time) |

Total: **1,004 Python files**.

## 2. Program entry point

- **GUI:** `python -m cli gui` → `python -m gui` → `gui/app.py:FaydGUI`.
- **CLI menu:** `python -m cli` → `cli/__main__.py` → `cli/menu.py:MENU`.
- **Demo:** `python -m cli demo` → `main.py`.
- **Training:** `python -m cli release` → `ml/release.py`.

## 3. Cognitive core

`core/cognitive_controller.py:CognitiveController` is the hub. It owns a
`KnowledgeGraph`, a `CognitiveMemory`, a `CognitiveState`, and a `StateHistory`,
and exposes the whole reasoning layer: `reason`, `discover_questions`,
`open_inquiries`, `recall_episodes`, `learn_from_prediction_error`,
`revise_belief`, `check_consistency`, `create_plan`, `decide`, `introspect`,
`analogize`, `extract_relations`, `learn_extracted_relations`, and more (50+
public methods). The controller is the one object the answer path and the
evaluation both build.

## 4. Entities, relations, evidence, beliefs

All present and rich:

- `core/entity.py` — `Entity`, `EntityType`.
- `core/relation.py` — `Relation` (typed, directed, with confidence).
- `core/evidence.py` — `Evidence`.
- `core/belief.py` — `Belief` (status lifecycle, confidence, evidence list).
- `core/ontology.py` — `Ontology` with a fast token→concept index (495 concepts,
  474 relations seeded), `match_concepts`, `add_concept_alias`.
- `core/knowledge_graph.py`, `core/relation_polarity.py`,
  `core/relation_consistency.py`, `core/relation_inference.py`.

## 5. Memory and storage

| Need | Existing module | Reuse decision |
| --- | --- | --- |
| General memory items | `core/memory.py:CognitiveMemory` | **Reuse** |
| Dump/restore | `core/memory_store.py:MemoryStore` | **Reuse** |
| Episodic | `core/episodic_memory.py:EpisodicMemory` | **Reuse** |
| Semantic | `core/semantic_memory.py:SemanticMemory` | **Reuse** |
| Procedural | `core/procedural_memory.py:ProceduralMemory` | **Reuse** |
| Associative | `core/memory_associator.py:MemoryAssociator` | **Reuse** |
| Working memory | `core/working_memory.py:WorkingMemory` | **Reuse** |
| Retrieval | `core/memory_retriever.py:MemoryRetriever` | **Reuse** |
| Consolidation | `core/memory_consolidator.py:MemoryConsolidator` | **Reuse** |
| Decay | `core/memory_decay.py:MemoryDecay` | **Reuse** |

Persistence backends: `storage/json_store.py:JsonStore` (namespaced, atomic,
Windows-safe key encoding) and `storage/sqlite_store.py:SqliteStore` (stdlib
SQLite). **Decision:** the new system uses `JsonStore` and **does not** create a
second database — the task forbids duplicating a database for the same purpose.

## 6. Goals and drives

- `core/goal.py` — `Goal`, `GoalStatus`, `PlanStep`.
- `core/planner.py` — `Planner.order_steps`.
- `core/plan.py`, `core/plan_evaluator.py`, `core/decision_engine.py`.
- `core/curiosity.py:Curiosity.discover(beliefs, required_confidence)` →
  `list[InformationGap]` — **the existing curiosity engine; reuse it.**
- `core/information_gap.py:InformationGap` with `.priority` and `.resolved`.
- `core/question_generator.py:QuestionGenerator.generate(gaps, limit)` →
  `list[Question]`, `.open_inquiries(gaps, limit)` → `list[Inquiry]`.
- `core/hypothesis_engine.py:HypothesisEngine.create(pattern)` → `Hypothesis`.
- `core/pattern_detector.py:PatternDetector.detect(...)` → `list[Pattern]`.
- `core/motivation` — **absent**; a new motivation engine is required.

## 7. Learning

- `core/learning_pipeline.py` — the controlled path: receive → extract →
  detect contradictions → confidence → store → (optional) graph update.
- `core/continuous_learning.py:ContinuousLearner` (explicit feedback only).
- `core/knowledge_update.py:KnowledgeUpdater` (provenance + decay).
- `core/ontology_learner.py:OntologyLearner` (bounded graph learning).
- `core/self_learning.py`, `core/self_improvement.py`, `core/metacognition.py`,
  `core/introspection.py`, `core/adaptive_learning.py`.
- `core/learning_records.py:LearningRecord` (frozen, provenance-carrying).
- `core/character_learning.py` — the seven characters' own files.

## 8. The Fayd Al-Ruh Model

`ml/llm/transformer.py` — a decoder-only transformer with **manual backprop**,
trained with **numpy only** (already a base dependency). Presets:

| Preset | layers | heads | embd | block | params (measured) |
| --- | --- | --- | --- | --- | --- |
| `small` | 3 | 4 | 96 | 64 | ~0.4 M |
| `release-1` (shipped) | 4 | 8 | 128 | 128 | **842,624** |
| `capable` | 6 | 8 | 192 | 128 | ~2.4 M |

Shipped checkpoint: `ml/llm/checkpoints/fayd-llm.npz` (6.4 MB, 842,624 params,
byte tokenizer, 4,000 steps, held-out perplexity **4.27**, bits/byte 2.12).

## 9. Model loading

- `core/model_interface.py` — `CognitiveModel` protocol, `ModelRequest`,
  `ModelResponse`, `CallableModel`.
- `core/model_runtime.py` — `RuntimeAdapter` protocol with four adapters:
  `LlamaCppAdapter`, `OllamaAdapter`, `InProcessTrainedAdapter`, and the cloud
  path. Each reports `capabilities`/`status`/`load`/`unload`/`infer`.
- `core/model_manager.py` — one API over the registry, the runtimes, and the
  catalog; `models_for_task`, `select_for_task`, `diagnostics`.
- `ml/llm/generator.py:LocalGenerator` — text generation from the trained model.

## 10. Model format

- The project's own model: **`.npz`** (numpy), loaded by `TransformerLM.load`.
- The mouths' external models: **GGUF** (llama.cpp) under `data/<mouth>/`.
- No GGUF is present in this environment (verified: `data/**/*.gguf` is empty).

## 11. Ollama

- `core/coder_model.py:OllamaCoderClient` — dependency-free `urllib` client.
- `core/model_runtime.py:OllamaAdapter`.
- `companion/model.py`, `hakeem/model.py` — GGUF-first with Ollama fallback.
- **Verified in this environment: Ollama is not reachable** (`coder_status`
  reports `server_reachable=False`).

## 12. Other language models

`core/language_models.py:all_language_models()` — a unified catalog:
**38 distinct models** across the mouths (Ultron, Vision, Banner, Hakeem,
Natasha/shared, Cloud). Verified available here: the retrieval mouth plus four
Ollama/LM-Studio entries in the catalog (Ollama itself unreachable).

## 13. The seven characters

`core/character_learning.py:CHARACTER_KEYS` = `sources`, `ultron`, `vision`,
`banner`, `natasha`, `hakeem`, `cloud`; `PROJECT_MODEL_KEY = "model"`.
`core/character_config.py` — five-section editable config per character.
`core/character_registry.py` — shipped defaults in `config/`, applied config in
`data/config/`. `core/character_identity.py` — one identity per character
(owner ↔ mouth). `core/learning_isolation.py` — structural isolation, Natasha
private.

## 14. Per-character learning files

`data/characters/learning/character_learning/<owner>.json` (facts ≤ 100,
exchanges ≤ 200). Isolated stores under `data/learning/fayd_al_ruh/...`.
**Rule enforced by the task and by this design:** the proactive mind must never
write into a character's learning file, and no character may write into Fayd
Al-Ruh's memory.

## 15. Services and background processes

**No background service exists.** There is no daemon, no Task Scheduler helper,
no auto-start. `core/agent_loop.py` and `core/cognitive_loop.py` are *in-process*
loops only. The GUI runs long work on worker threads with a queue drained by the
UI thread. **A new background runner is required.**

## 16. User interface

`gui/app.py:FaydGUI` — the Tk window. Sidebar actions in `gui/actions.py`
(`ACTIONS`, `ACTION_GROUPS`, `grouped_visible_actions`). Each action has a
`command` and `argv` and calls into `cli/menu.py`. Dialogs are opened from the
composer's ＋ menu. Every feature has a headless CLI twin.

## 17. Tests and dependencies

- **Tests:** `pytest` (327 files; 5,115 pass, 247 skip headless).
- **Quality:** `ruff` (E4/E7/E9/F/W/I/UP/B/C4/RUF), `mypy` (strict-ish).
- **Base dependencies** (`requirements.txt`): `requests`, `numpy`, `pypdf`,
  `python-pptx`, `pytest`, `ruff`, `mypy`. Everything heavy is an optional extra.

## 18. Windows 11 compatibility

Already handled throughout: `core/paths.py` (portable data dir),
`storage/json_store.py` (percent-encoded keys, `WinError 87` fix),
`core/speech.py` (OneCore/SAPI), `core/docs_studio.py` (`Path.as_uri`),
`core/window_controls.py` (`rcWork` work area). The new system follows the same
rules: only `pathlib`, `json`, `sqlite3`, `subprocess`, `ctypes` for Windows
integration, and no POSIX-only calls.

## 19. Resource compatibility

Target machine (per the task, verified against this container's real figures):

| Resource | Target (Dell E5570) | This container (measured) |
| --- | --- | --- |
| CPU | Intel Core i7 | 4 vCPU |
| RAM | 16 GB | 16 GB total, 14 GB available |
| GPU | integrated, no CUDA | none (`torch` absent) |
| Disk | — | 71.6 GB total, 59.9 GB free |
| Python | 3.13.3 | **3.13.15** |
| Ollama | when present | **not reachable** |
| `shared_models` GGUF | Qwen3.5-2B-Q4_K_M | **absent** |

## 20. What is missing (the gap this task closes)

1. **Proactive Mind** — no curiosity→question→hypothesis→idea pipeline that runs
   on its own; `Curiosity`/`QuestionGenerator`/`HypothesisEngine` exist but are
   only called from the controller and the evaluation.
2. **Association / idea / imagination / self-criticism / motivation engines** —
   absent as a coherent, persistent system.
3. **Thought scheduler** — absent.
4. **Cognitive Learning System** — the memory types exist individually; there is
   no single façade that records an experience, compares an expectation to its
   outcome, analyses the error, and stores a lesson with provenance and
   reliability.
5. **Teacher–Student system** — absent.
6. **Incremental training for the project's own model** — the trainer exists
   (`ml.llm.transformer`) but there is no experience→dataset→adapter→evaluate→
   accept/reject pipeline, and no versioning/rollback.
7. **Cognitive evaluation harness** — `evaluation/` measures the symbolic layer,
   not the trained model's own before/after ability.
8. **Background operation** — absent entirely.

## 21. Honest environment constraints

- **No `torch`/`transformers`/`peft`/`safetensors`** are installed, and the task
  forbids installing a training stack that would destabilise the machine. So
  **LoRA/QLoRA on an external GGUF/HF model cannot be executed here.** The
  project's **own** model trains with **numpy** (already present), which *is* a
  real incremental-training path, and it is what this system implements and runs.
- **No GGUF is present**, so the teacher model is exercised through a real
  adapter seam and an offline fallback; the wiring is complete and the honest
  next step is reported.
- The LoRA path is implemented as a **documented, guarded, ready-to-run** stage
  that reports the missing dependency rather than pretending to train.

## 22. Reuse summary (no duplication)

| New need | Reused existing component |
| --- | --- |
| Curiosity | `core/curiosity.py:Curiosity` |
| Gaps | `core/information_gap.py:InformationGap` |
| Questions | `core/question_generator.py:QuestionGenerator`, `core/question.py` |
| Hypotheses | `core/hypothesis_engine.py`, `core/hypothesis.py` |
| Patterns | `core/pattern_detector.py` |
| Episodic memory | `core/episodic_memory.py` |
| Semantic memory | `core/semantic_memory.py` |
| Procedural memory | `core/procedural_memory.py` |
| Associative memory | `core/memory_associator.py` |
| Beliefs/evidence | `core/belief.py`, `core/evidence.py` |
| Ontology | `core/ontology.py`, `core/ontology_learner.py` |
| Persistence | `storage/json_store.py:JsonStore` |
| Model runtime | `core/model_runtime.py`, `core/model_manager.py` |
| Student model | `ml/llm/transformer.py` + `ml/llm/generator.py` |
| Progress | `core/progress.py:ProgressReporter` |
| Best-effort | `core/best_effort.py` |
| Paths | `core/paths.py` |
| GUI actions | `gui/actions.py`, `cli/menu.py` |

The implementation follows in Parts 3–18 of the task; the delivery report is
`docs/FINAL_IMPLEMENTATION_REPORT.md`.
