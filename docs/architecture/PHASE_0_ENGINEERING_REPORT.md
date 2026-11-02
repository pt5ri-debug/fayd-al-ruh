# Fayd Al-Ruh — Phase 0 Final Engineering Report

**Project:** Fayd Al-Ruh · **Version:** 2.3.0 · **Commit:** `9c0b8f5`
**Report date:** 2026-11-02 · **Phase:** 0 (inspection, analysis, documentation)
**Method:** direct source inspection **and execution** — every figure in this
report and its companion documents was measured on the checkout, not copied from
the README.

> Phase 0 is **complete**. No source module, model weight, or personality
> learning dataset was modified. The only changes made are nine new documents
> under `docs/architecture/`. Phase 1 has **not** begun, per the rules.

---

## 1. Current project status

Fayd Al-Ruh is a **mature, well-engineered symbolic cognitive architecture** with
an unusually disciplined development process. It is far closer to the target
Cognitive Intelligence System than a typical research prototype: the knowledge
graph, reasoning layer, evidence/belief machinery, memory model, learning
pipeline, governance, and multi-personality system all exist and are tested.

| Dimension | Status |
| --- | --- |
| Version | 2.3.0 (single source in `core/version.py`, consistency-tested) |
| Branch / tree | `main`, clean, up to date with `origin/main` |
| Source (non-test) | 519 modules, ≈171,000 lines |
| Tests | 363 test files, 79,537 lines (≈46% of total) |
| Gates | ruff ✅ · mypy ✅ (872 files) · pytest ✅ (5824 passed, 15 skipped) · evaluation ✅ (55 metrics, overall 0.987721) |
| Offline | capable today; not yet *enforced or tested* |
| Target completeness | ≈ **80%** |

The four hard gates the project maintains (the sandbox, the groundedness gate,
the explicit-feedback rule, and the regression monitor) are intact and each can
refuse a change.

---

## 2. Existing functional capabilities

**Fully implemented and working (verified by reading and running):**

1. **Evidence-aware knowledge graph** — ontology (495 concepts / 474 relations /
   180 relation types), entities, typed relations, evidence, beliefs with a
   lifecycle, and a provenance ledger.
2. **Reasoning** — deductive / inductive / abductive reasoners, ontology
   entailment, causal reasoning (111 seeded causal edges, a closed causal-type
   set), counterfactuals, analogy by structure mapping, temporal reasoning,
   hypothesis lifecycle, and experiments.
3. **The grounded answer path** — retrieval (lexical + learned-semantic +
   ontology + BM25 bulk), a **groundedness gate that fails closed**, citations,
   and a user-visible reasoning trace.
4. **Reasoning-to-answer bridges** — plan, decision, belief, analogy, temporal,
   causal, connections, inference, and a composed confidence, each reaching the
   answer as data and each gated by a held-out metric with negative controls.
5. **Memory** — working, episodic, semantic (a learned distributional space), and
   procedural, with consolidation, decay, retention, and a fused retriever.
6. **Learning** — bounded cognitive learning (`OntologyLearner`,
   `relation_extraction`), experience-based learning (`cognition/`), and a
   staged, rollback-safe trainer for the in-project model.
7. **Model management** — a central `ModelManager`, a unified catalog, runtime
   adapters (llama.cpp / Ollama / in-process / cloud), task-based selection,
   memory estimation and preflight, and honest resource reporting.
8. **Perception** — document reading (TXT/MD/CSV/JSON/PDF/DOCX), OCR for scanned
   PDFs, a local vision-language model (Qwen3-VL), ASR (system + Qwen3-ASR), and
   four local TTS engines (system, Piper, Qwen3-TTS, VoiceTut/audio.cpp).
9. **Seven personalities + the project model** — each with its own persona,
   configuration, memory, learning store, model, and voice, with **structural**
   learning isolation (Natasha private).
10. **Governance** — deny-by-default permissions, a human-approval gate, an
    append-only audit log, a three-gate runtime, a root-confined tool sandbox,
    and governed model downloads.
11. **Two front ends** — a 4,279-line Tk window with 16 dialogs and 8 themes, and
    a CLI with 69 commands that is the headless twin of every window.
12. **Arabic support** — 100% concept-alias coverage, Arabic prose generation,
    Arabic answer bodies, and Arabic speech preparation (numbers/units spoken,
    not spelled).

---

## 3. Existing architecture and component relationships

The system is layered and **cycle-free between packages**:

```
front ends (gui/, cli/)  →  ai/ (mouths)  →  core/ (cognition)
                              ↓                ↑
                         cognition/  →  ml/, storage/, ingest/
```

- `core/cognitive_controller.CognitiveController` is the hub (70 methods),
  owning the ontology, graph, memory, state history, and causal graph, and
  seeding three declarative knowledge sources plus the Arabic glossary at start.
- `core/assistant.GroundedAssistant` is the one answer path.
- `core/model_manager.ModelManager` is the one model entry point.
- `core/learning_isolation.py` is the one source of truth for who exists and
  where their records live.
- `storage/` is stdlib-only; `core/` never imports `gui/` or `cli/`.

Full detail: `docs/architecture/CURRENT_ARCHITECTURE.md` and
`docs/architecture/COMPONENT_INVENTORY.md`.

---

## 4. Missing capabilities

Measured against the target (detail in `TECHNICAL_GAPS.md`):

| Missing | Priority | Nature |
| --- | --- | --- |
| **LoRA / PEFT controlled weight training** | P0 | new `core/finetune.py` + `ml/lora/` behind the existing version gate |
| **Event bus / pub-sub** | P0 | new `core/event_bus.py` |
| **Durable episodic/semantic memory** | P0 | extend memory + storage |
| **Enforced & tested offline operation** | P0 | new hard-off switch + `tests/test_offline.py` |
| **Automated (human-gated) self-improvement loop** | P0 | new `core/self_improvement_loop.py` over existing pieces |
| **Store recovery / backup-restore tests** | P0 | new tests |
| A usable in-project language model | P0 | extend `ml/llm` (BPE tokenizer, capacity, longer training) |
| Resource governor + general task queue | P1 | extend `model_resources` + `scheduler` |
| Uncertainty propagation into the composed answer | P1 | extend `answer_confidence` |
| Model-driven tool selection | P1 | extend `tool_agent` |
| Per-personality dataset/training-history/permissions | P1 | extend `versions` + `character_config` |
| Local embedding model; goal persistence; unified migrations | P2 | extensions |
| `self_model` name collision | P3 | rename |

---

## 5. Architectural weaknesses

1. **The generative path is the weak link.** The in-project LLM has 842,624
   parameters, a byte tokenizer, and produces incoherent English (measured, §6).
   Every *useful* answer today comes from retrieval + reasoning. The "language
   model as a component" cannot yet draft prose.
2. **No event bus.** Subsystems are wired by direct calls, so a monitor (needed
   for self-improvement), a live UI, and training hooks must each be bolted on.
3. **Memory is not fully durable.** Working and episodic memory live in-process;
   only learned *relations* survive a session boundary by default.
4. **The self-improvement pieces exist but are not connected.** Isolated
   worktrees (Code Studio), recommendations, and version rollback are all
   present; no coordinator drives them.
5. **Offline operation is a default, not a guarantee.** Web search is off by
   default and downloads are gated — but nothing fails if a path reaches the
   network.
6. **The trained artifact is stale.** The committed checkpoint was produced at
   revision `b615191`, many commits before HEAD `9c0b8f5` (measured), and nothing
   warns.

None of these is a design flaw; each is an *unfinished extension point*.

---

## 6. Duplicated functionality

No module is a behavioural duplicate of another (full analysis in
`COMPONENT_INVENTORY.md` §10). Three *near*-overlaps exist and are intentional or
harmless:

- `core/curiosity.py` vs `cognition/engines/curiosity.py` — the latter composes
  the former.
- `core/consistency_checker.py`, `core/relation_consistency.py`,
  `core/contradiction_detector.py` — three checks over three different inputs
  (beliefs / relations / evidence); consolidating them behind one validation
  service is a Phase-1 target, not a defect.
- `core/model_resources.py` vs `cognition/scheduler.py` — both read machine load;
  no shared governor (Phase 1 adds one).

**One genuine naming collision:** `core/self_model.py:SelfModel` (a declared
domain identity) and `cognition/metacognition.py:SelfModel` (a learned
self-model) — two different things, one name. Rename (P3).

---

## 7. Hardware and performance limitations

Measured on the audit host (Linux, 4 CPU, 16 GiB, **no GPU**) — see
`RESOURCE_ASSESSMENT.md`:

- The symbolic path (seed + retrieve + reason) costs **< 2 s and < 1 GiB**.
- The **target machine's GPUs cannot be used**: the AMD Radeon R7 M360 (GCN 1st
  gen, 2015, 2 GB) is not supported by a current ROCm/PyTorch build, and the
  Intel HD 520 (1 GB, shared) offers no useful acceleration for these stacks.
  **Plan for CPU inference.**
- **One large model at a time** on 16 GB. Feasible: 1.7B–4B Q4 comfortably;
  7–8B tight; two large models no.
- **Training:** the in-project numpy model trains in minutes; **CPU LoRA** on a
  1.7B–4B quantised model is feasible but slow (hours); full fine-tuning is not.
- Large-model operations (load, training, vision model) must be **sequential**;
  background thought cycles and bulk builds can run in **idle** periods.

The architecture is already CPU-first and GPU-agnostic, so a stronger machine
can be introduced later with no redesign.

---

## 8. Offline compatibility assessment

**The system is already offline-capable for normal operation.**

- The base install depends on `numpy`, `requests`, `pypdf`, `python-pptx` — all
  local libraries.
- Every internet use is an explicit, governed action: web search (off by
  default), the bulk Wikipedia build, model downloads (approval-gated), the
  update check, and the Cloud mouth (off by default).
- No optional engine (torch, llama.cpp, piper, qwen, diffusers) imports at module
  load; each reports the real next step when absent.

**The gap is enforcement, not capability:** there is no `FAYD_OFFLINE` hard-off
switch and no test asserting a base-install path opens no socket (G-8 / F-6).

---

## 9. Security and data integrity assessment

Strong. The full analysis is in `SECURITY_ASSESSMENT.md`; the essentials:

- **Deny-by-default permissions**, a **human-approval gate**, an **append-only
  audit log**, a **three-gate runtime**, and a **root-confined tool sandbox**.
- **Structural learning isolation** — the directory is resolved from the policy
  before any read/write; Natasha is the one private owner; sharing fails closed;
  a hand-edited config cannot enable her sharing. Pinned by a held-out gate with
  negative controls.
- **No secret is written to disk or a log** — the Cloud key is read from the
  environment at call time and redacted everywhere; SQL import redacts
  credentials.
- **Weaknesses found:** offline operation not enforced (F-6, P0); no
  store-recovery test (F-1, P0); no auto-backup before a destructive op (F-5,
  P1); GGUF validated by magic only (F-4, P2); no unified migration framework
  (F-2, P2); audit-log persistence depends on a caller (F-7, P2).
- **No recommendation weakens an existing control.**

---

## 10. Test baseline and actual results

**Gates (measured):**

| Gate | Result |
| --- | --- |
| `python -m ruff check .` | ✅ `All checks passed!` |
| `python -m mypy` | ✅ `Success: no issues found in 872 source files` |
| `python -m evaluation` | ✅ 55 metrics, **overall 0.987721** (causal/memory/petroleum/planning/reasoning/safety = 1.0; uncertainty 0.867935) |
| `python -m pytest -q` (under `xvfb-run`) | ✅ **5824 passed, 15 skipped, 0 failed, 0 errors** in 30:12 |

**Environment note (important for reproducibility):** run **without** a display,
the suite produces **4 collection errors** (`test_gui_typography.py`,
`test_progress.py`, `test_typography_settings.py`, `test_window_controls.py`)
because Tk cannot be imported. This is **environmental, not a code defect** — CI
installs Xvfb for exactly this reason, and the tests collect cleanly once Tk is
present. Recorded so a future headless run is not mistaken for a regression.

**Coverage against the audit's required topics:** knowledge consistency,
evidence validation, memory retrieval, learning isolation, model management,
reasoning, and task execution are **well covered** (many with held-out gates and
negative controls). The genuinely missing areas are **offline operation**,
**database recovery**, and **backup/restore** — all P0/P2 in the roadmap.

---

## 11. Prioritized implementation roadmap

Nine phases, ≈79–101 engineer-days, 11 new modules (all coordinators/facades),
preserving every existing capability. Full detail in
`IMPLEMENTATION_ROADMAP.md`.

| Order | Phase | New modules | Effort (ED) |
| ---: | --- | ---: | ---: |
| 1 | Cognitive Core Enhancement (event bus, governor, validation, uncertainty) | 3 | 6–8 |
| 2 | Long-Term Memory Architecture (durable episodic/semantic) | 1 | 8–10 |
| 3 | Cognitive & Experience-Based Learning | 0 | 6–8 |
| 4 | Reasoning & Inference Engine (facade, tool choice) | 1 | 6–8 |
| 5 | Seven Independent Personalities (datasets, history, permissions) | 0 | 8–10 |
| 6 | Controlled Model Weight Training (LoRA/QLoRA + LLM repair) | 4 | 15–20 |
| 7 | Local Multimodal Perception (perception→evidence bridge) | 1 | 10–12 |
| 8 | Autonomous Planning & Task Execution | 0 | 8–10 |
| 9 | Controlled Self-Improvement (the automated, human-gated loop) | 1 | 12–15 |

**Why this order:** Phase 1 is the smallest, lowest-risk change and the hard
dependency of both the governor and the self-improvement monitor. Phase 2 makes
memory durable, which Phases 3/7/8 consume. Phase 6 (the largest gap) depends on
Phase 5's per-owner datasets. Phase 9 consumes every prior phase's telemetry.

---

## 12. Exact recommended first task for Phase 1

> **Implement `core/event_bus.py` and publish the controller's lifecycle events
> through it.**

**Why this task, and why first:**

- **Necessary:** the target requires event-driven communication, and the
  self-improvement monitor (Phase 9) and the resource governor (Phase 1) both
  need a way to observe the system without being called from inside it.
- **Smallest and safest:** it is a *new*, dependency-free module (no existing
  behaviour changes), so it can be landed and verified in isolation with all four
  gates green.
- **Sets the pattern:** it establishes the observability convention every later
  phase reuses (bounded, `best_effort`, audited, a raising subscriber isolated).
- **New component, not an extension:** nothing in the repository provides this
  today (verified by search), and bolting it onto an existing module would
  violate the single-responsibility of the controller.

**Definition of done:** `core/event_bus.py` with synchronous + optional bounded
queue modes; a raising subscriber never reaches the publisher; an append-only
record for the audit; the controller publishes `cycle_started`,
`cycle_finished`, `belief_revised`, and `relation_learned`; unit tests for
ordering, isolation, and the bound; ruff/mypy/pytest/evaluation all green with no
change to any reasoning result.

---

## 13. Deliverables produced by Phase 0

All under `docs/architecture/`:

1. `CURRENT_ARCHITECTURE.md` — what exists today, subsystem by subsystem.
2. `TARGET_ARCHITECTURE.md` — the target design, mapped onto existing modules.
3. `COMPONENT_INVENTORY.md` — every module, its key API, and the duplication analysis.
4. `DEPENDENCY_MAP.md` — every third-party and system dependency, classified.
5. `TECHNICAL_GAPS.md` — 18 gaps (G-1…G-18) with evidence, priority, and fix.
6. `IMPLEMENTATION_ROADMAP.md` — the nine phases with effort, risks, tests, acceptance criteria.
7. `TEST_BASELINE.md` — the reproducible baseline and coverage analysis.
8. `RESOURCE_ASSESSMENT.md` — measured environment and target-machine guidance.
9. `SECURITY_ASSESSMENT.md` — the security posture and eight findings.
10. `PHASE_0_ENGINEERING_REPORT.md` — this report.

---

## 14. Compliance with the mandatory development rules

| Rule | Compliance |
| --- | --- |
| 1. Do not rebuild from scratch | ✅ no source changed |
| 2. Do not delete working modules | ✅ nothing deleted |
| 3. Do not duplicate functionality | ✅ the roadmap adds coordinators, not logic |
| 4. No destructive DB migrations | ✅ none performed |
| 5. Do not modify model weights | ✅ none modified (the stale-artifact finding is recorded, not fixed) |
| 6. Do not change personality datasets | ✅ none changed |
| 7. Do not weaken security controls | ✅ none changed; findings are additive |
| 8. Preserve Windows 11 compatibility | ✅ no code changed |
| 9. Preserve GUI functionality | ✅ no code changed |
| 10. Backward compatibility | ✅ no interface changed |
| 11. Document dependencies & decisions | ✅ `DEPENDENCY_MAP.md`, all decisions recorded |
| 12. Use actual test results | ✅ every figure measured |
| 13. No cloud services or mandatory external APIs | ✅ none introduced |
| 14. Do not begin Phase 1 until Phase 0 is reported | ✅ **Phase 1 has not begun** |

---

## 15. Conclusion

Fayd Al-Ruh already **is** an evidence-aware symbolic cognitive architecture with
a working knowledge graph, reasoning, memory, learning, governance, and seven
isolated personalities. Its answer path — retrieval + a fail-closed groundedness
gate + a visible reasoning trace + reasoning bridges — is genuinely strong, and
its engineering discipline (four gates, held-out metrics with negative controls,
a high test ratio) is exemplary.

To become the target **fully local Cognitive Intelligence System**, four things
are needed, in order: an **event bus and resource governor** to make the core
observable and arbitrated (Phase 1); **durable long-term memory** (Phase 2); a
**parameter-efficient, rollback-safe weight trainer** (Phase 6, the largest gap);
and the **automated but human-gated self-improvement loop** that ties the
existing pieces together (Phase 9). Eleven new modules — all small coordinators
or facades over code that already exists and is tested — close the gap.

**The project must not be rebuilt.** Every recommendation extends what is there.
Phase 0 is complete; Phase 1 may begin on the user's approval, starting with
`core/event_bus.py`.
