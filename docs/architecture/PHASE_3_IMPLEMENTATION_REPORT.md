# Fayd Al-Ruh — Phase 3 Implementation Report

**Version:** 2.3.0 · **Branch:** `feature/phase-3-learning` · **Date:** 2026-11-02

Phase 3 — **Cognitive and Experience-Based Learning** — unifies the project's two
symbolic learning paths (learning a *fact* from information, and learning a
*lesson* from experience) behind **one governed pipeline**, makes consolidation
and review **durable**, and **measures** whether learning improved anything —
all **without touching a single model weight** (that is Phase 6).

Every module is **additive**: nothing existing was deleted, renamed, or weakened.
The learning system is wired in *beneath* the layers already there, reusing the
Phase-1 cognitive core (validator, transactional integrator, bounded ontology
learner) and the Phase-2 long-term memory rather than building a second graph or
a second store.

This document records what was actually built and measured, so Phase 4 can build
on it without re-deriving it.

---

## 1. What was added

| Module | Kind | Purpose |
| --- | --- | --- |
| `core/learning_events.py` | **new** | The one record contract: `LearningEvent` carries a monotonic id, timestamp, **kind** (knowledge/experience/feedback/pattern/procedure/rollback), **source** + reference, the **target** acted on, a change summary, **before/after** references, the **outcome**, whether it was **applied**, an optional **evaluation** and **version**, and the **owner** — so per-personality isolation is structural and every act is auditable. |
| `core/learning_store.py` | **new** | The durable store over `sqlite3` from the **standard library**: append-only audit trail, a monotonic **version** counter, **append-only rollback** (events at/after a version are marked `rolled_back`, never deleted), **transactional** writes, versioned `LearnedProcedure` records, and corrupt-row tolerance. One file per owner. |
| `core/knowledge_learning.py` | **new** | Cognitive knowledge learning: receive an item → identify provenance → extract claims (through the real `RelationExtractor`) → find duplicates → weigh evidence → detect contradictions → integrate what is accepted into the Phase-1 graph **transactionally** → preserve what is disputed → record the event. Reuses `KnowledgeValidator` and `KnowledgeIntegrator`; never re-implements them. |
| `core/experience_learning.py` | **new** | Experience-based learning: capture a task's goal, approach, outcome, errors, and provenance as an **episodic** memory (Phase 2); compare intended vs actual; decide whether a result supports a reusable lesson; and promote a lesson to a **procedural** memory only when it has real support. |
| `core/feedback_processing.py` | **new** | Feedback processing: turn a stated reaction (correct/incorrect/missing/preferred/rejected/correction/task-success/task-failure) into a governed signal tied to the answer, task, memory, claim, or procedure it is about, with **correction** and **retraction** supported. |
| `core/pattern_learning.py` | **new** | Pattern and procedure learning: read the durable episodic memories, identify recurring approaches, count supporting **and** contradicting experiences **separately**, record applicability conditions and failure modes, and version the resulting `LearnedProcedure` — never promoting a weak pattern. |
| `core/learning_coordinator.py` | **new** | The **learning coordinator**: one governed entry point that wires the four pipelines over one `LearningStore` per owner. Exposes `learn_knowledge`, `learn_claim`, `learn_experience`, `record_feedback`, `extract_patterns`, `status`, `history`, `rollback`, and the controlled `learn_from` sharing interface. |
| `core/learning_evaluation.py` | **new** | The measurement: snapshot → run a learning action → snapshot again → report the **measured** delta (knowledge integrated, lessons promoted, procedures trusted, feedback processed) and a named metric set. A zero delta is reported as zero; a fall is a regression. |
| `core/learning_gate.py` | **new** | The held-out gate (`learning_accuracy`) over the whole architecture, with **negative controls**. |
| `cli/learning.py` | **new** | The headless twin: `python -m cli learning status\|knowledge\|experience\|feedback\|patterns\|history\|metrics\|evaluate\|rollback`. Drives the *same* coordinator the controller and the GUI use. |

### Extended

- **`core/cognitive_controller.py`** — accepts an optional learning coordinator,
  built **lazily** through `learning()`, reusing the controller's own graph,
  ontology, bounded learner, validator, integrator, and long-term memory. Adds
  `learn_knowledge`, `learn_claim`, `learn_experience`, `record_feedback`,
  `extract_learning_patterns`, `learning_status`, and `_publish_learning`, which
  forwards each learning act onto the Phase-1 event bus as `learning.recorded`. A
  controller that never learns pays nothing and behaves exactly as before.
- **`evaluation/benchmark.py` / `runner.py` / `baseline.json`** — the new metric
  `learning_accuracy` enters `overall` (0.988212 → **0.988444**).
- **`cli/menu.py` / `cli/__main__.py`** — the `learning` command and the menu
  entry *Learning: what the system has learned, and how to teach it*.
- **`gui/cognition.py`** — the Tk-free `CognitiveService` gains
  `learn_knowledge`, `record_experience`, `record_feedback`, `extract_procedures`,
  `learning_status`, `learning_metrics`, `learning_history`, and
  `rollback_learning`, so the cognition window and the CLI share one learning
  system.
- **`gui/cognition_dialog.py`** — a new **Learning & Experience** sidebar section
  (key `learning3`) exposes all seven Phase-3 controls with the same
  catalogue-driven buttons, notes, and tooltips as every other control: *Learn
  knowledge from text* (with an input box), *Record an experience and its lesson*
  (goal + outcome boxes), *Give feedback on an answer* (a kind menu + subject
  box), *Extract versioned procedures*, *What the Phase-3 learning holds*, *The
  learning audit trail*, and a **governed** *Roll the learning back a version*
  that asks for confirmation first.
- **`core/cognition_catalog.py`** — seven new bilingual `Capability` entries and a
  new section (*Learning & Experience (governed)*), and the gate now resolves a
  capability's command against the **real** subcommands of whichever CLI module
  it names (`cli.cognition` *or* `cli.learning`), so a `cli learning …` string is
  checked as strictly as a `cli cognition …` one. The window-button set grew with
  it, so the *every button has an entry, every entry has a button* judgement still
  holds.
- **`cli/learning.py`** — `rollback` is now genuinely **governed**: it refuses
  without `--approve` (exit `4`), matching the catalogue's `governed=True` claim
  and the project's convention for a durable reversal.

---

## 2. The one governed pipeline

Both symbolic learning paths now flow through the coordinator, which is built for
**exactly one owner** and resolves that owner's store through
`core.learning_isolation` **before any path**:

```
information ─▶ KnowledgeLearningPipeline ─┐
                                          ├─▶ LearningCoordinator ─▶ LearningStore
experience  ─▶ ExperienceLearningPipeline ─┤        (one owner, one store)
feedback    ─▶ FeedbackProcessor          ─┤
patterns    ─▶ PatternLearning            ─┘
```

- **Knowledge learning** reuses the Phase-1 `KnowledgeValidator` (structure,
  semantics, evidence, contradiction, temporal) and `KnowledgeIntegrator`
  (transactional graph write), and reads typed relations from the text through the
  real `RelationExtractor`.
- **Experience learning** writes an `EPISODIC` memory through the Phase-2
  `LongTermMemory`; a lesson strong enough to generalise becomes a `PROCEDURAL`
  memory and a versioned procedure.
- **Feedback processing** reuses the Phase-2 memory and the existing
  `ContinuousLearner`; a correction is a new record, never a mutation.
- **Pattern learning** reads the durable episodic memories and versions the
  procedures it derives.

Nothing in this layer loads a model, touches a weight, or reaches the network.

---

## 3. Honesty rules, each enforced in code and pinned by a test

| Rule | Where enforced | Pinned by |
| --- | --- | --- |
| An unsupported claim is **never** integrated | `KnowledgeValidator` evidence check in `knowledge_learning` | gate `knowledge-refuses-unsupported-claim` |
| An evidenced claim **is** integrated | transactional integrator | gate `knowledge-integrates-evidenced-claim` |
| Provenance survives integration | graph stores source + reference + evidence | gate `knowledge-preserves-provenance` |
| A duplicate **reinforces**, adding no second edge | duplicate detection in `knowledge_learning` | gate `knowledge-duplicate-reinforces` |
| A **contradiction is preserved**, not resolved | opposing claims kept side by side | gate `knowledge-preserves-contradiction` |
| A single success never promotes a procedure | `DEFAULT_MIN_SUPPORT` in `experience_learning` / `pattern_learning` | gate `no-premature-generalisation` |
| A failure is a caution, not a rule | failure-mode recording in `pattern_learning` | gate `failure-is-a-caution` |
| One feedback record never overrides evidence | `FeedbackProcessor` weight bound | gate `feedback-never-overrides-evidence` |
| An inferred signal is bounded below an explicit one | `FEEDBACK_INFERRED` weighting | gate `inferred-below-explicit` |
| Silence is not approval | `feedback_from_reaction` raises on blank | gate `silence-is-not-approval` |
| Feedback is correctable and retractable | correction/retraction in the store | gate `feedback-is-correctable` |
| Versions advance and roll back; the audit survives | `LearningStore.rollback` (append-only) | gate `versioning-and-rollback`, `audit-trail` |
| A refused act is recorded | `LearningEvent` outcome `rejected` | gate `rejected-act-is-recorded` |
| Natasha's learning is excluded from the main model's dataset | `dataset_owners` / `learn_from` policy | gate `natasha-is-excluded` |
| No model weight is ever touched | the whole layer writes only SQLite | gate `no-model-weights-touched` |

The gate's **22 judgements** run against the **real** pipelines over a temporary
folder, so a learning regression the unit tests still pass fails here.

---

## 4. Isolation and the seven personalities

Isolation is **structural**, not a policy flag. A coordinator is constructed for
one `owner` and calls `assert_owner` before resolving a directory, so a
cross-personality call is refused *before* a path exists. `learn_from` refuses a
private owner (Natasha) **before** resolving any path, and `dataset_owners` —
the owners whose records may enter an aggregated dataset — always excludes her.
Each personality's learning lives in its own SQLite file under
`data/learning/<owner>/`.

---

## 5. Measured results

| Item | Value |
| --- | --- |
| New modules | 9 core + 1 CLI |
| Extended modules | 6 (`cognitive_controller`, `benchmark`, `runner`, `baseline`, `menu`, `gui/cognition`) |
| New tests (`tests/test_learning_phase3.py`) | 79 |
| New tests (`tests/test_learning_cli.py`) | 15 |
| Held-out gate `learning_accuracy` | **1.0000** |
| Gate judgements | 22 (with negative controls) |
| `overall` (evaluation) | 0.988212 → **0.988444** |
| `ruff` | All checks passed |
| `mypy` | no issues |

Measured end to end on this machine (a temporary data folder):

```
$ python -m cli learning knowledge "porosity increases permeability" --source document --confidence 0.8
knowledge from document: 1 claim(s), 1 integrated, 0 duplicate(s), 0 conflict(s), 0 rejected

$ python -m cli learning metrics --json
{"knowledge_events": 1.0, "experience_events": 0.0, "feedback_events": 0.0,
 "procedures": 0.0, "trusted_procedures": 0.0, "lessons": 0.0}

$ python -m cli learning history
[knowledge] porosity —increases→ permeability -> accepted (applied, owner=fayd_al_ruh, source=document)
```

A second identical write is reported as a duplicate that **reinforces** (no
second edge); a claim with no evidence is reported `0 integrated`; an experience
whose lesson has support 1 is `promoted=False (needs 1 more corroborating
experience(s))`.

---

## 6. Acceptance criteria (from the roadmap) — met

| Criterion | Status |
| --- | --- |
| Cognitive and experience learning write one provenance record | ✅ `LearningEvent` carries source, reference, owner, before/after |
| Nothing auto-verifies | ✅ an unsupported claim is never integrated; a single success never promotes |
| Consolidation/review durable | ✅ `LearningStore` persists events, versions, patterns, procedures |
| Isolation holds (Natasha private) | ✅ `learn_from` refuses her before any path; gate `natasha-is-excluded` |
| Model weights untouched | ✅ the layer writes only SQLite; gate `no-model-weights-touched` |
| `evaluation` green | ✅ PASS, overall 0.988444, no regression |

---

## 7. What Phase 4 builds on

- The **`LearningEvent` audit trail** and its version counter are the record the
  reasoning layer can cite when it explains *why* it believes something.
- The **`LearningStore`** already holds versioned procedures, so the planning
  layer (Phase 8) can select a proven procedure rather than re-deriving one.
- The **isolation policy** is already extended to learning datasets, so Phase 5
  only has to add training history and permissions on the same owner vocabulary.
- The **`learning_accuracy` gate** is the negative-control baseline any Phase-4
  reasoning change must not lower.
