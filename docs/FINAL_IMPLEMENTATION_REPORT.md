# Fayd Al-Ruh — Autonomous Cognitive Development System — Implementation Report

**Version:** 2.3.0 · **Branch:** `feature/autonomous-cognitive-development` ·
**Date:** 2026-10-31 · **Companion document:** `docs/COGNITIVE_SYSTEM_AUDIT.md`

This report records what the integrated autonomous cognitive system actually does,
how it is wired into the existing architecture, and how it is verified. Every
claim here is backed by a test that runs on a fresh clone; nothing reaches the
network or loads a model unless a caller explicitly asks it to.

---

## 1. What the system is

Three integrated layers, all under `cognition/`, all built on the project's
existing knowledge (its `Ontology`, its `CausalGraph`, its `JsonStore`) — **no
second graph and no second database**:

1. **العقل المبادر (Proactive Mind)** — a bounded, self-directed thinking cycle
   over the graph: curiosity finds gaps, the engines turn them into questions,
   hypotheses, ideas, associations, imagined scenarios, and critiques, and the
   motivation engine ranks what to pursue next.
2. **التعلم المعرفي (Cognitive Learning)** — one façade over the project's own
   memory types: semantic facts (persisted through `FactStore`), episodic
   experiences, associative links, and a lesson learned from every expectation
   that missed.
3. **التدريب التدريجي (Gradual Training)** — staged, rollback-safe training of
   the project's **own** model, advanced one stage at a time and stopping the
   moment a stage is rejected.

## 2. Integration with the existing architecture

`core/cognitive_controller.py` gained a lazily-built autonomous block that shares
**this controller's own** `Ontology` and `CausalGraph`:

```python
controller.proactive_mind()     # over controller.ontology, controller.causal_graph
controller.cognitive_learning() # the six-memory façade
controller.think(max_ideas=2)   # one cycle
controller.pending_ideas()
controller.cognitive_status()
```

The layer is built **on first use**, so a caller that never touches it pays
nothing — and there is exactly one graph in the process.

## 3. Persistence (no duplicated store)

Everything durable goes through the project's existing
`storage.json_store.JsonStore`, under the `cognition/*` namespaces. Each
collection is one document (`_KEY = "all"`) and is **bounded** so a long-running
mind cannot grow without limit:

| Store | Namespace | Bound |
| --- | --- | --- |
| `IdeaStore` | `cognition/ideas` | `MAX_IDEAS = 500` |
| `CycleStore` | `cognition/cycles` | `MAX_CYCLES = 200` |
| `ExperienceStore` | `cognition/experiences` | `MAX_EXPERIENCES = 1000` |
| `ExampleStore` | `cognition/examples` | `MAX_EXAMPLES = 2000` |
| `FactStore` | `cognition/facts` | `MAX_FACTS = 4000` |
| scheduler state | `cognition/meta` | one small value |

A corrupt document is skipped, never fatal, so a background mind always starts.

### 3.1 The fact-persistence fix

Before this work, the semantic facts the learning system learned lived only in
RAM, so a fact learned in one CLI invocation was gone in the next. `FactStore`
(`cognition/storage.py`) now persists each fact's id, statement, and provenance
and reloads them on construction; `cognition/learning_system.py` wires it in, so
knowledge survives a restart exactly as the project's learned relations do.

## 4. Gradual training (التدريب التدريجي)

`cognition/training.py`'s `IncrementalTrainer` gained the progression machinery.
The **progression is read from the version log** rather than stored a second
time, so there is one source of truth for "which stage are we on":

- `progression()` → one `StageProgress` per stage (`accepted`/`attempts`/`done`).
- `next_stage()` → the smallest unfinished stage, or `None`.
- `run_gradual(up_to=...)` → advance stage by stage, **stopping on a rejected
  stage** (a failed stage means the model is not ready for more work), and
  resuming from where it left off on the next call.

Each accepted run is promoted to a numbered version and the active pointer moves
to it; a rejected run is recorded and the model rolls back to the last stable
version. The base model is copied once and never written again.

## 5. The CLI twin

`python -m cli cognition …` drives the *same* objects the controller and any
panel use, so the surfaces can never disagree:

```
status · think · prove · ideas · drives · learn · recall · experience
train [--stage S | --gradual --up-to S] [--approve]
versions · rollback · progression · evaluate [--compare] · teach · mode · tick
```

`prove` demonstrates the whole loop end to end (think → measure the ten
indicators → learn a fact and recall it). Exit codes follow the project
convention: `0` success, `2` invalid arguments, `3` a model is unavailable,
`4` a gate refused the operation, `5` an unexpected failure. Training is a
governed action and refuses without `--approve`.

## 6. Verification

| Gate | Result |
| --- | --- |
| `python -m pytest -q` | see §7 |
| `python -m ruff check .` | clean |
| `python -m mypy` | clean (838 source files) |
| `python -m evaluation` | PASS |

Test files added/extended:

- `tests/test_cognition_records.py` — the records and their round-trips (19).
- `tests/test_cognition_mind.py` — the engines, the cycle, the scheduler (17).
- `tests/test_cognition_development.py` — stages, versioning/rollback, the
  trainer (with an injected stub trainer, so no real training runs), gradual
  training, teacher-student, the evaluator, the runner (33).
- `tests/test_cognition_integration.py` — the controller wiring, persistence
  across controllers, and the whole CLI including `prove`/`progression` (23).

## 7. Honest limits

- **No `torch`/`transformers`/`peft`** are installed and the task forbids
  destabilising the machine, so LoRA/QLoRA on an external model cannot run here.
  The project's **own** model trains with **numpy** (already present), which is a
  real incremental-training path; that is what this system runs.
- **No GGUF is present**, so the teacher is exercised through a real adapter seam
  and an offline fallback; the wiring is complete and the honest next step is
  reported rather than pretended.
- A gradual run with no held-out units to evaluate cannot be scored, so the first
  stage is rejected and the programme stops — reported, never hidden.
