# Independent Character Intelligence — Architecture Analysis

**Version:** 2.3.0 · **Scope:** the seven answering characters (Jarvis, Ultron,
Vision, Banner, Natasha, Hakeem, Cloud) and the project's own **Fayd Al-Ruh
model**: their configuration, their learning stores, the isolation between
those stores, and the controlled knowledge-sharing interface.

This document is **Phase 1** of the upgrade: what already exists, what is
missing, and the plan that reuses the existing architecture rather than
replacing it. The implementation report (`CHARACTER_IMPLEMENTATION_REPORT.md`)
records what was actually built and measured.

---

## 1. Executive summary

The project already has, and has had for several releases, **seven answering
characters plus the project's own model**, each with a standing personality and
its own small learning file:

| Where | What it already gives us |
| --- | --- |
| `core/character_learning.py` | The seven `PERSONAS` (+ the project model), `CharacterLearning` (a per-character JSON record of **facts** and **exchanges**), and the corpus bridge that trains the project model on them. |
| `ai/characters.py` | `CharacterMouth` — the transparent wrapper that prepends a character's standing instruction and records every exchange. |
| `core/mouth_profile.py` | Per-mouth **name / logo / reading voice / language model**, persisted through `JsonStore`. |
| `core/language_models.py` | The **one** language-model catalog (every GGUF, the cloud providers, the trained checkpoint), de-duplicated by weight identity, with real availability. |
| `core/voices.py` / `core/speech.py` / `core/voice_moods.py` | The voice kinds (catalog / `SystemVoice` / `ModelVoice`), the engines, the mood prosody, and the honest per-engine support notes. |
| `core/knowledge_carry.py`, `core/ontology_learner.py`, `core/relation_extraction.py` | The graph's learned relations and the bounded write path. |
| `core/audit_log.py` | The append-only, tamper-evident trail every governed action writes to. |
| `gui/sidebar.py`, `gui/actions.py`, `gui/model_chat*.py` | The collapsible sidebar, the grouped actions, and the per-mouth chat windows. |

So the *characters* are real. What the task asks for, and what is **not** yet
present, is:

1. **A structured, editable, persisted configuration** per character (personality
   beyond a fixed tuple, language model *parameters*, voice parameters, learning
   settings) — today a personality is a hard-coded dataclass and the model/voice
   choices are single strings.
2. **A structured learning-record store** per character — today a character keeps
   facts and exchanges, but there is no record of *provenance*, *confidence*,
   *verification status*, or *source type*, and nothing distinguishes an
   unverified model claim from a user statement.
3. **Enforced isolation** between those stores, and a **controlled knowledge-sharing
   interface** with the **Natasha exclusion** enforced in the backend.
4. **A learning pipeline** that separates model-generated, internet-sourced,
   user-provided, shared, and system-generated information, and that never treats
   a model response as verified truth.
5. **A Characters management surface** in the sidebar with an APPLY that validates,
   activates, and rolls back.

The rest of this document establishes the constraints that shape that work.

---

## 2. Existing components (what must not be broken)

### 2.1 The cognitive core

`core/` is the evidence-aware symbolic architecture: the concept graph
(`core/ontology.py`, `core/concept.py`), entities/relations (`core/entity.py`,
`core/relation*.py`), evidence (`core/evidence.py`), beliefs
(`core/belief*.py`), causality (`core/causal_*.py`), and the reasoning bridges
(`core/answer_*.py`). The seven characters **read** this graph through the
grounded assistant; **nothing in this upgrade may write to it** except through the
existing bounded paths (`OntologyLearner`, `KnowledgeUpdater`). The new learning
stores are **not** the knowledge graph: they are per-character *records*, and the
main model reads permitted records from them — it does not merge them into the
symbolic graph behind the user's back.

### 2.2 The characters and their mouths

* `core.character_learning.CHARACTER_KEYS` = `("sources", "ultron", "vision",
  "banner", "natasha", "hakeem", "cloud")` — the seven; `PROJECT_MODEL_KEY =
  "model"` — the project's own model. These keys are the *answering-mouth* keys
  the chat already uses, so a character and its mouth are the same thing. **This
  must not change**: `ai.chat.ModelSwitcher`, `gui.service.ChatService`,
  `core.language_models.MOUTHS`, `core.mouth_profile.DEFAULT_NAMES`, the CLI, and
  the GUI all key off them.
* `PERSONAS[key]` is a `CharacterPersona` (`key`, `name`, `role`, `identity`,
  `traits`, `rules`) with a `system_prompt(learned_facts=...)`. This is the
  **default** personality; the new configuration system must be able to override
  it *without* losing the existing defaults.
* `ai.characters.CharacterMouth` wraps a mouth's assistant, prepends the
  personality, and records the exchange. It is deliberately transparent (unknown
  attributes are forwarded, `__class__` reports the wrapped mouth) — every
  existing caller and test depends on that.

### 2.3 Persistence

* `storage.json_store.JsonStore(root)` — namespaced, atomically-written JSON
  (`save` writes a `.tmp` and `os.replace`s it). Percent-encodes namespace/key so
  a logical name is a legal filename on Windows. **This is the project's atomic
  write primitive and should be reused.**
* `core.paths` — resolves bundled assets vs. writable data. Source checkouts keep
  learned data under a **git-ignored** `data/` folder beside the code;
  `FAYD_DATA_DIR` overrides. `characters_dir()` = `data/characters`,
  `character_learning_dir()` = `data/characters/learning`.
* `core.best_effort.best_effort(what)` — the one place a deliberate "must not fail
  the caller" operation lives; it records what was dropped (`swallowed_errors()`).
* `core.audit_log.AuditLog` — append-only `AuditRecord`s (`capability`, `target`,
  `outcome`, `reason`, `actor`, `metadata`), with `denials()`.

### 2.4 Model and voice assignment (today)

* **Model:** `core.mouth_profile.MouthProfile.model` is a single catalog key;
  `ai.mouths.build_mouth(mouth, model_key=...)` resolves it to a GGUF path (or a
  cloud provider + model) and hands it to the mouth's own loader.
  `gui.service.ChatService.set_mouth_model` remembers it and reloads the active
  mouth; `ai.chat.ModelSwitcher.invalidate_mouth` drops the cached mouth so the
  reload really takes the new file. **There is no temperature / context-length /
  max-tokens setting per character today** — the mouths' loaders take
  `max_new_tokens` but nothing persists it.
* **Voice:** `core.mouth_profile.MouthProfile.voice` is a voice key
  (`system:<engine>:<name>`, `model:<engine>:<name>`, or a catalog key).
  `gui.model_chat.ModelChatService` keeps a `TextReader` per mouth, so a voice is
  already exclusive to a mouth. **Speed / pitch / volume per character are not
  persisted** (the mood prosody is per *engine studio*, in
  `core.voice_studio_profile.py`).
* **Honesty rules that must be preserved:** `ENGINE_MOOD_SUPPORT` in
  `core.voice_moods.py` states which engine supports which prosody; the voice
  menus only offer what an engine really reports here
  (`core.voices.voice_sections`). A configuration must be **validated against the
  engine before it is applied**, not assumed.

### 2.5 The sidebar and the windows

`gui/sidebar.py` gives the collapsible `SidebarSection` / `SidebarColumn`.
`gui/actions.py` declares `ACTIONS` (each mapping to a `cli.menu.run_*` function),
`ACTION_GROUPS` (the named sections), `grouped_visible_actions()` (every visible
action in exactly one section; an unnamed action is appended to the last section
so it cannot vanish), and the `hidden` flag for actions reached from elsewhere.
`gui/app.py:_build_sidebar` draws them. `tests/test_sidebar.py` pins the grouping
invariants — so **adding a `CHARACTERS` section is safe and expected**, and the
new action must land in exactly one group.

### 2.6 The evaluation gate

`evaluation/` is the regression gate: `benchmark.py` declares `MetricSpec`s,
`runner.py` computes them, `baseline.json` is the reference, and
`python -m evaluation` must print PASS. **Adding a new capability means adding a
held-out metric with negative controls and updating the baseline** — the project's
own rule (AGENTS.md). The overall score is the mean of the metric list, so a new
metric enters it.

---

## 3. The gaps, and why the plan closes them

| # | Gap | What the plan adds |
| --- | --- | --- |
| 1 | Personality is a fixed dataclass; nothing is user-editable or persisted | `core/character_config.py` — a validated, serialisable `CharacterConfig` (General / Personality / Model / Voice / Learning) whose **defaults derive from the existing `PERSONAS` and catalogs**, so nothing is duplicated |
| 2 | No central registry; settings live in three unrelated stores | `core/character_registry.py` — the one loader/saver/validator, with APPLY, hot reload, and rollback |
| 3 | No structured learning records | `core/learning_records.py` + `core/character_learning_store.py` — append-only JSONL records with provenance, confidence, verification status, source type, evidence references, and learning status |
| 4 | No isolation between characters | `core/learning_isolation.py` — the store can only ever open its **own** directory; `Natasha` is private; a cross-character read raises `LearningAccessError` and is logged |
| 5 | No controlled knowledge sharing | `core/knowledge_sharing.py` — the main model reads the **six** shareable characters, and writes a **new** record with full provenance into its own store; it never edits the source record |
| 6 | No learning pipeline | `core/learning_pipeline.py` — receives a response, extracts statements, checks contradictions, scores confidence, stores with a status, and never auto-verifies |
| 7 | No Characters management surface | `gui/character_manager.py` (Tk-free service), `gui/character_dialog.py` (the window), `cli/characters.py` (the headless twin), and a `CHARACTERS` sidebar section |

---

## 4. Design decisions (and the constraint behind each)

1. **Additive, not a rewrite.** Every new module sits beside the existing ones and
   reuses them: `JsonStore` for atomic writes, `best_effort` for non-fatal
   cosmetic writes, `AuditLog`-style append-only records for the learning audit,
   `core.language_models` for model validation, `core.voices` /
   `core.voice_moods` for voice validation, `PERSONAS` for personality defaults,
   `CharacterLearning` for the existing facts/exchanges memory.

2. **Two learning layers, one per concern — not two competing engines.**
   `CharacterLearning` keeps the **memory and interaction history** (facts the
   character was told, exchanges it took part in) and is what the project model
   trains on. The new `CharacterLearningStore` keeps **learning records**
   (provenance, confidence, verification status, evidence). The pipeline writes
   *both* from one event, so they cannot disagree; the existing memory keeps its
   exact API and tests.

3. **Isolation is structural, not a policy flag.** A `CharacterLearningStore` is
   constructed for exactly one owner and resolves exactly one directory. There is
   no method that takes another owner's key. The sharing service is the **only**
   component that reads across characters, and it goes through
   `assert_shareable(key)`, which refuses Natasha before any path is built. The GUI
   cannot offer what the backend refuses, and the backend refuses regardless of the
   GUI.

4. **The main model's store is separate and writes its own records.** Sharing is
   *read the source, write my own*: the main store gets a new record carrying
   `source_character`, `source_record_id`, `source_date`, and the source's
   confidence; the original file is opened read-only and never written.

5. **Nothing is auto-verified.** A record's default status is `pending`; a
   language-model answer or an internet result can never enter as `verified`. The
   settings' `verification_threshold` / `confidence_threshold` and the
   `automatic`/`supervised` mode decide the *most* a new record may reach.

6. **Configuration is validated before it is applied, and rolled back if
   activation fails.** `CharacterRegistry.apply(key, draft)` validates the draft,
   snapshots the current config, saves the new one, tries to activate the affected
   components, and restores the snapshot if activation raises. A setting that
   cannot be hot-applied is reported as **restart-required** rather than silently
   ignored.

7. **Paths follow the project's own convention.** Shipped, read-only defaults live
   in the repository (`config/`); the user's saved configuration and all learning
   data live under the git-ignored `data/` folder, so an update can fast-forward
   without touching what the user configured or learned.

8. **The evaluation gains held-out metrics with negative controls** for isolation
   and sharing, so a regression (a character reading another's store, or Natasha's
   records leaking) fails the gate rather than passing unnoticed.

---

## 5. Storage layout

```
config/                              # shipped, tracked defaults (read-only)
  characters/
    jarvis.json ultron.json vision.json banner.json
    natasha.json hakeem.json cloud.json
  fayd_al_ruh.json                   # the project model's own defaults

data/                                # git-ignored, writable
  config/                            # the user's saved configuration (APPLY writes here)
    characters/*.json
    fayd_al_ruh.json
  learning/
    fayd_al_ruh/                     # the main model's independent learning directory
      main/                          #   its own records
        records.jsonl  stats.json
      characters/                    #   one directory per character, strictly isolated
        jarvis/   records.jsonl stats.json
        ultron/   ...
        vision/   ...
        banner/   ...
        natasha/  ...                #   PRIVATE — never read by the main model
        hakeem/   ...
        cloud/    ...
      audit/                         # append-only learning audit (incl. refused access)
        learning-audit.jsonl
  characters/learning/               # the existing facts/exchanges memory (unchanged)
```

`records.jsonl` is append-only: a record's later revision is another line, so
nothing is overwritten and a revised conclusion keeps its reason and its
predecessor. `stats.json` is a small, atomically-replaced metadata file so a
statistics view never has to read the whole store.

---

## 6. What is deliberately **not** done

* **No fine-tuning of the GGUF weights.** "Learning" here means persistent
  external memory and knowledge acquisition, exactly as the task states. The
  in-project model can still be *trained* on the characters' learning through the
  existing `--llm-characters` corpus bridge — that is a separate, existing path.
* **No change to the symbolic knowledge graph's write rules.** The stores are
  records; only the existing bounded learners touch the graph.
* **No new third-party dependency.** Everything uses the standard library plus
  what the base install already has.
* **No new mode or mouth.** The seven keys and the project model key stay exactly
  as they are.
