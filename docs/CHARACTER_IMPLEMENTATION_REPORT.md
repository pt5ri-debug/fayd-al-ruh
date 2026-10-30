# Independent Character Intelligence — Implementation Report

**Version:** 2.3.0 · **Branch:** `feature/independent-characters` · **Status:**
implemented, tested, lint- and type-clean.

This is the companion to `docs/CHARACTER_ARCHITECTURE.md` (the Phase-1 analysis).
It records what was actually built, where it lives, how the isolation and the
Natasha exclusion are enforced, and how to run and use it.

---

## 1. What was built, in one paragraph

Seven independent characters (Jarvis, Ultron, Vision, Banner, Natasha, Hakeem,
Cloud) and the project's own **Fayd Al-Ruh model** each get a **complete, editable
configuration** (general / personality / language model / voice / learning) and a
**strictly isolated learning store** on disk. A **controlled knowledge-sharing
interface** lets the main model learn from six of them — **never Natasha** — while
copying each permitted record into its *own* store with full provenance. A
**Characters** window (and a headless CLI twin) lets a user pick a character, edit
its configuration, and press **APPLY** to validate, save, activate what it can now,
and **roll back** on failure.

Nothing in the existing architecture was replaced: the knowledge graph, entities,
relations, evidence, beliefs, personas, mouth profiles, the language-model catalog,
the voice engines, and every existing window are unchanged and still used.

---

## 2. Architecture analysis (Phase 1, summarised)

The project already had, for several releases, seven answering characters plus the
project model, each with a standing personality and a small per-character learning
file. What it did **not** have was: a *structured, editable configuration* per
character; *independent learning stores* with enforced isolation; a *controlled
knowledge-sharing interface*; and a *management interface* with APPLY.

| Already existed (reused) | Where |
| --- | --- |
| The seven `PERSONAS` + the project model | `core/character_learning.py` |
| `CharacterMouth` — the transparent wrapper that prepends the standing instruction and records exchanges | `ai/characters.py` |
| Per-mouth name / logo / reading voice / language model | `core/mouth_profile.py` |
| The one language-model catalog (de-duplicated by weight identity) | `core/language_models.py` |
| Voice kinds, engines, moods, per-engine support notes | `core/voices.py`, `core/speech.py`, `core/voice_moods.py` |
| The bounded ontology write path and the relation reader | `core/ontology_learner.py`, `core/relation_extraction.py`, `core/knowledge_carry.py` |
| The append-only governed-action trail | `core/audit_log.py` |
| The collapsible sidebar and the grouped actions | `gui/sidebar.py`, `gui/actions.py` |
| Atomic JSON persistence, best-effort writes, path resolution | `storage/json_store.py`, `core/best_effort.py`, `core/paths.py` |

The upgrade is therefore **additive**: new modules beside the old ones, wired in at
the few points that already existed (the character wrapper, the sidebar, the CLI
menu, `core/paths.py`).

---

## 3. The independent learning architecture

### 3.1 One store per owner, resolved by the policy

`core/learning_isolation.py` is the **single source of truth** for who exists and
where their records live:

* `CHARACTER_OWNERS` = the seven; `MAIN_MODEL_OWNER` = `fayd_al_ruh`.
* `owner_for()` accepts **either vocabulary** (`sources`/`jarvis`,
  `model`/`fayd_al_ruh`), so no caller has to remember which it holds.
* `store_dir(owner)` / `store_dir_for(owner, root)` return the **one** path an
  owner may read or write:

```
data/learning/fayd_al_ruh/
  main/                       # the project model's own records
  characters/<key>/           # one strictly isolated directory per character
```

A `CharacterLearningStore` never accepts a path from a caller — it resolves its own
directory from the policy — so "a character writes only to its own store" is a
property of the code, not a promise.

### 3.2 The record

`core/learning_records.py` defines `LearningRecord`, carrying every field the task
requires: a unique `record_id`, the `owner`, the `source_type` and
`source_reference`, the `question`, the `answer`, the extracted `knowledge`, a
`created_at` timestamp, a `confidence`, a `verification` status, `evidence`
references, and a `learning_status` — plus the provenance fields
(`source_character`, `source_record_id`, `source_date`) a shared record needs. The
record is **frozen**: a revision is a *new* record, so the original is never
mutated and a superseded record keeps pointing at what replaced it.

The five source types (`language_model`, `internet`, `user`, `character`,
`system`) and the six learning statuses (`pending`, `provisional`, `verified`,
`rejected`, `contradicted`, `superseded`) are closed sets. `LearningRecord.trusted`
is true only for `provisional`/`verified`, so a pending or contradicted record is
stored and readable but is **not** knowledge.

### 3.3 The controlled pipeline

`core/learning_pipeline.py` is the one path from a response to a stored record. It
runs the task's nine steps: receive → extract statements → identify factual
statements → check contradictions → evaluate confidence → store → (optionally)
update the graph through the existing bounded learner → preserve the original and
its provenance → allow later correction.

Crucially, it **never auto-verifies**. A fresh record is `provisional` (above the
confidence threshold) or `rejected` (below it); the *only* way to `verified` is an
explicit `verify()` call. Contradictory information is **preserved**: the existing
record is untouched and the new one is marked `contradicted`, with the note naming
the record it disagrees with.

Learning is **not** fine-tuning: the GGUF weights are never touched. "Learning"
means persistent external memory and knowledge acquisition, using the existing
knowledge architecture.

---

## 4. Knowledge sharing and the Natasha exclusion

`core/knowledge_sharing.py` is the controlled interface. The exclusion is enforced
**at the architecture level**, not in the UI:

* `PRIVATE_OWNERS = {natasha}` and `SHAREABLE_OWNERS` = the other six, both derived
  from the policy, so no menu can disagree.
* `KnowledgeSharing.learn_from(character)` checks `PRIVATE_OWNERS` **before it
  resolves any path** — so Natasha's records are never even opened — and logs the
  refusal through the audit.
* For a permitted character, each trusted record becomes a **new** record owned by
  the main model, carrying `source_character`, `source_record_id`, `source_date`,
  and the source status in its metadata. The knowledge is kept **verbatim** (never
  paraphrased) and the copy starts `provisional` — learning from a character is not
  the same as verifying.
* The source store is opened **read-only** and is never written. A share that runs
  twice does not duplicate (the source id is the dedup key).
* **The knowledge is shared; the private exchange is not.** A plain share carries
  the extracted knowledge and the provenance, but the source's question/answer
  exchange is copied over **only** when the source character's own configuration
  explicitly permits it (`LearningSettings.share_conversation_history`). The check
  fails **closed** — an unreadable configuration means "do not expose the
  exchange" — and Natasha is refused before it is even consulted, so a private
  conversation can never leak through a share.
* `learn_from_all()` iterates `SHAREABLE_OWNERS` only, so Natasha can never be
  reached by "learn from everyone".

`CharacterConfig.effective_sharing` is the flag **and** `not is_private`, so even a
hand-edited configuration that asked Natasha to share reports `False` — the
backend's rule wins over the setting.

Every learning operation is logged (`core/learning_audit.py`), and a refused
cross-character access is logged with the actor and the owner it was refused.

---

## 5. Personality, language model, and voice

`core/character_config.py` is the structured configuration, one serialisable object
per character with five sections:

* **General** — name, description, role, active.
* **Personality** — traits, behaviour, communication style, reasoning style,
  formality, emotional expression, response length, preferred language
  (`match`/`en`/`ar`), system instructions, rules, knowledge domain, objectives.
* **Language model** — catalog key, provider, model identifier/path, temperature,
  top-p, context length, max tokens, model-specific system prompt, active.
* **Voice** — engine (a closed set from `core.voices`), voice, model path, language,
  gender, speed, pitch, volume, active. Which parameters each engine honours is
  *asked of the engine* (`core.voice_moods.support_note`), never assumed.
* **Learning** — enabled, learn from model, learn from internet, verification and
  confidence thresholds, mode (supervised/automatic), sharing flags.

The defaults are **derived, not duplicated**: a character's personality starts from
its existing `PERSONAS` entry, so the shipped behaviour is exactly what the project
had before this module existed. A configuration only ever records a *change*.

The configuration is separate from the language model, so **changing a model never
touches the personality, memory, or learning records** — pinned by a test.
Section-level `replace_general/replace_personality/replace_model/replace_voice/
replace_learning` helpers keep every editor (CLI, window) from reaching into a
section's fields by name.

`CharacterRegistry` persists the applied configuration under `data/config/` (the
git-ignored data folder), with the shipped defaults in the tracked `config/`
folder. The split means a program update can fast-forward the shipped defaults
without overwriting what the user configured.

---

## 6. The Characters sidebar and the management window

`gui/actions.py` gains a visible **Characters** entry, in its own **Characters**
section of the sidebar. Clicking it (or the second entry, *Who may share
knowledge*) opens `gui/character_dialog.py` — the management window — through
`FaydGUI._open_characters()`.

The window is built the way every other window in the project is built: the same
`Theme` tokens, `RoundedBox`, a `SidebarColumn` on the left, the three
`window_controls` title-bar buttons, and a worker thread whose result is handed
back through a queue the UI thread drains.

* The left column lists the seven characters and the project model (Natasha marked
  🔒, the main model marked *main*), plus a **Knowledge sharing** block stating who
  may share and who may not.
* Choosing a character draws its **complete, editable page**, section by section:
  A. General · B. Personality · C. Language model · D. Voice settings ·
  E. Learning settings · F. Memory and learning management.
* Section F shows the character's own statistics, a search over its records, an
  export/backup, and a reset that must be **explicitly confirmed**.
* **There is no control anywhere that lets one character edit another's private
  learning data** — each page reads and writes only the selected character's own
  store, through the service, which is itself bounded by the policy.

The Tk-free engine behind it is `gui/character_manager.CharacterManagerService`, so
the whole workflow is tested with no display; the window is only the view.

---

## 7. APPLY and hot reload

The **APPLY** button calls `CharacterManagerService.apply(draft)`, which:

1. **validates** every section (`CharacterConfig.validate()`); an invalid draft is
   reported and nothing is written;
2. **saves** the configuration atomically;
3. **activates** what it can now — the personality instruction (re-rendered on the
   next answer) and the learning settings (re-read per record) are named as
   *activated*;
4. reports anything it could not hot-apply — a model or voice swap — as
   **restart-required**, rather than pretending;
5. **rolls back** if activation fails: the previous configuration is restored and
   the report says so.

`CharacterRegistry.apply()` snapshots the current configuration before saving, runs
the caller's activation hook, and restores the snapshot if the hook raises — so a
character is never left half-configured. Both paths are pinned by tests.

The CLI's `set` is the same APPLY headless: it validates, applies, and exits `4`
on a refusal, leaving the stored configuration untouched.

---

## 8. Storage layout (as built)

```
config/                              # shipped, tracked defaults (read-only)
  characters/jarvis.json ultron.json vision.json banner.json
             natasha.json hakeem.json cloud.json
  fayd_al_ruh.json

data/                                # git-ignored, writable
  config/                            # the user's saved configuration (APPLY)
  learning/fayd_al_ruh/
    main/records.jsonl stats.json    # the project model's own store
    characters/<key>/…               # one strictly isolated store per character
    audit/learning-audit.jsonl       # the append-only learning audit
  characters/learning/…              # the existing facts/exchanges memory (unchanged)
```

---

## 9. Files created and modified

**Created**

| File | Purpose |
| --- | --- |
| `core/learning_isolation.py` | Owners, the private/shareable policy, and the one path resolver. |
| `core/learning_records.py` | `LearningRecord`, the source types, and the learning statuses. |
| `core/character_learning_store.py` | The per-owner, append-only, isolated store (+ stats, search, export). |
| `core/learning_pipeline.py` | The controlled learn → extract → contradict → store pipeline. |
| `core/knowledge_sharing.py` | The controlled knowledge-sharing interface (Natasha excluded). |
| `core/learning_audit.py` | The append-only learning audit, including refused access. |
| `core/character_config.py` | The structured, validated configuration and its defaults. |
| `core/character_registry.py` | Persistence, the APPLY, and the shipped defaults. |
| `core/character_isolation_gate.py` | The held-out isolation/sharing catalogue + `character_isolation_accuracy`. |
| `gui/character_manager.py` | The Tk-free engine behind the window and the CLI. |
| `gui/character_dialog.py` | The Characters management window. |
| `cli/characters.py` | The headless twin of the window. |
| `tests/test_character_system.py` | The isolation, sharing, pipeline, config, and CLI tests. |
| `tests/test_character_window.py` | The window tests. |
| `docs/CHARACTER_ARCHITECTURE.md` | The Phase-1 analysis. |
| `docs/CHARACTER_IMPLEMENTATION_REPORT.md` | This report. |
| `config/**` | The shipped default configuration files. |

**Modified**

| File | Change |
| --- | --- |
| `core/paths.py` | `config_dir`, `character_config_dir`, `bundled_config_dir`, `learning_root_dir`. |
| `core/character_learning.py` | A `root` accessor so the wrapper can find the records folder. |
| `ai/characters.py` | `CharacterMouth` takes a `registry` + `learning_store`, renders the *saved* configuration into the standing instruction, and exposes `reload_config()`. |
| `gui/actions.py` | The visible **Characters** action and its **Characters** group. |
| `gui/app.py` | `_open_characters()` and the action routing. |
| `evaluation/runner.py` | The `character_isolation_accuracy` metric value and field. |
| `evaluation/benchmark.py` | The `character_isolation_accuracy` `MetricSpec`. |
| `evaluation/baseline.json` | The updated reference (overall 0.9856 → 0.9860). |
| `cli/menu.py`, `cli/__main__.py` | The `characters` command and its menu entry. |
| `AGENTS.md` | The new modules and their rules. |

---

## 10. Tests and results

`tests/test_character_system.py` (82 tests) covers every required category:

* **A. Isolation** — each of the seven has its own store; a cross-character write
  is refused and logged; records survive a restart; the path is the policy's own.
* **B. Knowledge sharing** — the main model learns from each of the six; **Natasha
  is refused** and logged; the main model writes its *own* record; the source store
  is never modified; sharing twice does not duplicate; a plain share carries the
  knowledge but **not** the private exchange, and sharing the exchange needs an
  explicit per-character opt-in.
* **C. Learning sources** — a model answer and an internet result keep their
  provenance; an unverified answer is never auto-verified; a contradiction is
  preserved and the original is not rewritten; evidence references are kept.
* **D. Personality and configuration** — each character loads its own personality;
  changing one does not affect another; changing a model does not erase records;
  configuration persists across a restart; an invalid change is refused and rolled
  back.
* **E. Voice and language model** — a voice change does not affect another
  character; an unsupported engine or model is rejected; two characters sharing a
  model keep separate stores.
* **F. The management service** — the list, the sharing policy, APPLY and hot
  reload, rollback, statistics, search, export, a confirmed reset, and the
  verify/reject actions.

`tests/test_character_window.py` (8 tests, skipped without a display) covers the
sidebar entry, the character list, each character's page, APPLY, an invalid change
not corrupting the configuration, the Natasha private note, the learning
management section, and the window being opened once and reused.

`core/character_isolation_gate.py` adds the **held-out gate**: a catalogue of 19
hand-written judgements (`ISOLATION_JUDGEMENTS`) run against the *real* store,
sharing, and configuration -- the seven owners and their distinct directories, a
cross-character read refused, each of the six permitted shares allowed, **Natasha
refused** both directly and through "share all", a source record left untouched, the
provenance carried onto the main model's copy, no duplicate on a second share, a
plain share keeping the knowledge but not the private exchange, a hand-edited
Natasha configuration unable to enable sharing, and an unknown key refused. It is
wired into `evaluation` as the metric `character_isolation_accuracy`, so a
regression that opened one boundary is a **gate failure** rather than a passing
suite.

**Gates:** the full suite passes under `xvfb-run` (4557 passed, 15 skipped),
`ruff check .` is clean, `mypy` reports no issues (770 source files), and
`python -m evaluation` prints PASS with no regression
(`character_isolation_accuracy` 1.0000; overall 0.9860).

---

## 11. Running and using it

**The window:** launch `python -m cli gui`, then click **Characters** in the
sidebar. Pick a character, edit any section, press **APPLY**. The status line says
what activated now and what needs a restart. **Reset to default** returns the
character to its shipped configuration; **Reset learning data…** (in section F)
asks for confirmation before forgetting the character's records.

**The CLI:** the headless twin drives the same service and the same data.

```bash
python -m cli characters list                 # the seven + the project model
python -m cli characters show jarvis          # the full configuration
python -m cli characters set vision --temperature 0.2
python -m cli characters reset vision
python -m cli characters stats banner         # the character's own learning
python -m cli characters records banner
python -m cli characters search banner porosity
python -m cli characters export banner --out backup.json
python -m cli characters reset-learning banner --confirm
python -m cli characters share jarvis         # the main model learns from Jarvis
python -m cli characters share --all          # …from every permitted character
python -m cli characters main                 # what the main model learned
python -m cli characters audit --denials      # refused accesses
python -m cli characters shareable            # who may share, and who may not
```

Exit codes: `0` success, `2` invalid arguments, `3` the character or record is
unknown, `4` a gate refused the operation, `5` an unexpected failure.

---

## 12. What is deliberately **not** done

* **No fine-tuning of the GGUF weights** — "learning" is persistent external memory
  and knowledge acquisition, as the task states.
* **No change to the knowledge graph's write rules** — the stores are records; only
  the existing bounded learners touch the graph.
* **No new third-party dependency** — the standard library plus what the base
  install already has.
* **No new mode or mouth** — the seven keys and the project model key are unchanged.
* **No mechanism for one character to edit another's private learning data** — the
  isolation is in the backend services, not only in the interface.
