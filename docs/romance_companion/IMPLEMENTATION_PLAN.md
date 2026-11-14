# Romance & Companion Studio — Implementation Plan (Phase 0 deliverable)

**Companion to:** `docs/romance_companion/ARCHITECTURE_AUDIT.md`.

**Principle:** this is a *phased, reviewable* build. Each phase (a) inspects the
relevant existing code, (b) lists the files reused, (c) lists the files
modified, (d) lists the new files, (e) implements, (f) runs targeted tests,
(g) runs the full regression suite, (h) reviews, (i) updates documentation, and
(j) reports. No phase claims completion for anything that is only a mockup or a
placeholder. No phase weakens an existing test.

**Every phase must keep the four hard gates green:** `python -m pytest -q`,
`python -m ruff check .`, `python -m mypy`, `python -m evaluation` (PASS, no
regression).

**Naming assumption (to confirm):** the domain package is `core/romance/`, the
Tk-free service is `gui/romance.py`, the window is `gui/romance_dialog.py`, and
the CLI is `cli/romance.py` (`python -m cli romance …`). This mirrors
`core/workplace/` + `gui/workplace.py` + `gui/workplace_dialog.py` +
`cli/workplace.py`.

---

## Phase 0 — Architecture audit (THIS DELIVERABLE)

**Status:** complete, awaiting approval.

**Outputs:** `docs/romance_companion/ARCHITECTURE_AUDIT.md` (this folder) and
this plan. Contains the reuse matrix, the proposed component architecture, the
proposed database schema, the required changes to existing files, the list of
new files, and the risks/unresolved questions.

**No application code changed.**

---

## Phase 1 — Foundation and database

**Goal:** a versioned, transactional, isolated Studio database and its typed
repository/domain services — no UI yet.

**Inspect first:** `core/workplace/schema.py`, `core/workplace/database.py`,
`core/workplace/repository.py`, `core/workplace/domain.py`, `core/memory_db.py`,
`core/paths.py`, `storage/sqlite_store.py`, `core/learning_isolation.py`,
`core/character_identity.py`.

**Reuse:** the `TableSpec`/`Column`/`MIGRATIONS` pattern; `core.paths`
(`data_dir`, `ensure_data_layout`); `core.learning_isolation` for owner checks;
`core.best_effort` for cosmetic writes.

**New files:**
- `core/romance/__init__.py` — public surface.
- `core/romance/model.py` — frozen dataclasses (`RecordBase`-style) for every
  table in the schema (§17 of the audit).
- `core/romance/schema.py` — `SCHEMA_VERSION = 1`, `MIGRATIONS`, `TABLES`.
- `core/romance/database.py` — `RomanceDatabase` (open/migrate/transaction,
  `PRAGMA foreign_keys=ON`, integrity check, bounded reads, corrupt-row
  tolerance, backup/restore via SQLite's online-backup API).
- `core/romance/repository.py` — typed CRUD.
- `core/romance/domain.py` — audited create/update/**deactivate** (soft delete;
  `ON DELETE RESTRICT` everywhere, matching `core/workplace`).
- `core/romance/storage.py` — `CompanionStorageService` (resolve paths, export,
  delete-with-retention, backup).

**Modified files:** `core/paths.py` (add `romance_dir()`/`romance_db_path()`/
`romance_backups_dir()`), `pyproject.toml` (`include`, `[tool.mypy].files`),
`fayd-al-ruh.spec` (`collect_submodules`).

**Tests:** `tests/test_romance_storage.py` — schema creation, idempotent
re-open, migration forward, newer-schema refusal, transactional rollback,
corrupt-row tolerance, backup/restore (bad backup refused, good DB preserved),
soft-delete + `RESTRICT` behaviour, stable IDs, UTC timestamps, per-user/
per-character isolation.

**Gate:** full suite + a new `companion_studio_accuracy` metric stub registered
in `evaluation/benchmark.py` (initially 1.0 on the storage checks).

**Not in scope:** any UI, any model call, any memory retrieval.

---

## Phase 2 — Character Studio (library + personality + management)

**Goal:** create/edit/duplicate/import/export/archive/restore/delete companion
characters, each with the full profile field list, a personality-dial
configuration, and safe deletion with retention choices.

**Inspect first:** `core/character_config.py`, `core/character_registry.py`,
`core/character_identity.py`, `gui/character_manager.py`,
`gui/character_dialog.py`, `cli/characters.py`, `core/character_consistency.py`,
`core/personality_layer.py`.

**Reuse:** `core.character_identity` (one identity across vocabularies),
`core.character_config` section/validation patterns, `core.character_registry`
(`save`/`reset`/APPLY semantics), `gui.character_manager.CharacterManagerService`
as the service template, `gui/character_dialog.py` as the window template.

**Decision point (from audit risk #6):** Studio companions are **new records**
that reference an existing persona **template** for isolation, so the closed
owner set in `core/learning_isolation.py` is **not** changed in this phase.

**New files:**
- `core/romance/characters.py` — `CharacterManager` (library CRUD, duplicate,
  import/export JSON, archive/restore, delete-with-retention).
- `core/romance/personality.py` — `CharacterPersonalityEngine` (dials →
  readable prompt block, reusing the "tone guidance, never bare numbers" rule
  from `core/character_learning.CharacterPersona`).
- `core/romance/seed.py` — a small set of **clearly-labelled example**
  characters (no invented real people).

**Modified files:** none required beyond Phase 1's registrations.

**Tests:** `tests/test_romance_characters.py` (create/edit/duplicate/import/
export/archive/restore/delete + retention), `tests/test_romance_personality.py`
(dial validation, prompt rendering, consistency warnings), plus a window test
once Phase 10's shell exists (a headless service test lands here).

**Gate:** full suite; the character checks enter `companion_studio_accuracy`.

---

## Phase 3 — Local model integration

**Goal:** per-character and per-mode model assignment, load/unload, health,
memory estimation, and an honest model-evaluation facility — all routed through
the existing manager.

**Inspect first:** `core/model_manager.py`, `core/language_models.py`,
`core/model_runtime.py`, `core/model_resources.py`, `core/model_preflight.py`,
`core/model_selection.py`, `core/model_settings.py`, `core/model_download.py`,
`ai/mouths.py`, `gui/model_manager.py`.

**Reuse:** everything above. The Studio's router is a **thin** delegation; it
imports no runtime directly.

**New files:**
- `core/romance/routing.py` — `CompanionModelRouter` (`assign(character|mode,
  key)`, `resolve(...)`, `load`/`unload`, `status`, `health`, `cancel`, `timeout`,
  `estimate_memory`, `safe_defaults_for_hardware`).
- `core/romance/model_eval.py` — the model-evaluation facility (Arabic/English
  quality probes, personality consistency, emotional coherence, long-context,
  roleplay continuity, latency, RAM, VRAM, stability) writing a comparative
  report; it measures on the **user's** machine and reports honestly, and it
  never marks a model compatible merely because its files exist.

**Modified files:** possibly `core/model_selection.py` (add `companion`/`romance`
to `TASKS` if not already covered) — additive only.

**Tests:** `tests/test_romance_routing.py` (assignment resolution, honest
"no model" next step, memory estimate, load failure path), `tests/test_romance
_model_eval.py` (the report shape; real numbers only when a model is present,
otherwise an honest "not measured").

**Docs:** `docs/romance_companion/MODEL_INTEGRATION.md` +
`HARDWARE_COMPATIBILITY.md`.

---

## Phase 4 — Conversation modes

**Goal:** nine conversation modes, each with its own prompt template, style,
context rules, personality integration, safety settings, and history
continuity; switching modes never rewrites history.

**Inspect first:** `companion/prompt_builder.py`, `companion/companion_agent.py`,
`core/conversation.py`, `core/jarvis.py` (the `JarvisMode` choke point),
`ai/chat.py` (`format_turn`).

**Reuse:** the companion prompt builder's bounded-block structure and the
`SECTION_*` labels; `core.conversation` for storage; the project's text polish
(`core/text_polish.polish_answer`).

**New files:**
- `core/romance/modes.py` — `MODES` (general, romantic, dating_simulation,
  flirting_playful, adult, emotional_support, storytelling, character_roleplay,
  multi_character_roleplay), each with a `system_template`, style guidance,
  context-management rules, and safety settings; `mode_prompt(mode, character,
  memory, ...)` renders the request.
- `core/romance/conversation.py` — `ConversationManager` (Studio conversations,
  mode per conversation, mode recorded per message, resume/restore).

**Modified files:** none required (the Studio has its own conversation tables).

**Tests:** `tests/test_romance_modes.py` (each mode renders a distinct,
safety-carrying prompt; mode change preserves history and never mutates a prior
row), `tests/test_romance_conversation.py` (persistence + resume).

**Docs:** `docs/romance_companion/CONVERSATION_MODES.md`.

---

## Phase 5 — Relationship Engine

**Goal:** an independent, evidence-driven relationship state per character,
with transparent dimensions, configurable stages, events with timestamps and
message references, an explanation screen, and reset/edit.

**Inspect first:** `companion/relationship_manager.py` (`EVENT_KINDS`,
`RelationshipEvent`, `RelationshipSummary`), `companion/database.py`
(`relationship_events`), `core/interaction_reward.py`.

**Reuse:** the companion event model + the `relationship_events` table shape;
extend with dimensions, deltas, stage, and message references.

**New files:**
- `core/romance/relationship.py` — `RelationshipEngine`: `RelationshipState`
  (seven dimensions), `STAGES` (stranger → acquaintance → friend → close_friend
  → romantic_interest → romantic_partner, configurable), `observe(...)` (derive
  bounded deltas from an explicit event/evidence — never random), `explain()`
  (why the stage/state changed), `reset()`, `edit()`.

**Modified files:** none (Studio tables).

**Tests:** `tests/test_romance_relationship.py` (bounded deltas, determinism,
evidence requirement, stage thresholds, no-random-change negative control, the
explanation names the event, reset/edit audited).

**Docs:** `docs/romance_companion/RELATIONSHIP_ENGINE.md`.

---

## Phase 6 — Memory system

**Goal:** the five memory types with per-character/per-user isolation,
importance/recency, summarization, dedup, contradiction handling, search,
manual review, edit/delete, export, retention, source refs, and timestamps —
reusing the existing memory machinery.

**Inspect first:** `companion/memory_manager.py`, `companion/relationship_memory.py`,
`core/long_term_memory.py`, `core/memory_retrieval.py`, `core/memory_management.py`,
`core/memory_decay.py`, `core/character_memory_view.py`, `core/rag.py`,
`core/embeddings.py`.

**Reuse:** `MemoryManager` (kinds, sensitive-data refusal, `propose`),
`core.memory_retrieval` (explainable ranking), `core.embeddings` for semantic
recall, `core.memory_management` (edit/delete/export/backup).

**New files:**
- `core/romance/memory.py` — `CompanionMemoryManager`: the five types
  (short-term via the conversation window; episodic/semantic_personal/
  relationship/character durable), relevance-gated retrieval, dedup,
  contradiction preservation, review, retention.

**Modified files:** none.

**Tests:** `tests/test_romance_memory.py` (isolation — character A cannot read
B; importance/recency ordering; dedup; contradiction preserved; review/edit/
delete/export; retention; relevance gating; "not every statement becomes a
fact").

**Docs:** `docs/romance_companion/MEMORY_SYSTEM.md`.

---

## Phase 7 — Dating and roleplay studio

**Goal:** scenario builder + categories + interactive storytelling with
pause/resume/branch/save/restart, character consistency, story continuity, and
multi-character support.

**Inspect first:** `core/romance/{model,domain}.py` (Phase 1), the Phase 4/5/6
engines.

**Reuse:** conversation storage (Phase 4), relationship engine (Phase 5),
memory (Phase 6), the model router (Phase 3), isolation (Phase 1).

**New files:**
- `core/romance/roleplay.py` — `RoleplayEngine`: `Scenario`, `SCENARIO_CATEGORIES`,
  `ScenarioSession` (state machine: active/paused/ended; branch/save/restart),
  continuity/consistency enforcement, and multi-character turn-taking
  (**sequential**, bounded — one model, one turn at a time).
- `core/romance/orchestrator.py` — `CompanionOrchestrator` (single-character and
  multi-character turns; speaker identity is enforced so a character never
  impersonates another; scene context + previous events are assembled within a
  bounded budget).

**Modified files:** none.

**Tests:** `tests/test_romance_scenarios.py` (builder fields, categories, state
machine, branch/save/restart, continuity), `tests/test_romance_multi_character.py`
(distinct identities, no impersonation, sequential generation, add/remove
character mid-scenario).

**Docs:** `docs/romance_companion/ROLEPLAY_STUDIO.md`.

---

## Phase 8 — Voice integration

**Goal:** per-character voice, TTS/STT, push-to-talk, playback/stop/
interruption, segmentation, optional voice conversation mode — via a thin
adapter.

**Inspect first:** `core/reading.py`, `core/speech.py`, `core/voices.py`,
`core/voice_moods.py`, `core/character_config.VoiceConfig`,
`core/voice_studio_profile.py`, `core/mouth_profile.py`.

**Reuse:** everything above. The adapter never invents a capability and shows
only voices the engine really reports.

**New files:**
- `core/romance/voice.py` — `CompanionVoiceAdapter` (per-character voice config,
  `speak`/`stop`/`render`, `available_voices`, `describe`, honest next step when
  no engine exists).

**Modified files:** none.

**Tests:** `tests/test_romance_voice.py` (voice resolution, honest "no engine"
path, mood/prosody application, only-installed-voices rule) — engine-dependent
parts skip when no engine is installed, per project convention.

**Docs:** `docs/romance_companion/VOICE_INTEGRATION.md`.

---

## Phase 9 — Privacy and safety

**Goal:** enforce and *display* local-only operation; adult-content gating;
boundary enforcement; refusal logging; export/delete; explicit download
permission; and (optionally) local encryption.

**Inspect first:** `core/offline.py`, `core/permissions.py`,
`core/human_approval.py`, `core/learning_isolation.py`,
`companion/personality.py` (`BOUNDARIES`), `core/workplace/security.py`
(`is_ai_actor`), `core/audit_log.py`.

**Reuse:** the offline guard, the permission policy, the approval gate, the
audit log, and the existing companion `BOUNDARIES` as the content floor.

**New files:**
- `core/romance/safety.py` — `CompanionSafetyManager`: adult eligibility
  (disabled by default; explicit confirmation; per-character `adult_eligible`;
  refuses minors, age-ambiguous participants, coercion, exploitation, and
  non-consensual content), boundary enforcement, refusal + eligibility events
  into `companion_safety_events`, and the "never represent the AI as human" rule.

**Modified files:** possibly `core/preferences.py` (an `adult_eligible`
preference) — additive.

**Tests:** `tests/test_romance_safety.py` (adult mode off by default; eligibility
required; age-ambiguous refused; non-consensual refused; boundary bypass refused;
identity-confusion refused; memory isolation; data deletion; remote-provider
blocking under `FAYD_OFFLINE=1`).

**Docs:** `docs/romance_companion/PRIVACY_AND_SAFETY.md`.

---

## Phase 10 — Final UI, integration, and QA

**Goal:** the polished standalone Studio window, wired into navigation, with all
management screens (Model/Character/Memory/Settings), plus the full test suite,
docs, and the feature matrix.

**Inspect first:** `gui/workplace_dialog.py` (the window template),
`gui/actions.py`, `gui/app.py`, `gui/components.py`, `gui/sidebar.py`,
`gui/chat_composer.py`, `gui/dialog_base.py`, `gui/layout_direction.py`,
`cli/workplace.py` (the CLI template).

**New files:**
- `gui/romance.py` — `RomanceStudioService` (the one Tk-free engine the window
  and the CLI share).
- `gui/romance_dialog.py` — `RomanceStudioDialog` (Tk, imported lazily): left
  sidebar (Characters, Discover, Create, Recent Conversations, Dating Scenarios,
  Roleplay Library, Favorites, Archived), centre workspace (profile, history,
  composer, voice controls, mode selector, relationship indicator, activity),
  right info panel (profile, traits, interests, relationship status, milestones,
  memories, voice config, active scenario) — all collapsible/responsive.
- `cli/romance.py` — the headless twin.

**Modified files:** `gui/actions.py` (one action + group), `gui/app.py`
(`_open_romance_studio` + composer pill + `_open_dialogs`), `cli/menu.py`
(`run_romance` + row), `cli/__main__.py` (`_COMMANDS`), `gui/settings_hub.py`
(global Studio settings), `README.md`, `AGENTS.md`, `core/version.py` +
`pyproject.toml` (version bump in all three).

**Tests:** `tests/test_romance_window.py` (the window builds, the sidebar
sections exist, the composer works, a message round-trips, live re-theming),
`tests/test_romance_cli.py` (the headless twin), `tests/test_romance_end_to_end.py`
(the brief's 14-step workflow: launch → create → configure → assign model →
romantic conversation → save/restore → memory → close/reopen → resume →
personality/memory continuity → correct relationship state → voice when
available → local-only enforcement).

**Docs:** `docs/romance_companion/` — `INSTALL_AND_CONFIGURATION.md`,
`USER_GUIDE.md`, `DATABASE_SCHEMA.md`, `CHARACTER_SYSTEM.md`, `TEST_REPORT.md`,
`FEATURE_MATRIX.md` (implemented / partially / not implemented / requires
hardware / requires a model / requires a voice engine).

**Gate:** full suite (all existing tests still pass), `ruff`, `mypy`,
`python -m evaluation --update-baseline` with `companion_studio_accuracy`
entering `overall` and the regression gate green.

---

## Cross-cutting work present in every phase

- **Isolation:** every path resolves through `core.character_identity` +
  `core.learning_isolation` **before** touching a directory.
- **Gates:** any external action uses the permission policy + human-approval
  gate; the Studio adds no new egress path.
- **Best-effort:** cosmetic/optional failures go through
  `core.best_effort.best_effort`, never a bare `except: pass`.
- **Bilingual + RTL:** every label carries English + Arabic; the sidebar side
  follows `gui.layout_direction`.
- **No placeholders:** every button does real work; no simulated model output
  ships.

---

## Deliverables checklist (from the brief's §18)

| # | Deliverable | Phase | File |
|---|---|---|---|
| 1 | Architecture audit | 0 | `ARCHITECTURE_AUDIT.md` |
| 2 | Implementation plan | 0 | `IMPLEMENTATION_PLAN.md` |
| 3 | Database schema doc | 1 | `DATABASE_SCHEMA.md` |
| 4 | Character-system doc | 2 | `CHARACTER_SYSTEM.md` |
| 5 | Relationship-engine doc | 5 | `RELATIONSHIP_ENGINE.md` |
| 6 | Memory-system doc | 6 | `MEMORY_SYSTEM.md` |
| 7 | Model-integration doc | 3 | `MODEL_INTEGRATION.md` |
| 8 | Voice-integration doc | 8 | `VOICE_INTEGRATION.md` |
| 9 | Privacy & safety doc | 9 | `PRIVACY_AND_SAFETY.md` |
| 10 | Install & configuration guide | 10 | `INSTALL_AND_CONFIGURATION.md` |
| 11 | User guide | 10 | `USER_GUIDE.md` |
| 12 | Test report | 10 | `TEST_REPORT.md` |
| 13 | Hardware compatibility report | 3 | `HARDWARE_COMPATIBILITY.md` |
| — | Feature matrix | 10 | `FEATURE_MATRIX.md` |

---

## Adopted decisions (approved by the project owner)

The owner reviewed the audit's open questions and adopted the recommended answer
for each, with **one deliberate override** (question 2). These are now settled and
guide Phases 2+.

1. **Package placement** — `core/romance/` for the domain layer (as built in
   Phase 1), with `gui/romance.py` + `gui/romance_dialog.py` and `cli/romance.py`
   as the surfaces. Adopted.
2. **Character voice and the AI-disclosure rule — OVERRIDDEN by the owner.** The
   characters are **human-passing**: each character believes it is a human being
   and addresses the user as a human, and the character **never breaks the fiction
   with an "I am an AI / I am a virtual character" line inside the conversation**.
   This supersedes the existing `companion/personality.py` "she is a virtual
   character" wording *for the Romance & Companion Studio's own characters* (the
   existing Natasha and Hakeem mouths are **unchanged**). It is implemented under
   two rules so the product stays honest without breaking the roleplay:
   - **Diegetic (in-world) immersion.** Inside a conversation the character speaks
     as a person: it has a biography, feelings, opinions, and a life; it never
     narrates itself as a program and never claims a tool ran.
   - **Non-diegetic (application-level) disclosure.** The *application* states
     plainly, outside the fiction, that these are fictional AI companions: the
     first-run screen carries the **adult-eligibility confirmation** and a
     one-line "these are fictional AI characters, not real people" notice; the
     Studio's chrome and the character profile panel label each companion as an
     **AI character**; and the about/help text says the same. The character's own
     words stay in character; the product never *deceives*.
   - The **safety floor is unchanged and non-negotiable**: adults only (18+),
     explicit and revocable consent, no coercion, no exploitation, no
     non-consensual content, and no sexual content involving minors or
     age-ambiguous participants.
3. **Relationship dimensions** — normalised `0..1`, transparent and configurable,
   with documented default stage thresholds; changes are evidence-driven (an
   explicit event), never random. Adopted.
4. **Local database encryption** — deferred to Phase 9 (privacy). The database is
   local-only and git-ignored today; encryption is an opt-in hardening step with a
   properly managed key, not a Phase-1 requirement. Adopted.
5. **Ownership keys** — Phase 1 references a persona **template**; dedicated
   owner keys are revisited later. Adopted.
6. **Multi-character turns** — deterministic, sequential, bounded turn-taking over
   one local model (no uncontrolled concurrent generation). Adopted.
7. **Evaluation metric** — `companion_studio_accuracy` is added to `evaluation`
   (done in Phase 1; the baseline is re-recorded). Adopted.

**Phase 1 status: implemented, tested, and green** — see
`docs/romance_companion/PHASE_1_REPORT.md`. Work proceeds to Phase 2 (Character
Studio) under these decisions.

