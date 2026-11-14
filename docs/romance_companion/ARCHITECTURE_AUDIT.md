# Romance & Companion Studio — Phase 0 Architecture Audit

**Status:** audit only. No application code was modified to produce this
document (only this `docs/romance_companion/` folder is new).

**Repository:** `fayd-al-ruh` @ `main` (version 2.3.0, `core/version.py`).
The checkout is a shallow clone (`git rev-parse --is-shallow-repository` → `true`).

**Scope of this document:** what already exists in Fayd al-Ruh that the Romance
& Companion Studio must *reuse* rather than duplicate, what exists but is
incomplete, what is genuinely new, and what does not apply — plus the reuse
matrix, the proposed component architecture, the proposed database schema, the
required changes to existing files, the genuinely new files, and the risks.

---

## 0. How this audit was produced

Every claim below was read from the actual source tree, not assumed. The key
commands were:

- `find . -maxdepth 1 -type d` and directory listings of `core/ gui/ ai/ cli/
  companion/ hakeem/ storage/ ml/ ingest/ perception/ cognition/ evaluation/`.
- `grep -n`/`head`/`sed` over the modules named in the task brief
  (`core/documents.py`, `core/knowledge_base.py`, `core/knowledge_index.py`,
  `core/rag.py`, `core/assistant.py`, `core/model_manager.py`,
  `core/language_models.py`, `core/speech.py`, `core/reading.py`,
  `core/voices.py`, `core/character_*.py`, `companion/`, `hakeem/`, `gui/`,
  `cli/`, `evaluation/`).
- `pyproject.toml`, `.github/workflows/ci.yml`, `requirements*.txt`,
  `fayd-al-ruh.spec`, `tests/conftest.py`.

Two facts dominate the whole design:

1. **Fayd al-Ruh already contains a complete, local-first AI-companion layer.**
   The `companion/` package (character "Natasha") is a self-contained SQLite-
   backed companion with a stable personality, durable four-category memory,
   isolated persistent sessions, a simulated dialogue tone, an evolving
   relationship story, a structured relationship memory, an initiative engine,
   an anti-repetition engine, a linguistic polisher, and a local GGUF/Ollama
   model layer — plus `hakeem/` (a second such character). The Romance &
   Companion Studio is therefore largely an *orchestration + studio UI* over
   proven machinery, not a new companion engine.
2. **The project has a very strict, enforced architectural discipline.**
   Offline-first enforcement (`core/offline.py`), deny-by-default permissions
   (`core/permissions.py`), human-approval gates for every external action
   (`core/human_approval.py`), a single model catalog (`core/language_models.py`)
   and a single model manager (`core/model_manager.py`), a single speech layer
   (`core/speech.py`/`core/reading.py`/`core/voices.py`), an isolation policy
   (`core/learning_isolation.py`), and a held-out evaluation gate
   (`evaluation/`) that fails CI on regression. The Studio must fit this
   discipline or it will fail the build.

---

## 1. Application architecture and entry points

| Concern | Where it lives | Notes |
|---|---|---|
| Console entry point | `pyproject.toml` → `fayd-al-ruh = "cli.__main__:main"` | Also `python -m cli` |
| CLI dispatch | `cli/__main__.py` (`_COMMANDS` dict, ~90 aliases) | `main()` calls `ensure_offline()` then `ensure_data_layout()` before anything |
| CLI menu | `cli/menu.py` (`MENU` tuple, 136 rows; `run_*` functions) | The GUI sidebar mirrors numbered items via `gui/actions.py` |
| Desktop window | `gui/app.py` (`FaydGUI`, 4862 lines) | `python -m cli gui` / `python -m gui` |
| Domain-module precedent | `core/workplace/` + `gui/workplace.py` + `gui/workplace_dialog.py` + `cli/workplace.py` | **The template to copy**: a Tk-free service shared by a window and a CLI, over a versioned SQLite store |
| Second domain-module precedent | `core/training_center/` + `gui/training_center.py` + `gui/training_center_dialog.py` + `cli/training_center.py` | A "thin composition" workspace, catalog-driven controls, bilingual |
| Self-contained character precedent | `companion/`, `hakeem/` | Own package, own SQLite DB, own model layer, own personality |

**Entry-point rule to preserve:** every program entry point (`cli/__main__.py`,
`ai/chat.py`, `ai/jarvis.py`, `gui/app.py`) calls `core.offline.ensure_offline()`
and `core.paths.ensure_data_layout()` first. The Studio must be reached through
these paths so the offline guard and data layout are already in force.

---

## 2. Navigation and secondary-window patterns

- **Sidebar actions** live in `gui/actions.py`: a frozen `GuiAction(key, label,
  command, argv, hidden=False)` dataclass, an `ACTIONS` tuple (62 entries), and
  `_GROUP_SPECS` → `ACTION_GROUPS` which groups the *visible* actions into
  named collapsible sections. An action not named by any group is appended to
  the last group, so nothing silently vanishes.
- **Window management** in `gui/app.py`: each window has a `_open_<name>()`
  method that keeps one instance and re-raises it (`existing.window.deiconify();
  existing.window.lift()`), catches `tk.TclError` if it was closed, imports the
  dialog **lazily** (`from gui.workplace_dialog import WorkplaceDialog`), and
  passes a consistent set of constructor args: `theme=self._theme`,
  `font=self._font`, `mono=self._mono`, `overrides=self._profiles[SCOPE_WINDOWS]`,
  `list_overrides=self._profiles[SCOPE_LISTS]`, `on_close=...`.
- **Dialogs** are built on `gui.dialog_base.ThemedDialog` (live re-theming via
  `retheme`, re-typing via `retype`), the shared `gui/sidebar.SidebarColumn`
  (collapsible, persistent per-scope), `gui/components.py` (Button, Section,
  DialogHeader, StatusLine, WorkerPump, …), `gui.chat_composer.ChatComposer`
  (the shared rounded composer, with `add_mode_selector`, `add_mic`,
  `add_web_toggle`, `add_tool`), `gui.rich_text.insert_markdown`, and
  `gui.window_controls.install_window_controls`.
- **Composer pills**: the main window already places *Code Studio*, *Training*,
  and the Chat pill in the composer's bottom-left toolbar via
  `composer.add_mode_selector(icon=, label=, options=[...])`. This is the exact
  mechanism to add a **Romance & Companion Studio** pill.

**Reuse verdict:** EXISTING. Add one `GuiAction` + one `_GROUP_SPECS` entry +
one `_open_romance_studio()` + one composer pill + one `cli` command. No new
navigation framework.

---

## 3. Chat implementation and message storage

| Concern | Module | API |
|---|---|---|
| Turn model + bounded window | `core/conversation.py` | `Role` (`SYSTEM`/`USER`/`ASSISTANT`/`CONTEXT`), `Message`, `Conversation`, `ConversationMemory`, `JsonStore` persistence |
| Companion sessions | `companion/conversation_manager.py` | `ConversationManager`, `SessionInfo` |
| Companion messages + summaries | `companion/database.py` | `conversations`, `messages`, `conversation_summaries` tables |
| Workplace chats | `core/workplace/*` | Separate domain, its own DB |
| GUI chat session | `gui/service.py` | `ChatService`, `ChatReply`, `ai.jarvis.build_session(text_only=True)` + `ai.chat.ModelSwitcher` |
| Turn formatting | `ai/chat.py` | `format_turn`, `turn_payload` |

**Reuse verdict:** EXISTING. The Studio's conversations reuse the companion
session model (`companion/database.py` is already the right shape: messages,
summaries, emotion, relationship events, relationship memory). The Studio adds
*scenario/mode/character* columns around it (see §13), not a new message store.

---

## 4. Model manager and model discovery

| Concern | Module |
|---|---|
| Unified catalog of *every* language model | `core/language_models.py` (`all_language_models`, `models_for_choice`, `find_language_model`, `ready_language_model`, `download_model`, `can_download`, `MOUTHS`, `GGUF_MOUTHS`, `ENGINE_LABELS`) |
| Central manager (discover/register/inspect/load/unload/test) | `core/model_manager.py` (`ModelManager`, `list_models`, `ready_models`, `compatibility`, `load`, `unload`, `test`, `models_for_task`, `select_for_task`) |
| Durable registry | `core/model_registry.py` (`ModelRegistry`, `ModelRecord`) |
| Runtime adapters | `core/model_runtime.py` (`RUNTIME_LLAMA_CPP`, `RUNTIME_OLLAMA`, `RUNTIME_INPROCESS_TRAINED`, `RUNTIME_CLOUD`; `LlamaCppAdapter`, `OllamaAdapter`, `InProcessTrainedAdapter`) |
| Inference settings | `core/model_settings.py` (`ModelSettings` — temperature 0.7, top_p 0.95, ctx 4096, max_tokens 512) |
| GGUF header inspection (no execution) | `core/model_files.py` (`inspect_model_file`, reads architecture/context/params) |
| Task-based selection (explainable) | `core/model_selection.py` (`TASK_CAPABILITIES`, `rank_for_task`, `plan_fallback`, `task_profile`) |
| Machine figures (RAM/disk/CPU/GPU) | `core/model_resources.py` (`system_resources`, `disk_free`, `gpu_info`, `HONESTY_LINE`) |
| Memory estimate + preflight | `core/model_preflight.py` (`estimate_memory`, `preflight`, `PreflightReport`) |
| Managed process registry | `core/model_process.py` |
| Governed, resumable download | `core/model_download.py` (`.part` staging, digests, resume, `DownloadDenied`) + `core/model_sources.py` |
| Model Manager surfaces | `gui/model_manager.py` (`ModelManagerService`), `gui/model_manager_dialog.py`, `cli/models.py` |
| Per-mouth GGUF engines | `core/ultron.py`, `core/vision_model.py`, `core/banner.py`/`core/presentation_model.py`, `companion/model.py`, `hakeem/model.py`, `core/docs_model.py` |
| Trained-artifact bridge | `core/trained_artifacts.py` |

**Reuse verdict:** EXISTING (comprehensive). The Studio **must not** build a
second downloader, registry, or catalog. It routes every model choice through
`core.language_models` + `core.model_manager` and uses
`core/model_preflight`/`core/model_resources` for the hardware limits the brief
requires (16 GB RAM, AMD Radeon R7 M360 2 GB, Intel HD 520 1 GB — CPU-first).

---

## 5. Local inference (llama.cpp / GGUF)

- The in-process GGUF runtime is `llama-cpp-python`, imported lazily, wrapped by
  an injectable *runner seam* in each engine: `core/ultron.py`
  (`UltronRunner`/`LlamaCppRunner`), `core/vision_model.py` (`VisionRunner`),
  `core/banner.py`/`core/presentation_model.py` (`PresentationRunner`),
  `companion/model.py` (`LlamaCppNatashaRunner`), `core/docs_model.py`
  (`LlamaCppDocsRunner`).
- `core/coder_model.py` holds `OllamaCoderClient` (a dependency-free `urllib`
  client for Ollama's `/api/tags` and `/api/chat`).
- `core/model_runtime.py` exposes a uniform `RuntimeAdapter` protocol with
  `capabilities()`/`load()`/`unload()`/`infer()` and `LlamaCppAdapter`,
  `OllamaAdapter`, `InProcessTrainedAdapter`.
- The model files live under git-ignored per-engine folders resolved by
  `core/paths.py` (`ultron_models_dir`, `vision_models_dir`,
  `presentation_models_dir`, `natasha_models_dir`, `hakeem_models_dir`,
  `shared_models_dir`, `docs_models_dir`, …), each with a `model.json` registry
  and `find_model_file`/`model_is_cached` helpers.

**Reuse verdict:** EXISTING. The Studio's `CompanionModelRouter` is a *thin
router* over `core.model_manager` + `core.language_models`; it must not import
`llama_cpp` itself.

---

## 6. Voice engines, TTS and STT

| Concern | Module |
|---|---|
| Contracts + engines + mic recorder | `core/speech.py` (`SpeechSynthesizer`, `SpeechRecognizer`, `discover_synthesizer_name`, `load_synthesizer`, `discover_recognizer_name`, `load_recognizer`, `system_voices`, `MicrophoneRecorder`, `speech_available`, `microphone_install_hint`) |
| Voice catalog (3 kinds) | `core/voices.py` (`Voice`, `SystemVoice`, `ModelVoice`, `voice_sections`, `find_voice`, `find_system_voice`, `find_model_voice`, `group_system_voices`, `group_model_voices`) |
| Reading facade | `core/reading.py` (`TextReader` — `render`, `render_many`, `speak`, `save`, `set_voice`, `set_mood`, `set_prosody`, `system_voices`, `describe`, `split_for_speech`, `batch_chunks`) |
| Emotional states + prosody | `core/voice_moods.py` (`MOODS`, `Prosody`, `apply_mood`, `ENGINE_MOOD_SUPPORT`, `support_note`) |
| Neural engines | `core/piper_speech.py` (Piper, speaks Arabic), `core/audiocpp_speech.py` (VoiceTut-TTS/audio.cpp, Egyptian Arabic), `core/qwen_speech.py` (Qwen3-ASR + Qwen3-TTS) |
| Arabic speech preparation | `core/arabic_speech.py` (`prepare_for_speech` — numbers/units/ranges spelled for speech) |
| Per-engine studio | `gui/voice_studio.py` (`VoiceStudioService`), `gui/voice_studio_dialog.py`, `cli/voice_studio.py` |
| Per-mouth voice | `gui/model_chat.py` (`ModelChatService.start_reading`), `core/mouth_profile.py` |

**Reuse verdict:** EXISTING. The Studio's `CompanionVoiceAdapter` is a thin
adapter over `core.reading.TextReader` + `core.voices` + `core/voice_moods`.
The Studio must display **only** voices the engine really reports here
(`voice_sections` already guarantees this) and must never invent a capability.

---

## 7. Character, persona and profile management

This is the richest existing area; the Studio builds on it rather than replacing it.

| Concern | Module | API |
|---|---|---|
| The eight owners | `core/learning_isolation.py` | `CHARACTER_OWNERS`, `MAIN_MODEL_OWNER`, `PRIVATE_OWNERS` (Natasha), `SHAREABLE_OWNERS`, `store_dir`, `decide`/`assert_read`/`assert_owner` |
| One identity across two vocabularies | `core/character_identity.py` | `CharacterIdentity`, `identity_for`, `same_character`, `all_identities` |
| Editable configuration (5+1 sections) | `core/character_config.py` | `GeneralConfig`, `PersonalityConfig`, `ModelConfig`, `VoiceConfig`, `LearningSettings`, `KnowledgeConfig`, `CharacterConfig`, `SECTIONS`, `PERSONALITY_DEFAULTS`, `replace_*` |
| Persistence + APPLY | `core/character_registry.py` | `CharacterRegistry`, `apply`, `save`, `reset`, `default_registry` |
| Standing personas | `core/character_learning.py` | `CHARACTER_KEYS`, `PERSONAS`, `CharacterPersona.system_prompt` |
| Per-character memory + learning | `core/character_learning.py`, `core/character_learning_store.py`, `core/learning_records.py`, `core/learning_pipeline.py` | `CharacterLearning`, `LearningRecord`, `LearningPipeline` |
| Per-character knowledge corpus | `core/character_knowledge.py`, `core/character_knowledge_gate.py` | `CharacterKnowledge`, `reference_block` |
| Trait-consistency validator | `core/character_consistency.py` | `check_consistency`, `ConsistencyReport` |
| Memory view + recall/decay | `core/character_memory_view.py` | `CharacterMemoryView`, `recall`, `fact_decay` |
| Character manager engine | `gui/character_manager.py` | `CharacterManagerService` |
| Characters window | `gui/character_dialog.py` | `CharacterManagerDialog` |
| CLI twin | `cli/characters.py` | `list/show/set/reset/stats/records/...` |
| The seven-agent framework | `core/agents/` | `PersonalityRegistry`, `AgentSystem`, capabilities, routing, collaboration, conflict, monitoring, isolation |
| The companion (Natasha) | `companion/` | `CompanionAgent`, `CompanionConfig`, personality, memory, sessions, tone, relationship |
| The scholar (Hakeem) | `hakeem/` | `HakeemAgent`, `HakeemConfig` |

**Reuse verdict:** EXISTING BUT INCOMPLETE for the Studio's needs. The existing
character system is a *petroleum/cognitive* character system: it models
personality dials, memory, learning, and a *relationship story* for **one**
companion. What is missing for a Romance & Companion Studio is:

- **Multi-character companion library** — many user-created companion
  characters (each with the full profile field list in §5 of the brief),
  beyond the eight fixed owners.
- **A structured, numeric relationship *state*** with dimensions + stages +
  a history explaining why it changed (the existing `companion/relationship_manager.py`
  is an *event log* + a "story", not a scored multi-dimension state machine).
- **Conversation *modes*** as first-class objects with their own prompt
  templates.
- **Dating/roleplay scenarios** as durable records (there is no scenario model
  today).
- **Adult-content eligibility gating** as a first-class, disabled-by-default,
  audited switch (the existing `companion/personality.py` `BOUNDARIES` and the
  `explicitness`/`intimacy` dials are the *content* rules, but there is no
  age-eligibility gate, no per-character adult flag, and no safety-event log).

These five are the genuinely new domain components; everything underneath them
(personality, memory, isolation, model routing, voice, storage) is reused.

---

## 8. Document and knowledge-management services

| Concern | Module |
|---|---|
| Citable documents + corpus | `core/documents.py` (`Document`, `DocumentStore`, `build_knowledge_documents`, `build_document_store`) |
| Bulk knowledge base (Wikipedia/Gutenberg/Wikisource) | `core/knowledge_base.py` (`KnowledgeBaseStore`, `build_knowledge_base`, `chunk_text`) |
| FTS5 index | `core/knowledge_index.py` (`KnowledgeTextIndex`) |
| Ingestion pipeline | `ingest/` (`IngestionPipeline`, connectors, dedup ledger, jobs) |
| Import Center | `gui/import_center.py`, `gui/import_dialog.py`, `cli/import_center.py` |
| Knowledge Base Manager | `gui/kb_manager.py`, `gui/kb_manager_dialog.py`, `cli/kb_manager.py` |
| Document reader | `core/document_reader.py` (+ `core/docx_extract.py`, `core/pdf_extract.py`, `core/pdf_ocr.py`) |

**Reuse verdict:** EXISTING. The Studio reuses the document reader for
importing a character's background/biography text and the ingestion pipeline
for optional per-character reference corpora. It must not build a second
ingestion or indexing path.

---

## 9. RAG and semantic retrieval

| Concern | Module |
|---|---|
| Retriever | `core/rag.py` (`RagRetriever`, `BulkCorpusRetriever`, `RetrievedPassage`) |
| Learned semantic space | `core/embeddings.py` (`cooccurrence_counts`, PPMI, truncated SVD, `evaluate_held_out`) |
| Grounded assistant | `core/assistant.py` (`GroundedAssistant`, `build_default_assistant`, `AssistantAnswer`) |
| Groundedness gate | `core/grounding.py` (`check_grounding`) |
| Answer confidence | `core/answer_confidence.py` |
| Reasoning trace | `core/reasoning_report.py` |

**Reuse verdict:** EXISTING. The Studio's memory retrieval reuses
`core.rag.RagRetriever` for any per-character reference corpus and reuses
`core.embeddings` for semantic recall (already used by
`core/character_memory_view.recall`). It must not build a second RAG engine.

---

## 10. SQLite database infrastructure

Three proven patterns, all **standard-library `sqlite3`**:

1. `storage/sqlite_store.py` — the base `SqliteStore` (the simplest precedent).
2. `core/memory_db.py` — one versioned, transactional, backup/restore DB per
   owner; `SCHEMA_VERSION`, `MIGRATIONS`, corrupt-row tolerance, bounded paging.
3. `core/workplace/` — the richest precedent: `schema.py` (`TableSpec`,
   `Column`, `MIGRATIONS`, `SCHEMA_VERSION`, `ON DELETE RESTRICT` foreign keys),
   `database.py` (`WorkplaceDatabase`), `repository.py` (typed CRUD),
   `domain.py` (audited create/update/deactivate with soft delete).
4. `companion/database.py` — `CompanionDatabase`, `SCHEMA_VERSION = 2`, a
   v1→v2 migration, timestamped bounded backups, `ON DELETE CASCADE` for
   conversation→message.

**Reuse verdict:** EXISTING (patterns). The Studio uses the **workplace schema
pattern** (`TableSpec`/`Column`/`MIGRATIONS`) for its own versioned database, and
the **companion database pattern** for conversation/message tables. It does not
add a new database engine or an ORM.

---

## 11. Configuration, logging, events, background tasks

| Concern | Module |
|---|---|
| Paths / data layout | `core/paths.py` (`data_dir`, `*_dir`, `ensure_data_layout`) |
| Preferences | `core/preferences.py` (`UserPreferences`, `PreferenceRecord`) |
| Per-mouth profile | `core/mouth_profile.py` (`MouthProfile`, `load_profile`, `save_profile`) |
| Per-engine studio profile | `core/voice_studio_profile.py` |
| JSON persistence | `storage/json_store.py` (`JsonStore`) |
| Settings hub (global registry) | `gui/settings_hub.py` (`SETTINGS_ENTRIES`, `CONTEXTUAL_SETTINGS`, `current_values`) |
| Audit trail | `core/audit_log.py`, `core/learning_audit.py` |
| Event bus | `core/event_bus.py` |
| Background work in the GUI | `gui/components.py` `WorkerPump`; the "worker thread + `queue.Queue` drained by the UI thread" pattern in every dialog |
| Progress | `core/progress.py` (`ProgressReporter`) |
| Best-effort swallows | `core.best_effort.best_effort` + `swallowed_errors()` |

**Reuse verdict:** EXISTING. The Studio uses `JsonStore` for settings, the
`WorkerPump`/queue pattern for background work, `best_effort` for cosmetic
failures, and registers any global setting in `gui/settings_hub.py`.

---

## 12. GUI design system, components, themes, i18n/RTL

| Concern | Module |
|---|---|
| Component library | `gui/components.py` (`Button`/`button`, `Section`, `DialogHeader`, `HeroHeader`, `StatusLine`, `EmptyState`, `LoadingState`, `SearchInput`, `SettingsRow`, `SettingsSection`, `Notification`, `WorkerPump`, `FormValidator`, `DataTable`) |
| Design tokens | `gui/design.py` (`SPACE`, `RADIUS`, `SIZE`, `VARIANT_*`) |
| Themes (8, tkinter-free, WCAG-checked) | `gui/themes.py` (`Theme`, `THEMES`, `validate_theme`, `contrast_ratio`, `load_theme_name`) |
| Typography (role-based scale, overridable) | `gui/typography.py`, `gui/typography_settings.py` |
| Markdown painting | `gui/rich_text.py` |
| Collapsible sidebar | `gui/sidebar.py` (`SidebarSection`, `SidebarColumn`) |
| Live re-theming / re-typing | `gui/dialog_base.py` (`ThemedDialog`) |
| RTL/LTR + script detection | `gui/layout_direction.py` (`Direction`, `side_for`, `default_alignment`, `detect_direction`) |
| Bilingual labels | Per-module `name`/`name_ar` fields + `label(language=)` helpers (e.g. `gui/workplace.py`, `core/workplace/records.py`) |
| Arabic text pipeline | `core/arabic_prose.py` (`arabic_render`, `render_arabic_text`), `core/arabic_tashkeel.py`, `core/text_polish.py` |
| Window controls | `gui/window_controls.py` (`install_window_controls`, work-area maximize, Escape-to-close) |
| Per-message provenance label | `gui/message_footer.py` (`add_provenance_label`) |

**Reuse verdict:** EXISTING. The Studio window is built from the same
components, the same `Theme`, the same `SidebarColumn`, the same composer, and
the same RTL helper as every other window. The brief's "light and dark themes,
Arabic and English, full RTL/LTR, keyboard navigation, accessibility" are all
already project-wide features the Studio inherits for free.

---

## 13. Tests, fixtures, packaging, and the quality gates

- **Tests:** 434 files under `tests/`, plus sub-packages (`agent/`, `causal/`,
  `decision/`, `end_to_end/`, `integration/`, `learning/`, `memory/`,
  `metacognition/`, `planning/`, `reasoning/`, `temporal/`). `tests/conftest.py`
  counts `tkinter.Tk` constructions and reaps a dead root on the main thread.
  Window tests skip without a display (`TK_AVAILABLE`).
- **CI** (`.github/workflows/ci.yml`): `ruff check .`, `mypy`,
  `xvfb-run -a python -m pytest -q`, `python -m evaluation` (regression gate),
  `python -m ml --episodes 60` (smoke), a wheel build + console-script smoke
  test, and a PyInstaller portable build. **CI installs only `requirements.txt`**
  (which bundles `pytest`, `ruff`, `mypy`, `pypdf`, `python-pptx`).
- **Evaluation gate:** `evaluation/` with `benchmark.METRIC_SPECS`,
  `runner.run_suite`, `baseline.json`. `python -m evaluation` must print PASS
  with no metric regression; `--update-baseline` writes the new reference.
- **Packaging:** `pyproject.toml` (setuptools, `[tool.setuptools.packages.find]`
  with an explicit `include` list — **a new top-level package must be added
  there**), `fayd-al-ruh.spec` (PyInstaller `hiddenimports` via
  `collect_submodules` — **a new package should be added there too**).
- **Version:** one string in `core/version.py`, mirrored in `pyproject.toml`
  and the README status line; `tests/test_version_consistency.py` enforces it.
- **Lint config:** `ruff` selects `E4/E7/E9/F/W/I/UP/B/C4/RUF`; `mypy` checks
  `core, storage, ai, evaluation, ml, cli, gui, scripts, tests, ingest,
  cognition, perception, main.py` — **a new package must be added to
  `[tool.mypy] files`**.

**Reuse verdict:** EXISTING. The Studio follows the same workflow: feature
branch, `pytest`, `ruff`, `mypy`, `evaluation`, and one held-out gate metric.

---

## 14. Privacy, offline and safety enforcement (mandatory for §14 of the brief)

| Concern | Module | Guarantee |
|---|---|---|
| Network egress guard | `core/offline.py` (`ensure_offline`, `install_guard`, `NetworkDisabled`) | With `FAYD_OFFLINE=1`, every connection to an **external** host is blocked; loopback stays open |
| Capability policy | `core/permissions.py` (`Capability.EXTERNAL_TOOL`, deny-by-default `PermissionPolicy`) | No external tool runs without an explicit grant |
| Human approval | `core/human_approval.py` (`HumanApprovalGate`, `ApprovalRequest`) | Every governed external action (web, cloud, model download) needs an explicit approval |
| Isolation policy | `core/learning_isolation.py` | Natasha's records are private; a cross-owner read is refused **before** any path resolves |
| Content boundaries | `companion/personality.py` (`BOUNDARIES`: adults-only 18+, revocable consent, never claim to be human, no manipulation) | The existing companion already enforces the brief's core content rules |
| AI-is-not-an-approver | `core/workplace/security.py` (`is_ai_actor`) | An AI can never be recorded as an approver |
| Audit trail | `core/audit_log.py`, `core/learning_audit.py` | Refusals and actions are recorded |

**Reuse verdict:** EXISTING for offline/permissions/approval/audit.
**EXISTING BUT INCOMPLETE** for *adult-content eligibility*: the content rules
exist, but there is no age-eligibility confirmation gate, no per-character adult
flag, and no `companion_safety_events` log. These are new (see §7).

---

## 15. Capability classification matrix

Legend: **EXISTING** = reuse as-is · **INCOMPLETE** = extend the existing
implementation · **NEW** = implement a new component · **N/A** = not applicable.

### 15.1 Studio shell and navigation

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Standalone Studio window | EXISTING | Copy `gui/workplace_dialog.py` structure; add `_open_romance_studio()` in `gui/app.py` |
| Sidebar navigation entry | EXISTING | Add a `GuiAction` + `_GROUP_SPECS` entry in `gui/actions.py`; add a `cli` command in `cli/menu.py` + `cli/__main__.py` |
| Composer pill entry | EXISTING | `composer.add_mode_selector(...)` in `gui/app.py` |
| Collapsible side panels, responsive | EXISTING | `gui.sidebar.SidebarColumn` |
| Light/dark themes | EXISTING | `gui/themes.py` (8 themes) |
| Arabic/English + RTL/LTR | EXISTING | `gui/layout_direction.py`, bilingual labels, `core/arabic_prose.py` |
| Keyboard navigation / accessibility | EXISTING | `gui/window_controls.py` (Escape), the project-wide Tk patterns |
| Left sidebar: Characters/Discover/Create/Recent/Scenarios/Roleplay/Favorites/Archived | NEW (records) + EXISTING (UI) | New `companion_characters` + `companion_scenarios` records; the panel is a `SidebarColumn` |
| Main workspace (profile/history/composer/voice/mode/relationship/status) | EXISTING (parts) + NEW (wiring) | `ChatComposer`, `TextReader`, new mode selector, new relationship indicator |
| Right info panel (profile/traits/interests/status/milestones/memories/voice/scenario) | NEW (view) + EXISTING (data) | Read from character config, relationship state, memory, voice profile |

### 15.2 Character Studio

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Character profile fields (id/name/nickname/age/adult/gender/biography/nationality/languages/personality/traits/interests/style/humour/romantic prefs/boundaries/relationship style/voice/appearance/avatar/mode/model/created/version) | INCOMPLETE | Extend the existing `core/character_config.CharacterConfig` section model with a companion-specific **profile** section, or add a parallel `CompanionCharacter` record that *references* an owner key |
| Personality configuration dials | INCOMPLETE | Reuse `core/character_config.PersonalityConfig` (formality, emotional_expression, response_length, preferred_language) + `core/personality_layer` registers; add the brief's dials (introversion, seriousness, warmth, confidence, humour, initiative, affection, romantic intensity, formality, dialect) |
| Create/Edit/Duplicate/Import/Export/Archive/Restore/Delete | INCOMPLETE | Reuse `core/character_registry` (`save`/`reset`) + `gui/character_manager.CharacterManagerService`; add duplicate/import/export/archive |
| Data-retention choice on delete | NEW | A retention policy + explicit prompt (reuse `core/retention_policy.py` if suitable) |
| Private memories never shared | EXISTING | `core/learning_isolation.py` + `core/character_config.effective_sharing` |

### 15.3 Conversation modes

| Brief capability | Class | Reuse / extend |
|---|---|---|
| 9 modes with own system prompt + style + context rules + personality integration + safety + continuity | NEW | New `conversation_modes` catalog + a mode-aware prompt builder that extends `companion/prompt_builder.py` |
| Change mode without losing history / no silent rewrite | NEW (policy) + EXISTING (storage) | Store mode per message/turn; never mutate prior rows |
| Adult-content controls (disabled by default, eligibility confirmation, 18+ only, refuse minors/coercion/non-consent) | NEW | `core/companion_safety.py` + `companion_safety_events` table + a per-character `adult_eligible` flag |
| Not representing the AI as human | EXISTING | `companion/personality.py` `BOUNDARIES` already states this |

### 15.4 Relationship Engine

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Independent relationship state per character | NEW | New `relationship_states` table + `core/relationship_engine.py` |
| Dimensions (familiarity/trust/closeness/comfort/affection/shared experiences/stability) | NEW | Bounded, transparent, configurable numeric ranges |
| No random changes | NEW (rule) | Every change derives from an explicit event/evidence |
| Relationship events (first conversation, shared interests, milestones, conflicts, reconciliation) | INCOMPLETE | Reuse `companion/relationship_manager.py` `EVENT_KINDS` + `RelationshipEvent`; extend with new kinds |
| Events stored with timestamps + message refs | INCOMPLETE | `companion/database.py` `relationship_events` already has `occurred_at`; add a message reference column |
| Configurable stages (Stranger…Romantic Partner) | NEW | New `relationship_stages` catalog |
| Progression depends on evidence | NEW | Derive stage from dimension thresholds + event evidence |
| Not dependent/manipulative/possessive/coercive | EXISTING | `companion/personality.py` `BOUNDARIES` |
| Relationship-history screen explaining changes | NEW | Window panel over `relationship_events` + a change reason |
| Reset/edit relationship states | NEW | Service methods + audited |

### 15.5 Companion memory system

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Short-term memory | EXISTING | `core/conversation.Conversation` bounded window |
| Episodic memory | EXISTING | `companion/memory_manager.py` + `core/long_term_memory.py` (`EPISODIC`) |
| Semantic personal memory | EXISTING | `companion/memory_manager.py` (`KIND_FACT`/`KIND_PREFERENCE`) + `core/memory_record.py` |
| Relationship memory | INCOMPLETE | `companion/relationship_memory.py` (`DOMAIN_USER_FACT`/`DOMAIN_FICTION`/`DOMAIN_DIALOGUE_PREF`) exists; the *scored* relationship state is new |
| Character memory (biography/personality/continuity) | EXISTING | `core/character_config.GeneralConfig` + `core/character_learning` facts |
| Per-character + per-user isolation | EXISTING | `core/learning_isolation.py` + `PersonalityMemoryRegistry` |
| Importance/recency scoring, summarization, dedup, contradiction, search, manual review, edit/delete, export, retention, source refs, timestamps | EXISTING | `core/memory_retrieval.py`, `core/memory_management.py`, `core/memory_decay.py`, `core/long_term_memory.py` |
| Retrieve only what is relevant | EXISTING | `core/character_memory_view.recall` (embedding + lexical) |
| Do not copy all history into every prompt | EXISTING | `companion/prompt_builder.py` bounded budget |
| Do not auto-treat every statement as fact | EXISTING | `MemoryManager.propose`/`is_save_request`; `looks_sensitive` refusal |

### 15.6 Local model integration

| Brief capability | Class | Reuse / extend |
|---|---|---|
| GGUF discovery, selection, per-character/per-mode assignment | EXISTING + INCOMPLETE | `core/language_models` + `core/model_manager`; per-mode assignment is new wiring |
| Quantization info, context-length config | EXISTING | `core/model_files.inspect_model_file`, `core/model_settings` |
| CPU inference, optional GPU offload | EXISTING | `core/model_runtime.LlamaCppAdapter` |
| Memory estimation | EXISTING | `core/model_preflight.estimate_memory` |
| Load/unload controls | EXISTING | `core/model_manager.load`/`unload` |
| Runtime health, inference speed, cancellation, timeout, OOM recovery | INCOMPLETE | Health via `runtime_status`; speed/cancellation/timeout/OOM recovery are new *thin* wiring over the adapter |
| Compatibility validation | EXISTING | `core/model_manager.compatibility` |
| Hardware profile (16 GB RAM, 2 GB + 1 GB GPUs, CPU-first) | EXISTING | `core/model_resources` + `core/model_preflight`; document safe defaults |
| Model evaluation facility (Arabic/English quality, consistency, coherence, long-context, roleplay continuity, latency, RAM, VRAM, stability) + comparative report | NEW | A new evaluation harness *over* the existing runtime, writing a report under `docs/romance_companion/` |
| Trusted download with checksums/license/resume | EXISTING | `core/model_download.py` + `core/model_sources.py` (never a second downloader) |

### 15.7 Voice and audio

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Reuse TTS/STT; companion-specific adapter | EXISTING | `core/reading.TextReader` + `core/speech`; the adapter is a thin new class |
| Per-character voice settings (engine/voice/language/dialect/speed/pitch/volume/emotion/presets/enable) | EXISTING | `core/character_config.VoiceConfig` + `core/voice_moods` + `core/voice_studio_profile` |
| TTS, STT, push-to-talk, playback, stop, interruption, segmentation, optional voice mode | EXISTING | `TextReader`, `MicrophoneRecorder`, `split_for_speech`, `batch_chunks` |
| Show only installed/compatible voices | EXISTING | `core/voices.voice_sections` |

### 15.8 Dating and roleplay studio

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Scenario builder (title/description/location/time/participants/roles/background/situation/objectives/narrative style/content level/boundaries) | NEW | New `companion_scenarios` + `scenario_participants` records |
| Scenario categories | NEW | A catalog of category constants |
| Interactive storytelling (consistency, continuity, scene context, relationships, previous events, perspective, user choices) | NEW (engine) + EXISTING (data) | A `RoleplayEngine` over the existing conversation + memory + relationship state |
| Pause/resume/branch/save/restart | NEW | Scenario state machine + stored branches |

### 15.9 Multi-character conversations

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Independent profiles/personalities/memories/dialogue identity | EXISTING | `core/learning_isolation`, `core/character_config`, `companion`/`hakeem` |
| Orchestrator determines speaking turns | NEW | New turn-taking in the `CompanionOrchestrator` |
| Characters must not impersonate one another | NEW (rule) + EXISTING (isolation) | A speaker-identity rule in the prompt + isolation guard |
| State persists between sessions | EXISTING | SQLite + `JsonStore` |
| Add/remove characters during a scenario | NEW | Scenario participant management |
| Sequential generation to bound memory | NEW (policy) + EXISTING (runtime) | One model, one turn at a time |

### 15.10 Database, privacy, management, testing

| Brief capability | Class | Reuse / extend |
|---|---|---|
| Versioned schema + migrations + FKs + indexes + transactions + stable IDs + UTC + integrity + safe delete + migration tests | EXISTING (pattern) | Copy the `core/workplace/schema.py` pattern |
| Privacy: local-only, no remote conv API, no telemetry, no remote sync, explicit download permission, export/delete, optional local encryption | EXISTING + INCOMPLETE | `core/offline.py` etc. already enforce it; local DB encryption is NEW (optional) |
| Display active inference provider + local runtime status | INCOMPLETE | `core/answer_provenance.py` names the model; add a runtime-status line |
| Model Manager / Character Manager / Memory Manager / Settings screens | EXISTING + NEW | Reuse `gui/model_manager_dialog.py`, `gui/character_dialog.py`; new Memory Manager + Settings panels for the Studio |
| Unit/integration/safety/regression tests | EXISTING (framework) | New test files following project conventions |
| Final feature matrix, docs, reports | NEW | This folder |

---

## 16. Proposed component architecture

The brief proposes a `romance_companion/` package. To respect the project's
conventions (a `core/` sub-package for domain logic, a `gui/` module for the
Tk-free service + window, and a `cli/` module for the headless twin), the
proposal is **`core/romance/`** for the domain, with the UI and CLI beside the
other surfaces. (The exact package name is a decision for the user; this
document uses `core/romance/` as the working name, matching `core/workplace/`.)

```
core/romance/                 # domain (standard library + project modules only)
    __init__.py               # public surface
    model.py                  # frozen dataclasses (Character, Scenario, RelationshipState, Memory, Mode)
    schema.py                 # TableSpec/Column/MIGRATIONS/SCHEMA_VERSION
    database.py               # RomanceDatabase (versioned, transactional, backup/restore)
    repository.py             # typed CRUD (like core/workplace/repository.py)
    domain.py                 # audited create/update/deactivate (soft delete)
    seed.py                   # shipped example characters/scenarios (clearly marked as examples)
    characters.py             # CharacterManager (library CRUD, duplicate/import/export/archive)
    personality.py            # CharacterPersonalityEngine (dials -> prompt block)
    modes.py                  # conversation-mode catalog + prompt templates
    relationship.py           # RelationshipEngine (dimensions, stages, events, explanations)
    memory.py                 # CompanionMemoryManager (per-character isolation, importance/recency)
    roleplay.py               # RoleplayEngine (scenarios, branches, continuity)
    routing.py                # CompanionModelRouter (thin over core.model_manager)
    voice.py                  # CompanionVoiceAdapter (thin over core.reading)
    safety.py                 # CompanionSafetyManager (adult eligibility, refusals, audit)
    orchestrator.py           # CompanionOrchestrator (single + multi-character turns)
    storage.py                # CompanionStorageService (paths, export/delete, backup)
    gate.py                   # held-out gate: companion_studio_accuracy

gui/romance.py                # RomanceStudioService (Tk-free; the one engine)
gui/romance_dialog.py         # RomanceStudioDialog (Tk, imported lazily)
cli/romance.py                # headless twin (python -m cli romance ...)

docs/romance_companion/       # this folder + later phase docs
tests/test_romance_*.py       # unit/integration/safety/window tests
```

**Boundary rules (non-negotiable):**
- The domain package imports **no Tk** and **no `llama_cpp`** (lazy seams only),
  matching `core/workplace/`.
- The GUI service (`gui/romance.py`) is the single engine the window and the CLI
  both drive, so they can never disagree.
- Every external action (a model download, a web lookup) passes the existing
  permission + human-approval gates; the Studio adds no new egress path.
- Isolation is structural: the Studio resolves every character through
  `core/character_identity` + `core/learning_isolation` before touching a path.

---

## 17. Proposed database schema

The Studio uses its **own** versioned SQLite database
(`data/romance/romance.db`, git-ignored, resolved through `core/paths`), built
on the `core/workplace/schema.py` pattern. It does **not** duplicate the shared
model/voice/document services (those keep their own stores and are referenced by
key).

```sql
-- v1 -------------------------------------------------------------------
PRAGMA foreign_keys = ON;

-- One companion character. ``owner`` links to the character-identity vocabulary
-- (sources/ultron/.../natasha or a new ``romance:<slug>`` owner) so a Studio
-- character can *wrap* an existing persona OR be a brand-new companion.
CREATE TABLE companion_characters (
    record_id        TEXT PRIMARY KEY,          -- CHR-<uuid>
    owner            TEXT NOT NULL,             -- identity key (isolation policy)
    name             TEXT NOT NULL,
    nickname         TEXT NOT NULL DEFAULT '',
    fictional_age    INTEGER,                   -- NULL when unspecified
    adult_eligible   INTEGER NOT NULL DEFAULT 0,-- 1 only after explicit confirmation
    gender           TEXT NOT NULL DEFAULT '',
    biography        TEXT NOT NULL DEFAULT '',
    nationality      TEXT NOT NULL DEFAULT '',
    languages        TEXT NOT NULL DEFAULT '[]',-- JSON array
    appearance       TEXT NOT NULL DEFAULT '',
    avatar_path      TEXT NOT NULL DEFAULT '',
    relationship_style TEXT NOT NULL DEFAULT '',
    default_mode     TEXT NOT NULL DEFAULT 'general',
    model_key        TEXT NOT NULL DEFAULT '',  -- core.language_models key
    archived         INTEGER NOT NULL DEFAULT 0,
    profile_version  INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,             -- UTC ISO-8601
    updated_at       TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_char_owner  ON companion_characters(owner);
CREATE INDEX idx_char_active ON companion_characters(active, archived);

-- Personality dials (0..1) + free instructions, one row per character.
CREATE TABLE character_personalities (
    record_id        TEXT PRIMARY KEY,
    character_id     TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    introversion     REAL NOT NULL DEFAULT 0.5,
    playfulness      REAL NOT NULL DEFAULT 0.5,
    warmth           REAL NOT NULL DEFAULT 0.5,
    expressiveness   REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 0.5,
    humour           REAL NOT NULL DEFAULT 0.5,
    initiative       REAL NOT NULL DEFAULT 0.5,
    affection        REAL NOT NULL DEFAULT 0.5,
    romantic_intensity REAL NOT NULL DEFAULT 0.5,
    formality        REAL NOT NULL DEFAULT 0.5,
    response_length  TEXT NOT NULL DEFAULT 'balanced',
    dialect          TEXT NOT NULL DEFAULT '',
    instructions     TEXT NOT NULL DEFAULT '',
    traits           TEXT NOT NULL DEFAULT '[]',-- JSON array
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

-- Free-form profile items (interests, hobbies, communication style, boundaries...).
CREATE TABLE character_profiles (
    record_id    TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    kind         TEXT NOT NULL,   -- interest | hobby | style | preference | boundary | goal
    label        TEXT NOT NULL,
    value        TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_profile_char ON character_profiles(character_id, kind);

-- Explicit conversation boundaries (user-controlled limits).
CREATE TABLE character_boundaries (
    record_id    TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    kind         TEXT NOT NULL,   -- topic | act | language | content_level
    value        TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

-- The scored relationship state: one live row per (character, user).
CREATE TABLE relationship_states (
    record_id     TEXT PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    user_id       TEXT NOT NULL DEFAULT 'local',  -- per-user isolation
    familiarity   REAL NOT NULL DEFAULT 0.0,
    trust         REAL NOT NULL DEFAULT 0.0,
    closeness     REAL NOT NULL DEFAULT 0.0,
    comfort       REAL NOT NULL DEFAULT 0.0,
    affection     REAL NOT NULL DEFAULT 0.0,
    shared_experiences REAL NOT NULL DEFAULT 0.0,
    stability     REAL NOT NULL DEFAULT 0.5,
    stage         TEXT NOT NULL DEFAULT 'stranger',
    updated_at    TEXT NOT NULL,
    UNIQUE (character_id, user_id)
);

-- Why the relationship changed (append-only; carries the evidence).
CREATE TABLE relationship_events (
    record_id     TEXT PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    kind          TEXT NOT NULL,   -- first_conversation | shared_interest | milestone | conflict | reconciliation ...
    title         TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT '',
    dimension     TEXT NOT NULL DEFAULT '', -- which dimension moved, if any
    delta         REAL NOT NULL DEFAULT 0.0,
    message_id    TEXT NOT NULL DEFAULT '', -- supporting message reference
    conversation_id TEXT NOT NULL DEFAULT '',
    occurred_at   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_relev_char ON relationship_events(character_id, occurred_at);

-- Companion memories (short-term lives in the conversation window; the durable
-- kinds are episodic | semantic_personal | relationship | character).
CREATE TABLE companion_memories (
    record_id     TEXT PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.5,
    confidence    REAL NOT NULL DEFAULT 0.5,
    source        TEXT NOT NULL DEFAULT 'user',
    source_message_id TEXT NOT NULL DEFAULT '',
    fictional     INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_mem_char ON companion_memories(character_id, kind);
CREATE INDEX idx_mem_created ON companion_memories(character_id, created_at);

-- One conversation (mode + scenario are conversation-scoped).
CREATE TABLE companion_conversations (
    record_id     TEXT PRIMARY KEY,          -- CNV-<uuid>
    character_id  TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    title         TEXT NOT NULL DEFAULT 'Conversation',
    mode          TEXT NOT NULL DEFAULT 'general',
    scenario_id   TEXT NOT NULL DEFAULT '',  -- optional
    created_at    TEXT NOT NULL,
    last_active   TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX idx_conv_char ON companion_conversations(character_id, last_active);

-- Messages (append-only; a mode change never rewrites an old row).
CREATE TABLE companion_messages (
    record_id       TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES companion_conversations(record_id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,   -- user | assistant | system | context | character:<key>
    speaker         TEXT NOT NULL DEFAULT '', -- which character spoke (multi-character)
    mode            TEXT NOT NULL DEFAULT '', -- mode in force when this message was written
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (conversation_id, seq)
);

-- Scenarios (dating/roleplay).
CREATE TABLE companion_scenarios (
    record_id     TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    category      TEXT NOT NULL DEFAULT 'first_meeting',
    description   TEXT NOT NULL DEFAULT '',
    location      TEXT NOT NULL DEFAULT '',
    setting_time  TEXT NOT NULL DEFAULT '',
    background    TEXT NOT NULL DEFAULT '',
    starting_situation TEXT NOT NULL DEFAULT '',
    objectives    TEXT NOT NULL DEFAULT '[]',
    narrative_style TEXT NOT NULL DEFAULT '',
    content_level TEXT NOT NULL DEFAULT 'general',
    boundaries    TEXT NOT NULL DEFAULT '[]',
    is_real_place INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE scenario_participants (
    record_id   TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL REFERENCES companion_scenarios(record_id) ON DELETE CASCADE,
    character_id TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    role        TEXT NOT NULL DEFAULT '',
    order       INTEGER NOT NULL DEFAULT 0
);

-- A running roleplay (pause/resume/branch/save/restart).
CREATE TABLE scenario_sessions (
    record_id     TEXT PRIMARY KEY,
    scenario_id   TEXT NOT NULL REFERENCES companion_scenarios(record_id) ON DELETE RESTRICT,
    conversation_id TEXT NOT NULL REFERENCES companion_conversations(record_id) ON DELETE CASCADE,
    state         TEXT NOT NULL DEFAULT 'active', -- active | paused | ended
    branch_of     TEXT NOT NULL DEFAULT '',       -- parent session when branched
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

-- Model assignment per character / per mode (references the shared catalog by key).
CREATE TABLE companion_model_profiles (
    record_id   TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,   -- character | mode
    scope_id    TEXT NOT NULL,   -- character id or mode key
    model_key   TEXT NOT NULL,   -- core.language_models key
    context_length INTEGER NOT NULL DEFAULT 4096,
    max_tokens  INTEGER NOT NULL DEFAULT 512,
    temperature REAL NOT NULL DEFAULT 0.7,
    top_p       REAL NOT NULL DEFAULT 0.95,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

-- Voice profile per character (references core.voices / core.speech by key).
CREATE TABLE character_voice_profiles (
    record_id   TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES companion_characters(record_id) ON DELETE RESTRICT,
    engine      TEXT NOT NULL DEFAULT '',   -- system | piper | audio.cpp | qwen3-tts | synthetic
    voice_key   TEXT NOT NULL DEFAULT '',   -- system: | model: | catalog key
    language    TEXT NOT NULL DEFAULT '',
    dialect     TEXT NOT NULL DEFAULT '',
    speed       REAL NOT NULL DEFAULT 1.0,
    pitch       REAL NOT NULL DEFAULT 0.0,
    volume      REAL NOT NULL DEFAULT 1.0,
    mood        TEXT NOT NULL DEFAULT 'neutral',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1
);

-- Studio settings (key/value, like companion_settings).
CREATE TABLE companion_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Safety events (refusals, eligibility confirmations) -- append-only.
CREATE TABLE companion_safety_events (
    record_id   TEXT PRIMARY KEY,
    character_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL,   -- adult_eligibility | refusal | boundary_set | age_ambiguous
    reason      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
```

**Migration discipline (copied from `core/workplace/schema.py`):**
`SCHEMA_VERSION` + numbered `MIGRATIONS`; a database from a *newer* build is
refused; every migration runs in its own transaction; a migration with no path
is refused rather than half-applied. Migration tests are required (§16 of the
brief).

---

## 18. Required changes to existing files

| File | Change | Risk |
|---|---|---|
| `gui/actions.py` | Add one `GuiAction("romance-studio", …)`; add it to a group in `_GROUP_SPECS` | Low (additive) |
| `gui/app.py` | Add `_open_romance_studio()`; add a composer pill `add_mode_selector`; register the dialog in `_open_dialogs` for live re-theming | Low (additive) |
| `cli/menu.py` | Add `run_romance()` and a `MENU` row | Low (additive) |
| `cli/__main__.py` | Add `"romance": run_romance` (and an alias) to `_COMMANDS` | Low (additive) |
| `core/paths.py` | Add `romance_dir()`/`romance_db_path()`/`romance_backups_dir()` (+ `__all__`) | Low (additive) |
| `pyproject.toml` | Add `romance` to `[tool.setuptools.packages.find].include` and to `[tool.mypy].files` | Low |
| `fayd-al-ruh.spec` | Add `collect_submodules("romance")` (or the chosen package name) | Low |
| `gui/settings_hub.py` | Register any *global* Studio setting (default model/mode/content level) | Low |
| `core/character_config.py` | *(Optional)* add a companion profile section, **or** keep Studio characters in the new package and reference an owner key | Medium — prefer the new-package option to avoid disturbing the enforced 6-section model |
| `README.md`, `AGENTS.md`, `core/version.py` + `pyproject.toml` | Document the new studio; bump version in all three | Low but must be consistent |

**Deliberately NOT changed:** the cognitive core, the model manager, the speech
layer, the isolation policy, the evaluation baseline (except by adding new
metric entries through the sanctioned `--update-baseline` flow), the existing
companion/hakeem behaviour, and any existing test.

---

## 19. Genuinely new files (first cut)

Domain: `core/romance/{__init__,model,schema,database,repository,domain,seed,
characters,personality,modes,relationship,memory,roleplay,routing,voice,safety,
orchestrator,storage,gate}.py`.

Surfaces: `gui/romance.py`, `gui/romance_dialog.py`, `cli/romance.py`.

Tests: `tests/test_romance_*.py` (characters, personality, relationship, memory,
isolation, conversation persistence, scenarios, routing, model-load failure,
voice adapter, safety, migrations, and an end-to-end workflow test mirroring
§16 of the brief).

Docs: this folder — `ARCHITECTURE_AUDIT.md`, `IMPLEMENTATION_PLAN.md`, and the
phase docs named in §18 of the brief (database, characters, relationship,
memory, model integration, voice integration, privacy/safety, install guide,
user guide, test report, hardware compatibility, feature matrix).

---

## 20. Risks and unresolved technical questions

1. **Package name and placement.** The brief suggests `romance_companion/`; the
   project convention is `core/<domain>/` + `gui/<domain>.py` + `cli/<domain>.py`.
   *Question for the user:* which name should be used? This document assumes
   `core/romance/` + `gui/romance*.py` + `cli/romance.py`.
2. **Adult-content policy boundary.** The existing companion already allows
   consensual adult expression with hard safety rules. The Studio adds an
   **eligibility gate** (disabled by default, explicit confirmation, per-character
   adult flag, refusal of minors/ambiguity/coercion/non-consent, safety-event
   log). *Question:* is the existing `companion/personality.py` boundary set the
   accepted policy, and should the Studio *reuse* it verbatim as the content
   floor? (Recommended: yes.)
3. **Relationship scoring transparency.** The brief forbids random changes and
   demands an explanation screen. The proposed design derives every delta from an
   explicit event with evidence; *question:* what should the default dimension
   ranges and stage thresholds be (0..100? 0..1?), and who may edit them?
4. **Hardware reality.** 16 GB RAM with 2 GB + 1 GB GPUs means CPU-first, one
   model resident, and **sequential** multi-character generation. A 7B Q4 model
   is ~4.5 GB and workable; a larger model may not fit alongside the OS. The
   model-evaluation facility must measure this honestly on the target machine
   (we cannot measure the user's exact machine here).
5. **Local database encryption.** The brief lists it as "optional". Standard-
   library `sqlite3` has no built-in encryption; a real implementation needs
   SQLCipher (a new dependency) or an app-level cipher over field values.
   *Question:* is this in scope, and if so which approach is acceptable? The
   default plan is to **not** add a dependency and instead document the
   limitation plus provide export/delete and a key-file-based field cipher only
   if the user approves a dependency.
6. **New-character ownership keys.** Studio characters need an isolation owner
   key. The existing policy is a closed set of eight owners. Adding
   `romance:<slug>` owners touches `core/learning_isolation.py` (a security-
   sensitive module). *Question:* should Studio companions be *new* owners, or
   should each Studio companion *reference* an existing owner (e.g. run as the
   Natasha/companion persona)? The recommended, lowest-risk option is: Studio
   characters are new records that reference an existing persona **template**
   for isolation, so no change to the closed owner set is needed in Phase 1.
7. **Multi-character turn-taking.** Deterministic, sequential, bounded turn
   selection (no parallel generation) is required by the hardware profile; the
   policy details (who speaks when, round-robin vs. addressed) are a design
   decision to confirm.
8. **Scenario persistence vs. the existing companion DB.** Two reasonable
   options: a *new* `core/romance` database (recommended, clean separation) or
   extending `companion/database.py` (would couple the Studio to one character).
   This document recommends the new database.
9. **Evaluation integration.** The project requires a held-out gate metric.
   The Studio adds `companion_studio_accuracy` (with negative controls) and
   re-runs `python -m evaluation --update-baseline`. *Question:* is adding a
   new metric (which changes `overall`) acceptable? (The project's own history
   shows this is the norm.)
10. **Scope and phasing.** This is a large, ten-phase build. It must be delivered
    in independently reviewable phases (the brief's §17), each with its own
    tests and docs, and each keeping the four hard gates green.

---

## 21. Summary of the reuse decision

- **Reuse wholesale (do not duplicate):** model manager + catalog + download +
  resources + preflight; speech/TTS/STT + voices + reading + moods; RAG +
  embeddings + documents + ingestion + knowledge index; SQLite patterns
  (`memory_db`, `workplace`, `companion`); isolation + identity; permissions +
  approval + offline + audit; GUI components + themes + typography + sidebar +
  composer + dialog base + RTL; the evaluation gate and CI.
- **Extend (existing but incomplete):** character configuration/registry and the
  companion relationship event log; per-mode model assignment; runtime
  health/speed/cancellation/timeout/OOM wiring; the answer-provenance/runtime
  status line.
- **New:** the Studio domain package (characters library, personality engine,
  conversation modes, relationship engine, companion memory manager, roleplay
  engine, model router, voice adapter, safety manager, orchestrator, storage),
  the Studio window, the Studio CLI, the Studio database, the model-evaluation
  facility, and the documentation set.
- **Not applicable:** a second RAG engine, a second model downloader/registry,
  a second TTS/STT engine, a second document-ingestion system, a second generic
  knowledge store, and any cloud/remote inference path.

**This is Phase 0 only. No application code was modified. Awaiting approval
before Phase 1.**
