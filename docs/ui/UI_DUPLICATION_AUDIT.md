# Fayd Al-Ruh — UI Duplication Audit (Phase 0)

> Where the same setting or the same visual component is implemented more than
> once. Measured with `ast` over the real `gui/` package. The rule applied
> throughout the restructuring: **centralize a *global* setting; keep a
> *context-dependent* one distinct.** This file records both what to merge and
> what must stay separate.

## 1. Duplicated visual components (code duplication)

Methods defined independently in many dialog classes — these are the same
component re-implemented per window:

| Method | Copies | What it draws | Verdict |
|--------|-------:|---------------|---------|
| `_button` | **17** | a flat `tk.Button` in the theme's raised surface | **centralize** → `gui.components.button` |
| `_build_header` | 13 | title + subtitle + spinner + window controls | **centralize** → `gui.components.DialogHeader` |
| `_build_sidebar` | 14 | a `SidebarColumn` + sections | already shared (`gui.sidebar`) — the *call* is duplicated, not the widget |
| `_add_message` | 10 | a transcript row (bubble + footer) | **centralize** → `gui.components.TranscriptRow` |
| `_build_composer` | 11 | the shared `ChatComposer` | already shared; the wiring is duplicated |
| `_build_conversation` | 9 | transcript + composer frame | **centralize** → `gui.components.ConversationPane` |
| `_pump_results` | 12 | drain a worker queue on the UI thread | **centralize** → `gui.components.WorkerPump` |
| `_run_background` | 6 | start a worker thread | **centralize** (same helper as above) |
| `_schedule_pump` | 8 | `after`-based drain scheduling | **centralize** |
| `_start_work` | 7 | busy state + spinner | **centralize** |
| `_set_busy_buttons` | 5 | enable/disable a button list | **centralize** |
| `_finish` | 5 | clear the busy state | **centralize** |
| `_section` | 5 | a titled card frame | **centralize** → `gui.components.Section` |
| `_column_pad` / `_available_column` / `_bubble_*` / `_reflow` / `_fit_height` | 5–11 each | the centred transcript column metrics | **centralize** → `gui.components.TranscriptMetrics` |
| `_scroll_to_bottom` / `_toggle_scrollbar` / `_on_transcript_configure` / `_on_bubble_wheel` / `_on_mousewheel` | 6–11 each | transcript scrolling | **centralize** |
| `_make_read_only` / `_insert_rich` / `_avatar` | 6–8 each | message rendering | **centralize** |
| `_initial_geometry` | 6 | opening size + work-area clamp | **centralize** → `gui.components` |
| `_close` | 17 | window close path | thin; keep per-window but route through a helper |
| `_set_status` | 9 | a status line label | **centralize** → `gui.components.StatusLine` |

## 2. Duplicated settings (data)

### 2a. Global settings that *should* have one home

| Setting | Stored at | Edited in | Verdict |
|---------|-----------|-----------|---------|
| Theme | `gui/appearance` | Main window *Themes* section | **already single-homed** (Phase 63) |
| Typography (3 scopes) | `gui/typography` | Main window *Fonts & paragraphs* | **already single-homed** (Phase 64) |
| Reading voice (main window) | `gui/reading_voice` | Main *Reading Voice* section | single-homed for the main chat |
| Auto-speak | `gui/auto_speak` | Main *Reading Voice* section | single-homed |
| Operating mode | `gui/operating_mode` | (retired to one mode) | single value now |
| Cloud provider | `cloud` namespace | Main *Settings* section | **DUP risk**: also settable from the CLI `cloud use` — same store, OK |
| Cloud model | `cloud` namespace | Main *Settings* section | same store, OK |
| Internet switch | `jarvis`/`preferences` (`web`) | Main *Settings* switch, the chat composer toggle, the Training Center | **was duplicated** (chat read `PreferenceRecord.web`, the Training Center its own `training_center/web`) — **fixed in Phase 3**: one `core.preferences` value, read by all three |
| Sidebar section states | `gui/sidebar_sections[:scope]` | every window | per-window by design |

### 2b. Context-dependent settings that must **stay distinct**

These look like duplicates but are deliberately separate — merging them would be
a regression:

| Setting | Global | Per-context | Why keep apart |
|---------|--------|-------------|----------------|
| Reading voice | main window's `gui/reading_voice` | each mouth window (`mouths` namespace) and each engine studio (`voice_studios`) | a character's voice and an engine studio's voice are exclusive; a global switch must not overwrite them (Phases 30/32) |
| Language model | each mouth's `mouths` profile | each character's `CharacterConfig.model` | a personality-specific override is not the global default |
| Permissions | global sandbox allow-list | each agent's capability ceiling | a personality's ceiling is `requested ∩ ceiling`, never wider |
| Resource limits | global `MonitorConfig` | task-specific limits in the execution engine | a task may set a tighter bound, never a looser one |

### 2c. Genuine duplication to eliminate

1. **`_button` ×17** → one shared factory with a small variant enum.
   *(done — Phase 1; each window's own helper is now a 4-line delegation.)*
2. **`_build_header` ×13 + hero band ×5** → `DialogHeader` + `HeroHeader`.
   *(done — Phase 4; all 18 dialogs build their band from one component, and a
   guard test fails if a new window reintroduces the copy.)*
3. **worker-thread/queue plumbing ×12** → one `WorkerPump` (start + drain +
   cancel) so a window cannot get the thread-safety wrong. *(the shared class
   exists; migrating each window is Phase 5 work, kept separate so each window's
   bespoke busy/cancel/progress behaviour is preserved.)*
4. **transcript row/column metrics ×5–11** → one `TranscriptMetrics` +
   `TranscriptRow`.
5. **status line ×9** → one `StatusLine`.
6. **card/section frame ×5** → one `Section`.

## 3. Controls that are *views* of one source (not duplicates)

These are intentionally repeated because they are quick switches, and they all
funnel into **one** handler, so they can never disagree:

- The **mouth switch** appears three ways: the header selector, the composer's
  Chat pill, and the 7 model chips — all call `_switch_mouth`/`_open_model_chat`.
- The **section open/close** is driven by the header click *and* the
  programmatic `_open_section`, which deliberately does not persist.
- The **theme** appears as the header ☾/☀ toggle *and* the *Themes* list — both
  call `_apply_theme`.

## 4. Rule adopted for Phase 3

- **Centralize**: theme, typography, the internet switch, model download paths,
  and the log — one authoritative store each, with every window reading it.
- **Do not merge**: per-mouth voice/model, per-character model, per-agent
  permissions, per-task resource limits. Add *contextual links* from a dialog to
  the authoritative page instead.

## 5. Phase 3 outcome

The rule is now enforced, not merely stated:

- `gui/settings_hub.py` is the **registry**: every global setting names its one
  authoritative `reader`/`writer`/`store`, and `tests/test_settings_hub.py`
  proves every name resolves to a real function and that no two settings share a
  store they do not belong to.
- The **one internet switch** was de-duplicated: the Training Center no longer
  keeps its own `web` preference — `TrainingCenterService.web_enabled`/`set_web`
  read and write `core.preferences` through the hub, so the chat toggle, the
  Training Center, and the new Settings switch are one value (a legacy
  Training-Center value is migrated on first read).
- The **Settings page** now reports every global setting and its live value, each
  with a link to the section that owns its editor, so the page *shows* a setting
  rather than keeping a second copy of it.
- The **context-dependent** settings remain separate and are named in
  `CONTEXTUAL_SETTINGS`, so a future change cannot quietly merge them.
