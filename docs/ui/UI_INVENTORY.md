# Fayd Al-Ruh — UI Inventory (Phase 0)

> Baseline audit of the entire user interface before the restructuring. Every
> figure here was **measured** on the real checkout (commit `34e19d6`, version
> `2.3.0`), never estimated. This file is the reference the later phases verify
> against.

## 1. Framework and architecture

- **Framework**: Tkinter, part of the CPython standard library. The GUI adds
  **no dependency** (`pyproject.toml`'s base `dependencies` are only `numpy` and
  `requests`).
- **Package**: `gui/` — 44 modules, **41 519 lines**, **845 widget
  constructions** (every `tk.*` / `ttk.*` widget call, counted with `ast`).
- **Layering**:
  - **Tk-free service layer** (testable with no display): `gui/service.py`
    (`ChatService`, 1 442 lines), `gui/model_chat.py`, `gui/voice_studio.py`,
    `gui/training_center.py`, `gui/code_studio.py`, `gui/presentation.py`,
    `gui/docs_studio.py`, `gui/scene3d.py`, `gui/image_gen.py`,
    `gui/image_edit.py`, `gui/import_center.py`, `gui/kb_manager.py`,
    `gui/sql_import.py`, `gui/character_manager.py`, `gui/model_manager.py`,
    `gui/cognition.py`.
  - **Tk widget layer** (imported lazily, never by `cli/`): `gui/app.py` (the
    main window, 4 321 lines) and the 18 dialog windows.
  - **Shared components**: `gui/chat_composer.py`, `gui/sidebar.py`,
    `gui/rich_text.py`, `gui/markdown.py`, `gui/themes.py`,
    `gui/typography.py`, `gui/typography_settings.py`, `gui/dialog_base.py`,
    `gui/window_controls.py`, `gui/message_footer.py`, `gui/clipboard.py`,
    `gui/image_preview.py`, `gui/choice_dialog.py`, `gui/actions.py`,
    `gui/modes.py`, `gui/options.py`.
- **Threading**: every window runs long work on a worker thread and hands the
  result back through a `queue.Queue` the UI thread drains (Tk is not
  thread-safe). 19 modules start worker threads.
- **Entry points**: `python -m cli gui`, `python -m gui`, and `python -m cli`
  (135 numbered menu entries — the headless twin of every GUI feature).

## 2. Windows (18 dialogs + the main window)

Every dialog is a `tk.Toplevel`, subclasses `ThemedDialog` (live re-theme /
re-type), builds a `SidebarColumn` of collapsible sections, uses the shared
`ChatComposer` where it has an input, and installs the three
`gui.window_controls` title-bar buttons.

| # | Window | Module | Title | Sidebar sections | Widgets | Worker | Messages |
|---|--------|--------|-------|------------------|--------:|:------:|:--------:|
| 0 | Main window | `gui/app.py` | — (brand) | 8 groups + Chats + Reading Voice + Themes + Fonts + Settings | 103 | ✓ | ✓ |
| 1 | Characters | `gui/character_dialog.py` | — | Characters, Knowledge sharing, (pages A–F) | 51 | ✓ | ✓ |
| 2 | Code Studio | `gui/code_studio_dialog.py` | Code Studio | Workspaces, Explorer, Editor, Find & Replace, Source Control, Tasks, Notes, Review and apply | 67 | ✓ | ✓ |
| 3 | Autonomous Cognitive System | `gui/cognition_dialog.py` | Autonomous Cognitive System | Mind, Study, Depth, Metacognition, Consolidation, Cognitive Learning, Learning & Experience, Training, Characters, Indicators, System | 51 | ✓ | ✓ |
| 4 | Doc's Studio | `gui/docs_dialog.py` | Doc's Studio | Open, Documents, Text and reading, Model | 36 | ✓ | ✓ |
| 5 | Image generation | `gui/image_gen_dialog.py` | Generate an image from text | (cards, no sidebar) | 32 | ✓ | ✓ |
| 6 | Image editing | `gui/image_edit_dialog.py` | Edit an image from text | (cards) | 26 | ✓ | ✓ |
| 7 | Knowledge Import Center | `gui/import_dialog.py` | Knowledge Import Center | (cards) | 35 | ✓ | ✓ |
| 8 | Knowledge Base Manager | `gui/kb_manager_dialog.py` | Knowledge Base Manager | Character, Documents, Web, Corpus | 37 | ✓ | ✓ |
| 9 | Model chat (per mouth ×7) | `gui/model_chat_dialog.py` | — | Character, Reading voice, Language model | 31 | ✓ | ✓ |
| 10 | Model Manager | `gui/model_manager_dialog.py` | Model Manager | Summary, Tasks, Library, Runtimes, Downloads, Resources, Trained, Diagnostics, Log | 45 | ✓ | ✓ |
| 11 | Presentation Studio | `gui/presentation_dialog.py` | Presentation Studio | Projects, Workflow, Slides, Deck look, Export, Course content | 70 | ✓ | ✓ |
| 12 | Course content | `gui/presentation_course_dialog.py` | Course content | Matches, Concepts, Use a concept | 23 | ✓ | ✓ |
| 13 | Final presentation editor | `gui/presentation_finalize_dialog.py` | Final presentation editor | Slides, Deck look, Export | 54 | ✓ | ✓ |
| 14 | Scene Studio | `gui/scene3d_dialog.py` | Scene Studio | Scene, Figures, Props, Poses | 41 | ✓ | ✓ |
| 15 | SQL Database Import | `gui/sql_import_dialog.py` | SQL Database Import | (numbered cards 1–4) | 40 | ✓ | ✓ |
| 16 | Training Center | `gui/training_center_dialog.py` | Training Center | Dashboard, Sources, Knowledge, Q&A, Datasets, Jobs, Fine-Tuning, Pretraining, Evaluation, KB, Pipelines, History, Resources, Settings | 26 | ✓ | ✓ |
| 17 | Voice studio (per engine ×4) | `gui/voice_studio_dialog.py` | — | Engine, Reading voice, My voice (clone), Mood, Document, Engine setup | 43 | ✓ | ✓ |
| 18 | Choice dialog | `gui/choice_dialog.py` | — (modal) | — | 2 | — | — |

## 3. Interactive widget census

| Widget | Count | Where it dominates |
|--------|------:|--------------------|
| `tk.Label` | 176 | everywhere (no shared label factory) |
| `tk.Frame` | 258 | everywhere |
| `tk.Text` | 103 | transcripts, editors, code, previews |
| `tk.Button` | 58 | 17 hand-rolled `_button` factories |
| `tk.Entry` | 42 | forms, search boxes |
| `tk.Canvas` | 27 | rounded cards, avatars, swatches, viewports |
| `tk.Listbox` | 14 | pickers (voices, models, files, slides) |
| `tk.Menu` | 13 | popup menus |
| `ttk.Combobox` | 8 | dropdowns |
| `tk.OptionMenu` | 11 | model/provider/backend selectors |
| `tk.Radiobutton` | 8 | exclusive choices |
| `tk.Checkbutton` | 13 | toggles |
| `tk.Menubutton` | 4 | mode pills |
| `tk.Spinbox` | 4 | batch counts, sizes |
| `ttk.Treeview` | 3 | model table, SQL tables |
| `ttk.Progressbar` | 5 | busy spinners |
| `tk.Scrollbar` | 6 | scroll regions |

## 4. Main window regions

- **Title bar** (`Theme.titlebar` band): the model button (returns to the
  project's own model), the model selector (`MOUTH_CHOICES`, 7 mouths), the
  theme toggle, and the three window controls.
- **Sidebar** (`_SIDEBAR_WIDTH = 288`, collapsible, scrollbar on overflow):
  brand + version, **New chat**, search box, then collapsible sections:
  *Chats* (dated history), *Reading Voice*, *Themes*, *Fonts & paragraphs*, the
  **8 action groups** (§5), and *Settings* (Cloud provider/model, Save, Clear).
- **Transcript**: a centred column (`_COLUMN_MAX = 820`), user bubbles on the
  right, assistant rows with a gold avatar, Markdown rendering, per-message
  Copy / Read-aloud footers, a live "Thinking…" indicator.
- **Composer** (shared `ChatComposer`): input, ＋ attach, three mode pills
  (*Code Studio*, *Training*, *Chat*), mic, web toggle, Send disc; below the
  card, the **model row** (7 model chips + 4 voice-engine chips).

## 5. Sidebar action groups (59 actions, 45 visible)

| Group | Icon | Visible actions |
|-------|------|-----------------|
| Assistant | ✨ | What the AI can do, Autonomous mind, Learn from a teacher |
| Characters | 🧑 | Characters |
| Language models | 🧩 | Model Manager, Shared models ×3 |
| Knowledge & Web | 📚 | Learn from feedback, Learn from the internet, Ask the web, Build KB, KB status, KB Manager, Ingest sources, Ingestion status |
| Companion | 💬 | Natasha Traits, Mood, Initiative, Relationship memory |
| Speech Models | 🔊 | Qwen3 ×3, Piper ×2, VoiceTut ×3 |
| Training & Learning | 🧠 | Training Center, Train LM, Train on chats, Train on characters, ML pipeline, Release ×2, Curriculum, Demo |
| System | ⚙ | Install support, Evaluation, Self-improvement ×4, Self-check, Updates |

14 actions are `hidden=True` (kept resolvable for the CLI, not drawn): the 7
mouth switches (the header selector offers them), Import Center, Code Studio,
Presentation Studio, Doc's Studio, Image generation, Characters-shareable.

## 6. Persisted UI state (`JsonStore`, data dir)

| Namespace / key | Owner module | Holds |
|-----------------|--------------|-------|
| `gui` / `appearance` | `gui/themes.py` | chosen theme name |
| `gui` / `reading_voice` | `gui/app.py` | reading-voice key |
| `gui` / `auto_speak` | `gui/app.py` | auto-speak flag |
| `gui` / `sidebar_sections` | `gui/app.py` | section open/closed |
| `gui` / `operating_mode` | `gui/modes.py` | mode key |
| `gui` / `typography` | `gui/typography_settings.py` | 3 typography profiles |
| `gui` / `sidebar_sections:<scope>` | `gui/sidebar.py` | per-dialog section state |
| `models` / `manager_selection` | `gui/model_manager.py` | selected model |
| `training_center` / `settings` | `gui/training_center.py` | web switch, etc. |
| `cloud` / provider, model | `core/cloud_model.py` | chosen provider + model |
| `mouths` | `core/mouth_profile.py` | per-mouth name/logo/voice/model |
| `voice_studios` | `core/voice_studio_profile.py` | per-engine name/logo/voice/mood |

## 7. Tests

- 400 test files, ~6 969 test functions. GUI-related: `test_gui.py` (61),
  `test_gui_design.py` (61), `test_typography_settings.py` (36),
  `test_window_controls.py` (19), `test_themes.py` (17), `test_sidebar.py` (16),
  `test_gui_typography.py` (14), plus 12 per-window test files.
- CI runs pytest under `xvfb-run`, so the real Tk windows are exercised, and
  `tests/conftest.py` reaps a dead `tk.Tk` on the main thread.
