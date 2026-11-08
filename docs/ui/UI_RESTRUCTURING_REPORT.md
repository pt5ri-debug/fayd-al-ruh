# Fayd Al-Ruh — UI Restructuring Report (Phases 0-8)

> The final report the restructuring plan (`UI_RESTRUCTURING_PLAN.md`) requires.
> Every claim below is backed by a test or a measurement taken on the real
> checkout; nothing is asserted that a command cannot reproduce. The "before"
> figures come from `UI_BASELINE.md` (branch `main`, commit `34e19d6`, version
> `2.3.0`).

## 1. What changed, phase by phase

The restructuring was **additive and de-duplicating**: no established window,
mode, or backend function was removed; every relocated control keeps its
functionality reachable. The one operating mode (`FAYD_MODEL_MODE`) and the
header model selector are unchanged.

| Phase | Deliverable | Where |
|-------|-------------|-------|
| 0 — Audit | UI inventory, functional registry, duplication audit, gaps, plan, baseline | `docs/ui/*.md` |
| 1 — Design system | One component library + design tokens | `gui/components.py`, `gui/design.py` |
| 2 — Shell/navigation | Hierarchical sidebar groups + Home dashboard | `gui/actions.py`, `gui/dashboard.py`, `gui/app.py` |
| 3 — Centralized settings | One authoritative home per global setting; one internet switch | `gui/settings_hub.py` |
| 4 — Windows/dialogs | Shared title band in all 18 dialogs | `gui/components.py` (`DialogHeader`/`HeroHeader`) |
| 5 — Missing essential UI | Personalities, Maintenance (backup/diagnostics/shortcuts/memory), notifications, model-fit note | `gui/agents*.py`, `gui/maintenance*.py`, `gui/notifications.py` |
| 6 — Perf/a11y/l10n | `best_effort` routing, Escape-to-close, LTR/RTL direction, window-size memory, non-blocking UI | `gui/layout_direction.py`, `gui/app.py`, `gui/window_controls.py` |
| 7 — Regression/integration | Full suite + gates + coverage matrix | `UI_COVERAGE_MATRIX.md` |
| 8 — Verification/docs | This report; README/AGENTS updates | `docs/ui/UI_RESTRUCTURING_REPORT.md` |

## 2. Unified design system

- **One component library** (`gui/components.py`, 946 lines): `Button`/`button`
  (variants), `Section`, `DialogHeader`, `HeroHeader`, `StatusLine`,
  `EmptyState`, `LoadingState`, `SearchInput`, `SettingsRow`, `SettingsSection`,
  `Notification`/`NotificationCenter`, `WorkerPump`, `FormValidator`, `DataTable`
  helper. A guard test (`tests/test_ui_components_guard.py`) reads the real
  `gui/` source with `ast` and fails if a dialog grows back its own `_button` or
  `_section`.
- **Design tokens** (`gui/design.py`): the spacing / radius / type accessors
  implied by `gui.themes` + `gui.typography`, named once.
- **Themes**: eight complete palettes (`gui/themes.py`), one switch restyles
  **every** open window (main window rebuilt; each dialog re-themed in place
  through `gui/dialog_base.ThemedDialog`).
- **Typography**: three scopes (windows / lists / chat), one type scale
  (`gui/typography.py`) with Word-like overrides (`gui/typography_settings.py`).
- **Direction**: one LTR/RTL preference (`gui/layout_direction.py`) that moves
  the navigation and supplies the default paragraph alignment.

## 3. Navigation and layout

- **Home / Dashboard** (`gui/dashboard.py`): what is answering, the session, the
  recent conversations, the knowledge figure, the machine's own figures, and
  quick actions — read from the existing services, never fabricated.
- **Hierarchical sidebar**: 45 visible actions in 8 named, collapsible groups
  (*Assistant*, *Knowledge & Web*, *Companion*, *Speech Models*, *Training &
  Learning*, *System*, …), declared once in `_GROUP_SPECS`; an action no group
  names is appended to the last group, so nothing can vanish.
- **Reusable layout**: the shared `SidebarColumn`/`SidebarSection`
  (`gui/sidebar.py`), the shared `ChatComposer` (`gui/chat_composer.py`), and the
  `ThemedDialog` skeleton (`gui/dialog_base.py`) are the one implementation every
  window builds on.

## 4. Centralized settings (no duplicate copies)

`gui/settings_hub.py` is the registry of every **global** setting with its one
authoritative `reader`/`writer`/`store`:

| Setting | Authoritative home |
|---------|--------------------|
| Appearance theme | `gui.themes` |
| Fonts & paragraphs | `gui.typography_settings` |
| Reading voice / auto-speak | `gui.app.load/save_reading_voice`, `load/save_auto_speak` |
| Reading direction | `gui.layout_direction` |
| Internet switch | `core.preferences` (one value: chat composer, Training Center, Jarvis, Settings) |
| Cloud provider / model | `core.cloud_model` |

Settings that are deliberately **context-dependent** are kept distinct and named
in `CONTEXTUAL_SETTINGS` (per-mouth voice/model, per-character model, per-agent
permissions, per-task limits). A shared writer serves exactly one store, so no
two surfaces keep an independent copy — pinned by `tests/test_settings_hub.py`.

## 5. Performance, accessibility, localization

- **Best-effort routing.** Every deliberate swallow goes through
  `core.best_effort.best_effort`, so what was dropped is *recorded*
  (`swallowed_errors()`) rather than lost. Converted in `gui/app.py`,
  `gui/settings_hub.py`, `gui/themes.py`, `gui/typography.py`,
  `gui/typography_settings.py`, `gui/window_controls.py`, `gui/model_chat.py`,
  `gui/model_chat_dialog.py`, `gui/voice_studio_dialog.py`,
  `gui/presentation_dialog.py`, `gui/character_evaluation.py`, and
  `gui/layout_direction.py`.
- **Keyboard navigation.** Escape closes the active dialog (the standard
  affordance), listed in the shortcut catalogue (`gui/maintenance.SHORTCUTS`).
- **RTL/LTR.** The navigation sits left for LTR and right for RTL; the direction
  supplies the default paragraph alignment (an explicit Fonts & paragraphs choice
  still wins); `detect_direction` reads the dominant script of mixed text.
- **Window-size memory.** The main window's size (and maximized flag) is saved on
  close and restored on launch; an unusable saved size falls back to the default.
- **Non-blocking UI.** `WorkerPump` runs long work off the UI thread; a test
  measures the event loop keeps turning while a slow worker runs.

## 6. Testing and verification

| Gate | Baseline (Phase 0) | Final (Phase 7) |
|------|--------------------|-----------------|
| `xvfb-run -a python -m pytest -q` | 7168 passed, 16 skipped | **7333 passed, 16 skipped** |
| `python -m ruff check .` | clean | clean |
| `python -m mypy` | clean (1068 files) | clean (1086 files) |
| `python -m evaluation` | PASS, `overall = 0.9909` | PASS, `overall = 0.9909` |

- **No test was weakened or removed.** The 16 skips are the same
  environment-dependent ones the baseline documents (`espeak-ng` missing,
  optional-extra absences).
- **New tests** (all in `tests/`): `test_components.py` (+1), `test_dashboard.py`,
  `test_agents_service.py`, `test_agents_window.py`, `test_maintenance.py`,
  `test_maintenance_window.py`, `test_notifications.py`,
  `test_layout_direction.py`, `test_settings_hub.py`, `test_ui_components_guard.py`,
  plus additions to `test_gui_design.py`, `test_window_controls.py`,
  `test_typography_settings.py`, `test_model_chat.py`, `test_sidebar.py`.

## 7. Files added / modified

**Added** (source): `gui/components.py`, `gui/design.py`, `gui/dashboard.py`,
`gui/settings_hub.py`, `gui/agents.py`, `gui/agents_dialog.py`,
`gui/maintenance.py`, `gui/maintenance_dialog.py`, `gui/notifications.py`,
`gui/layout_direction.py`, `cli/maintenance.py`.

**Added** (docs): `docs/ui/UI_INVENTORY.md`, `UI_FUNCTIONAL_REGISTRY.md`,
`UI_DUPLICATION_AUDIT.md`, `UI_GAPS_AND_MISSING_FEATURES.md`,
`UI_RESTRUCTURING_PLAN.md`, `UI_NAVIGATION.md`, `UI_COVERAGE_MATRIX.md`,
`UI_BASELINE.md`, `UI_RESTRUCTURING_REPORT.md`.

**Modified**: `gui/app.py`, `gui/actions.py`, the 18 `gui/*_dialog.py` (shared
title band), `gui/typography.py`, `gui/typography_settings.py`, `gui/themes.py`,
`gui/window_controls.py`, `gui/model_chat.py`, `gui/model_chat_dialog.py`,
`gui/voice_studio_dialog.py`, `gui/presentation_dialog.py`,
`gui/character_evaluation.py`, `cli/__main__.py`, `cli/menu.py`, `core/paths.py`,
and the test files named above.

## 8. Remaining limitations (honest)

- **CLI-only subsystems** remain so: `perception/`, `core.planning`/
  `core.execution` (`python -m cli autonomy`), `core.analogy`,
  `core.embeddings`, and the evaluation viewer. Each has a working CLI and the
  answer path already reaches analogy and the evaluation metrics; surfacing them
  is a future phase. They are **not** claimed as verified GUI surfaces.
- **RTL mirroring is navigation + default alignment**, not a full mirror of every
  dialog's internal layout — stated plainly in `gui/layout_direction.py` and the
  Settings switch's tooltip.
- **Tk draws no full justification**, so the Fonts & paragraphs alignment offers
  left / center / right only (`gui.typography_settings.ALIGNMENT_NOTE`).

## 9. How to launch and test

```bash
# Launch the window (needs a display)
python -m cli gui          # or: python -m gui

# Headless twin of every feature
python -m cli              # the numbered menu

# The gates the plan requires
xvfb-run -a python -m pytest -q
python -m ruff check .
python -m mypy
python -m evaluation

# UI-specific suites
xvfb-run -a python -m pytest tests/test_gui_design.py tests/test_components.py \
  tests/test_dashboard.py tests/test_settings_hub.py tests/test_layout_direction.py \
  tests/test_ui_components_guard.py tests/test_maintenance_window.py \
  tests/test_agents_window.py tests/test_notifications.py -q
```

The application stays runnable at every commit; each phase is one branch merged
to `main` only after its gate is green.
