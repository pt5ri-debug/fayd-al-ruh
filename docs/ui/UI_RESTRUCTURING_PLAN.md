# Fayd Al-Ruh — UI Restructuring Plan (Phase 0)

> The ordered plan the later phases execute. Each phase ends with: tests run,
> regressions fixed, functionality verified, changes documented, and a
> recoverable git checkpoint (a small commit on a phase branch, merged to `main`
> after the phase's gate is green).

## Guiding constraints (from `AGENTS.md`)

- Work on a feature branch; never commit to `main` directly.
- Before finishing a phase: `python -m pytest -q`, `python -m ruff check .`,
  `python -m mypy`, `python -m evaluation` (must PASS, no regression).
- **Additive, minimal changes.** No established window, mode, or backend
  function is removed. A relocated control keeps its functionality reachable.
- Everything stays local and Windows-11 compatible.

## Phase 0 — Audit and inventory *(this phase)*

**Deliverables:** `UI_INVENTORY.md`, `UI_FUNCTIONAL_REGISTRY.md`,
`UI_DUPLICATION_AUDIT.md`, `UI_GAPS_AND_MISSING_FEATURES.md`,
`UI_RESTRUCTURING_PLAN.md`, plus a baseline test run recorded in
`UI_BASELINE.md`.

**Gate:** the baseline pytest run is green; the inventory matches the code.

## Phase 1 — Design system and shared components

**Goal:** one place for the primitives every window re-implements.

- `gui/components.py` (new): `Button` (variants: primary/secondary/ghost/small),
  `Section`, `DialogHeader`, `StatusLine`, `EmptyState`, `LoadingState`,
  `Notification`/`NotificationCenter`, `WorkerPump` (start + drain + cancel),
  `SearchInput`, `SettingsRow`, `SettingsSection`, `DataTable` helper,
  `FormValidator`.
- Adopt the component in **at least one** dialog first (prove it), then migrate
  the rest window by window, keeping every existing attribute name so tests and
  handlers are untouched.
- `gui/design.py` (new): the spacing/radius/typography token accessors already
  implied by `gui.themes` + `gui.typography`, named once.

**Gate:** the new components have their own tests; a migrated window's existing
tests still pass unchanged; `ruff`/`mypy` clean.

## Phase 2 — Main shell and navigation *(done)*

**Goal:** a professional, hierarchical navigation without inventing windows.

- Reorganized `gui/actions.py`'s `ACTION_GROUPS` to mirror the requested
  structure — **AI Interaction**, **Cognitive System**, **Personalities**,
  **Model Management**, **Knowledge & Data**, **Learning & Training**,
  **Voice & Audio**, **System** — **as groups over the existing actions**, so no
  action is lost (`grouped_visible_actions` still appends any orphan). The
  section list is declared once in `_GROUP_SPECS` (title, icon, keys).
- Added a **Home / Dashboard** section to the main window: the answering model,
  the session, the recent chats, the knowledge figure, the machine's own figures,
  and quick actions, read through the new Tk-free `gui/dashboard.py`
  (`build_dashboard`/`dashboard_for`), from the existing services (`ChatService`,
  `core.model_resources`, `core.knowledge_base`).
- Kept the header model selector and composer row exactly as they are.

**Gate:** every visible action still resolves and appears in exactly one group;
`tests/test_sidebar.py` + `tests/test_dashboard.py` green.

See `docs/ui/UI_NAVIGATION.md` for the full sidebar map.

## Phase 3 — Centralize settings *(done)*

**Goal:** one authoritative home per **global** setting; context-dependent ones
stay distinct.

- `gui/settings_hub.py` (new, Tk-free): the **registry** of every global setting
  with its one authoritative `reader`/`writer`/`store`, the list of settings that
  are deliberately **context-dependent** (`CONTEXTUAL_SETTINGS`), and the one
  **internet switch** (`read_internet`/`write_internet`/`internet_status`) that
  every surface consults.
- The **internet switch is one preference** now: `TrainingCenterService.web_enabled`/
  `set_web` read and write `core.preferences` through the hub instead of keeping a
  private `training_center/web` value, so the chat composer toggle, the Training
  Center, and the new Settings switch are one value (a legacy value is migrated).
- The main window's **Settings section** became the centralized settings page: it
  reports every global setting and its live value, each with a link to the section
  that owns its editor (Themes, Fonts & paragraphs, Reading Voice), plus the one
  internet switch — so it *shows* a setting rather than keeping a second copy.
- **Not merged** (kept distinct, and named in the registry): per-mouth voice/model,
  per-character model, per-agent permissions, per-task limits.

**Gate:** `tests/test_settings_hub.py` asserts every entry names a real
reader/writer, that a shared writer only serves one store, that the
context-dependent settings are separate, and that the one internet switch is the
same value the Training Center and a chat session read; the window tests assert
the Settings page reports each setting and flips the shared switch.

## Phase 4 — Restructure existing windows and dialogs *(done)*

**Goal:** every dialog uses the shared components and the same skeleton.

- The **title band** was the largest copy: 13 dialogs wrote a near-identical
  `_build_header` (bold title, dim subtitle, the three window controls) and 5 more
  wrote a large "hero" band by hand. All 18 now build their band through one of
  two shared components in `gui/components.py`:
  - `DialogHeader` (title / subtitle / window-control trio, an optional busy
    spinner, an optional `Cancel` button shown exactly while busy, and an `extra`
    slot for a window's own header widget -- a live title label, a text spinner).
  - `HeroHeader` (the `heading1` variant with a subtitle and dim note lines).
  - The band is painted with `Theme.titlebar`/`titlebar_edge` in one place, so a
    window can no longer drift from the rest; a guard test
    (`test_every_dialog_builds_its_title_bar_from_the_shared_header`) scans every
    `gui/*_dialog.py` and fails if a new window reintroduces the copy.
- Every window's own `_button` helper remains a *thin* 4-line delegation to
  `gui.components.button` (the shared factory), not a hand-rolled button: the
  real duplication (button styling) was removed in Phase 1. The per-window helper
  is kept because each window applies its own defaults (`small`/`anchor`), and
  folding those into one signature would be a worse abstraction than four lines.
- Every window's public attributes (`_window_controls`, `_spinner`,
  `_cancel_button`, `_header_title`, `_close_button`) are preserved, so no
  existing test or caller changed.

**Gate:** the per-window tests pass unchanged; the shared-header guard passes.

## Phase 5 — Complete missing essential UI

**Goal:** close the gaps in `UI_GAPS_AND_MISSING_FEATURES.md`.

- Settings hub sections: **Backup/Restore**, **Diagnostics/Logs**,
  **Keyboard shortcuts**.
- A read-only **Memory & Knowledge** viewer over `core.character_memory_view`
  and `core.memory_management`.
- A **Personalities (agents)** view over `core.agents` (read-only + route).
- A **Notification centre** fed by the worker pumps.
- Model-fit **preflight** note where a model is chosen.

**Gate:** each new surface has tests and a row in `UI_COVERAGE_MATRIX.md`.

## Phase 6 — Performance, accessibility, localization *(done)*

- **Broad swallows routed through `core.best_effort.best_effort`.** Every place
  the interface deliberately ignores a failure (a live internet-session switch,
  a per-mouth model activation, stopping a recording/reading on close, the
  Windows/macOS/X11 work-area probes, the installed-font listing) now goes
  through the one helper, so what was dropped is *recorded* rather than lost.
- **Keyboard navigation.** `install_window_controls` binds **Escape** to close
  the active window/dialog (the standard affordance), and the shortcut catalogue
  (`gui.maintenance.SHORTCUTS`) now lists it -- `tests/test_maintenance.py`'s
  binding check confirms the key is really bound.
- **RTL/LTR layout direction.** `gui/layout_direction.py` (new, Tk-free) is the
  one preference: it moves the navigation column to the **right** for RTL and
  left for LTR, chooses the default paragraph alignment, and offers a
  `detect_direction` for mixed text. The main window reads it once at build and
  re-packs the sidebar/main frame on the chosen side; the Settings section has
  the one switch (`_toggle_direction`), which rebuilds the window (the
  transcript is snapshotted first). The direction's default alignment reaches the
  type scales: `Typography.with_overrides` takes a `fallback_alignment`, and
  `FaydGUI._resolve_type_scales` (the one place the three scales are built) passes
  it, so plain text follows the direction while an explicit Fonts & paragraphs
  alignment always wins. The preference is registered in
  `gui.settings_hub` as a global setting with one authoritative home.
- **Mixed Arabic-English rendering.** Pinned by test: the Arabic-script run is
  detected as RTL even with a Latin term inside, an English sentence with an
  Arabic term stays LTR, and a letters-free string uses the fallback rather than
  guessing.
- **Window-size persistence.** The main window's size (and maximized flag) is
  saved on close through `save_window_geometry` and restored at startup through
  `load_window_geometry`; an unusable saved size (below the minimum) falls back
  to the default opening size.
- **Long work never blocks the UI thread.** Pinned by
  `test_worker_pump_never_blocks_the_ui_thread`: a slow worker leaves the Tk
  event loop turning (the pump runs it on its own thread and hands the result
  back through the queue), so the window stays responsive.

**Gate:** `tests/test_layout_direction.py` (new), `tests/test_window_controls.py`,
`tests/test_components.py`, `tests/test_gui_design.py`, and
`tests/test_settings_hub.py` extended; `ruff`/`mypy` clean; `evaluation` PASS.

## Phase 7 — Full regression and integration

- Run the whole suite, `ruff`, `mypy`, `evaluation`; compare against the Phase-0
  baseline in `UI_BASELINE.md`.
- Write `UI_COVERAGE_MATRIX.md` mapping every original control → new location →
  backend → test → regression status.

**Gate:** evaluation PASS with no metric below baseline.

## Phase 8 — Final verification and documentation

- `UI_RESTRUCTURING_REPORT.md`: what changed, files added/modified, performance
  comparison, remaining limitations, and how to launch and test.
- Update `README.md` and `AGENTS.md` with the new UI architecture.

**Gate:** the report's claims are each backed by a test or a measurement.

## Recoverable checkpoints

Each phase is one branch (`ui/phase-N-<name>`), a small commit per logical step,
merged to `main` only after the phase gate is green. The main application stays
runnable at every commit (every change is additive).
