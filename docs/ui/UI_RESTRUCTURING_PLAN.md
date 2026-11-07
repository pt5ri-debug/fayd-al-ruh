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

## Phase 3 — Centralize settings

**Goal:** one authoritative home per **global** setting; context-dependent ones
stay distinct.

- `gui/settings_hub.py` (new, Tk-free model) + `gui/settings_dialog.py` (new
  window): a single Settings window whose sections *link to* the authoritative
  editors (Themes, Fonts, Reading Voice, Cloud, the internet switch, Backup,
  Diagnostics) rather than re-implementing them.
- Ensure the **internet switch** is one preference read by the main composer,
  the Training Center, and Jarvis.
- **Do not merge** per-mouth voice/model, per-character model, per-agent
  permissions, or per-task limits; add contextual links instead.

**Gate:** a test asserts each global setting has exactly one writer; the
context-dependent settings are asserted to remain separate.

## Phase 4 — Restructure existing windows and dialogs

**Goal:** every dialog uses the shared components and the same skeleton.

- Migrate each of the 18 dialogs to `DialogHeader` + `Section` + `WorkerPump` +
  `TranscriptRow`, removing the 17 `_button` copies and the 12 worker-thread
  copies.
- Preserve every window's `TITLE`, sections, service, and public attributes.

**Gate:** the per-window tests pass unchanged; a test asserts no dialog defines
its own `_button` any more.

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

## Phase 6 — Performance, accessibility, localization

- Route the broad `except Exception: pass` blocks through
  `core.best_effort.best_effort`.
- Keyboard navigation: logical tab order, documented shortcuts, Escape to close.
- RTL layout preference for Arabic; mixed Arabic-English rendering.
- Persist window sizes; confirm long work never blocks the UI thread.

**Gate:** `test_gui_typography.py`/`test_gui_design.py` extended; a test proves
the UI thread is never blocked.

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
