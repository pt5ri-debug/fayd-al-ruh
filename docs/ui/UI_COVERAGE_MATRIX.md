# Fayd Al-Ruh — UI Functional Coverage Matrix (Phases 1-5)

> The verification matrix the plan (`UI_RESTRUCTURING_PLAN.md`) requires: every
> functional control accounted for, with its original location, its function, its
> new location, its backend, and its test/regression status. **A control is only
> listed as verified when a real test exercises it** — never because it is drawn.

## How to read this matrix

| Column | Meaning |
|--------|---------|
| **Control** | the original control |
| **Original** | the window/section it lived in |
| **Function** | the backend operation it performs |
| **New location** | where it is now (unchanged unless stated) |
| **Backend** | the module/service it calls |
| **Test** | the test file that exercises it |
| **Regression** | whether the original test still passes unchanged |

The restructuring was **additive**: no existing control moved or was removed, so
"New location" equals "Original" for everything already present. The rows below
the "Phase 5 additions" heading are the *new* surfaces.

## Phase 1 — Design system (no control moved)

| Control | Original | Function | New location | Backend | Test | Regression |
|---------|----------|----------|--------------|---------|------|-----------|
| every dialog title band | each `gui/*_dialog.py` | title/subtitle/window controls | `gui.components.DialogHeader`/`HeroHeader` | `gui/components.py` | `tests/test_gui_design.py` | OK (unchanged) |
| every window button | each window's `_button` | styled button | `gui.components.button` | `gui/components.py` | `tests/test_gui_design.py` | OK |

## Phase 2 — Main shell / navigation (groups only; no action removed)

| Control | Original | Function | New location | Backend | Test | Regression |
|---------|----------|----------|--------------|---------|------|-----------|
| 45 visible sidebar actions | one flat list | run the action | grouped into 8 named sections | `gui.actions.grouped_visible_actions` | `tests/test_sidebar.py` | OK |
| Home dashboard | (new) | overview | main-window Home section | `gui/dashboard.py` | `tests/test_dashboard.py` | n/a |

## Phase 3 — Centralized settings (by reference; no second copy)

| Setting | Original home | Function | New location | Backend | Test | Regression |
|---------|---------------|----------|--------------|---------|------|-----------|
| Appearance theme | Themes section | read/write theme | Settings page links to Themes | `gui.themes` | `tests/test_settings_hub.py` | OK |
| Fonts & paragraphs | Fonts section | read/write profiles | Settings page links to Fonts | `gui.typography_settings` | `tests/test_settings_hub.py` | OK |
| Reading voice | Reading Voice section | read/write voice | Settings page links to Reading Voice | `gui.app.load/save_reading_voice` | `tests/test_settings_hub.py` | OK |
| Internet switch | 3 places | one preference | one value every surface reads | `core.preferences` via `gui.settings_hub` | `tests/test_settings_hub.py` | OK |

## Phase 4 — Shared dialog skeleton (attributes preserved)

| Control | Original | Function | New location | Backend | Test | Regression |
|---------|----------|----------|--------------|---------|------|-----------|
| `_build_header` (13 dialogs) | each dialog | build the title band | `gui.components.DialogHeader` | `gui/components.py` | `tests/test_gui_design.py` | OK |
| `_window_controls` etc. | each dialog | public attributes | preserved | — | per-window tests | OK |

## Phase 5 additions (new surfaces)

| Control | Window | Function | Backend | Test | Regression |
|---------|--------|----------|---------|------|-----------|
| **Personalities (agents)** action | main sidebar → System | open the agents window | `gui.agents.AgentsService` | `tests/test_agents_window.py` | n/a (new) |
| personality list + capabilities | Personalities window | list profiles, ceilings | `core.agents.registry`/`capabilities` | `tests/test_agents_service.py` | n/a |
| route a question | Personalities window | which personality answers | `core.agents.routing` | `tests/test_agents_service.py` | n/a |
| isolation report | Personalities window | who may read whom | `core.learning_isolation` | `tests/test_agents_service.py` | n/a |
| **Maintenance** action | main sidebar → System | open the maintenance window | `gui.maintenance.MaintenanceService` | `tests/test_maintenance_window.py` | n/a |
| Back up a store | Maintenance window | copy memory/learning/knowledge | `core.memory_management`/`learning_store`/`knowledge_base` | `tests/test_maintenance.py` | n/a |
| Restore a store (governed) | Maintenance window | replace with a backup (asks first) | same | `tests/test_maintenance.py` | n/a |
| Diagnostics | Maintenance window | machine + model figures | `core.model_resources`/`model_manager` | `tests/test_maintenance.py` | n/a |
| Operation log | Maintenance window | recent Model Manager events | `core.model_log.ModelLog` | `tests/test_maintenance.py` | n/a |
| Keyboard shortcuts | Maintenance window | the bindings the app installs | `gui.maintenance.SHORTCUTS` | `tests/test_maintenance.py` | n/a |
| Memory & Knowledge — memory | Maintenance window | one personality's memory (read-only) | `core.character_memory_view` | `tests/test_maintenance.py` | n/a |
| Memory & Knowledge — search | Maintenance window | search the bulk KB (read-only) | `core.knowledge_index` | `tests/test_maintenance.py` | n/a |
| Notification feed | main window | record background outcomes | `gui.notifications.NotificationFeed` | `tests/test_notifications.py` | n/a |
| Model-fit note | each per-mouth window | "will it fit?" estimate | `core.model_files`/`model_preflight` | `tests/test_model_chat.py` | n/a |

## Phase 6 additions (accessibility, localization, persistence)

| Control / behaviour | Location | Function | Backend | Test | Regression |
|---------------------|----------|----------|---------|------|-----------|
| Reading direction (LTR/RTL) | Settings section | move the navigation, choose alignment | `gui.layout_direction` | `tests/test_layout_direction.py`, `tests/test_gui_design.py` | n/a |
| Direction default alignment | type scales (all scopes) | plain text follows the direction; an explicit choice still wins | `gui.typography.Typography.with_overrides` | `tests/test_typography_settings.py` | n/a |
| Escape to close | every window's title bar | close the active window/dialog | `gui.window_controls.install_window_controls` | `tests/test_window_controls.py` | OK (unchanged) |
| Window size memory | main window | restore the last size on launch | `gui.app.load_window_geometry`/`save_window_geometry` | `tests/test_gui_design.py` | n/a |
| UI never blocks | every worker pump | run long work off the UI thread | `gui.components.WorkerPump` | `tests/test_components.py` | OK (unchanged) |
| Escape listed | Maintenance window | shortcut catalogue | `gui.maintenance.SHORTCUTS` | `tests/test_maintenance.py` | OK (unchanged) |

## Target vs. measured

| Target | Measured |
|--------|----------|
| 100% inventory coverage | every control in `UI_INVENTORY.md` is accounted for above |
| 100% mapping of existing functional controls | no control removed; each keeps its backend and test |
| No unintentional feature removal | all existing tests pass unchanged |
| No unresolved duplicate settings | `UI_DUPLICATION_AUDIT.md` + `gui.settings_hub` (one writer per global setting) |
| No unimplemented production buttons | every new button has a callback and a test |
| No unexplained regression failures | the full suite is green (see `UI_BASELINE.md`) |

The matrix is honest about scope: the CLI-only subsystems named in
`UI_GAPS_AND_MISSING_FEATURES.md` §6 ("Still open") are **not** claimed as
verified surfaces.
