# Fayd Al-Ruh — UI Gaps and Missing Features (Phase 0)

> What the backend can do that the interface cannot yet reach, and what is
> missing from both. Every claim was verified by grepping the real `gui/` package
> for the backend module. **No gap is listed as "missing" merely because it is
> technically possible** — each is either a documented backend feature with no
> GUI surface, or a genuine usability hole.

## 1. Backend features with NO GUI surface (verified)

| Backend | GUI reference | Impact | Priority |
|---------|---------------|--------|----------|
| `core.agents` (the seven personalities as independent agents: registry, capabilities, routing, collaboration, conflict) | **none** | A whole Phase-5 subsystem is CLI-only (`python -m cli agents`); a user cannot see or route personalities from the window | **High** |
| `perception/` (document/image/audio/speech → evidence) | **none** | The unified perception pipeline is CLI-only (`python -m cli perception`); the Import Center and Doc's Studio cover parts of it but not the evidence bridge | Medium |
| `core.planning` + `core.execution` (`python -m cli autonomy`) | **none** | Goal→plan→execute→verify is CLI-only; the window cannot run or watch a task plan | Medium |
| `core.analogy` (`python -m cli analogy`) | **none** | Reasoning by analogy is reachable from the answer path (Phase 46) but has no dedicated explorer | Low |
| `core.embeddings` (`python -m cli embeddings`) | **none** | The learned semantic space has no explorer; `similar`/`compare` are CLI-only | Low |
| `core.answer_audit` + the 72 evaluation metrics (`python -m evaluation`) | none (the *System* group runs the CLI, no viewer) | A user cannot see *which* reasoning layer an answer carried | Medium |

## 2. Usability holes (the backend exists; discovery is poor)

| Hole | Evidence | Fix (phase) |
|------|----------|-------------|
| **No single Settings page.** Appearance, typography, the internet switch, cloud, reading voice, and per-mouth settings are spread across the main sidebar's *Settings*, *Themes*, *Fonts*, *Reading Voice* sections and each dialog. | `gui/app.py` builds 4 separate settings sections | A **Settings hub** window (Phase 3) that links to each authoritative editor |
| **The internet switch lives in three places** (main composer, Training Center, Jarvis `web on`). | `set_web`, `TrainingCenterService.set_web`, `JarvisMode.set_web` | One preference (`PreferenceRecord.web`) with contextual switches that all read it (Phase 3) |
| **No context-length / memory-requirement surface in the chat** — only the Model Manager's per-model settings and the training feasibility report. | `gui/model_manager_dialog.py` settings row | Surface a "will this model fit?" preflight note where a model is chosen (Phase 5) |
| **No memory inspection** beyond the Characters window's per-character memory. The long-term memory (Phase 2), knowledge conflicts (Phase 27), and the unified memory view are CLI-only. | `core.memory_management`, `core.character_memory_view` | A **Memory & Knowledge** viewer (Phase 5), read-only, over the existing stores |
| **No system diagnostics page** beyond the Model Manager's diagnostics. Logs, resource figures, and self-check results are CLI-only. | `core.model_log.ModelLog.tail`, `python -m cli selfcheck` | A **System** section in the Settings hub (Phase 3/5) |
| **No backup/restore control.** `core.memory_management.backup/restore`, `core.learning_store.backup/restore`, and the companion's `backup` exist; only the CLI reaches some. | grep above | Backup/Restore buttons in the Settings hub (Phase 5) |
| **No keyboard-shortcut help.** Shortcuts exist (Ctrl+N/S/C/A) but are undocumented in the UI. | `gui/app.py` bindings | A *Keyboard shortcuts* entry (Phase 6) |
| **No RTL layout switch.** Arabic text renders, but the layout is LTR-fixed. | no `justify`/RTL toggle in the shell | A layout-direction preference (Phase 6) |

## 3. Genuine feature gaps (missing from backend AND UI)

| Feature | Purpose | Essential? | Verdict |
|---------|---------|-----------|---------|
| Unified **Settings hub** window | one place to find every global setting | **Yes** | Implement (Phase 3) |
| **Dashboard / Home** panel | system overview: personality, model, recent chats, resources, notifications | **Yes** | Implement as a main-window Home section (Phase 2) |
| **Notification centre** | a single place for background results (a finished training run, a failed download) | Yes | Implement a small one (Phase 5) |
| **Empty/loading/error state components** | consistent feedback | Yes | Implement (Phase 1) |
| **Personality agents view** | the Phase-5 agent system | Yes (backend exists) | Implement read-only + route (Phase 5) |

## 4. Explicitly **not** in scope

To avoid inventing work, these are deliberately excluded:

- **No new model runtime, no new training path** — the request is UI/UX.
- **No second settings store** — Phase 3 centralizes by *reference*, never by a
  parallel copy.
- **No removal of any established mode or window** — *Jarvics*/*AI* modes are
  already unified to one; every existing window is kept.
- **No unrelated backend change** — the only backend edits are the ones a UI
  feature genuinely needs (e.g. exposing an existing method).

## 6. Phase 5 — what was closed (measured)

Each gap below was closed with a real surface, a test, and a row in
`UI_COVERAGE_MATRIX.md`. All changes are additive; no existing control moved.

| Gap | Closed by | Surface | Tests |
|-----|-----------|---------|-------|
| `core.agents` had no GUI | `gui/agents.py` (Tk-free `AgentsService`) + `gui/agents_dialog.py` | the **Personalities (agents)** window (list, capabilities, route, activity, isolation) | `tests/test_agents_service.py`, `tests/test_agents_window.py` |
| No backup/restore control | `gui/maintenance.py` (`MaintenanceService`) + `cli/maintenance.py` | the **Maintenance** window + `python -m cli maintenance backup/restore` | `tests/test_maintenance.py`, `tests/test_maintenance_window.py` |
| No system diagnostics page | same service | the **Maintenance** window's *Diagnostics* section + `python -m cli maintenance diagnostics` | same |
| No keyboard-shortcut help | `gui/maintenance.py`'s `SHORTCUTS` | the **Maintenance** window's *Keyboard shortcuts* section + `python -m cli maintenance shortcuts` | same |
| No memory inspection | `gui/maintenance.py`'s `memory_lines`/`memory_summary` (over `core.character_memory_view`) | the **Maintenance** window's *Memory & Knowledge* section + `python -m cli maintenance memory` | `tests/test_maintenance.py` |
| No knowledge-base search surface | `gui/maintenance.py`'s `knowledge_search` (over `core.knowledge_index`) | the **Maintenance** window's *Memory & Knowledge* section + `python -m cli maintenance knowledge` | same |
| No notification centre | `gui/notifications.py` (`NotificationFeed`) | the main window's Home dashboard reads the feed; background outcomes are recorded | `tests/test_notifications.py` |
| No model-fit preflight note where a model is chosen | `gui/model_chat.py`'s `model_fit` (over `core.model_files`/`core.model_preflight`) | each per-mouth window's model note | `tests/test_model_chat.py` |

### Still open (honest)

- **`perception/`**, **`core.planning`/`core.execution`**, **`core.analogy`**,
  **`core.embeddings`**, and the **evaluation viewer** remain CLI-only. They are
  lower priority (each has a working CLI, and the answer path already reaches the
  analogy and the evaluation metrics); a future phase may surface them.
- **RTL layout switch** — deferred to Phase 6 (localization), where it belongs.

## 7. Verification method for each fix

Every gap closed in Phase 5 gets: a backend call through an existing service
method, a widget, a test in `tests/test_gui*.py`, and a row in
`docs/ui/UI_COVERAGE_MATRIX.md` naming the original location and the new one.
