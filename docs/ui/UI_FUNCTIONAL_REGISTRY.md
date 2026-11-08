# Fayd Al-Ruh — UI Functional Registry (Phase 0)

> The control-by-control map of the interface: what each control is, where it
> lives, what it is wired to, and whether that wiring is real. Measured on the
> real checkout, not assumed. **A control is only listed as functional when its
> callback resolves to real backend code**, not because it is drawn.

## 1. How to read this registry

Each row records: **ID · window · label · backend · status · duplicates ·
verdict**. "Backend" names the module/function the control calls. Status is one
of:

- **OK** — wired to real backend code, exercised by tests.
- **OK¹** — wired, but no dedicated test asserts its callback.
- **DEAD** — drawn but its command does nothing / raises.
- **GAP** — the backend exists but the control does not.
- **DUP** — the same setting is also edited in another window.

## 2. Main window controls

| ID | Window | Label | Backend | Status |
|----|--------|-------|---------|--------|
| MW-01 | Main / title bar | Model button (Fayd Al-Ruh Model) | `_on_mode_menu` → `_switch_mouth(MOUTH_MODEL)` | OK |
| MW-02 | Main / title bar | Model selector | `_on_source_menu` → `_switch_mouth` | OK |
| MW-03 | Main / title bar | ☾/☀ theme toggle | `_toggle_theme` → `_apply_theme` | OK |
| MW-04 | Main / title bar | – □ ✕ window controls | `gui.window_controls.install_window_controls` | OK |
| MW-05 | Main / sidebar | ☰ sidebar toggle | `_toggle_sidebar` | OK |
| MW-06 | Main / sidebar | ＋ New chat | `_on_new_chat` | OK |
| MW-07 | Main / sidebar | search box | `refresh_history` | OK |
| MW-08 | Main / sidebar | Chats history rows | `_on_resume_session` | OK |
| MW-09 | Main / sidebar | Reading Voice list | `_on_voice_change` → `ChatService.set_voice` | OK |
| MW-10 | Main / sidebar | auto-speak check | `_on_auto_speak_toggle` | OK |
| MW-11 | Main / sidebar | Test the reading voice | `_test_voice` | OK |
| MW-12 | Main / sidebar | Themes rows | `_apply_theme` | OK |
| MW-13 | Main / sidebar | Fonts & paragraphs (3 scopes) | `_set_font_*`/`_set_alignment`/… → `typography_settings` | OK |
| MW-14 | Main / sidebar | 45 action buttons | `cli.menu.run_*` | OK |
| MW-15 | Main / sidebar | Cloud provider picker | `_on_cloud_provider_change` → `ChatService.set_cloud_provider` | OK |
| MW-16 | Main / sidebar | Cloud model picker | `_set_cloud_model` | OK |
| MW-17 | Main / sidebar | Free providers… | `_show_free_cloud_providers` | OK |
| MW-18 | Main / sidebar | Save conversation / Clear transcript | `_on_save` / `_clear_transcript` | OK |
| MW-19 | Main / composer | input + Send | `_on_send` → `ChatService.ask` | OK |
| MW-20 | Main / composer | ＋ attach menu | `_on_attach_clicked` | OK |
| MW-21 | Main / composer | mic | `_toggle_mic` | OK |
| MW-22 | Main / composer | web toggle | `_on_web_toggled` → `ChatService.set_web` | OK |
| MW-23 | Main / composer | Code Studio / Training / Chat pills | `_open_code_studio` / `_open_training_center` / `_switch_mouth` | OK |
| MW-24 | Main / model row | 7 model chips | `_open_model_chat` | OK |
| MW-25 | Main / model row | 4 voice-engine chips | `_open_voice_studio` | OK |
| MW-26 | Main / transcript | Copy / Read aloud (per message) | `_copy_label` / `_speak_label` | OK |
| MW-27 | Main / transcript | right-click menu | `_transcript_menu_extras` | OK |
| MW-28 | Main / transcript | Ctrl+N/S, Ctrl+C/A | bindings | OK |
| MW-29 | Main / status | status line | `_status_text` | OK |
| MW-30 | Main / confirm bar | Run / Cancel (tool approval) | `_resolve_confirmation` | OK |

## 3. Dialog controls (representative; the full set is per-window)

Every dialog follows the same four-part shape — **header** (title, subtitle,
spinner, window controls), **SidebarColumn** of collapsible sections,
**conversation/transcript**, **composer** — so the controls below are grouped by
that shape rather than repeated per window.

| ID | Window | Label | Backend | Status |
|----|--------|-------|---------|--------|
| DL-01 | all dialogs | – □ ✕ controls | `install_window_controls` | OK |
| DL-02 | all dialogs | section headers | `SidebarColumn.add` / `SidebarSection.toggle` | OK |
| DL-03 | all dialogs | composer Send | per-service `ask`/`run_*` | OK |
| CH-01 | Characters | character list | `CharacterManagerService.summaries` | OK |
| CH-02 | Characters | pages A–F fields | `CharacterManagerService.apply` | OK |
| CH-03 | Characters | APPLY | `apply` (validates, snapshots, rolls back) | OK |
| CH-04 | Characters | What this character remembers | `remember_fact`/`facts`/`forget_facts` | OK |
| CH-05 | Characters | Knowledge corpus page | `core.character_config.KnowledgeConfig` | OK |
| CS-01 | Code Studio | 8 sections (workspaces…apply) | `CodeStudioService` | OK |
| CS-02 | Code Studio | Run / Apply to project | `run_change` / `merge` (gated) | OK |
| CS-03 | Code Studio | file editor panel | `FileEngine` via service | OK |
| CG-01 | Cognition | 12 sections, catalogue-driven | `CognitiveService` | OK |
| CG-02 | Cognition | governed buttons (confirm) | `catalogue` `governed` + `--approve` | OK |
| DS-01 | Doc's Studio | open/read/save/ask | `DocsStudioService` | OK |
| DS-02 | Doc's Studio | page preview | `gui.image_preview.load_scaled` | OK |
| IG-01 | Image gen | prompt/backend/Generate/Save | `ImageGenerationService` | OK |
| IE-01 | Image edit | choose/instruction/Apply/Undo/Save | `ImageEditService` | OK |
| IC-01 | Import Center | files/folder/jobs/OCR | `KnowledgeImporter` | OK |
| KB-01 | KB Manager | character/files/web/board/paste | `KnowledgeBaseManagerService` | OK |
| KB-02 | KB Manager | web approval switch | `set_web_approval` (governed) | OK |
| MC-01 | Model chat | identity/voice/model | `ModelChatService` | OK |
| MC-02 | Model chat | Add/Download/Forget model | `add_model`/`download_model`/`remove_model` | OK |
| MM-01 | Model Manager | 9 sections incl. search/diagnostics | `ModelManagerService` | OK |
| MM-02 | Model Manager | Cancel (cooperative) | `_cancel` (drops result, honest) | OK |
| PS-01 | Presentation | 8-step workflow | `PresentationService` | OK |
| PS-02 | Presentation | Approve all / Approve content | `approve_slides*` | OK |
| PS-03 | Presentation | Command box | `apply_slide_command` | OK |
| PC-01 | Course content | Use missing concepts | hands to Studio composer | OK |
| PF-01 | Final editor | edit/reorder/re-approve/export | `PresentationService` | OK |
| SC-01 | Scene Studio | scene/figures/props/poses | `SceneStudioService` | OK |
| SC-02 | Scene Studio | render/preview/wireframe | `render` / `render_to_bytes` | OK |
| SQ-01 | SQL import | 4 numbered sections | `SqlImportService` | OK |
| SQ-02 | SQL import | Cancel | `request_cancel` marker | OK |
| TC-01 | Training Center | 14 sections, catalogue-driven | `TrainingCenterService` | OK |
| TC-02 | Training Center | streaming job progress | `_start_streamed_job` | OK |
| VS-01 | Voice studio | engine/voice/mood/document | `VoiceStudioService` | OK |
| VS-02 | Voice studio | Engine setup (install/download) | `install_runtime`/`download_voices` | OK |
| VS-03 | Voice studio | My voice (clone/import) | `core.audiocpp_speech` import/clone | OK |
| SH-01 | Main window · Settings | report every global setting | `gui.settings_hub.current_values` | OK (Phase 3) |
| SH-02 | Main window · Settings | the one internet switch | `gui.settings_hub.write_internet` | OK (Phase 3) |
| SH-03 | Main window · Settings | link to a setting's own editor | `_settings_link_command` | OK (Phase 3) |

## 4. Findings

- **No dead buttons were found.** Every drawn control resolves to a real
  callback; the 59 sidebar actions all map to a `cli.menu.run_*` function; the
  dialogs drive their Tk-free service. There is **no placeholder dialog in
  production** and no `command=lambda: None`.
- **The `_button` factory is duplicated 17×** with three variants (font
  `2xs`/`xs`, padding, `anchor`). Behaviour is *nearly* identical but not
  identical, so a shared component is warranted (§Phase 1).
- **`tk.Label` is used 176×** with no shared label factory, so typography and
  colour drift is possible window to window.
- **13 `except …: pass` blocks** in `gui/` silently swallow an error. Most are
  legitimate Tk-lifecycle guards (`tk.TclError` on a destroyed widget), but four
  swallow a broad `Exception` (`model_chat_dialog.py:1042`, `voice_studio_dialog.py:1291`,
  `typography_settings.py:330`, `window_controls.py:114/140/172`) — these should
  route through `core.best_effort.best_effort` so a real bug leaves a trace
  (Phase 6).
- **Per-window state is stored under the `gui` namespace** with several keys and
  several `scope` variants; there is no single settings page (§Phase 3).
  **Phase 3 closed this**: `gui/settings_hub.py` is the registry of every global
  setting and the main window's *Settings* section is the single page that reports
  them; the one genuinely duplicated preference (the internet switch, which the
  Training Center kept a private copy of) now reads/writes `core.preferences`.
