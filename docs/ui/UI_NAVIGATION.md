# Fayd Al-Ruh — Navigation (Phase 2)

> The main window's sidebar, after the Phase-2 restructuring: a **Home
> dashboard** followed by eight named, collapsible sections. Every section is a
> group *over the existing actions* — no action was removed, none is duplicated,
> and any action no group names is still appended to the last section by
> `gui.actions.grouped_visible_actions`, so a newly added action can never
> silently vanish.

## Layout, top to bottom

1. **Brand** — the project's name and version.
2. **＋ New chat** — the primary action.
3. **Search chats** — filters the saved-conversation list.
4. **Home** (open by default) — the dashboard (below).
5. **Chats** (open by default) — saved conversations, grouped by date.
6. **Reading Voice** — the one place a reading voice is chosen.
7. **Themes** — the eight shipped appearances.
8. **Fonts & paragraphs** — the Word-like font/paragraph settings, per scope.
9. **AI Interaction** · **Cognitive System** · **Personalities** ·
   **Model Management** · **Knowledge & Data** · **Learning & Training** ·
   **Voice & Audio** · **System** — the answering-assistant and tool actions.
10. **Settings** — the Cloud provider/model pickers and the session controls.
11. **Status line** — the session and the shortcuts.

## The Home dashboard

The dashboard is the window's front page. Its rows are read from the *live*
session and the machine by `gui.dashboard` (Tk-free), and it is rebuilt every
time the section is opened (and after a model switch), so it never names a
stale model or shows an out-of-date figure.

| Section | Rows | Source |
| --- | --- | --- |
| **Home** | Model, Session, Mode | `ChatService.source`/`describe_mouth_model`, `session`, `operating_mode` |
| **Recent conversations** | up to 5 saved chats with their date | `ChatService.saved_session_summaries` |
| **Knowledge & learning** | the bulk knowledge-base figure | `gui.dashboard.knowledge_base_summary` → `KnowledgeBaseStore` |
| **This machine** | system, CPU cores, memory, disk free | `core.model_resources.system_resources` |
| **Quick actions** | What the AI can do, Autonomous mind, Training Center, Knowledge Base Manager, Model Manager | the real `gui.actions` catalogue |

**Honest by construction.** A missing figure reads `unknown`, never a fabricated
zero; an empty recent/knowledge/resource group is *dropped*, not drawn blank; and
each quick action resolves against the real action catalogue (an action that left
the catalogue is simply skipped).

## Section keys and titles

| Key | Title | Holds |
| --- | --- | --- |
| `home` | Home | the dashboard (built in the window) |
| `chats` | Chats | saved conversations |
| `voices` | Reading Voice | the voice picker |
| `themes` | Themes | the appearance catalogue |
| `fonts` | Fonts & paragraphs | the type settings |
| `group:assistant` | AI Interaction | `insight`, `ask-teacher` |
| `group:cognitive` | Cognitive System | `cognition` |
| `group:characters` | Personalities | `characters`, `characters-shareable`, the four Natasha entries |
| `group:models` | Model Management | `models`, the three `shared-models` entries |
| `group:knowledge` | Knowledge & Data | `learn`, `internet-learner`, `web`, the knowledge/ingest/import entries |
| `group:training` | Learning & Training | the training entries, `curriculum`, `demo` |
| `group:speech` | Voice & Audio | the Qwen3/Piper/audio.cpp entries |
| `group:system` | System | `deps`, `evaluation`, the self-improvement entries, `selfcheck`, `update` |
| `settings` | Settings | the Cloud pickers and the session controls |

The open/closed state of each section persists per scope through
`gui.sidebar.load_section_states`/`save_section_states`, so a user's layout
survives a restart.

## What did not change

- The **header model selector** and the **composer model row** are exactly as
  they were: the header switches *this* conversation's mouth; the row opens each
  model's own window.
- **Themes**, **Fonts & paragraphs**, and the **Reading Voice** keep their single
  home; the **Settings** section still does not repeat them.
- Every existing action keeps its key, its label, its sidebar label, and its
  backend; only its *group title* and its *neighbourhood* changed.
