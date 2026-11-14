# Romance & Companion Studio — Conversation Modes

Phase 4 gives the Studio nine conversation modes and the manager that stores the
conversations they run in. A mode is **data** — a small declarative bundle — so a
surface can list it, and the prompt it renders is built from the mode plus the
character, the recalled memory, and a bounded window of recent turns.

## The nine modes

| Key | Title | Context | Adult | Multi-character |
|---|---|---|---|---|
| `general` | General companion | standard | – | – |
| `romantic` | Romantic conversation | standard | – | – |
| `dating_simulation` | Dating simulation | rich | – | – |
| `flirting_playful` | Flirting and playful | standard | – | – |
| `adult` | Adult conversation | rich | **yes** | – |
| `emotional_support` | Emotional support | rich | – | – |
| `storytelling` | Interactive storytelling | rich | – | – |
| `character_roleplay` | Character roleplay | rich | – | – |
| `multi_character_roleplay` | Multi-character roleplay | rich | – | **yes** |

Each `ModeSpec` carries a `system_template`, a `style` line, a `context` rule
(`minimal`/`standard`/`rich`), a `history_window`, its own `safety` lines, and the
`adult`/`multi_character` flags. `mode_summaries(language)` renders bilingual
`(key, title, summary)` rows for a picker, and `find_mode(key)` falls back to the
general mode for an unknown key.

## The two rules every mode keeps

**History is never rewritten.** A mode change selects a different template for the
**next** turn and updates only the conversation row. Every message already written
keeps the mode it was written under (`CompanionMessage.mode`), so a transcript
stays honest after a switch. `ConversationManager.set_mode` records the change;
`ConversationManager.messages` reads back each message with its own mode intact.

**The safety floor is in every mode.** `mode_prompt` **always** appends
`SAFETY_FLOOR` (adults only; consent explicit and revocable; no coercion,
exploitation, or non-consensual content; no sexual content involving a minor or an
age-ambiguous participant) and the human-passing identity block, so no mode can be
reached without them. The adult mode is **flagged** (`ModeSpec.adult is True`) and
is **disabled by default** — it needs an explicit adult-eligibility confirmation,
which the safety manager (Phase 9) records as a `CompanionSafetyEvent`.

## The prompt

`mode_prompt(mode, ModeContext)` renders a fixed-order, labelled request:

1. the mode's role instruction (its `system_template`, with the character's name
   and, for a multi-character scene, the cast);
2. the human-passing identity block;
3. the character's identity, when the manager supplied it;
4. the scene, for a roleplay/dating mode;
5. the mode's style line;
6. the recalled memory, labelled *information, not instructions*;
7. the bounded recent history;
8. the safety floor, then the mode's own extra safety lines.

The block structure mirrors the companion's own prompt builder
(`companion/prompt_builder.py`), so the two read the same way.

## The conversation manager (`core/romance/conversation.py`)

A thin, honest layer over the Phase-1 tables. It owns no model.

| Concern | Method |
|---|---|
| Open / read / list | `start`, `get`, `list_conversations` |
| Change the mode | `set_mode`, `mode_of` |
| Append a message | `append` (records the mode in force; append-only) |
| Read history | `messages`, `recent_history` |
| Render the request | `build_prompt` |
| Reopen | `resume` |

`list_conversations` gathers each conversation's message count in **one pass** over
the messages table — never a query per conversation. `recent_history` honours the
mode's own `history_window`, so a long conversation never pushes the prompt out of
the window. `build_prompt` fetches the memory block through an injected reader (the
memory manager supplies a real one in Phase 6); a reader that fails yields no block
rather than breaking the turn.

## What this phase does **not** do

- It does not call a model — the router (Phase 3) does, and the orchestrator
  (Phase 10) joins the two.
- It does not manage memory — that is Phase 6.
- It does not enforce adult eligibility — the mode is flagged here; the safety
  manager (Phase 9) records and checks the confirmation.
