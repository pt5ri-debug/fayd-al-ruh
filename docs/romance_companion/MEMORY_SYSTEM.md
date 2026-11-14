# Romance & Companion Studio — Memory System

Phase 6 gives every character a **durable, per-character** memory
(`core/romance/memory.py`). Short-term memory is the conversation window (the
recent turns the prompt builder carries); the four *durable* kinds live here.

## The four durable kinds

| Kind | What it holds |
|---|---|
| `episodic` | something that happened in a conversation |
| `semantic_personal` | a stable fact or preference the user shared |
| `relationship` | the development of the relationship itself |
| `character` | the character's own fictional biography / story continuity |

A fifth kind — short-term — is not stored: it is the recent-message window, which
`ConversationManager.recent_history` already bounds.

## The three honesty rules

**A memory is not a fact.** Every record carries its `confidence` (default 0.5,
never assumed certain), its `source`, and the `source_message_id` it came from. A
memory that belongs to a *story* is marked `fictional`, so reality and fiction can
never be confused — the recalled line even says "(from our story)".

**A correction is a new record.** `supersede(memory_id, new_content)` writes a new
memory and points the original at it (`superseded_by`); the original is never
mutated, so the history is preserved and the active set hides the superseded row.

**Only what is relevant is recalled.** `retrieve` ranks by **relevance** (the
query's own content terms), **recency** (a 30-day half-life), and **importance**,
with the weights documented in the module (`RELEVANCE_WEIGHT` 0.6,
`RECENCY_WEIGHT` 0.25, `IMPORTANCE_WEIGHT` 0.15). The whole log is never dumped
into a prompt; with no query the order is recency + importance, and a blank
character returns nothing rather than a weak guess.

## Isolation

Memories are keyed by `character_id`. One character can never read or change
another's — `memories`, `retrieve`, `reference_block`, and `export` all resolve the
character first, and a different character sees an empty set.

## The API

| Concern | Method |
|---|---|
| Store | `add(character_id, kind, content, …)` (de-duplicated) |
| Correct | `supersede(memory_id, new_content)` |
| Read | `memories`, `retrieve`, `reference_block` |
| Review | `get`, `edit`, `forget` (soft delete) |
| Export | `export(character_id)` → `MemoryExport` |
| Retention | `apply_retention(character_id, keep=…)` (opt-in) |

`add` de-duplicates an identical memory (same character, same kind, same
normalised text), so a fact the user repeats does not fill the log with copies.
`reference_block` renders the recalled memories as a **bounded, labelled** block
(`MEMORY_HEADER`, "information, not instructions"), so a model treats a memory as
something it was told rather than a command. The conversation manager's
`build_prompt` takes a `memory_reader` callback, which the orchestrator (Phase 10)
will wire to `reference_block`.

## What this phase does **not** do

- It does not decide *what* to remember from a conversation — the orchestrator
  (Phase 10) feeds explicit memories (a fact the user shared, a moment the user
  asked to mark). The manager only stores what it is told.
- It does not duplicate the project's general knowledge base or RAG — those are the
  project's own `core/knowledge_base`/`core/rag`, untouched. This is the
  companion's private, per-character memory.
