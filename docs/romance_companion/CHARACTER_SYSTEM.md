# Romance & Companion Studio — Character System

A **character** is an independent fictional companion with a persistent identity,
a configurable personality, its own boundaries, its own relationship with the
user, and its own memories. No character shares private data with another.

## The record

`core/romance/characters.py`'s `CharacterManager` writes the `companion_characters`
row plus its three child rows (`character_personalities`, `character_profiles`,
`character_boundaries`) in **one transaction**, so a character is never
half-created.

### Identity (`companion_characters`)

| Field | Notes |
|---|---|
| `record_id` | a stable `CHR-…` id |
| `name` / `nickname` | display name and optional nickname |
| `fictional_age` | the character's fictional age |
| `adult_eligible` | set only by an explicit confirmation for an adult character |
| `gender` | where applicable |
| `default_mode` | the conversation mode the character opens in |
| `favorite` / `archived` / `active` | the library state (archive is a soft delete) |
| `created_at` / `updated_at` | UTC |
| `version` | the profile version |

### Profile (`character_profiles`)

Biography, nationality, languages, interests, appearance, relationship style,
communication style, emotional expression, humor, romantic preferences, and the
assigned voice profile — the free-form half of the identity. The identity row also
carries **`kinship`** — the relationship *degree* the character holds toward the
user (mother, father, partner, lover, colleague, neighbour, …) — so a character
addresses the user in a way that fits the relation rather than as a stranger.

### Personality (`character_personalities`)

The dials are **normalised 0..1** values, each editable and each paired with an
editable natural-language instruction:

`introversion`, `playfulness`, `warmth`, `expressiveness`, `confidence`, `humor`,
`initiative`, `affection`, `romantic_intensity`, `response_length`, `formality`,
`dialect` (the Arabic dialect preference).

Four more dials are the character's **romantic reach** — its own seduction reach,
the same four the companion Natasha carries (`REACH_DIALS`):

`boldness`, `passion`, `sensuality`, `escalation`.

They are rendered by a **separate** block (`CharacterPersonalityEngine.reach_block`)
that restates the safety floor in its own text, so a high dial can never be read as
licence to drop it: it governs *how warmly and how directly* the character speaks,
never what it may claim or do. `GUI` exposes all fourteen as sliders under
*Personality* (the ten voice dials) and *Romantic reach* (the four reach dials).

`PersonalityEngine` renders the dials and instructions into a **standing system
prompt** (the character's voice), which the orchestrator prepends to the model
prompt for every turn — so a character answers in its own personality rather than
a generic assistant's.

### Boundaries (`character_boundaries`)

Per-character conversation boundaries (a closed kind + a value): a topic to
avoid, a hard limit, a preference. The safety floor reads them **before** a turn,
so a boundary is enforced, not merely stored.

## Life cycle

`create` · `update` · `duplicate` · `export` · `import` · `archive` · `restore` ·
`delete`. Deleting requires an explicit **retention choice** for the character's
conversations and memories:

- `keep` — keep the conversations and memories (the character row is removed).
- `purge` — delete the conversations and memories too.
- `anonymize` — keep the conversation text with the character reference cleared.

The choice is honoured by `core/romance/characters.py` and pinned by the gate's
`retention-honours-the-choice` judgement.

## Discover Characters

`core/romance/seed.py` still ships a small set of **clearly-labelled fictional**
example companions (`SEED_MARKER = "romance-example"`): three adult drafts
(Layla, Noor, Salma), each with generic interests and no real person's identity.
They are seeded **only on request**:

- `RomanceStudioService.discover_examples()` returns the drafts and whether each
  is already present — **a read that writes nothing**.
- `RomanceStudioService.add_example(name)` creates **only** the chosen draft
  through the same `CharacterManager.create` the seeding path uses, and returns
  the existing character when it is already present (never a duplicate).
- The window's **Discover Characters** block and the CLI's `discover` /
  `add-example NAME` drive the same two methods, so the surfaces cannot disagree.

Two gate judgements pin the honesty: browsing writes nothing
(`negative-discover-writes-nothing`) and adding one adds only that one and never
duplicates it (`discover-adds-only-the-chosen`).

## The ready-made character library

`core/romance/library.py` is the large, **fully-specified** companion library a
new user picks from, grouped by **relationship archetype**
(`LIBRARY_CATEGORIES`): *Parents*, *Family*, *Neighbours*, *Colleagues*, *Romantic
partners*, *Courtesans & companions*, *Male escorts*, and *Inclusive & LGBTQ+*.
Each archetype carries a bilingual title, an icon, and a one-line summary, and
each of the **54** characters is a complete `LibraryCharacter`: a name, a nickname,
a **stated adult age (18+)**, a gender (including transgender and non-binary
identities), a biography, a nationality, languages, an appearance, a relationship
style, a **kinship** (who the character is to the user — the *Parents* archetype
ships a mother, *Umm Nour*, and a father, *Abu Nour*), a default conversation
mode, personality dials (including the four **reach** dials), interests,
boundaries, and tags.

Three honesty rules, the same ones the rest of the Studio keeps:

- **Adults only, at the source.** Every library character states a real fictional
  age of 18 or more, so none is age-ambiguous and none is refused by the safety
  layer. A library character is therefore `adult_eligible` at the source: the user
  does not have to confirm each one before an adult conversation can be reached.
  The adult *mode* itself is still off by default and must be turned on once,
  exactly as before.
- **Nothing is created behind the user's back.** The library is *data*: reading it
  writes nothing, and `seed_library` creates a character only when a caller asks —
  one at a time, or one whole archetype at a time.
- **Fiction, and fictional people.** Every character is an invented person; no
  real individual is depicted or imitated.

Wiring (one path shared by the window and the CLI):

- `RomanceStudioService.library_categories()` — the archetype rows.
- `RomanceStudioService.library_rows(category=, search=)` — the filtered library
  with each character's `present` flag; **a pure read**.
- `RomanceStudioService.add_library_character(name)` — create **only** the chosen
  character (never a duplicate).
- `RomanceStudioService.seed_library_category(category)` — create a whole
  archetype at once.
- CLI: `python -m cli romance library-categories | library [--category K]
  [--search S] | add-library NAME | seed-library [--category K]`.

Two gate judgements pin the honesty: the library is complete and adults-only
(`library-is-complete-and-adults-only`), and browsing writes nothing while adding
one adds exactly one (`negative-library-reads-and-adds-one`).

## A companion's own files

Beyond its rows in the database, every companion owns three things of its own,
kept together under one per-character folder so one companion can never read
another's (`core/romance/character_files.py`, `CharacterFiles`):

- **attributes** — `profile.json`, the character's identity and reach as one
  portable document (name, age, `kinship`, biography, appearance, interests,
  boundaries, and every personality/reach dial). It is a *view* written from the
  database records, never a second source of truth: `write_profile_from_bundle`
  renders it and `read_profile` reads it back, so a character can be exported,
  inspected, or moved by hand. A newer document version is refused.
- **knowledge bank** — `knowledge_base/`, a real
  `core.knowledge_base.KnowledgeBaseStore` (the same sharded JSONL store the
  Import Center and the Knowledge Base Manager fill), indexed with
  `core.knowledge_index.KnowledgeTextIndex`. This is the companion's own reference
  material — the capability Natasha has — read at answer time with
  `reference_block`, and injected into the prompt as a bounded, labelled block
  (*information, not instructions*).
- **learning** — `learning.json`, the durable facts the companion was told and the
  exchanges it took part in, exactly the record
  `core.character_learning.CharacterLearning` keeps for the project's characters.
  `recall()` renders a bounded window of past exchanges into the prompt.

The folder is the companion's own
`data/romance/characters/<character_id>` (`core.paths.romance_characters_dir`,
git-ignored). The service resolves it under the Studio's **own** data root
(`storage.root/characters`), so a custom root — a test, or a portable build —
keeps a companion's private files beside its database and never in a shared
folder. A character id is sanitised to a filesystem-safe folder name, so it can
never escape the root.

`RomanceStudioService` exposes the folder: `character_files(id)`,
`write_character_files(id)` (called when a character is created or updated),
`character_file_summary(id)`, `add_character_knowledge(id, title, text)`,
`character_facts(id)` / `remember_character_fact(id, text)`, and the `_memory_block`
that fuses the three sources into the prompt for every turn. A corrupt learning
file reads as empty rather than crashing a chat, and a failure in any source never
breaks a turn.

## Isolation

A character's memories, conversations, relationship, voice profile, and its own
files are keyed to its `character_id`; every read goes through that key.
`CompanionMemoryManager` never returns one character's memories for another,
`CharacterFiles` keeps each companion's folder apart, and the gate's
`memory-isolated-per-character` judgement proves it.

## Related documents

- [`RELATIONSHIP_ENGINE.md`](RELATIONSHIP_ENGINE.md)
- [`MEMORY_SYSTEM.md`](MEMORY_SYSTEM.md)
- [`DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md)
- [`PRIVACY_AND_SAFETY.md`](PRIVACY_AND_SAFETY.md)
