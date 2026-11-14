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
assigned voice profile — the free-form half of the identity.

### Personality (`character_personalities`)

The dials are **normalised 0..1** values, each editable and each paired with an
editable natural-language instruction:

`introversion`, `playfulness`, `warmth`, `expressiveness`, `confidence`, `humor`,
`initiative`, `affection`, `romantic_intensity`, `response_length`, `formality`,
`dialect` (the Arabic dialect preference).

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

`core/romance/seed.py` ships a small set of **clearly-labelled fictional**
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

## Isolation

A character's memories, conversations, relationship, and voice profile are keyed
to its `character_id`; every read goes through that key. `CompanionMemoryManager`
never returns one character's memories for another, and the gate's
`memory-isolated-per-character` judgement proves it.

## Related documents

- [`RELATIONSHIP_ENGINE.md`](RELATIONSHIP_ENGINE.md)
- [`MEMORY_SYSTEM.md`](MEMORY_SYSTEM.md)
- [`DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md)
- [`PRIVACY_AND_SAFETY.md`](PRIVACY_AND_SAFETY.md)
