# Romance & Companion Studio — Character Library & Window Redesign

This pass adds a large, ready-made **character library** to the Studio and gives
the window a professional redesign around it. It is additive: the existing
character system, conversation, relationship, memory, voice, and safety layers are
unchanged, and the earlier **Discover Characters** examples remain available.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/library.py` (new) | `LIBRARY_CATEGORIES` (8 relationship archetypes) and `LIBRARY_CHARACTERS` (54 fully-specified adults), plus `library_drafts`, `library_by_category`, `find_library_character`, `category_keys`, `find_category`, and `seed_library` |
| `core/romance/seed.py` (edit) | Re-exports the library so the seed layer and the library are one surface |
| `gui/romance.py` (edit) | `LibraryRow`; `library_categories`, `library_rows(category=, search=)`, `add_library_character`, `seed_library_category` |
| `cli/romance.py` (edit) | `library-categories`, `library [--category] [--search]`, `add-library NAME`, `seed-library [--category]` |
| `gui/romance_dialog.py` (edit) | Hero header with a live stat strip, an avatar banner, a **Character Library** browser (archetype picker + search + preview), chat-style message bubbles, a composer card, and a relationship closeness bar |
| `core/romance/character_files.py` (new) | `CharacterFiles` — a companion's own private folder (`profile.json`, `knowledge_base/`, `learning.json`): `read_profile`, `write_profile_from_bundle`, `add_knowledge`, `reference_block`, `add_fact`/`forget_fact`/`facts`, `remember`/`recall` |
| `core/romance/gate.py` (edit) | 4 new judgements (78 total), still 1.0 |
| `tests/test_romance_library.py` (new), `tests/test_romance_companion_files.py` (new), `tests/test_romance_window.py` (edit) | 21 library/service/CLI tests, 24 own-files tests, 3 new window tests (library) and 2 more (own files) |

## The library

The library is **data**, grouped by **relationship archetype**
(`LIBRARY_CATEGORIES`): *Parents* (2), *Family* (12), *Neighbours* (6),
*Colleagues* (8), *Romantic partners* (8), *Courtesans & companions* (8), *Male
escorts* (4), and *Inclusive & LGBTQ+* (6). Every character is a complete
`LibraryCharacter` — a name, a nickname, a stated adult age (22–74), a gender (27
women, 21 men, 3 non-binary, 2 trans women, 1 trans man), a biography, a
nationality, languages, an appearance, a relationship style, a **kinship** (who the
character is to the user, e.g. *mother*/*father*/*partner*/*companion*), a default
conversation mode, personality dials (including its own **romantic reach**),
interests, boundaries, and tags.

Three honesty rules:

- **Adults only, at the source.** Every character states a real fictional age of
  18 or more and is `adult_eligible=True`, so none is age-ambiguous and none is
  refused by the safety layer — while the adult *mode* itself stays off by default.
- **Nothing is created behind the user's back.** Reading the library writes
  nothing; `seed_library` creates a character only when a caller asks — one at a
  time, or one whole archetype at a time.
- **Fiction, and fictional people.** Every character is an invented person; no real
  individual is depicted or imitated.

## The window

- A **hero header** built through the shared `DialogHeader` (its `extra` slot
  carries the live stat strip: companions, chats, memories, scenarios) plus the
  primary **New companion…**, **Settings…**, and **Right panel** controls.
- A **Character Library** sidebar section: an archetype picker, a live search, the
  filtered list, a preview pane, and **Add the chosen** / **Add the whole group**.
- A **centre banner** with the character's circular avatar (its initial), name, and
  a summary subtitle, over the transcript.
- A **chat-style transcript**: user turns are right-aligned bubbles, companion
  turns are left-aligned rows, and the honesty disclosure stays at the top,
  outside the fiction.
- A **composer card** with a primary **Send** disc.
- A **right panel** enriched with a relationship closeness bar above the state,
  and a **Her Own Knowledge & Memory** block that reports the companion's private
  folder (its profile, knowledge bank, facts, and remembered exchanges) and offers
  *Add knowledge…*, *Tell a fact…*, *What she remembers…*, and *Open the folder…*.

## Gate

`companion_studio_accuracy` gains four judgements:

- `library-is-complete-and-adults-only` — every archetype is present and every
  character is a fully-specified adult.
- `negative-library-reads-and-adds-one` — browsing the library writes nothing, and
  adding one adds exactly one.
- `own-files-round-trip` — a companion's own profile, knowledge bank, and durable
  facts round-trip and reach the block its prompt is built from.
- `negative-own-files-isolated` — one companion can never read another's files.
