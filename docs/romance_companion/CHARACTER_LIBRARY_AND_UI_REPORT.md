# Romance & Companion Studio — Character Library & Window Redesign

This pass adds a large, ready-made **character library** to the Studio and gives
the window a professional redesign around it. It is additive: the existing
character system, conversation, relationship, memory, voice, and safety layers are
unchanged, and the earlier **Discover Characters** examples remain available.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/library.py` (new) | `LIBRARY_CATEGORIES` (7 relationship archetypes) and `LIBRARY_CHARACTERS` (52 fully-specified adults), plus `library_drafts`, `library_by_category`, `find_library_character`, `category_keys`, `find_category`, and `seed_library` |
| `core/romance/seed.py` (edit) | Re-exports the library so the seed layer and the library are one surface |
| `gui/romance.py` (edit) | `LibraryRow`; `library_categories`, `library_rows(category=, search=)`, `add_library_character`, `seed_library_category` |
| `cli/romance.py` (edit) | `library-categories`, `library [--category] [--search]`, `add-library NAME`, `seed-library [--category]` |
| `gui/romance_dialog.py` (edit) | Hero header with a live stat strip, an avatar banner, a **Character Library** browser (archetype picker + search + preview), chat-style message bubbles, a composer card, and a relationship closeness bar |
| `core/romance/gate.py` (edit) | 2 new judgements (76 total), still 1.0 |
| `tests/test_romance_library.py` (new), `tests/test_romance_window.py` (edit) | 21 library/service/CLI tests and 3 new window tests |

## The library

The library is **data**, grouped by **relationship archetype**
(`LIBRARY_CATEGORIES`): *Family* (12), *Neighbours* (6), *Colleagues* (8),
*Romantic partners* (8), *Courtesans & companions* (8), *Male escorts* (4), and
*Inclusive & LGBTQ+* (6). Every character is a complete `LibraryCharacter` — a
name, a nickname, a stated adult age (22–74), a gender (26 women, 20 men, 3
non-binary, 2 trans women, 1 trans man), a biography, a nationality, languages, an
appearance, a relationship style, a default conversation mode, personality dials,
interests, boundaries, and tags.

Three honesty rules:

- **Adults only, at the source.** Every character states a real fictional age of
  18 or more, so none is age-ambiguous and none is refused by the safety layer.
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
- A **right panel** enriched with a relationship closeness bar above the state.

## Gate

`companion_studio_accuracy` gains two judgements:

- `library-is-complete-and-adults-only` — every archetype is present and every
  character is a fully-specified adult.
- `negative-library-reads-and-adds-one` — browsing the library writes nothing, and
  adding one adds exactly one.
