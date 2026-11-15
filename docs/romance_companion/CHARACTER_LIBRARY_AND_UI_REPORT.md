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
| `gui/romance.py` (edit) | `LibraryRow` (now carries the existing `character_id` and a `chat_ready` property); `library_categories`, `library_rows(category=, search=)`, `add_library_character`, `open_library_character`, `seed_library_category` |
| `cli/romance.py` (edit) | `library-categories`, `library [--category] [--search]`, `add-library NAME`, `open-library NAME`, `seed-library [--category]` |
| `gui/romance_dialog.py` (edit) | Hero header with a live stat strip, an avatar banner, a **Character Library** browser (archetype picker + search + preview), chat-style message bubbles, a composer card, and a relationship closeness bar |
| `core/romance/character_files.py` (new) | `CharacterFiles` — a companion's own private folder (`profile.json`, `knowledge_base/`, `learning.json`): `read_profile`, `write_profile_from_bundle`, `add_knowledge`, `reference_block`, `add_fact`/`forget_fact`/`facts`, `remember`/`recall` |
| `core/romance/gate.py` (edit) | 6 new judgements (80 total), still 1.0 |
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
  **Choosing a companion from the library brings it into the Studio and selects
  it**, so the chat area immediately shows its name and it can be talked to at
  once — exactly like a companion under **My companions**. A companion that is not
  yet in the Studio is created once (never duplicated); one that already exists is
  simply re-selected, so its conversations and memories are preserved. The
  archetype picker, the search, the preview, and the **Add the whole group** action
  are unchanged, and a plain *browse* (moving the filter or typing a search) still
  writes nothing — only an explicit choice does.
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

`companion_studio_accuracy` gains six judgements:

- `library-is-complete-and-adults-only` — every archetype is present and every
  character is a fully-specified adult.
- `negative-library-reads-and-adds-one` — browsing the library writes nothing, and
  adding one adds exactly one.
- `own-files-round-trip` — a companion's own profile, knowledge bank, and durable
  facts round-trip and reach the block its prompt is built from.
- `negative-own-files-isolated` — one companion can never read another's files.
- `library-open-selects-and-adds-once` — opening a ready-made library companion
  makes it the selected one the chat uses, adding it exactly once and never
  duplicating it.
- `negative-library-open-no-duplicate` — re-opening an already-added library
  companion re-selects it without adding a second copy or losing its conversation.

## Fix — choosing a library companion selects it (2026-11-15)

**Reported:** in the Studio window's **Character Library** section, clicking a
ready-made companion did not display the chosen character's name in the chat area,
so the user could not converse with it — unlike a companion under **My
companions**, which works.

**Cause:** the library is a *catalogue of drafts*. Clicking a row only updated the
small **preview** label; it never selected the character into the chat. Selection
required first pressing **Add the chosen**, which the user did not expect to have
to do before the companion appeared.

**Fix:** `RomanceStudioService.open_library_character(name)` brings a library
companion into the Studio and returns it — an already-added companion is returned
as it is (never duplicated), and a not-yet-added one is created once, exactly like
**Add the chosen**. `LibraryRow` now carries the existing `character_id` (and a
`chat_ready` property), so the window knows whether a row is already a companion.
The window's `_on_library_selected` calls it and then selects the result through
the new shared `_select_character` (the same routine **My companions** uses), so
the transcript, panels, mode, and status line can never drift apart. Clicking a
library row always still previews it, so a click is never a no-op. A programmatic
selection guard (`_suppress_select`) keeps a list rebuild from being mistaken for a
user click. The CLI gained the headless twin `python -m cli romance open-library
NAME`. Two window tests (`test_choosing_a_library_companion_selects_it_for_the_chat`,
`test_the_composer_sends_to_a_companion_chosen_from_the_library`), four
service/CLI tests, and the two new gate judgements pin the behaviour.
