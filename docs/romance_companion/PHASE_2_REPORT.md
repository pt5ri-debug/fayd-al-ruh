# Romance & Companion Studio — Phase 2 Report (Character Studio)

Phase 2 adds the **character library** and the **personality engine** over the
Phase-1 storage. It is additive: two new modules (`core/romance/personality.py`,
`core/romance/characters.py`), one seed module (`core/romance/seed.py`), the
package exports, the gate's Phase-2 judgements, and tests. No application
behaviour elsewhere changed.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/personality.py` | `CharacterPersonalityEngine` (dials → tone guidance), `SAFETY_FLOOR`, `human_passing_identity`, validation/clamping |
| `core/romance/characters.py` | `CharacterManager` + `CharacterDraft`/`CharacterBundle`/`CharacterSummary` |
| `core/romance/seed.py` | Three clearly-labelled **example** characters (all adults; no real person depicted) |
| `tests/test_romance_characters.py` | 26 tests (personality, manager, seed, gate) |
| `core/romance/gate.py` (edit) | 12 new judgements → 34 total, still 1.0 |

## The two rules this phase owns

**Human-passing (the owner's decision).** Inside a conversation the character is
**a person** — a life, a past, feelings, and opinions — and speaks with the user
as one person to another. `human_passing_identity()` renders that opening, and it
states plainly that the character must *never* describe itself as a program, an
assistant, a model, or an AI. The **application** still discloses, outside the
fiction, that these are fictional AI companions (that disclosure lives in the
Studio chrome, which arrives with the UI phase); the character's own words stay in
character.

**The floor that never moves.** `SAFETY_FLOOR` is appended to every prompt block
and restates itself in its own words: adults only (18+), consent explicit and
revocable, no coercion or pressure, no exploitation, and nothing sexual involving
a minor or an age-ambiguous participant. The manager enforces the adult rule at
the source: a character may be `adult_eligible` **only** when its fictional age is
a known number ≥ 18, so an unknown age is refused as age-ambiguous.

## Character management

- **Create** — validates the name and the adult rule; writes the character plus
  its personality, profile items, and boundaries in one call.
- **Edit** — `update` edits the character's own fields (re-validating the age rule
  when it changes); `set_personality`/`add_profile_item`/`add_boundary`/
  `set_voice`/`set_model` manage the children.
- **Duplicate** — copies the source's personality, items, boundaries, voice, and
  model assignment onto a new character.
- **Import/Export** — `export_character` writes one portable JSON file;
  `import_character` re-creates a **new** character from it and refuses a foreign
  file rather than half-reading it.
- **Archive/Restore** — `archive` hides a character from the default listing;
  `restore_from_archive` brings it back.
- **Delete** — `delete(character_id, choice=...)` requires an explicit retention
  choice (`keep`/`purge`/`anonymize`) and delegates to the Phase-1 storage service.
- **Library** — `list_characters` returns `CharacterSummary` rows with each
  character's **real** conversation and memory counts, gathered in one pass per
  supporting table (never a query per character), with owner/archived/favorite/
  search filters.

## Personality configuration

Ten bounded `0..1` dials (introversion, playfulness, warmth, expressiveness,
confidence, humour, initiative, affection, romantic intensity, formality) plus a
response length, an Arabic dialect preference, free traits, and free
instructions. `CharacterPersonalityEngine.render` turns them into the readable
block a model is given — each dial a *sentence*, the block headed by "tone
guidance — this shapes your voice, never what you may claim". `validate` refuses
an unknown field or answer length and clamps every dial.

## Verification

- `python -m pytest tests/test_romance_characters.py tests/test_romance_storage.py -q`
  — **55 passed** (26 new).
- `python -m ruff check core/romance tests/test_romance_characters.py` — clean.
- `python -m mypy core/romance` — Success, 11 source files.
- Gate: `companion_studio_accuracy()` — **34 judgements, 1.0**.

## Honest limitations / deferred

- The Studio **window** and **CLI** arrive in the UI phase; Phase 2 is the headless
  engine both will drive.
- The personality dials are stored and rendered, but their *effect on a reply*
  needs the model router and the conversation engine (Phases 3–4).
- Voice and model assignment are **recorded** on the character; actually loading a
  model or a voice is Phase 3 (model) and Phase 8 (voice).
- Relationship state and memories are Phase-1 tables the character scopes to;
  their update rules are Phases 5–6.
