# Romance & Companion Studio — Phase 8 Report (Voice Integration)

Phase 8 gives every character a voice by **reusing** the project's existing speech
engines, never by writing a second TTS/STT subsystem. It is additive: one new
module (`core/romance/voice.py`), the package exports, the gate's Phase-8
judgements, tests, and docs. One latent Phase-7 bug was fixed in passing.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/voice.py` | `CompanionVoiceAdapter` — per-character voice settings, TTS, STT, playback/stop |
| `tests/test_romance_voice.py` | 16 tests (adapter, profiles, engine validation, fallbacks) |
| `core/romance/gate.py` (edit) | 4 new judgements → 63 total, still 1.0 |
| `core/romance/roleplay.py` (edit) | **bugfix**: participant order |

## The adapter, not a new engine

`CompanionVoiceAdapter` sits on top of the project's own
`core.speech.TextReader`, `core.voices` (the voice catalog and the real installed
voices), and `core.voice_moods` (the emotional speaking states). It does **not**
synthesize audio itself:

- `voices(character_id)` returns only voices the machine **really has** —
  `core.voices.voice_sections` drops a source whose engine is absent — so a menu
  built from it can never offer a silent voice.
- `set_profile`/`update_profile` validate the engine (`ENGINE_NAMES`) and the
  mood against the real catalogs; an unknown value is a `VoiceAdapterError`, never
  a silent no-op.
- `speak`/`read_aloud` render and play through the project's `TextReader`;
  `transcribe` accepts either a saved `AudioClip`/path or a live recording and runs
  the project's recognizer.
- `stop` interrupts playback.

Everything is **best-effort**: a missing engine (no espeak-ng, no Piper, no
Qwen3-TTS) never breaks a scene — the adapter reports the honest next step and the
turn proceeds without audio. The gate's `voice-never-breaks-a-scene` judgement
pins exactly this (a reader whose engine raises returns `None`, not an exception).

## The bugfix (found by the Phase-8 full run)

The full suite surfaced an **intermittent** failure in a Phase-7 test
(`test_add_participants_in_order_and_read_the_cast`): a scenario's cast came back
in the wrong order about half the time. The cause was a real SQL bug —
`RoleplayEngine.participants` ordered by `"order"`, but the column is stored as
`sort_order` (the schema renames the reserved word `order`), so `ORDER BY "order"`
sorted by a **string literal** and returned rows in arbitrary order. The fix is
one line (`order_by="sort_order, record_id"`), and the test now passes 8/8 in a
tight loop. This is the kind of latent defect a phase-scoped test run can hide and
a full run catches.

## What this phase does **not** do

- It does not add a voice engine, a model download, or a new audio format.
- It does not auto-download a voice: fetching one stays with the existing
  governed engines behind their own approval gates.
- It does not force a voice on a character that has none; the Studio shows the
  real next step instead.
