# Romance & Companion Studio — Test Report

Every figure below is **measured** on the real code, not asserted.

## The Studio's own suite

`tests/test_romance_*.py` — **242 tests**, all passing:

| File | Tests | Covers |
|---|---|---|
| `test_romance_storage.py` | 29 | the SQLite service, schema derivation, versioning, transactions, migrations, backup/restore, retention |
| `test_romance_characters.py` | 26 | create/edit/duplicate/export/import/archive/restore/delete, personality, boundaries |
| `test_romance_modes.py` | 25 | the nine modes, prompts, context rules, history continuity across a mode change |
| `test_romance_safety.py` | 21 | adult-mode default-off, the age floor, hard limits, boundary enforcement, local-only |
| `test_romance_memory.py` | 20 | the five memory kinds, scoring, dedup, isolation, review/edit/delete/export |
| `test_romance_relationship.py` | 20 | the seven dimensions, evidence-driven change, stages, events, history |
| `test_romance_routing.py` | 19 | model discovery/selection/assignment, load/unload, settings, generation |
| `test_romance_studio.py` | 18 | the service the window and CLI share, the CLI, and the end-to-end workflow |
| `test_romance_roleplay.py` | 17 | scenarios, sessions, casts, pause/resume/branch |
| `test_romance_voice.py` | 16 | the voice adapter, per-character voice, engine/language filtering |
| `test_romance_orchestrator.py` | 15 | one turn, ordering, refusal, multi-character turns, resume |
| `test_romance_model_eval.py` | 9 | the model-evaluation facility |
| `test_romance_window.py` | 7 | the real Tk window (skipped with no display; runs under `xvfb` in CI) |

Run them with:

```bash
python -m pytest tests/test_romance_*.py -q
```

## The end-to-end workflow

`test_end_to_end_workflow_across_a_restart` runs the brief's integration workflow
against a **real on-disk database**: create an adult character → start a romantic
conversation → send a turn → store an important memory → close → reopen → resume
the conversation with its full history → confirm personality and memory
continuity → confirm the relationship state persisted → confirm local-only is
enforced. It is the "reopen and continue" proof, not a mock.

## The held-out gate

`core/romance/gate.py`'s `companion_studio_accuracy` runs **74 judgements** on the
real code, including **negative controls**:

- the schema is derived from the record contracts; a newer schema is refused; a
  bad backup is refused with the good database preserved;
- retention honours the caller's choice; memory is isolated per character;
- the safety floor fails closed on a broken check and never crashes a turn;
- the orchestrator keeps each speaker distinct in a multi-character turn;
- **Discover writes nothing** (a negative control) and adding one example adds
  only that one and never duplicates it.

It is **1.0000**, wired into `evaluation` as the metric `companion_studio_accuracy`.

## Regression: the whole project

- `python -m pytest -q` — **7763 passed, 484 skipped, 0 failed** (in ~44 min on
  the target machine; the skips are the window tests with no display locally and
  the optional-extra tests, both of which run under `xvfb`/CI).
- `python -m ruff check .` — clean.
- `python -m mypy` — **Success: no issues found in 1184 source files**.
- `python -m evaluation` — **PASS**, `overall` **0.9918** (unchanged; no
  regression), with `companion_studio_accuracy` 1.0000 and 18 evaluation
  scenarios passing, 0 failing.

No existing test was weakened to make the Studio pass.

## Local-only / safety tests

The safety suite (`test_romance_safety.py`) proves: adult mode is off by default;
turning it on needs an explicit confirmation; a character with no stated age or
an age under 18 is refused; a hard-limit term is refused even with adult mode on;
a per-character boundary is enforced; and local-only mode refuses a remote
provider.
