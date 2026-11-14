# Romance & Companion Studio — Model Integration

Phase 3 routes the Studio's conversations to a **local** language model through the
project's existing Model Manager. It owns **no** downloader, registry, or runtime
of its own — the router is a thin delegation, and it never opens a second path to
the network.

## The router (`core/romance/routing.py`)

`CompanionModelRouter` resolves a *choice* (a character's or a conversation mode's
assigned model) to the exact file the manager already knows, and drives the load,
unload, and test operations through that one manager.

| Concern | Method | Honesty rule |
|---|---|---|
| Which models exist | `choices()` | every entry carries its real `available` flag and next step |
| Resolve a key | `resolve(key)` | an unknown/stale key falls back to the first *available* model, never a crash |
| Can it answer now | `status()` / `health()` | `loaded`/`ready` read from the manager's own state, never assumed |
| Load / unload | `load()` / `unload()` | a failed load is returned as an `OperationResult`, not raised |
| Test it answers | `test()` | runs the manager's own tiny probe |
| Memory | `memory_estimate()` / `preflight()` | an *estimate* with its uncertainty and unknowns stated |
| Machine figures | `resources()` | each figure `None` when the platform cannot report it |
| Rank / fallback | `rank()` / `fallback_plan()` | the manager's explainable ranking, against the `companion` task |
| Download | `download_hint()` | the governed command as a **string**; the router never runs it |
| Cancel / timeout | `begin_generation()` / `should_stop()` | a handle that can be cancelled and that expires |

### Assignments

`assign_character(character_id, key)` and `assign_mode(mode, key)` persist the
choice through an injected writer, so the character table (`character_model_profiles`)
and the mode table remain the single record of the assignment; the router only
resolves it.

### The catalog seam

The router reads the models a companion may use from the existing catalog
(`core.language_models.models_for_choice("natasha")`) through a `catalog` seam — a
callable returning the model objects. A test injects a fake catalog, so the whole
router is exercised with no weights. The router imports no runtime directly.

## Safe defaults for this hardware

`SAFE_CONTEXT` (4096) and `SAFE_MAX_TOKENS` (512) are the shipped defaults, and
`gpu_layers` stays `0` unless a caller raises it explicitly. That is the honest
default for a CPU-first machine: a large enough window for a real conversation,
small enough that the KV cache does not crowd out the weights.

## The evaluation facility (`core/romance/model_eval.py`)

`ModelEvaluator` runs a small, fixed battery of probes through the resolved model
and reports what it actually did. The generation seam (`runner`) is injectable, so
the facility runs with no model.

Dimensions measured (see `DIMENSIONS`):

- `arabic_quality` — a probe written in Arabic answers.
- `english_quality` — a probe written in English answers.
- `personality_consistency` — a warm/playful instruction produces a fitting line.
- `emotional_coherence` — a fitting reply to a hard day.
- `long_context` — a short-context instruction is honoured.
- `roleplay_continuity` — an in-character greeting.
- `latency` — mean seconds per reply (lower is better).
- `memory` — free system memory before the run (measured when the platform reports it).
- `stability` — the share of probes that completed without error.

**Two honesty rules.** Every figure is *measured on this machine*, and when no
model is installed the report says `not measured` and names the real next step —
it never prints a fabricated score. `compare(keys)` renders a comparative report
and says plainly when nothing could be measured. VRAM is not reported per model by
the local runtime, and the report says so rather than inventing a number.

## What this phase does **not** do

- It does not download models (the existing governed engines do, behind their own
  permission and human-approval gates).
- It does not implement a second registry or runtime.
- It does not assume a model is installed — every surface reports the real next step.
