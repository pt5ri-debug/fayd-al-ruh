# Romance & Companion Studio — Dating & Roleplay Engine

Phase 7 gives the Studio fictional **scenarios** and running **sessions**
(`core/romance/roleplay.py`): a dating simulation or a roleplay scene that the
user can pause, resume, branch, save, and restart.

## Scenarios

A scenario is a small, declarative bundle:

| Field | Meaning |
|---|---|
| `title` | the scenario's name (required) |
| `category` | one of ten (below) |
| `description` / `background` | the story's own words |
| `location` / `setting_time` | where and when the scene happens |
| `starting_situation` | how the scene opens |
| `objectives` | what the scene is trying to reach |
| `narrative_style` | the register the scene is written in |
| `content_level` | `general` / `romantic` / `adult` (default `general`) |
| `boundaries` | the scene's own limits |
| `is_real_place` | **fictional by default**; set only if the place is real |

### The ten categories

`first_meeting`, `coffee_shop`, `virtual_date`, `romantic_evening`,
`travel_story`, `daily_life`, `friendship`, `emotional_conversation`,
`adventure`, `multi_character`.

## The cast

`add_participant(scenario_id, character_id, role=…)` adds a character to the
scenario's cast with its own role and a stable order. A character **cannot** join
the same scenario twice (the engine refuses, and `cast()` never lists a duplicate),
so the model can never be told two characters are one. `remove_participant`
deactivates a cast member (soft delete).

## Sessions

A session is one running play of a scenario.

| Operation | Effect |
|---|---|
| `start(scenario_id, …)` | open a session; a conversation is opened for it (a scene is its own conversation) |
| `pause` / `resume` / `end` | the session's state |
| `branch(session_id)` | a **new** session that points back at the one it forked from |
| `restart(session_id)` | a fresh session for the same scenario and cast |

**A branch never rewrites the original.** `branch` creates a new
`scenario_sessions` row whose `branch_of` names the parent; the original's
`branch_of` stays empty. A story's history is preserved, so branching is safe.

`start` opens the scene's conversation in the mode the category implies
(`dating_simulation`, or `multi_character_roleplay` for a multi-character story),
so the scene's turns land in a conversation of their own.

## What this phase does **not** do

- It does not generate the scene's prose — that is the orchestrator + the local
  model (Phase 10). This phase is the scenario/session model the orchestrator
  drives.
- It does not decide adult eligibility — the safety manager (Phase 9) gates the
  `adult` content level. The scenario records the level; the gate enforces it.
