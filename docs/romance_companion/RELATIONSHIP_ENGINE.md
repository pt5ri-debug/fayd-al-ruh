# Romance & Companion Studio — Relationship Engine

Phase 5 gives every character an **independent** relationship with the user, scored
on seven transparent dimensions that move only from explicit, bounded events. The
engine (`core/romance/relationship.py`) stores the state and an append-only event
log, and can explain exactly why the relationship is where it is.

## The seven dimensions

| Dimension | What it tracks |
|---|---|
| `familiarity` | how much the two have talked |
| `trust` | how much the character can rely on the user |
| `closeness` | emotional closeness |
| `comfort` | ease of talking together |
| `affection` | warmth and liking |
| `shared_experiences` | things done together |
| `stability` | how steady the relationship is (starts at 0.5) |

Every dimension is bounded `0..1`. There is deliberately **no** dependency,
possessiveness, or jealousy dimension, so the engine can never reward pressure.

## The three rules

**No random change.** `observe(character_id, kind, message_id=…)` records an
explicit event and moves the state by that event's own deltas, each bounded by
`MAX_STEP` (0.12). A dimension can never jump. Reading the state (`snapshot`)
changes nothing.

**Evidence is required.** An event carries the message it came from. A milestone
with no evidence is still recorded — it is a mark the user asked for — but it moves
nothing. The same `(kind, message_id)` observed twice is idempotent: a replayed
message does not double-count.

**Never coercive.** A `conflict` moves closeness, trust, and stability **down**.
Reconciliation moves them back up. There is no mechanism by which the relationship
improves from pressure, and no "love score" that could be gamed.

## Events

`EVENT_EFFECTS` names the events and their default deltas:

`first_conversation`, `introduction`, `shared_interest`, `personal_share`,
`meaningful_conversation`, `shared_experience`, `affection_expressed`, `conflict`,
`reconciliation`, `milestone`.

Each recorded `RelationshipEventRecord` carries the timestamp (`occurred_at`), the
message reference (`message_id`), the conversation, the dimension(s) moved, and the
delta — so the explanation is built from stored evidence, never a reconstruction.

## Stages

`STAGES` maps the overall closeness (the mean of the seven dimensions) onto a
configurable ladder: **stranger → acquaintance → friend → close_friend →
romantic_interest → romantic_partner**. The engine accepts its own `stages` tuple,
so a caller can rename or re-threshold the ladder. `stage_for(dimensions)` returns
the highest threshold the score passes; a stage change is recorded on the state row.

## Explain, reset, edit

- `explain(character_id)` renders *why* the relationship is where it is — the
  events in order, each with the dimension it moved. A fresh relationship says so
  plainly.
- `reset(character_id)` returns the state to its starting point (the event log is
  preserved, so the history is not lost).
- `edit(character_id, **dimensions)` sets a dimension directly — the user's own
  explicit correction. Values are clamped and an unknown dimension is refused.

## Isolation

Every character has its own relationship state and its own event log, keyed by
`character_id` (and `user_id`). One character's relationship can never read or
change another's — the state row is resolved per `(character_id, user_id)`.

## What this phase does **not** do

- It does not read the conversation to *decide* an event — the orchestrator
  (Phase 10) will feed explicit events (a shared interest the user named, a
  conflict) from the conversation. The engine only records what it is told.
- It does not store memories — that is Phase 6.
