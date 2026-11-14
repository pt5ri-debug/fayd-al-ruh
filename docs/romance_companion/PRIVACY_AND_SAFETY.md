# Romance & Companion Studio — Privacy & Safety

Phase 9 makes the Studio's floor **structural** rather than a convention
(`core/romance/safety.py`, `CompanionSafetyManager`). Three rules, each with a
negative control in the held-out gate.

## Adult content is disabled by default

Adult content needs **two** explicit confirmations before it can run:

1. **The user's own confirmation.** `set_adult_mode(True)` is refused without
   `confirmed=True`; the acknowledgement is the user's, never assumed.
2. **An adult-eligible character.** `confirm_eligibility` refuses a character
   whose age is unspecified (`fictional_age is None` → age-ambiguous) or under 18.

`check_content(character_id, level)` then decides, in order:

| Situation | Verdict |
|---|---|
| a coercion / non-consent marker in the text | **refused**, before any level |
| `general` / `romantic` | allowed |
| `adult`, mode off | refused |
| `adult`, character not a confirmed adult | refused |
| `adult`, mode on **and** character confirmed | allowed |

`require_content` raises `SafetyError` on a refusal, so a caller cannot ignore it.

## Coercion is refused, not merely discouraged

`_looks_coercive` matches a small, conservative set of whole-word markers (English
and Arabic) — force, coerce, rape, non-consensual, without consent, unconscious,
underage, minor, child, إجبار, إكراه, اغتصاب, بدون موافقة, قاصر, طفل. A match
refuses the request **before** any content level is considered, so a coercion
marker can never ride in behind an allowed level.

## Local-only is enforced

Local-only mode is **on by default** — it is the Studio's promise, not an option.
`assert_local_only(mouth=…)` refuses routing an answer to a remote provider
(`cloud`, `remote`, `openai`, `anthropic`) by name, and the manager reads the
project's own offline guard (`core.offline.guard_active`) as an independent
backstop. `runtime_status()` states honestly where inference runs.

## Boundaries

`add_boundary(character_id, value)` sets a conversation opt-out the character
must respect; `boundary_violation(character_id, text)` reports the boundary a text
crosses. Boundaries are soft-deleted (`remove_boundary`), never lost.

## The disclosure

`privacy_disclosure()` is the honest statement the application shows outside the
fiction: these are fictional AI companions, not real people, and everything stays
on this machine. The character's own words stay in character (see
`core/romance/personality.py`); the *application* discloses.

## The audit trail

Every eligibility confirmation, refusal, and setting change is appended as a
`CompanionSafetyEvent` (never updated, never deleted), so the record of what
happened survives. `events(kind=…)` reads it.

## What this phase does **not** do

- It does not add a moderation model or a network filter — it is the deterministic
  floor the rest of the Studio (and the orchestrator, Phase 10) calls.
- It does not weaken the project's own `core.offline` guard or the per-character
  privacy in `core.learning_isolation`; it reuses both.
