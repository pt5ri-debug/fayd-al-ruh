# Romance & Companion Studio — Phase 9 Report (Privacy & Safety)

Phase 9 makes the Studio's floor **structural**. It is additive: one new module
(`core/romance/safety.py`), two small helpers on the storage service, the package
exports, the gate's Phase-9 judgements, tests, and docs.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/safety.py` | `CompanionSafetyManager` — adult gating, boundaries, local-only, refusal events, the disclosure |
| `core/romance/storage.py` (edit) | `get_setting`/`set_setting` (the settings store the manager reads and writes) |
| `tests/test_romance_safety.py` | 21 tests (adult mode, eligibility, coercion, boundaries, local-only, events, disclosure) |
| `core/romance/gate.py` (edit) | 4 new judgements → 67 total, still 1.0 |

## The rules, and why they are structural

**Adult content is off by default, and needs two confirmations.** `set_adult_mode`
refuses without `confirmed=True` (the *user's* acknowledgement), and
`check_content(…, "adult")` refuses unless the character is a **confirmed adult** —
`confirm_eligibility` refuses an unspecified age (age-ambiguous) or one under 18.
A character whose `fictional_age` is `None` can therefore never reach adult
content, which is the brief's "explicitly adults, at least 18" made an invariant.

**Coercion is refused before any level is considered.** `_looks_coercive` matches a
small, conservative whole-word set (English + Arabic); a match returns a refusal
*before* the level switch, so a coercion marker can never ride in behind an
allowed level. The gate's `negative-safety-coercion-refused` judgement turns adult
mode **on** and confirms eligibility, then proves the coercive text is still
refused — the negative control that a naive "mode on ⇒ allowed" implementation
fails.

**Local-only is on by default.** `assert_local_only(mouth=…)` refuses a remote
provider by name (`cloud`, `remote`, `openai`, `anthropic`) and reads the
project's own `core.offline.guard_active` as an independent backstop;
`runtime_status()` states where inference runs. This is the brief's §14: a
conversation must never fall back to a remote provider.

**Boundaries are opt-outs the character must respect.** `add_boundary` sets one,
`boundary_violation` reports the boundary a text crosses, `remove_boundary`
soft-deletes it (never lost).

**Every act is audited.** A confirmation, a refusal, a setting change, or a
local-only block is appended as a `CompanionSafetyEvent` (append-only; the
repository has no update or delete for the table), so the record survives.

## The disclosure

`privacy_disclosure()` is the honest statement the *application* shows: these are
fictional AI companions, not real people, and everything stays on this machine.
The character's own words stay in character (`human_passing_identity`, Phase 2);
the disclosure is the application's, outside the fiction — exactly the split the
owner's decision (#2) requires.

## What this phase does **not** do

- It adds no moderation model and no network filter: it is the deterministic floor
  the orchestrator (Phase 10) calls before any generation.
- It does not weaken `core.offline` or the per-character privacy in
  `core.learning_isolation`; it reuses both.
- It does not implement database encryption here (deferred; noted in the plan).
