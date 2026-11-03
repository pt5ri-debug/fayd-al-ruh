# Phase 5 — Isolation, Capabilities, and the Natasha Guarantee

**Version:** 2.3.0 · **Date:** 2026-11-02

This document states exactly what a personality may and may not do in the Phase-5
multi-agent architecture, and where each rule is enforced. It is the reference for
the hard constraint of the Phase-5 mission:

> **Natasha's private learning and records must never leak into aggregated
> training datasets.**

## 1. The two vocabularies, and one identity

The chat speaks **mouth** keys (`sources`, `model`); the Characters window, the
CLI, and the per-character learning store speak **owner** keys (`jarvis`,
`fayd_al_ruh`). Phase 5 never invents a third: `core.learning_isolation.owner_for`
and `core.agents.registry.PersonalityRegistry.resolve` map **either** vocabulary
to **one** `PersonalityProfile`. An unknown key resolves to the project's own
model — never to a made-up personality — so a stale saved choice is harmless.

| Mouth key | Owner key | Name |
| --- | --- | --- |
| `sources` | `jarvis` | Jarvis |
| `ultron` | `ultron` | Ultron |
| `vision` | `vision` | Vision |
| `banner` | `banner` | Banner |
| `natasha` | `natasha` | Natasha |
| `hakeem` | `hakeem` | Hakeem |
| `cloud` | `cloud` | Cloud |
| `model` | `fayd_al_ruh` | Fayd Al-Ruh (the project model) |

## 2. Capabilities

A capability is a closed token (`AgentCapability`): `answer`, `retrieve`,
`reason`, `run_tools`, `manage_goals`, `receive_delegation`,
`contribute_training`, `search`, `write_files`, `read_all_memory`.

The rule is a **ceiling**, not a grant:

```python
effective_capabilities(owner, requested=...) == requested & capability_ceiling(owner)
```

- A personality can never hold a capability its ceiling omits, **no matter what a
  configuration asks for** — the intersection is applied on every read.
- `SEARCH` and `WRITE_FILES` are in the vocabulary but in **no** shipped ceiling;
  they are refused by default and exist only so a future phase can grant them
  deliberately.
- `CONTRIBUTE_TRAINING` is absent from **Natasha's** ceiling. She cannot
  contribute to any dataset even if a caller asks her to.

`CapabilityGateway` decides and **audits** every request
(`gateway.decisions`, `gateway.refusals()`); `gateway.require(...)` raises
`CapabilityDenied` on the strict path.

## 3. Memory access

`PersonalityMemoryAccess` is the only reader of a personality's memory from the
agent layer. Its guard runs **before any directory is resolved**:

| Reader | May read | May **not** read |
| --- | --- | --- |
| a character | its **own** memory | any other character's memory |
| the project's model | every **shareable** character's memory | **Natasha's** memory |

A refused read raises `LearningAccessError`. There is no method that takes an
arbitrary owner and returns another's records.

## 4. Learning scope

`core/agents/learning_scope.py` answers three questions:

1. **What may a personality learn from?** Its own records, plus (for the project's
   model) the shareable characters' records.
2. **Who may contribute to the aggregated training dataset?**
   `aggregated_dataset_owners()` — the project's model **and the six shareable
   characters**.
3. **Who is excluded?** Natasha. `aggregated_dataset_is_clean()` returns `True`
   exactly when no private owner is present.

## 5. The Natasha guarantee, at three levels

| Level | Where | What it guarantees |
| --- | --- | --- |
| **Ceiling** | `core/agents/capabilities.py` | `decide_capability("natasha", CONTRIBUTE_TRAINING).allowed is False` |
| **Scope** | `core/agents/learning_scope.py` | `"natasha" not in aggregated_dataset_owners()` |
| **Corpus** | `core/character_learning.py` | `training_keys()` returns six owners; the corpus functions default to it |

The training corpus (`ml.llm.corpus`) reads the character records only through
`character_learning_sentences`/`character_learning_units`, both of which default
to `training_keys()`. A test writes a private fact to Natasha and asserts it does
**not** appear in:

- `character_learning_sentences(root=...)`
- `character_learning_units(root=...)`
- `ml.llm.corpus.build_corpus(include_characters=True)`

## 6. Delegation and collaboration

- A delegation must target a **character** (never the project's model).
- Delegation **depth** is bounded (default 2) and the **budget** per
  collaboration is bounded (default 4). Exceeding either raises
  `DelegationRefused`.
- A delegate receives only the caller's **payload** — never the caller's private
  memory, which the memory access layer would refuse anyway.

## 7. Conflict resolution

When two personalities disagree, the default policy is **preserve**: the
disagreement is reported, every position is kept, and **no winner is named**. The
opt-in `evidence` policy selects a position only when it is **clearly stronger**
than every other (strictly more evidence, and not a near-tie); otherwise it, too,
names no winner. A personality is never silently overruled.

## 8. What this document is not

It does not describe a *policy flag* that a caller can flip. Every rule above is a
**structural** check — a ceiling intersection, a pre-path guard, or a fixed owner
list — so it holds no matter which surface (CLI, GUI, test) calls the layer.
