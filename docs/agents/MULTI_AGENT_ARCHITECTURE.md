# Phase 5 — Multi-Agent Architecture for the Seven Personalities

**Version:** 2.3.0 · **Branch:** `feature/phase-5-agents` · **Date:** 2026-11-02

## 1. What Phase 5 is (and is not)

Phase 5 turns the project's **seven personalities** — Jarvis, Ultron, Vision,
Banner, Natasha, Hakeem, and Cloud — plus the project's own model
(`fayd_al_ruh`) into **independent cognitive agents** that share **one framework**.

Before this phase a "personality" was a *character*: a persona, a configuration,
a learning record, and a mouth. Each was distinct, but there was no single object
that owned a personality's **identity, capabilities, working context, goals,
memory access, and learning scope together**, and nothing that could **route** a
request to the right personality, **delegate** between them, or **resolve** their
disagreements.

Phase 5 adds that framework. It is deliberately **not**:

- seven separate applications, or seven copies of a chat loop;
- a second knowledge graph, a second memory store, or a second learning pipeline;
- a rewrite of the character system, the mouths, or the isolation policy.

Every module is **additive**. The existing character system
(`core/character_identity.py`, `core/character_config.py`,
`core/character_registry.py`, `core/learning_isolation.py`,
`core/character_learning.py`) is **reused**, not replaced. The agent layer sits
*above* it and composes the pieces that were already there.

## 2. The one design rule

> **Shared system, independent agents.**

The *framework* is shared: one registry, one runtime, one router, one
collaboration bus, one conflict resolver, one monitor, one capability gate, one
memory-access layer, one learning-scope layer. The *agents* are independent:
each has its own identity, its own capabilities (bounded by a ceiling), its own
working context, its own goals and task tree, its own memory, and its own
learning records.

A character can therefore be **reasoned about as an agent** without any of its
private state being readable by another agent.

## 3. The modules

| Module | Kind | Purpose |
| --- | --- | --- |
| `core/agents/registry.py` | **new** | The **personality registry**. One `PersonalityProfile` per owner — identity, role, specialisation, style, model, capabilities, memory scope, learning scope, privacy, configuration version. Resolves either vocabulary (`sources`/`jarvis`, `model`/`fayd_al_ruh`) to **one** profile; an unknown key resolves to the project's own model, never to a made-up personality. |
| `core/agents/capabilities.py` | **new** | The **capability model**. A closed `AgentCapability` vocabulary, a per-owner **hard ceiling** (`capability_ceiling`), an effective set that is always `requested ∩ ceiling`, a `CapabilityGateway` that decides and audits every request, and `CapabilityDenied` for the strict path. |
| `core/agents/memory_access.py` | **new** | The **memory access layer**. A personality may read its **own** memory; the project's model may read a **shareable** character's memory; a character may **never** read another character's memory, and the project's model may **never** read Natasha's. The guard runs **before any path is resolved**. |
| `core/agents/learning_scope.py` | **new** | The **learning scope**. Which owners a personality may learn from, which may **contribute** to the aggregated training dataset, and which are **excluded** (Natasha). `aggregated_dataset_owners()` is the one function the training corpus reads, and `aggregated_dataset_is_clean()` proves no private owner is present. |
| `core/agents/context.py` | **new** | The **independent working context**. One bounded, owner-scoped turn log per personality (`ContextManager.context_for`), never shared between agents. |
| `core/agents/goals.py` | **new** | The **goal and task manager**. Goals and tasks per owner, a parent/child task tree, a depth bound, and the `TaskStatus` lifecycle (`pending → running → succeeded/failed/refused/timed_out/cancelled`). |
| `core/agents/runtime.py` | **new** | The **agent runtime**. Executes one task end to end: build a task, check capabilities, request a resource grant from the governor, run an injected executor, enforce a timeout and a cancellation token, validate the result, record the task, and publish lifecycle events. |
| `core/agents/routing.py` | **new** | The **task router**. Classifies a request into a `TaskType`, scores each suitable personality on **named factors** (declared specialisation, capability coverage, task preference), keeps the chosen personality across a follow-up turn, honours a manual override, and is **honest** when no personality is suitable. |
| `core/agents/collaboration.py` | **new** | **Bounded collaboration**. Delegation between personalities with a **depth bound**, a per-collaboration **budget**, a character-only target rule, and a provenance-carrying result. |
| `core/agents/conflict.py` | **new** | **Conflict resolution**. Compares the positions several personalities took on one question and **preserves** the disagreement by default; an opt-in `evidence` policy selects only a **clearly stronger** case and never decides a near-tie. |
| `core/agents/communication.py` | **new** | The **communication interface**. A bounded, logged message bus between personalities, with broadcast, threads, and recipient validation. |
| `core/agents/monitoring.py` | **new** | **Monitoring and evaluation**. Each personality's measured activity — tasks, successes, failures, refusals, success rate — read from the real goal manager and capability gateway. |
| `core/agents/system.py` | **new** | The **unifying facade**. Composes the registry, runtime, router, collaboration, conflict resolver, monitor, bus, memory-access layer, and learning scope over **one** shared configuration, so the CLI, the tests, and any GUI drive the same system. |
| `core/agents/gate.py` | **new** | The **held-out gate** (`personality_agents_accuracy`) over the real wiring, with **negative controls** and **discrimination tests**. |
| `cli/agents.py` | **new** | The **headless twin**: `python -m cli agents list\|show\|capabilities\|route\|run\|collaborate\|conflicts\|monitoring\|isolation`. |

### Extended

- **`core/character_learning.py`** — the **Natasha leak fix**. `training_keys()`
  returns the six shareable characters and **excludes** Natasha; the default of
  `character_learning_units` is now the shareable set; `persona_for` was corrected
  to key by the **mouth** vocabulary so a persona lookup never misses.
- **`evaluation/benchmark.py` / `runner.py` / `baseline.json`** — the new metric
  `personality_agents_accuracy` enters `overall`.
- **`cli/menu.py` / `cli/__main__.py`** — the `agents` command and three menu
  entries (*the seven personalities as independent agents*, *which personality
  should answer a question*, *the memory and learning isolation policy*).

## 4. What each personality owns

| Owner | Mouth | Private? | Contributes to training? | Default capabilities |
| --- | --- | --- | --- | --- |
| Jarvis | `sources` | no | yes | answer, retrieve, reason, run tools, manage goals, receive delegation, contribute training |
| Ultron | `ultron` | no | yes | as Jarvis |
| Vision | `vision` | no | yes | as Jarvis |
| Banner | `banner` | no | yes | as Jarvis |
| **Natasha** | `natasha` | **yes** | **no** | answer, retrieve, reason, run tools, manage goals, receive delegation *(no training)* |
| Hakeem | `hakeem` | no | yes | as Jarvis |
| Cloud | `cloud` | no | yes | as Jarvis |
| **Fayd Al-Ruh** (project model) | `model` | no | owns the aggregate | as Jarvis + read every shareable character's memory |

`SEARCH` and `WRITE_FILES` are **not** granted to any personality by default;
they exist in the vocabulary and can only be granted where a ceiling allows them
(which, for the shipped personalities, is nowhere).

## 5. Isolation, structurally

Isolation is not a flag — it is a **ceiling and a guard**.

- **Capability isolation.** `effective_capabilities(owner)` is always a subset of
  `capability_ceiling(owner)`. A configuration cannot widen it, because the
  intersection is applied on every read.
- **Memory isolation.** `PersonalityMemoryAccess.can_read(reader, owner)` is
  checked **before any directory is resolved**; a refused read raises
  `LearningAccessError`.
- **Learning isolation.** `aggregated_dataset_owners()` never contains Natasha,
  and the training corpus (`ml.llm.corpus`) reads it through
  `character_learning_sentences`/`training_keys`, so her records cannot reach a
  dataset even if she is asked for them.

## 6. The Natasha guarantee

The Phase-5 mission's hard constraint — **Natasha's private learning must never
leak into aggregated training datasets** — is enforced at three levels:

1. **Ceiling** — `AgentCapability.CONTRIBUTE_TRAINING` is absent from Natasha's
   ceiling, so `decide_capability("natasha", CONTRIBUTE_TRAINING).allowed` is
   `False`.
2. **Scope** — `learning_scope("natasha").contributes` is `False`;
   `aggregated_dataset_owners()` excludes her; `aggregated_dataset_is_clean()`
   is `True`.
3. **Corpus** — `training_keys()` returns six owners, not seven, and
   `character_learning_sentences()`/`character_learning_units()` default to it.

A dedicated test writes a private fact to Natasha, then asserts the training
corpus, the corpus units, and `build_corpus(include_characters=True)` all omit it.

## 7. Verification

- `tests/test_agents_phase5.py` — 92 tests across the registry, capabilities,
  runtime, context, goals, memory access, learning scope, the Natasha guarantee,
  routing, collaboration, conflict resolution, communication, monitoring, the
  composed system, and the gate.
- `tests/test_agents_cli_phase5.py` — 19 tests over the CLI.
- `core/agents/gate.py` — 42 held-out judgements, score **1.0** on the shipped
  wiring, **0.0** for an empty catalogue, and **< 1.0** when a guard is removed
  (proven to discriminate).

## 8. Deliberately out of scope

Phase 5 is the **agent framework**. It does not:

- train model weights (Phase 6);
- add a second knowledge graph or memory store (the Phase-1/Phase-2 layers are
  reused);
- give a personality a new model (the mouths are unchanged);
- run anything on its own (a task runs only when `run` is called).
