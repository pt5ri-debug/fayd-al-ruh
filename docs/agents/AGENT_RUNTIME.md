# Phase 5 — Runtime, Routing, and Collaboration

**Version:** 2.3.0 · **Date:** 2026-11-02

This document describes how a request becomes an answer in the Phase-5
multi-agent architecture: how a personality is chosen, how a task is executed,
how personalities delegate, and how their disagreements are handled.

## 1. The task lifecycle

A task moves through a closed set of statuses
(`core/agents/goals.py::TaskStatus`):

```
pending ──▶ running ──▶ succeeded
                     ├─ failed
                     ├─ refused
                     ├─ timed_out
                     └─ cancelled
```

`TaskStatus.terminal` is true for every outcome except `pending`/`running`.
A task also carries its owner, an optional goal, an optional parent task, a
`delegated_by` field, its depth, its description, any error, and its timestamps.

## 2. The runtime

`AgentRuntime.run(owner, request, *, question=, capability=, timeout=, token=,
memory_bytes=)` executes one task:

1. **Build the task** in the goal manager (`pending`).
2. **Check the capability** — if the owner does not hold it, the task is
   `refused` and `CapabilityDenied` is available on the strict path.
3. **Ask the governor** for a resource grant when a `memory_bytes` estimate is
   given; a declined request leaves the task `pending` with a readable
   `deferred_reason` (the task is *not* failed).
4. **Run the injected executor** — the seam a test injects, so the runtime is
   exercised with no model.
5. **Enforce the timeout** (the task becomes `timed_out`) and the **cancellation
   token** (`cancelled`).
6. **Validate the result** — a successful-but-empty answer is invalid
   (`default_validator`), so a mouth that produced nothing is a failure, not a
   success.
7. **Record** the task's terminal status and **publish** `agent.task.started` /
   `agent.task.finished` on the event bus.

The runtime holds **no model** and touches **no network**.

## 3. Routing

`TaskRouter.route(question, *, conversation=, task_type=, override=,
user_choice=)`:

1. **Classify** the question into a `TaskType` (`PETROLEUM`, `PRESENTATION`,
   `RELIGION`, `GENERAL`, …) by matching the personality's own declared hints.
2. **Score** every suitable personality on **named factors**:
   - *specialisation* — is this the personality declared for the task type?
   - *capabilities* — does it hold the capabilities the task needs?
   - *task preference* — does its configuration name a matching objective?
3. **Keep** the personality chosen for a conversation's previous turn, unless the
   new turn is clearly a different topic.
4. **Honour a manual override** (a character key) or a **user choice** — but
   refuse an override that names a **non-character** (the project's model is not
   a routable personality).
5. **Be honest** when no personality is suitable: `suitable` is `False`,
   `selected` is `""`, and the reason names the miss.

The decision is **traceable**: `RoutingDecision.scores` carries each candidate's
named contributions, and `describe()` renders them.

## 4. Collaboration

`AgentCollaboration` runs a bounded delegation tree:

- **Depth bound** (default 2) — a delegation deeper than the bound raises
  `DelegationRefused`.
- **Budget** (default 4) — the number of delegations in one collaboration is
  bounded; `budget_used(shared_id)` reports the count.
- **Character-only targets** — the project's model is never a delegation target.
- **Provenance** — every `DelegationResult` names the sender, the target, the
  shared id, and its depth.

`collaborate(sender, question, targets)` asks several personalities the same
question and returns a `CollaborationReport` that keeps **each answer separate** —
never merged into one voice.

## 5. Conflict resolution

`ConflictResolver.resolve(question, positions, policy=)`:

- Each `Position` carries an owner, its answer, and its evidence.
- `compare_positions` reports the dimensions that **differ** (answer, evidence,
  confidence), so a reader sees *what* the disagreement is about.
- The default `PRESERVE` policy names **no winner** and keeps every position.
- The opt-in `EVIDENCE` policy selects a position only when it is **strictly**
  stronger (more evidence than every other, and not a near-tie); otherwise it too
  names no winner.

This is deliberate: the program does not know which personality is right, so it
must not pretend to.

## 6. Communication

`MessageBus` is a bounded, logged channel between personalities:

- `send(sender, recipient, kind, body, shared_id=)` validates the recipient
  (a character or the project's model) and the body (non-empty).
- `BROADCAST` reaches every owner.
- `inbox(owner)` and `thread(shared_id)` read it back.

A message is **information**, never an instruction a personality must obey.

## 7. Monitoring

`AgentMonitor.report()` reads each personality's **real** activity from the goal
manager and the capability gateway: tasks, successes, failures, refusals, and a
success rate (a refusal is counted as a *refusal*, not a failure — nothing was
attempted). The report covers every personality in the registry, so a quiet
personality reads as a zero tally rather than being absent.

## 8. The composed system

`AgentSystem` wires all of the above over **one** shared configuration, so the
CLI, the tests, and any GUI drive the same registry, runtime, router,
collaboration, conflict resolver, monitor, bus, memory-access layer, and learning
scope. `AgentSystem(mouth=...)` uses a mouth-backed executor; an injected
executor makes the whole system testable with no model.
