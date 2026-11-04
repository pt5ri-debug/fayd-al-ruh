# Phase 8 — Autonomous Planning and Task Execution

**Status:** implemented, gated, and wired.
**Metric:** `task_execution_accuracy` (held-out, 37 judgements, 1.0000).
**Baseline:** `evaluation/baseline.json` `overall` 0.9902 (was 0.9900).
**Branch:** `feature/phase-8-autonomous-planning`.

## 1. Goal

Interpret a goal, determine what it requires, decompose it into executable tasks,
build and **validate** a plan, select a permitted tool and (optionally) an agent
for each task, execute the permitted actions under the sandbox, monitor progress,
verify the outcome, recover from failure, revise the plan, and record the
experience — **without ever widening the system's authority** and without ever
reporting a tool's success as the goal's success.

The whole layer is CPU-first, dependency-free, Tk-free, and offline: a coordinator
built with no arguments plans and executes an entirely local run against the
project's existing sandbox.

## 2. Pre-implementation review (roadmap + sections 2/3)

1. **Phases 1–6 accepted.** `python -m evaluation` runs green on the branch base
   (`overall` 0.9900), so the baseline is a real reference and every earlier gate
   (memory, learning, agents, training, perception) stays at 1.0.
2. **Roadmap read.** `docs/architecture/IMPLEMENTATION_ROADMAP.md` §Phase 8 names
   the required modifications, the reused modules, and the metric
   `task_execution_accuracy` — the metric implemented here.
3. **Existing modules inspected.** `core/goal.py`, `core/planner.py`,
   `core/plan_evaluator.py`, `core/action*.py`, `core/state*.py`, `core/tools.py`,
   `core/tool_agent.py`, `core/sandbox_*.py`, `core/permissions.py`,
   `core/human_approval.py`, `core/progress.py`,
   `cognition/{runner,scheduler,learning_system}.py`, `core/agents/*`,
   `core/learning_coordinator.py`, `core/experience_learning.py`,
   `core/perception_gate.py` (the gate pattern), and `evaluation/*`.
4. **Sandbox and permission gates inspected.** `core.tools.build_sandbox` supplies
   the allow-list (`calculator`, `list_dir`, `read_file`, `search`, `write_file`)
   and the `HumanApprovalGate` (`sandbox.approvals`); `core.agents.capabilities`
   supplies each personality's hard ceiling.
5. **Dependencies and reusable pipelines identified.** The tool handlers, the
   sandbox policy, the approval gate, the capability gateway, the Phase-3 learning
   coordinator, and the Phase-5 agent runtime all already existed; this phase adds
   the structured goal/task/plan records, the deterministic decomposer, the tool
   catalogue, the execution engine, the verifier, the recovery manager, the audit
   log, and the coordinator that composes them.
6. **Windows 11 compatibility.** Nothing here needs a native library, a GPU, or a
   model: it is pure standard-library Python over the existing sandbox. The one
   platform-sensitive path (the sandbox root) is a `pathlib.Path`.
7. **Hardware.** Intel/AMD x86_64, 16 GB RAM, no usable CUDA GPU. The CPU-first
   rule is respected: no model is loaded, no inference runs, and memory is bounded
   by `MAX_TASKS`, the history limit, and the recovery retry bound.
8. **Checkpoint.** A checkpoint commit precedes the work; all gates were green
   before it.
9. **Compatibility problems reported, not assumed.** No library or model is
   required, so there is no unsupported feature to report; the one honest
   limitation is that a *summarise* or *compare* step would need a language model,
   which the decomposition engine records as a note rather than emitting a task
   that could not really run (section 5).

## 3. What was already there (reused, not rebuilt)

Phase 8 **extends** the architecture. The sandbox, the tool handlers, the approval
gate, the capability ceilings, the learning coordinator, the agent runtime, and the
JSON store all existed; this phase adds the planning/execution layer over them.

| Existing module | What Phase 8 reuses |
| --- | --- |
| `core/tools.py` | `ToolSpec`, `build_sandbox`, `build_sandbox_tools`, the allow-list policy, `ToolResult` |
| `core/permissions.py` | `Capability` (the closed capability vocabulary) |
| `core/human_approval.py` | `HumanApprovalGate`, `ApprovalRequest` (the live approval) |
| `core/agents/capabilities.py` | `CapabilityGateway`, `capability_ceiling` (an agent's hard ceiling) |
| `core/agents/runtime.py` | `CancellationToken` (cooperative cancellation) |
| `core/agents/collaboration.py` | `AgentCollaboration` (bounded delegation, section 14) |
| `core/learning_coordinator.py` | `LearningCoordinator.learn_experience` (the Phase-3 record) |
| `core/experience_learning.py` | `ExperienceRecord` (the experience shape) |
| `core/learning_events.py` | `SOURCE_SYSTEM` (provenance for the recorded lesson) |
| `core/goal.py` | the symbolic `Goal`/`GoalStore` (BFS planning) — **extended** |
| `core/planner.py` | `Planner` (BFS + `order_steps`) — **extended** with persistence |
| `core/tool_agent.py` | `ToolAgent` — **extended** to run a chosen plan and record steps |
| `core/paths.py` | the writable data folder for the goal/plan/history stores |
| `storage/json_store.py` | the durable JSON document store |
| `core/best_effort.py` | the "must not fail the caller" helper for a persistence write |
| `evaluation/` | the metric/benchmark/baseline gate |

## 4. New modules

### The `planning/` package (goal → validated plan)

| Module | Responsibility |
| --- | --- |
| `planning/goals.py` | `Goal`, `GoalKind`, `GoalStatus`, `VerificationCriteria`, `GoalManager` — the structured goal, its durability, and the **authorization rule** (only a user goal or a subgoal of one authorizes execution) |
| `planning/tasks.py` | `Task`, `TaskState` — one unit of work with its preconditions, inputs, outputs, dependencies, tool, arguments, resources, permissions, success criteria, and failure handling; a task's life is a sequence of **immutable versions** |
| `planning/tools.py` | `ToolDescriptor`, `ToolCatalogue`, `ToolSelection`, `RiskCategory`, `select_tool` — the centralized tool registry and its deterministic, explainable selection |
| `planning/decomposition.py` | `DecompositionEngine` — the rule-driven goal→tasks engine, **bounded** (`MAX_TASKS`) and **shallow** (`MAX_DEPTH`), which never emits a task with no tool |
| `planning/dependencies.py` | `DependencyGraph`, `DependencyError` — the DAG, its topological order, and the sequential/independent/parallelizable classification |
| `planning/plan.py` | `TaskPlan`, `PlanStatus`, `PlanValidation`, `validate_plan` — the versioned plan and its validator |
| `planning/coordinator.py` | `PlanningCoordinator`, `default_planner` — decompose → select → validate → plan, plus **replanning** (section 6) and persistence |
| `planning/storage.py` | `GoalStore`, `PlanStore` — durable JSON persistence with corrupt-document tolerance |

### The `execution/` package (plan → verified outcome)

| Module | Responsibility |
| --- | --- |
| `execution/engine.py` | `ExecutionEngine`, `TaskExecution`, `ExecutionReport` — the dependency-ordered scheduler, argument validation and `{task_id}` output chaining, bounded retries, cooperative cancellation, and the injected `runner` seam |
| `execution/permissions.py` | `PermissionManager`, `PermissionRecord`, `ActionCategory` — the permission/approval gate over the sandbox allow-list, the human-approval gate, and an agent's capability ceiling |
| `execution/verification.py` | `OutcomeVerifier`, `Outcome`, `VerificationResult` — verify the **goal's** criteria (artifacts, success criteria), never merely a tool's success |
| `execution/recovery.py` | `RecoveryManager`, `RecoveryDecision`, `RecoveryAction`, `Checkpoint` — bounded, **safety-aware** recovery (never retry an irreversible action) and checkpoint/resume |
| `execution/history.py` | `ExecutionHistory`, `ExecutionEntry`, `redact` — the append-only, **credential-redacting** audit log |
| `execution/monitor.py` | `ExecutionMonitor` — live progress (completed/total, elapsed, ETA) |
| `execution/coordinator.py` | `ExecutionCoordinator`, `ExecutionResult`, `default_executor` — plan → execute → verify → recover → **record the experience** |

### The gate and the surfaces

| Module | Responsibility |
| --- | --- |
| `core/planning_gate.py` | The held-out gate: 37 judgements over the **real** layer, with negative controls |
| `cli/autonomy.py` | The headless twin: `plan`/`run`/`tools`/`goals`/`show`/`history`/`gate` |
| `cli/menu.py`, `cli/__main__.py` | The menu entries and the `autonomy`/`autonomous` commands |

## 5. The safety model (section 9)

Authority is never widened. Five mechanisms hold, and each is pinned by a
**negative control** in the gate:

1. **Only an authorized goal runs.** `GoalManager.authorizes` is true only for a
   user goal or a subgoal of one; a **suggested** goal is refused by the validator
   (`plan-suggested-refused`).
2. **Only a permitted tool runs.** `ToolCatalogue.available` is bounded by the
   sandbox allow-list, and `select_tool` refuses a tool outside it
   (`select-respects-allow-list`).
3. **A risky action needs approval.** `PermissionManager` classifies an action
   (`LOW`/`MEDIUM`/`HIGH`); an irreversible action escalates to `HIGH`; a
   high-risk action is **refused without a live approval** and runs once one is
   granted (`perm-high-refused-without`, `perm-approval-unblocks`,
   `perm-irreversible-escalates`).
4. **An agent cannot exceed its ceiling.** `PermissionManager` consults the
   Phase-5 `CapabilityGateway`, so an action outside a personality's ceiling is
   refused (`perm-agent-ceiling`).
5. **A tool's success is not the goal's.** `OutcomeVerifier` checks the goal's own
   artifacts and success criteria, so a completed tool whose goal is unmet is a
   **failure** (`verify-tool-not-goal`, `verify-missing-artifact`).

Two more honesty rules: an **invalid plan is never executed**
(`exec-invalid-not-run`) and a **failure is never hidden** — the run reports it
and the verifier names the failed task (`honesty-failure-reported`).

## 6. Verification and recovery (sections 10, 11, 13)

- **Verification** compares the run against the goal's `VerificationCriteria`
  (artifacts that must exist, success criteria that must hold). The outcome is one
  of `VERIFIED_SUCCESS`, `PARTIAL_SUCCESS`, `FAILED`, `CANCELLED`, `UNVERIFIED`.
- **Recovery** is **safety-aware**: a reversible failure may be retried (bounded
  by `max_recovery_retries`), but an **irreversible** action is **never retried**
  automatically — it is escalated for a human decision
  (`recover-no-retry-irreversible`). A verified success and a cancelled run need no
  recovery.
- **Checkpoints** (`Checkpoint.from_report`/`restore`) keep completed tasks, so a
  resumed run never redoes finished work.
- **Cancellation** is cooperative: a `CancellationToken` stops the run at the next
  task boundary and the outcome is `CANCELLED`, never a silent stop.

## 7. Learning and agent integration (sections 14, 15)

- **Learning (section 15).** A finished run becomes a
  `core.experience_learning.ExperienceRecord` and is recorded through the Phase-3
  `LearningCoordinator`, so a lesson is learned the same way every other lesson is
  — **never** by fine-tuning and never as a trusted fact. A coordinator with no
  learning service records nothing, and a learning failure is swallowed *and
  written to the history* rather than breaking the run.
- **Agents (section 14).** A task's actor is a Phase-5 personality whose
  capability ceiling is consulted through the same permission gate, so an agent
  never runs a tool its owner does not hold; bounded delegation is the existing
  `AgentCollaboration`.

## 8. Resource management (section 12, 13)

- **CPU-first, offline.** No model is loaded and no network is touched.
- **Bounded.** `MAX_TASKS` (12) bounds a plan; `MAX_DEPTH` (2) bounds nesting; the
  history has a fixed `limit`; recovery retries are bounded.
- **No unnecessary loading.** The sandbox is built lazily; a plan is validated
  before anything runs, so an invalid plan costs nothing.
- **Cancellation and timeout.** A `CancellationToken` stops a run; a tool has its
  own `timeout_seconds`.
- **Persistence is best-effort.** A goal/plan/history write to a read-only folder
  is recorded through `core.best_effort` rather than crashing a run.

## 9. Surfaces

- **CLI:** `python -m cli autonomy plan|run|tools|goals|show|history|gate`
  (aliases `autonomy`/`autonomous`), mirrored as five menu entries.
  `run` returns exit `4` when the goal is not achieved, so a script can detect it.
- **Menu:** *plan a goal*, *run a goal under the permission and approval gate*,
  *list the tools*, *the execution audit trail*, *the held-out gate*.
- **Evaluation:** `task_execution_accuracy` (1.0000) enters `overall`.

## 10. Tests

`tests/test_autonomy.py` (86 tests) covers: the structured goal and its
durability; the decomposition engine (order, dependency, bounds, refusals, the
honest unrecognised-goal case, the arithmetic-only expression); the tool catalogue
and selection (pinned, allow-list, honest no-match, risk ceiling); the plan
validator (unknown tool, suggested goal, no tool, empty, duplicate ids); the
execution engine (read/write, dependency output chaining, invalid-not-run,
cancellation, argument validation, retry, no-retry-irreversible, dependency
blocking); the permission manager (low/high, irreversible escalation, approval
unblock, agent ceiling); verification (success, tool-not-goal, missing artifact,
cancelled, failed-task naming); recovery (noop cases, retry, no-retry, bound,
checkpoint restore); the history (recording, redaction, round trip, bound);
learning (recorded experience, no-learning, learning failure); persistence across
a restart; the CLI; and the gate (perfect, empty→0.0, discriminating, negative
controls present).

## 11. Acceptance checklist

| # | Criterion | Status |
| --- | --- | --- |
| 1 | The planning/execution architecture is modular | ✅ ten `planning/` + eight `execution/` modules, each with a stable API and an injection seam |
| 2 | Goals are persisted and survive a restart | ✅ `GoalStore`/`PlanStore`; pinned by `test_a_goal_and_plan_survive_a_restart` |
| 3 | A goal is decomposed into executable tasks | ✅ `DecompositionEngine`, bounded and shallow |
| 4 | Each task is bound to a **permitted** tool | ✅ `select_tool` refuses outside the allow-list |
| 5 | A plan is validated before it runs | ✅ `validate_plan`; an invalid plan is never executed |
| 6 | Execution runs under the permission/approval gate | ✅ `PermissionManager` + the human-approval gate |
| 7 | The outcome is verified against the goal | ✅ `OutcomeVerifier` (a tool's success is not the goal's) |
| 8 | Failure is recovered from safely | ✅ `RecoveryManager` (never retries an irreversible action) |
| 9 | The plan can be revised | ✅ `PlanningCoordinator.replan` |
| 10 | Progress is monitored and the run can be cancelled | ✅ `ExecutionMonitor`, `CancellationToken` |
| 11 | The experience is recorded | ✅ through the Phase-3 `LearningCoordinator` |
| 12 | An agent cannot exceed its ceiling | ✅ the capability gateway is consulted |
| 13 | Resource limits are respected | ✅ bounded tasks, depth, history, retries; CPU-first |
| 14 | Offline runtime | ✅ no model, no network |
| 15 | Tests pass and the gate is green | ✅ 86 tests; `task_execution_accuracy` 1.0000 |
| 16 | Previous phases remain stable | ✅ memory/learning/agents/training/perception gates stay 1.0 |

## 12. Known limitations

- **No language-model step.** A *summarise* or *compare* goal needs a model, which
  is not a sandbox tool; the decomposition engine records that honestly as a note
  rather than emitting a task it could not run.
- **Rule-driven decomposition.** The engine recognises a goal's shape from its own
  words (read/write/compute). A goal outside those shapes yields one honest task
  with no tool, marked blocked — never a fabricated plan.
- **No parallel execution by default.** The engine runs tasks in dependency order;
  the scheduler supports independent tasks, but the default is sequential, which is
  what a single-CPU, single-disk machine wants.
- **Recovery is bounded and conservative.** It retries a reversible failure a
  bounded number of times and escalates otherwise; it never invents a fix.
