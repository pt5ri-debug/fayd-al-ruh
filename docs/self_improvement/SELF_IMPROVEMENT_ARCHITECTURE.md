# Self-Improvement Architecture (Phase 9)

Controlled self-improvement for Fayd Al-Ruh: an automated loop that **monitors**,
**detects**, **proposes**, **tests in isolation**, **measures**, **accepts or
rejects**, and **rolls back** — always stopping at a human decision before
anything changes production.

Self-improvement here means *controlled engineering and optimization*. It does
**not** establish consciousness, human-like self-awareness, or unrestricted
autonomy. The system cannot remove its own approval requirement, grant itself a
permission, disable a control, edit its audit log, delete a backup, change the
isolation rules, bypass the sandbox, or modify production code or model weights
without approval.

## The one loop

Everything lives under `core/improvement/`, and the one facade the rest of the
project uses is `core/improvement/loop.py`'s `ImprovementLoop` — reached through
`core.self_improvement.improvement_loop(root, state_root=...)`. The loop does
nothing the pieces it composes do not already do; it only **orders** them:

| stage | component reused |
|---|---|
| monitor | `core/improvement/monitor.py` (the event bus) |
| detect | `core/improvement/weakness.py` + `health.py` |
| propose | `core/improvement/proposals.py` |
| prioritize | `core/improvement/prioritization.py` |
| isolate | `core/improvement/sandbox.py` (git worktrees) |
| measure | `core/improvement/evaluation.py` + the evaluation suite |
| accept / reject | `cognition.versions.ModelRegistry.should_accept` |
| approve | `core/improvement/approval.py` (a human) |
| deploy / roll back | `core/improvement/deployment.py` / `rollback.py` |
| record | `core/improvement/history.py` |

```
        ┌─────────────────────────────────────────────────────────────┐
        │                     ImprovementLoop                          │
        └─────────────────────────────────────────────────────────────┘
  monitor ──▶ detect ──▶ propose ──▶ prioritize ──▶ isolate ──▶ measure
     │           │           │            │             │           │
  metrics    weaknesses   proposals     ranking      worktree   baseline
  (bounded)  (evidence)   (structured)  (named)      + throwaway vs candidate
     │           │           │            │          data root      │
     ▼           ▼           ▼            ▼             ▼           ▼
  history ◀────────────────────────────────────────────────── accept/reject
     ▲                                                             │
     │                          ┌──────────────────────────────────┘
     │                          ▼
  append-only            AWAITING APPROVAL ──(human)──▶ deploy ──▶ monitor
     ▲                          │                            │
     └─────────────────── rollback ◀───── regression ─────────┘
```

A cycle **never** merges or deploys without an explicit `deploy=True` **and** a
recorded approval. With the default arguments `run_cycle(...)` ends at
`LoopStage.AWAITING_APPROVAL`, which is the phase's acceptance criterion.

## The components

| module | responsibility | stable API |
|---|---|---|
| `metrics.py` | the structured metric vocabulary | `MetricSpec`, `MetricSample`, `MetricKind`, `Provenance`, `metric_spec`, `metric_names`, `METRIC_CATALOGUE` |
| `monitor.py` | collect operational metrics | `PerformanceMonitor`, `MonitorConfig`, `MetricSummary`, `catalogue`, `subscribe_controller` |
| `health.py` | one honest reading of how the system is doing | `analyse_health`, `HealthReport`, `HealthStatus`, `HealthAlert` |
| `weakness.py` | evidence-backed findings, never guesses | `detect_weaknesses`, `Weakness`, `WeaknessKind`, `Severity`, `FailureRecord` |
| `proposals.py` | a structured, reviewable change request | `generate_proposals`, `ImprovementProposal`, `ProposalCategory`, `ProposalStatus` |
| `prioritization.py` | a ranking with named criteria | `rank_proposals`, `score_proposal`, `PriorityScore`, `FACTOR_WEIGHTS` |
| `sandbox.py` | one isolated place a candidate may run | `SandboxManager`, `SandboxSession`, `SandboxLimits`, `SandboxError` |
| `experiments.py` | one candidate change, run and measured | `ExperimentManager`, `Experiment`, `CandidateChange`, `ExperimentStatus` |
| `regression.py` | baseline comparison and regression detection | `compare_metrics`, `RegressionVerdict`, `MetricDelta`, `load_baseline_metrics` |
| `evaluation.py` | the ten-step reproducible comparison | `evaluate_candidate`, `EvaluationReport`, `CandidateRun`, `CandidateStatus` |
| `approval.py` | a human decision, and only a human's | `ChangeApprovalManager`, `ApprovalOutcome`, `ApprovalDecision` |
| `deployment.py` | versioned, checked, backed up, reversible | `DeploymentCoordinator`, `DeploymentRecord`, `DeploymentStatus`, `REFUSED_TOUCHES` |
| `rollback.py` | restore the prior state, never erase history | `RollbackManager`, `RollbackRecord`, `RollbackStatus` |
| `history.py` | the append-only audit trail | `ImprovementHistory`, `HistoryEntry`, `EntryKind` |
| `loop.py` | the one controlled, human-gated loop | `ImprovementLoop`, `CycleResult`, `LoopStage`, `FORBIDDEN_OPERATIONS`, `SecurityInvariants` |
| `gate.py` | the held-out gate | `self_improvement_loop_accuracy`, `SELF_IMPROVEMENT_JUDGEMENTS` |

## Reused, never rebuilt

The phase **reuses** the existing infrastructure rather than competing with it:

- **Sandboxing** reuses `core/code_studio.py`'s git worktrees (one branch per
  candidate, cut from the base branch) and adds only a throwaway `FAYD_DATA_DIR`,
  so a candidate's writes cannot reach real learned data.
- **Permission checks** are the ones already in place: the deployment coordinator
  refuses a sensitive touch outright, and a source change is routed to the
  human's Code Studio merge — never applied by the loop.
- **The acceptance rule** is `cognition.versions.ModelRegistry.should_accept`
  (the Phase-6 training gate), so the loop and the model gate cannot disagree
  about what "an improvement" is.
- **Isolation** is `core/learning_isolation.py`; the phase reads it and preserves
  it, and Natasha's exclusion from aggregated training data
  (`core/character_learning.training_keys()`) is unchanged.
- **The evaluation suite** is `evaluation/`; the loop compares against the same
  `evaluation/baseline.json` the project already ships.

## Security boundaries

The security invariants are enforced **outside** the proposal mechanism. A
proposal is *data*; it has no path to any of the forbidden operations, so the
guarantee is structural rather than a policy flag. `ImprovementLoop.assert_secure()`
checks each invariant against the live loop:

- no self-approval path (the loop has no method that approves its own change);
- deployment refuses model weights and datasets;
- source changes are never automatic;
- experiments are isolated;
- no forbidden operation is exposed (`FORBIDDEN_OPERATIONS`);
- the audit log is append-only (the history has no delete or edit path);
- a backup precedes a deployment.

See [SECURITY_BOUNDARIES.md](SECURITY_BOUNDARIES.md) for the full list and the
tests that pin it.

## Offline, CPU-first

Nothing in `core/improvement/` loads a model, reaches the network, or requires a
GPU. Metrics are collected locally, experiments run in a local git worktree with
bounded resource limits, and the whole package imports with no third-party
dependency beyond what the base install already carries. See
[PERFORMANCE_MONITORING.md](PERFORMANCE_MONITORING.md) for the resource
management story.

## Where to read next

- [PERFORMANCE_MONITORING.md](PERFORMANCE_MONITORING.md)
- [WEAKNESS_DETECTION.md](WEAKNESS_DETECTION.md)
- [IMPROVEMENT_PROPOSALS.md](IMPROVEMENT_PROPOSALS.md)
- [EXPERIMENT_SANDBOX.md](EXPERIMENT_SANDBOX.md)
- [EVALUATION_AND_REGRESSION.md](EVALUATION_AND_REGRESSION.md)
- [APPROVAL_AND_DEPLOYMENT.md](APPROVAL_AND_DEPLOYMENT.md)
- [ROLLBACK.md](ROLLBACK.md)
- [SECURITY_BOUNDARIES.md](SECURITY_BOUNDARIES.md)
- [SELF_IMPROVEMENT_API.md](SELF_IMPROVEMENT_API.md)
- [SELF_IMPROVEMENT_TESTS.md](SELF_IMPROVEMENT_TESTS.md)
