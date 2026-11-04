# Phase 9 — Controlled Self-Improvement: Implementation Report

An automated, **human-gated** loop for Fayd Al-Ruh:
monitor → detect → propose → isolated test → measure → accept/reject → rollback.

Self-improvement here is controlled engineering and optimization. It does not
establish consciousness, human-like self-awareness, or unrestricted autonomy, and
it cannot widen its own authority.

## 1. Existing components inspected

| area | inspected | reused? |
|---|---|---|
| Phase 1 cognitive core | `core/cognitive_controller.py`, `core/ontology.py`, `core/learning_pipeline.py` | read-only |
| Phase 2 long-term memory | `core/memory_db.py`, `core/long_term_memory.py`, `core/memory_gate.py` | read-only |
| Phase 3 learning | `core/learning_events.py`, `core/learning_coordinator.py`, `core/learning_evaluation.py` | read-only |
| Phase 5 agents | `core/agents/` (registry, capabilities, memory access, isolation) | read-only |
| Phase 6 training | `ml/training/`, `cognition/versions.py` (`ModelRegistry.should_accept`) | **reused** (acceptance rule) |
| Phase 7 perception | `perception/`, `core/perception_bridge.py` | read-only |
| Phase 8 autonomy | `core/planning/`, `core/execution/`, `core/plan_evaluator.py` | read-only |
| isolation | `core/learning_isolation.py`, `core/character_learning.training_keys()` | **reused** (preserved) |
| evaluation | `evaluation/runner.py`, `evaluation/benchmark.py`, `evaluation/baseline.json` | **extended** |
| sandboxing | `core/code_studio.py` (git worktrees, `run_checks`) | **extended** |
| audit | `core/learning_audit.py`, `storage/json_store.py` | pattern reused |

## 2. Components reused

- **Sandbox** — `core/code_studio.py`'s git worktrees (one branch per candidate,
  cut from the base branch). The phase adds only a throwaway `FAYD_DATA_DIR`, so a
  candidate's writes cannot reach real learned data. `CodeStudio.run_checks` gained
  `command`/`timeout`/`env` parameters so the sandbox can run the project's own
  checks with the throwaway environment.
- **Acceptance rule** — `cognition.versions.ModelRegistry.should_accept`. The
  loop converts its measured deltas into the registry's own `Evaluation` records
  and asks the same rule, so the loop and the model gate cannot disagree.
- **Baseline** — the project's own `evaluation/baseline.json`, read by
  `load_baseline_metrics()`.
- **Isolation** — `core/learning_isolation.py` and
  `core/character_learning.training_keys()` are read and preserved unchanged.
- **Metric specs** — `evaluation/benchmark.METRIC_SPECS` supply each metric's
  direction and tolerance to the comparison.

## 3. Components created or modified

**Created** — `core/improvement/` (17 modules):

`metrics.py`, `monitor.py`, `health.py`, `weakness.py`, `proposals.py`,
`prioritization.py`, `sandbox.py`, `experiments.py`, `regression.py`,
`evaluation.py`, `approval.py`, `deployment.py`, `rollback.py`, `history.py`,
`loop.py`, `gate.py`, `__init__.py`.

**Created** — `cli/improve.py` (the headless twin), and the tests
`tests/test_self_improvement_phase9.py` (72 tests) and
`tests/test_improve_cli.py` (12 tests), plus 11 documentation files under
`docs/self_improvement/`.

**Modified**:

| file | change |
|---|---|
| `core/self_improvement.py` | added `improvement_loop()`, `SelfImprovement.loop()`, `SelfImprovement.analyse()` — the existing `requires_human_review=True` behaviour is unchanged |
| `core/code_studio.py` | `run_checks(workspace, command=None, *, timeout=900.0, env=None)` |
| `cognition/versions.py` | `ModelRegistry.rollback()` returns the last accepted `ModelVersion` |
| `core/paths.py` | `self_improvement_dir()`, `self_improvement_metrics_dir()`, `self_improvement_experiments_dir()`, `self_improvement_backups_dir()` (all under the git-ignored `data/`) |
| `evaluation/runner.py`, `evaluation/benchmark.py` | the `self_improvement_loop_accuracy` metric (enters `overall`) |
| `evaluation/baseline.json` | overall 0.9902 → 0.9903 |
| `cli/menu.py`, `cli/__main__.py` | `run_improve`, the menu entries, and the `improve` / `self-improvement` / `self_improvement` aliases |
| `gui/actions.py` | four read-only self-improvement actions in the **System** group |

## 4. Monitoring metrics

26 metrics in `METRIC_CATALOGUE` covering task success/failure, reasoning
accuracy, retrieval quality, knowledge integration errors and duplicates, memory
retrieval latency, inference latency, tool execution success, planning
efficiency, repeated failures, resource consumption, memory/CPU, model load and
unload, and user corrections and feedback. Every sample carries a timestamp and a
provenance (`measured` / `estimated` / `reported`); the store is a bounded local
ring.

## 5. Weakness detection results

`python -m cli improve detect --failure execution:read_file --failure
execution:read_file --failure execution:read_file` produces:

```
1 weakness(es) detected:
  [high] WK-59adc037 repeated_failure (execution, n=3, conf=0.53): 'read_file' failed 3 times in execution.
      impact: The same operation keeps failing, so the affected path is unreliable.
      investigate: Inspect the last 3 failures for 'read_file' and their error details.
      evidence: failure[0] read_file, failure[1] read_file, failure[2] read_file
```

A single failure produces nothing. Every finding carries its id, evidence,
component, frequency, severity, confidence (< 1.0), impact, suggested
investigation, and related metrics.

## 6. Improvement proposals generated

`generate_proposals(weaknesses)` maps each finding to a proposal with all 14
required fields plus `source_weakness_id`, so an improvement is traceable to the
problem it addresses. A kind with no planner yields no proposal (nothing is
invented). `rank_proposals` scores each on nine **named** factors
(`severity` 0.20, `frequency` 0.10, `evidence` 0.10, `benefit` 0.15,
`complexity` 0.10, `regression_risk` 0.10, `resource_cost` 0.05,
`reversibility` 0.10, `user_impact` 0.10) and honours a user override.

## 7. Sandbox and experiment results

The sandbox is a git worktree with a throwaway `FAYD_DATA_DIR`. A candidate write
lands in the worktree, never the checkout; an escape or a forbidden file
(`.git`, `pyproject.toml`, `requirements.txt`, `.github`) raises `SandboxError`;
an over-budget session refuses to continue; cleanup removes the worktree.

`ExperimentManager.run` writes the change, runs the project's checks in the
sandbox, runs the candidate's `runner` seam, evaluates against the baseline, and
records the experiment (diff, check output, evaluation) in the append-only
history.

## 8. Baseline comparison results

`compare_metrics(baseline, candidate)` uses the evaluation suite's own direction
and tolerance. A candidate that improves is `improved`; one that regresses is
`regressed`; one with no metrics is `inconclusive`. An empty candidate, an empty
baseline, or a baseline sharing no metric is `inconclusive` — never a pass.
`EvaluationReport.accepted_by_registry()` confirms the verdict with the Phase-6
rule.

## 9. Deployment and rollback tests

- An unapproved deployment returns `AWAITING_APPROVAL` with **no side effect**.
- A `model_weights` / `training_dataset` / `security` / `permissions` touch is
  `REFUSED` and routed to the Phase-6 training governance or the security
  governance.
- A `source_code` touch returns `MANUAL_MERGE_REQUIRED` — the human merges it.
- A configuration change is applied after a backup, with a version.
- A rollback restores the exact pre-deployment file content, preserves the
  deployment and rollback entries in the history, and verifies its own restore.

## 10. Security test results

`python -m cli improve security` prints every invariant as `held`:

```
Security invariants: held
  ok: no self-approval path (the loop has no method that approves its own change)
  ok: deployment refuses model weights and datasets (the deployment coordinator refuses a weight or dataset change)
  ok: source changes are never automatic (a source change is routed to the human's Code Studio merge)
  ok: experiments are isolated (every experiment runs in a git worktree with a throwaway data root)
  ok: no forbidden operation is exposed (the loop exposes none of the forbidden self-modification operations)
  ok: the audit log is append-only (the history has no delete or edit path)
  ok: a backup precedes a deployment (a deployment backs up every file it will change)
```

## 11. Resource measurements

Measured on the target class (Windows 11, Intel i7, 16 GB, no GPU): the whole
package imports with no model and no third-party dependency; the 84 Phase-9 tests
run in ~4 s; the evaluation gate stays at 0.9903. Experiments are bounded by
`SandboxLimits` (default timeout 900 s, 4000 files) and never load a model or
reach the network.

## 12. Known limitations

- The loop's experiments run the **project's own checks**; a candidate that needs
  a specific external service cannot be tested here.
- `runner` is a seam: a caller supplies the workload. The package ships a
  default runner that reports the baseline unchanged, so a caller must supply a
  real one to measure a real change.
- The GUI exposes only the **read-only** commands. Mutating commands
  (experiment/approve/deploy/rollback) are CLI-only on purpose, so the
  confirmation is explicit.
- The container this phase was verified in has no Tk shared library, so six
  pre-existing Tk tests (in `test_gui_design.py`, `test_sidebar.py`,
  `test_import_center.py`) cannot run here; they are unrelated to this phase and
  fail identically on `HEAD`.

## 13. Acceptance checklist

| # | criterion | status |
|---|---|---|
| 1 | Performance monitoring is operational | ✅ 26 metrics, measured/estimated provenance, bounded local store |
| 2 | Weakness detection produces traceable findings | ✅ `Weakness` with id/evidence/component/confidence |
| 3 | Improvement proposals are structured and reviewable | ✅ 14 fields + lifecycle + source weakness |
| 4 | Experiments run in isolation | ✅ git worktree + throwaway data root |
| 5 | Candidate changes are evaluated against baselines | ✅ `evaluate_candidate` (ten steps) |
| 6 | Regression detection works | ✅ `compare_metrics` + `RegressionVerdict` |
| 7 | Production changes require approval | ✅ `ChangeApprovalManager` refuses self-approval |
| 8 | Deployment is versioned and reversible | ✅ `DeploymentRecord` version + backup |
| 9 | Rollback is tested | ✅ three gate judgements + full-loop test |
| 10 | Personality isolation is preserved | ✅ read-only over `core/learning_isolation` |
| 11 | Natasha's training-data exclusion remains enforced | ✅ `training_keys()` excludes her |
| 12 | Security boundaries are tested | ✅ seven live invariants + gate |
| 13 | Offline operation is verified | ✅ no network, no model |
| 14 | Tests pass and limitations are documented | ✅ 84 tests; limitations above |
| 15 | Previous phases remain stable | ✅ evaluation PASS, overall 0.9903 |

## 14. Offline verification

```
$ python -m cli improve status
Controlled self-improvement loop
  root: .
  baseline metrics: 68
  proposals: 0
  history: 0 entr(ies)
Security invariants: held
  ...
```

No command in this phase makes a network call or loads a model.

## 15. Implemented vs. simulated

**Implemented and verified**: the metric catalogue and monitor, weakness
detection, proposal generation and prioritization, the sandbox, the comparison,
the approval manager, the deployment coordinator, the rollback manager, the
append-only history, the loop, the held-out gate, the CLI, and the GUI actions.

**Simulated in tests only**: individual candidate experiments use an injected
`runner` seam (a test's own function), so no real production change is ever made
by a test. **No production change was made by this phase** — the phase delivers
the framework, its gate, its CLI, and its documentation.

## 16. Stop

Phase 9 is complete. Per the brief, development stops here and the final system
status awaits the user's review.
