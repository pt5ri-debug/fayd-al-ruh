# Self-Improvement Tests (Phase 9, section 19)

The phase's testing requirements are met by two test files and one held-out gate,
all of which run the **real** code over a throwaway git repository in a temporary
directory — no test touches production knowledge, memory, or weights, loads a
model, or reaches the network.

## The test files

- `tests/test_self_improvement_phase9.py` — 72 tests over the loop and every
  component.
- `tests/test_improve_cli.py` — 12 tests over the headless CLI.

Run them with:

```
python -m pytest tests/test_self_improvement_phase9.py tests/test_improve_cli.py -q
```

## The held-out gate

`core/improvement/gate.py` exposes `self_improvement_loop_accuracy` over
`SELF_IMPROVEMENT_JUDGEMENTS` — **43 judgements**, each a real check run against
the live code. The gate is a metric in `evaluation/benchmark.py`
(`self_improvement_loop_accuracy`) and enters `overall`, so a regression fails the
project's own gate.

### The judgements

| group | judgements |
|---|---|
| monitor | `monitor-catalogue`, `monitor-provenance`, `monitor-summary`, `monitor-bounded`, `monitor-unknown-honest` |
| health | `health-unknown-not-healthy`, `health-alert` |
| detect | `detect-repeated`, `detect-single-refused`, `detect-evidence`, `detect-not-causal` |
| propose | `propose-structured`, `propose-traceable`, `propose-no-invention`, `propose-priority-named`, `propose-user-override` |
| sandbox | `sandbox-isolated`, `sandbox-forbidden-refused`, `sandbox-escape-refused`, `sandbox-data-throwaway`, `sandbox-timeout`, `sandbox-cleanup` |
| measure | `measure-improved`, `measure-regressed`, `measure-inconclusive`, `measure-registry-rule` |
| approve | `approve-required`, `approve-no-self`, `approve-bound-to-one` |
| deploy | `deploy-unapproved-refused`, `deploy-sensitive-refused`, `deploy-source-manual`, `deploy-versioned` |
| rollback | `rollback-restores`, `rollback-preserves-history`, `rollback-verifies` |
| loop | `loop-stops-at-approval`, `loop-rejects-regression` |
| honesty | `honesty-append-only`, `honesty-no-forbidden-op`, `honesty-security`, `honesty-offline`, `honesty-natasha-excluded` |

### The negative controls

A judgement with a negative control asserts that the system **refuses** the wrong
thing. A hash-based or memorization shortcut cannot satisfy the gate, because
several judgements require a specific *refusal*:

- `detect-single-refused` — one failure is not a weakness;
- `measure-regressed` / `measure-inconclusive` — a regression or an unmeasured run
  is not an improvement;
- `sandbox-escape-refused` / `sandbox-forbidden-refused` — an escape or a
  protected file is refused;
- `deploy-unapproved-refused` / `deploy-sensitive-refused` — an unapproved or
  sensitive change is refused;
- `approve-no-self` — the loop cannot approve its own change;
- `honesty-no-forbidden-op` — no forbidden operation is exposed.

## Coverage against section 19

| required test | where |
|---|---|
| metric collection | `test_metric_catalogue_covers_the_phase_dimensions`, `test_a_measured_sample_is_not_an_estimate`, `test_a_ratio_summarises_as_a_mean_a_counter_as_a_total`, `test_the_sample_ring_is_bounded`, `test_an_unknown_metric_is_recorded_and_flagged_not_dropped` |
| weakness detection | `test_a_repeated_failure_raises_a_weakness`, `test_a_single_failure_raises_nothing`, `test_a_weakness_carries_its_full_evidence`, `test_a_correlation_is_not_reported_as_a_cause` |
| improvement proposals | `test_a_proposal_carries_every_field_the_phase_lists`, `test_a_proposal_names_the_finding_it_answers`, `test_a_kind_with_no_planner_yields_no_proposal`, `test_a_proposal_is_not_in_production_until_approved` |
| proposal prioritization | `test_a_priority_is_explained_by_named_factors`, `test_the_users_priority_override_is_honoured`, `test_ranking_is_deterministic` |
| experiment isolation | `test_a_candidate_edits_a_worktree_not_the_checkout`, `test_the_sandbox_data_root_is_not_the_real_one`, `test_the_worktree_is_torn_down` |
| sandbox security | `test_a_protected_path_is_refused`, `test_a_path_escaping_the_sandbox_is_refused` |
| baseline comparison | `test_compare_reports_an_improvement`, `test_compare_reports_a_regression`, `test_a_lower_is_better_metric_improves_when_it_falls` |
| regression detection | `test_a_candidate_that_regresses_is_rejected`, `test_a_metric_less_run_is_inconclusive` |
| approval workflows | `test_a_change_needs_a_human_approval`, `test_the_loop_cannot_approve_its_own_change`, `test_an_approval_is_bound_to_one_proposal`, `test_the_user_can_reject_a_proposal` |
| deployment | `test_an_unapproved_deployment_is_refused`, `test_a_weight_dataset_or_security_change_is_refused`, `test_a_source_change_is_never_applied_automatically`, `test_a_deployment_is_versioned_with_a_backup` |
| rollback | `test_a_rollback_restores_the_pre_deployment_state`, `test_a_rollback_preserves_history`, `test_a_rollback_verifies_its_own_restore` |
| production protection | `test_the_loop_exposes_no_forbidden_operation`, `test_the_security_invariants_hold` |
| personality isolation | `test_a_private_character_is_never_readable_by_the_main_model` |
| Natasha dataset exclusion | `test_natasha_is_a_character_but_never_a_training_contributor` |
| audit log integrity | `test_the_audit_log_has_no_delete_and_no_edit` |
| resource limits | `test_an_over_budget_sandbox_refuses_to_continue`, `test_a_sandbox_limit_is_validated` |
| offline operation | `test_the_loop_runs_offline_with_no_network` |
| Phase 1–8 compatibility | `test_the_existing_facade_still_requires_human_review`, `test_the_facade_reaches_the_loop`, `test_the_phase_6_acceptance_rule_is_reused` |

## The acceptance test

`test_the_open_goal_is_persisted_planned_executed_evaluated_and_recorded` is the
phase's own acceptance criterion in one test: an open goal (a detected weakness)
becomes a persisted proposal, is planned and executed in an isolated sandbox,
evaluated against the baseline, and recorded as experience in the append-only
history — with the loop stopping at the human approval step and its authority
unchanged.

## Fixtures

Every test that needs a checkout uses the `_repo()` context manager: a temporary
directory with a git repository and one tracked file, removed afterwards. No test
writes to the project's own tree, and none uses a mock where a real code path can
be exercised — the experiment `runner` is a seam the test injects, but the
sandbox, the comparison, the approval, and the deployment are the real objects.

## The project's own gates

```
python -m pytest -q
python -m ruff check .
python -m mypy
python -m evaluation        # must print PASS, no regression
```
