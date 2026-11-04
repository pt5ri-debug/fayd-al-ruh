# Evaluation and Regression (Phase 9, sections 9 and 14)

Every candidate is evaluated against a **reproducible baseline**, and a
deployment is monitored afterwards for a regression. The loop **reuses the
evaluation suite's own comparator** where it can, so the loop and the project's
gate agree on what an improvement is.

Modules: `core/improvement/regression.py` (comparison) and
`core/improvement/evaluation.py` (the ten-step comparison).

## The ten-step evaluation

`evaluate_candidate(proposal_id=, problem=, session=, baseline_metrics=, run=,
criteria=)` runs exactly the steps the phase lists:

1. **define the problem** — `problem`;
2. **establish baseline metrics** — `baseline_metrics`;
3. **define success criteria** — `EvaluationCriteria`;
4. **prepare a workload** — the sandbox and the `run` seam;
5. **run the candidate in isolation** — `run(session)`;
6. **measure results** — the `CandidateRun`'s metrics;
7. **compare candidate and baseline** — `compare_metrics`;
8. **detect regressions** — `RegressionVerdict.regressions`;
9. **assess resource impact** — the run's elapsed time and memory reading;
10. **produce an evaluation report** — `EvaluationReport`.

## The outcome

`CandidateStatus` names the verdict: `improved`, `unchanged`, `regressed`,
`inconclusive`, or `error`. `EvaluationReport.acceptable` is true **only** for a
measured improvement that met the criteria — never for an inconclusive run.

**An inconclusive run is not a pass.** A candidate that produces no metrics, or
that shares no metric with the baseline, is marked `inconclusive` and is never
offered for approval. The gate pins this with the negative control
`measure-inconclusive`.

## The direction rule

A metric's direction is read from the evaluation suite's own `METRIC_SPECS` (so
`brier_score` improves when it *falls*, while `overall` improves when it rises).
A metric not in the suite falls back to a local table and the verdict names the
assumption. `MetricDelta.delta` is always the raw movement; `regressed` and
`improved` apply the direction.

## The project's own acceptance rule

The phase requires the loop not to invent a second definition of "an
improvement". `EvaluationReport.version_evaluations()` converts the measured
deltas into the model registry's own `cognition.versions.Evaluation` records, and
`EvaluationReport.accepted_by_registry()` hands them to
`ModelRegistry.should_accept` — the **Phase-6 training acceptance gate**. So a
change the loop calls an improvement is one the project's own rule calls an
improvement. The gate pins this with `measure-registry-rule`.

## Regression after deployment

Section 14 requires post-deployment monitoring. `compare_metrics(before, after)`
is the same comparison, so a deployment can be compared against the state it
replaced. A regression raises a `WeaknessKind.REGRESSION` finding, which is what
triggers a rollback (see [ROLLBACK.md](ROLLBACK.md)).

## What is measured

The comparison reads whichever metrics the candidate's `run` reports, from the
catalogue's own list: accuracy, reliability, task success, retrieval quality,
reasoning quality, latency, memory consumption, CPU usage, error recurrence, and
regression rate. `EvaluationCriteria` bounds it:

- `min_improvement` — the smallest gain that counts;
- `max_regression` — the largest tolerated fall;
- `required_metrics` — metrics that must be present;
- `max_memory_bytes` — an optional memory ceiling.

## The honesty rule

**No improvement is claimed unless the measurements support it.** A run that
reports no metrics is `inconclusive`; a run that regresses is `regressed`; and a
run that only matches the baseline is `unchanged`. The gate pins the regressing
case with `measure-regressed`.
