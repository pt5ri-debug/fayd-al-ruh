# Self-Improvement API (Phase 9)

The stable Python surface and the headless CLI. Everything lives under
`core/improvement/`; the one facade is `ImprovementLoop`.

## The loop

```python
from core.self_improvement import improvement_loop, SelfImprovement

# The one entry point. root = the checkout the loop may isolate;
# state_root = where the loop keeps its own data (defaults to the project's).
loop = improvement_loop(root=".", state_root="data")

# Or through the class that already existed:
loop = SelfImprovement().loop(root=".", state_root="data")
```

### `ImprovementLoop`

| method | does |
|---|---|
| `record_metrics()` | read the machine's own figures into the monitor |
| `analyse(*, failures=(), minimum_frequency=2)` | detect weaknesses and rank proposals |
| `run_cycle(*, proposal=None, change=None, failures=(), runner=None, check_command=None, deploy=False)` | one cycle, stopping at the human approval step |
| `approve(proposal_id, *, approver="user")` | record a human approval |
| `reject(proposal_id, *, reason="")` | record a rejection |
| `deploy_approved(proposal_id)` | deploy a proposal that has an approval (refuses otherwise) |
| `rollback_deployment(deployment_id)` | reverse a deployment from its own backup |
| `proposals()` | the proposals this loop has seen |
| `assert_secure()` | check every security invariant against the live loop |
| `describe()` | a readable status block |

`run_cycle` returns a `CycleResult` with the `stage` it reached, the `weaknesses`,
the `ranked` proposals, the `experiment`, the `deployment`, the `rollback`, and
the `security` report. `CycleResult.stopped_at_approval` is true at the default
end point.

### `LoopStage`

`monitor`, `detect`, `propose`, `isolate`, `measure`, `awaiting_approval`,
`deploy`, `rollback`.

## The components

```python
from core.improvement.metrics import METRIC_CATALOGUE, metric_spec, metric_names
from core.improvement.monitor import PerformanceMonitor, MonitorConfig
from core.improvement.health import analyse_health
from core.improvement.weakness import detect_weaknesses, FailureRecord
from core.improvement.proposals import generate_proposals
from core.improvement.prioritization import rank_proposals
from core.improvement.sandbox import SandboxManager, SandboxLimits
from core.improvement.experiments import ExperimentManager, CandidateChange
from core.improvement.regression import compare_metrics, load_baseline_metrics
from core.improvement.evaluation import evaluate_candidate, EvaluationCriteria
from core.improvement.approval import ChangeApprovalManager
from core.improvement.deployment import DeploymentCoordinator, REFUSED_TOUCHES
from core.improvement.rollback import RollbackManager
from core.improvement.history import ImprovementHistory, EntryKind
from core.improvement.gate import self_improvement_loop_accuracy
```

Key signatures:

```python
PerformanceMonitor(config=MonitorConfig()).record(name, value, *, provenance=Provenance.MEASURED, labels={})
PerformanceMonitor.summary(name) -> MetricSummary | None
PerformanceMonitor.attach(bus, events=(...)) -> tuple[str, ...]
PerformanceMonitor.sample_resources() -> tuple[MetricSample, ...]

detect_weaknesses(monitor_or_summaries, *, failures=(), minimum_frequency=2) -> tuple[Weakness, ...]
generate_proposals(weaknesses) -> tuple[ImprovementProposal, ...]
rank_proposals(proposals, *, severities={}, overrides={}) -> tuple[RankedProposal, ...]

SandboxManager(root=...).create(task=..., limits=SandboxLimits()) -> SandboxSession
SandboxManager.write(session, path, content)   # refuses an escape or a forbidden file
SandboxManager.diff(session) / changed_files(session) / cleanup(session)

evaluate_candidate(*, proposal_id, problem, session, baseline_metrics, run, criteria=EvaluationCriteria()) -> EvaluationReport
compare_metrics(baseline, candidate, *, tolerance=0.0, specs=None) -> RegressionVerdict

ChangeApprovalManager().request(proposal) / approve(proposal, approver=) / reject(proposal, reason=) / is_approved(id)
DeploymentCoordinator(state_root=, backups_dir=, history=, approvals=).deploy(proposal, *, experiment=, files=, approved=False)
RollbackManager(state_root=, history=).rollback(deployment, reason=)
```

## The held-out gate

```python
from core.improvement.gate import self_improvement_loop_accuracy
self_improvement_loop_accuracy()          # 1.0 on the shipped wiring
self_improvement_loop_accuracy(())        # 0.0 for an empty catalogue
```

The metric is wired into `evaluation` as `self_improvement_loop_accuracy` (it
enters `overall`), so a regression in the loop fails the project's own gate.

## The headless CLI

```
python -m cli improve status            # the loop and its security invariants
python -m cli improve metrics           # the metric catalogue
python -m cli improve monitor --record retrieval.quality=0.4 [--estimated]
python -m cli improve health            # the health report
python -m cli improve detect [--failure component:op]... [--min-frequency N]
python -m cli improve propose [--failure ...]... [--json]
python -m cli improve experiment <id> --approve [--file path=content]... [--overall V] [--check-command CMD]
python -m cli improve approve <id> [--approver NAME]
python -m cli improve reject <id> [--reason TEXT]
python -m cli improve deploy <id> [--approve]
python -m cli improve rollback <deployment-id>
python -m cli improve history [--limit N] [--json]
python -m cli improve security          # the invariants, checked live
python -m cli improve gate              # the held-out gate
```

Aliases: `improve`, `self-improvement`, `self_improvement`.

Exit codes: `0` success, `2` invalid arguments, `3` the loop is unavailable, `4` a
gate refused the operation, `5` an unexpected failure.

## The GUI

The four read-only commands are in the sidebar's **System** group
(`gui/actions.py`):

- *Self-improvement: what the loop can do, and its security invariants*;
- *Self-improvement: the system health report*;
- *Self-improvement: detect weaknesses from the metrics*;
- *Self-improvement: the security invariants, checked live*.

They drive the *same* `run_improve` the CLI uses, so the window and the CLI can
never disagree. The mutating commands (experiment, approve, deploy, rollback) are
deliberately **not** in the GUI — a change is offered and approved from the
terminal, where the confirmation is explicit.
