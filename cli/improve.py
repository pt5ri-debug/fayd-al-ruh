"""CLI for Phase 9 -- controlled self-improvement.

    python -m cli improve status            # what the loop can do, and its security
    python -m cli improve metrics           # the metric catalogue
    python -m cli improve monitor           # record a sample / read the summaries
    python -m cli improve health            # the health report
    python -m cli improve detect            # detect weaknesses from the metrics
    python -m cli improve propose           # generate and rank proposals
    python -m cli improve experiment <id>   # run a candidate in an isolated sandbox
    python -m cli improve approve <id>      # the user's approval (a human decision)
    python -m cli improve reject <id>       # refuse a proposal
    python -m cli improve deploy <id>       # deploy an approved change (never a source one)
    python -m cli improve rollback <id>     # reverse a deployment from its backup
    python -m cli improve history           # the append-only audit trail
    python -m cli improve security          # the security invariants, checked live
    python -m cli improve gate              # the held-out gate

**Nothing here deploys on its own.** A source change is never applied by the loop
-- it is merged by the user through the Code Studio -- and a deployment needs an
explicit ``--approve`` plus a recorded approval. Every command drives the *same*
:class:`core.improvement.loop.ImprovementLoop` the window uses.

Exit codes: ``0`` success, ``2`` invalid arguments, ``3`` the loop is unavailable,
``4`` a gate refused the operation, ``5`` an unexpected failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from core.improvement.evaluation import CandidateRun
from core.improvement.gate import self_improvement_loop_accuracy
from core.improvement.health import analyse_health
from core.improvement.history import EntryKind
from core.improvement.metrics import METRIC_CATALOGUE
from core.improvement.monitor import PerformanceMonitor
from core.improvement.proposals import ProposalCategory
from core.improvement.weakness import FailureRecord, Severity

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_GATE = 4
EXIT_FAILURE = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="improve", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="What the loop can do, and its security invariants.")

    sub.add_parser("metrics", help="The metric catalogue the loop reads.")

    monitor = sub.add_parser("monitor", help="Record a metric sample, or read the summaries.")
    monitor.add_argument("--record", metavar="NAME=VALUE", help="Record a sample, e.g. retrieval.quality=0.4")
    monitor.add_argument("--estimated", action="store_true", help="Mark the sample as estimated, not measured.")

    health = sub.add_parser("health", help="The health report from the metrics.")
    health.add_argument("--quality", type=float, help="A retrieval.quality sample to record first.")

    detect = sub.add_parser("detect", help="Detect weaknesses from the metrics and any failures.")
    detect.add_argument("--failure", action="append", default=[], metavar="COMPONENT:SUBJECT",
                        help="Report a failure (repeatable), e.g. execution:read_file")
    detect.add_argument("--min-frequency", type=int, default=2, help="The floor below which nothing is reported.")

    propose = sub.add_parser("propose", help="Generate and rank improvement proposals.")
    propose.add_argument("--failure", action="append", default=[], metavar="COMPONENT:SUBJECT")
    propose.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    experiment = sub.add_parser("experiment", help="Run a candidate change in an isolated sandbox.")
    experiment.add_argument("proposal_id", help="The proposal to experiment with.")
    experiment.add_argument("--file", action="append", default=[], metavar="PATH=CONTENT",
                            help="A candidate change (repeatable), e.g. core/x.py='y = 2'.")
    experiment.add_argument("--overall", type=float, help="A simulated 'overall' metric for the candidate.")
    experiment.add_argument("--check-command", default="", help="The check command (default: pytest -q).")
    experiment.add_argument("--approve", action="store_true", help="confirm the experiment may run")

    approve = sub.add_parser("approve", help="Approve a proposal (a human decision).")
    approve.add_argument("proposal_id")
    approve.add_argument("--approver", default="user", help="Who is approving.")

    reject = sub.add_parser("reject", help="Reject a proposal.")
    reject.add_argument("proposal_id")
    reject.add_argument("--reason", default="")

    deploy = sub.add_parser("deploy", help="Deploy an approved change (never a source change).")
    deploy.add_argument("proposal_id")
    deploy.add_argument("--approve", action="store_true", help="confirm the deployment")

    rollback = sub.add_parser("rollback", help="Reverse a deployment from its own backup.")
    rollback.add_argument("deployment_id")

    history = sub.add_parser("history", help="The append-only improvement history.")
    history.add_argument("--proposal", default="", help="Show only one proposal's story.")
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--json", action="store_true")

    sub.add_parser("security", help="The security invariants, checked live.")
    sub.add_parser("gate", help="The held-out self-improvement gate.")

    # Every command that builds the loop accepts the checkout it may isolate.
    for name in (
        "status", "monitor", "health", "detect", "propose", "experiment",
        "approve", "reject", "deploy", "rollback", "history", "security",
    ):
        sub.choices[name].add_argument("--root", default=".", help="The checkout the loop may isolate.")

    return parser


def _loop(args: argparse.Namespace):
    """The loop over the requested checkout, using the project's own data folder."""
    from core.paths import data_dir
    from core.self_improvement import improvement_loop

    root = Path(getattr(args, "root", ".") or ".").resolve()
    return improvement_loop(root, state_root=data_dir())


def _failures(pairs: Sequence[str]) -> tuple[FailureRecord, ...]:
    records: list[FailureRecord] = []
    for pair in pairs:
        component, _, subject = pair.partition(":")
        if not subject:
            raise ValueError(f"a failure must be COMPONENT:SUBJECT, got {pair!r}")
        records.append(FailureRecord(component.strip(), subject.strip()))
    return tuple(records)


def run_status(args: argparse.Namespace) -> int:
    loop = _loop(args)
    print(loop.describe())
    return EXIT_OK


def run_metrics(args: argparse.Namespace) -> int:
    print(f"{len(METRIC_CATALOGUE)} metric(s):")
    for spec in METRIC_CATALOGUE:
        direction = "higher is better" if spec.higher_is_better else "lower is better"
        print(f"  {spec.name:<32} [{spec.component}] {spec.kind.value}, {direction} - {spec.description}")
    return EXIT_OK


def run_monitor(args: argparse.Namespace) -> int:
    monitor = PerformanceMonitor()
    if args.record:
        name, _, raw = args.record.partition("=")
        if not name or not raw:
            raise ValueError("--record must be NAME=VALUE")
        from core.improvement.metrics import Provenance

        provenance = Provenance.ESTIMATED if args.estimated else Provenance.MEASURED
        monitor.record(name.strip(), float(raw), provenance=provenance)
    monitor.sample_resources()
    print(monitor.describe())
    for summary in monitor.summaries():
        print(f"  {summary.describe()}")
    return EXIT_OK


def run_health(args: argparse.Namespace) -> int:
    monitor = PerformanceMonitor()
    if args.quality is not None:
        monitor.record("retrieval.quality", args.quality)
        monitor.record("retrieval.quality", args.quality)
    report = analyse_health(monitor)
    print(report.describe())
    return EXIT_OK


def run_detect(args: argparse.Namespace) -> int:
    from core.improvement.weakness import describe_weaknesses, detect_weaknesses

    monitor = PerformanceMonitor()
    findings = detect_weaknesses(
        monitor, failures=_failures(args.failure), minimum_frequency=args.min_frequency
    )
    print(describe_weaknesses(findings))
    return EXIT_OK


def run_propose(args: argparse.Namespace) -> int:
    from core.improvement.prioritization import describe_ranking, rank_proposals
    from core.improvement.proposals import generate_proposals
    from core.improvement.weakness import detect_weaknesses

    monitor = PerformanceMonitor()
    findings = detect_weaknesses(monitor, failures=_failures(args.failure))
    proposals = generate_proposals(findings)
    severities = {f.weakness_id: f.severity for f in findings}
    ranked = rank_proposals(proposals, severities=severities)
    if args.json:
        print(json.dumps([p.to_primitive() for p in proposals], ensure_ascii=False, indent=2))
        return EXIT_OK
    print(describe_ranking(ranked))
    return EXIT_OK


def run_experiment(args: argparse.Namespace) -> int:
    if not args.approve:
        print("refused: an experiment needs --approve (it runs a candidate change)")
        return EXIT_GATE
    loop = _loop(args)
    change: dict[str, str] = {}
    for pair in args.file:
        path, _, content = pair.partition("=")
        if not path:
            raise ValueError("--file must be PATH=CONTENT")
        change[path.strip()] = content

    runner = None
    if args.overall is not None:
        overall = args.overall

        def runner(session: object) -> CandidateRun:
            return CandidateRun(metrics={"overall": overall}, elapsed_seconds=0.0)

    command = tuple(args.check_command.split()) if args.check_command else None
    record = loop.experiments.run(
        proposal_id=args.proposal_id,
        problem=f"a candidate for {args.proposal_id}",
        change=change,
        runner=runner,  # type: ignore[arg-type]
        check_command=command,
    )
    print(record.describe())
    return EXIT_OK if record.status.value != "failed" else EXIT_GATE


def run_approve(args: argparse.Namespace) -> int:
    loop = _loop(args)
    # Register the proposal so the approval is bound to a real record.
    if args.proposal_id not in {p.proposal_id for p in loop.proposals()}:
        loop.analyse()
    outcome = loop.approve(args.proposal_id, approver=args.approver)
    print(outcome.describe())  # type: ignore[attr-defined]
    return EXIT_OK if outcome.approved else EXIT_GATE  # type: ignore[attr-defined]


def run_reject(args: argparse.Namespace) -> int:
    loop = _loop(args)
    if args.proposal_id not in {p.proposal_id for p in loop.proposals()}:
        loop.analyse()
    outcome = loop.reject(args.proposal_id, reason=args.reason)
    print(outcome.describe())  # type: ignore[attr-defined]
    return EXIT_OK


def run_deploy(args: argparse.Namespace) -> int:
    loop = _loop(args)
    record = loop.deploy_approved(args.proposal_id)
    print(record.describe())
    if record.status.value in {"awaiting_approval", "refused"}:
        return EXIT_GATE
    return EXIT_OK


def run_rollback(args: argparse.Namespace) -> int:
    loop = _loop(args)
    record = loop.rollback_deployment(args.deployment_id)
    print(record.describe())
    return EXIT_OK if record.status.value == "rolled_back" else EXIT_GATE


def run_history(args: argparse.Namespace) -> int:
    loop = _loop(args)
    if args.proposal:
        if args.json:
            print(json.dumps([e.to_primitive() for e in loop.history.for_proposal(args.proposal)],
                             ensure_ascii=False, indent=2))
            return EXIT_OK
        print(loop.history.trace(args.proposal))
        return EXIT_OK
    entries = loop.history.entries(limit=args.limit)
    if args.json:
        print(json.dumps([e.to_primitive() for e in entries], ensure_ascii=False, indent=2))
        return EXIT_OK
    for entry in entries:
        print(f"{entry.at.isoformat()} [{entry.kind.value}] {entry.summary}")
    print(loop.history.describe())
    return EXIT_OK


def run_security(args: argparse.Namespace) -> int:
    loop = _loop(args)
    invariants = loop.assert_secure()
    print(invariants.describe())
    return EXIT_OK if invariants.secure else EXIT_GATE


def run_gate(args: argparse.Namespace) -> int:
    score = self_improvement_loop_accuracy()
    print(f"self_improvement_loop_accuracy: {score:.4f}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    if not args.command:
        parser.print_help()
        return EXIT_OK
    handlers = {
        "status": run_status,
        "metrics": run_metrics,
        "monitor": run_monitor,
        "health": run_health,
        "detect": run_detect,
        "propose": run_propose,
        "experiment": run_experiment,
        "approve": run_approve,
        "reject": run_reject,
        "deploy": run_deploy,
        "rollback": run_rollback,
        "history": run_history,
        "security": run_security,
        "gate": run_gate,
    }
    try:
        return handlers[args.command](args)
    except ValueError as error:
        print(f"invalid: {error}")
        return EXIT_USAGE
    except KeyError as error:
        print(f"unknown: {error}")
        return EXIT_UNAVAILABLE
    except FileNotFoundError as error:
        print(f"not available: {error}")
        return EXIT_UNAVAILABLE
    except Exception as error:  # pragma: no cover - defensive
        print(f"unexpected failure: {type(error).__name__}: {error}")
        return EXIT_FAILURE


__all__ = ["main"]

# Kept importable so a caller can build a proposal without the parser.
_CATEGORY = ProposalCategory
_SEVERITY = Severity
_ENTRY = EntryKind


if __name__ == "__main__":
    raise SystemExit(main())
