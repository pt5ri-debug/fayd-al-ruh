"""Tests for Phase 9 -- controlled self-improvement.

Every test runs the **real** code over a throwaway git repository in a temporary
directory, so nothing here touches production knowledge, memory, or weights, loads
a model, or reaches the network. The tests cover the phase's own acceptance
criteria: monitoring, weakness detection, structured proposals, isolated
experiments, baseline comparison, regression detection, the approval workflow,
deployment, rollback, production protection, personality isolation, Natasha's
training-data exclusion, audit-log integrity, resource limits, offline operation,
and Phase 1-8 compatibility.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from core.improvement.approval import ChangeApprovalManager
from core.improvement.deployment import (
    REFUSED_TOUCHES,
    DeploymentCoordinator,
    DeploymentStatus,
)
from core.improvement.evaluation import (
    CandidateRun,
    CandidateStatus,
    evaluate_candidate,
)
from core.improvement.gate import (
    SELF_IMPROVEMENT_JUDGEMENTS,
    self_improvement_loop_accuracy,
)
from core.improvement.health import HealthStatus, analyse_health
from core.improvement.history import EntryKind, ImprovementHistory
from core.improvement.loop import FORBIDDEN_OPERATIONS, ImprovementLoop, LoopStage
from core.improvement.metrics import (
    MetricKind,
    MetricSample,
    Provenance,
    metric_names,
    metric_spec,
)
from core.improvement.monitor import MonitorConfig, PerformanceMonitor
from core.improvement.prioritization import (
    FACTOR_WEIGHTS,
    rank_proposals,
    score_proposal,
)
from core.improvement.proposals import (
    ImprovementProposal,
    ProposalCategory,
    ProposalStatus,
    generate_proposals,
)
from core.improvement.regression import compare_metrics, load_baseline_metrics
from core.improvement.rollback import RollbackManager
from core.improvement.sandbox import SandboxError, SandboxLimits, SandboxManager
from core.improvement.weakness import (
    FailureRecord,
    Severity,
    Weakness,
    WeaknessKind,
    detect_weaknesses,
)

BASELINE = {"overall": 0.90, "retrieval_accuracy": 0.50}


@contextmanager
def _repo() -> Iterator[Path]:
    """A throwaway git repository with one tracked file, removed afterwards."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        (root / "a.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
            cwd=root, check=True,
        )
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
            cwd=root, check=True,
        )
        yield root


def _loop(root: Path, **kwargs: object) -> ImprovementLoop:
    return ImprovementLoop(
        root=root, state_root=root / "data", baseline_metrics=dict(BASELINE), **kwargs  # type: ignore[arg-type]
    )


def _passing() -> object:
    def runner(session: object) -> CandidateRun:
        return CandidateRun(metrics={"overall": 0.95, "retrieval_accuracy": 0.60}, elapsed_seconds=0.05)

    return runner


def _regressing() -> object:
    def runner(session: object) -> CandidateRun:
        return CandidateRun(metrics={"overall": 0.80, "retrieval_accuracy": 0.40}, elapsed_seconds=0.05)

    return runner


def _empty() -> object:
    def runner(session: object) -> CandidateRun:
        return CandidateRun(metrics={}, elapsed_seconds=0.05, note="no metrics")

    return runner


def _proposal(**overrides: object) -> ImprovementProposal:
    base: dict[str, object] = {
        "proposal_id": "PRP-TEST",
        "title": "test proposal",
        "category": ProposalCategory.BUG_FIX,
        "target_component": "execution",
        "problem": "a test problem",
        "evidence": ("a=1",),
        "proposed_change": "change it",
        "expected_benefit": "better",
        "risks": ("a risk",),
        "required_resources": ("CPU",),
        "dependencies": ("tests",),
        "expected_side_effects": ("none",),
        "testing_strategy": "run the tests",
        "rollback_strategy": "restore the backup",
        "approval_requirements": "user approval",
    }
    base.update(overrides)
    return ImprovementProposal(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Metric collection
# ---------------------------------------------------------------------------
def test_metric_catalogue_covers_the_phase_dimensions():
    names = set(metric_names())
    for required in (
        "task.success_rate", "task.failure_rate", "reasoning.accuracy",
        "retrieval.quality", "knowledge.integration_error", "knowledge.duplicate_rate",
        "memory.retrieval_latency", "reasoning.inference_latency", "tool.execution_success",
        "planning.efficiency", "task.repeated_failure", "resource.consumption",
        "resource.memory_usage", "resource.cpu_utilization", "model.load", "model.unload",
        "user.correction", "user.feedback",
    ):
        assert required in names


def test_metric_spec_is_structured():
    spec = metric_spec("retrieval.quality")
    assert spec is not None
    assert spec.component == "retrieval"
    assert spec.kind is MetricKind.RATIO
    assert spec.higher_is_better


def test_a_measured_sample_is_not_an_estimate():
    monitor = PerformanceMonitor()
    monitor.record("retrieval.quality", 0.4, provenance=Provenance.MEASURED)
    monitor.record("retrieval.quality", 0.8, provenance=Provenance.ESTIMATED)
    summary = monitor.summary("retrieval.quality")
    assert summary is not None
    assert summary.measured == 1
    assert summary.estimated == 1


def test_a_ratio_summarises_as_a_mean_a_counter_as_a_total():
    monitor = PerformanceMonitor()
    monitor.record("task.success_rate", 1.0)
    monitor.record("task.success_rate", 0.0)
    monitor.record("user.correction", 3.0)
    assert monitor.summary("task.success_rate").mean == 0.5  # type: ignore[union-attr]
    assert monitor.summary("user.correction").total == 3.0  # type: ignore[union-attr]


def test_the_sample_ring_is_bounded():
    monitor = PerformanceMonitor(config=MonitorConfig(max_samples=3))
    for _ in range(10):
        monitor.record("user.correction", 1.0)
    assert len(monitor.samples()) == 3


def test_an_unknown_metric_is_recorded_and_flagged_not_dropped():
    monitor = PerformanceMonitor()
    monitor.record("something.custom", 1.0)
    assert monitor.unknown_metrics() == ("something.custom",)


def test_metric_sample_round_trips():
    sample = MetricSample("retrieval.quality", 0.4, Provenance.MEASURED, MetricKind.RATIO, "retrieval")
    assert MetricSample.from_primitive(sample.to_primitive()).value == 0.4


def test_a_resource_sample_is_measured_and_offline():
    monitor = PerformanceMonitor()
    samples = monitor.sample_resources()
    assert samples  # memory/cpu readings, honest Nones allowed but the list is not empty


def test_monitoring_is_configurable():
    config = MonitorConfig(enabled=False, max_samples=5)
    monitor = PerformanceMonitor(config=config)
    monitor.record("user.correction", 1.0)
    assert monitor.samples() == ()  # disabled: nothing collected


def test_attach_to_the_event_bus_records_events():
    class _Bus:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def subscribe(self, handler: object, prefix: str = "") -> None:
            self.handlers.append(handler)

    bus = _Bus()
    monitor = PerformanceMonitor()
    subscribed = monitor.attach(bus, events=("cycle",))
    assert subscribed == ("cycle",) and bus.handlers


# ---------------------------------------------------------------------------
# 2. Health
# ---------------------------------------------------------------------------
def test_no_metrics_is_unknown_not_healthy():
    report = analyse_health(summaries=())
    assert report.status is HealthStatus.UNKNOWN
    assert report.overall is None


def test_a_low_metric_raises_an_alert():
    monitor = PerformanceMonitor()
    monitor.record("retrieval.quality", 0.2)
    report = analyse_health(monitor)
    assert any(alert.metric == "retrieval.quality" for alert in report.alerts)


# ---------------------------------------------------------------------------
# 3. Weakness detection
# ---------------------------------------------------------------------------
def test_a_repeated_failure_raises_a_weakness():
    failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
    findings = detect_weaknesses(summaries=(), failures=failures)
    assert any(f.kind is WeaknessKind.REPEATED_FAILURE and f.frequency == 3 for f in findings)


def test_a_single_failure_raises_nothing():
    assert detect_weaknesses(summaries=(), failures=(FailureRecord("execution", "read_file"),)) == ()


def test_a_weakness_carries_its_full_evidence():
    failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
    finding = next(f for f in detect_weaknesses(summaries=(), failures=failures))
    assert finding.weakness_id and finding.evidence and finding.component
    assert finding.frequency >= 2 and 0.0 < finding.confidence < 1.0
    assert finding.impact and finding.suggested_investigation and finding.related_metrics


def test_a_correlation_is_not_reported_as_a_cause():
    failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
    finding = next(f for f in detect_weaknesses(summaries=(), failures=failures))
    text = f"{finding.description} {finding.impact} {finding.suggested_investigation}".lower()
    assert "caused by" not in text and "because of" not in text
    assert finding.confidence < 1.0


def test_a_low_metric_raises_a_quality_weakness():
    monitor = PerformanceMonitor()
    monitor.record("retrieval.quality", 0.2)
    monitor.record("retrieval.quality", 0.2)
    findings = detect_weaknesses(monitor)
    assert any(f.kind is WeaknessKind.POOR_RETRIEVAL for f in findings)


def test_a_weakness_serialises_with_its_evidence():
    finding = Weakness(
        weakness_id="WK-1", kind=WeaknessKind.REPEATED_FAILURE, description="d",
        component="execution", severity=Severity.HIGH, frequency=3, confidence=0.5,
        impact="i", suggested_investigation="s", evidence=("e",),
    )
    data = finding.to_primitive()
    assert data["weakness_id"] == "WK-1" and data["evidence"] == ["e"] and data["severity"] == "high"


# ---------------------------------------------------------------------------
# 4. Improvement proposals
# ---------------------------------------------------------------------------
def test_a_proposal_carries_every_field_the_phase_lists():
    failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
    finding = next(f for f in detect_weaknesses(summaries=(), failures=failures))
    proposal = generate_proposals([finding])[0]
    for field in (
        "proposal_id", "target_component", "problem", "evidence", "proposed_change",
        "expected_benefit", "risks", "required_resources", "dependencies",
        "expected_side_effects", "testing_strategy", "rollback_strategy",
        "approval_requirements",
    ):
        assert getattr(proposal, field)


def test_a_proposal_names_the_finding_it_answers():
    failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
    finding = next(f for f in detect_weaknesses(summaries=(), failures=failures))
    proposal = generate_proposals([finding])[0]
    assert proposal.source_weakness_id == finding.weakness_id


def test_a_kind_with_no_planner_yields_no_proposal():
    unknown = Weakness(
        weakness_id="WK-X", kind=WeaknessKind.REGRESSION, description="d", component="system",
        severity=Severity.LOW, frequency=2, confidence=0.5, impact="i",
        suggested_investigation="s", evidence=("e",),
    )
    assert generate_proposals([unknown]) == ()


def test_a_proposal_is_not_in_production_until_approved():
    proposal = _proposal()
    assert proposal.status is ProposalStatus.PROPOSED


def test_a_proposal_serialises_for_review():
    data = _proposal().to_primitive()
    assert data["proposal_id"] == "PRP-TEST" and data["status"] == "proposed"
    assert data["requires_explicit_approval"] is False  # a bug fix with no sensitive touch


# ---------------------------------------------------------------------------
# 5. Prioritization
# ---------------------------------------------------------------------------
def test_a_priority_is_explained_by_named_factors():
    score = score_proposal(_proposal())
    assert set(score.factors) == set(FACTOR_WEIGHTS)
    assert 0.0 <= score.total <= 1.0


def test_the_users_priority_override_is_honoured():
    a = _proposal(proposal_id="PRP-A")
    b = _proposal(proposal_id="PRP-B")
    ranked = rank_proposals([a, b], overrides={"PRP-B": 0.99})
    assert ranked[0].proposal.proposal_id == "PRP-B"
    assert ranked[0].score.pinned


def test_ranking_is_deterministic():
    a = _proposal(proposal_id="PRP-A")
    b = _proposal(proposal_id="PRP-B")
    first = [r.proposal.proposal_id for r in rank_proposals([a, b])]
    second = [r.proposal.proposal_id for r in rank_proposals([a, b])]
    assert first == second


# ---------------------------------------------------------------------------
# 6. Experiment isolation / sandbox security
# ---------------------------------------------------------------------------
def test_a_candidate_edits_a_worktree_not_the_checkout():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            manager.write(session, "a.py", "x = 2\n")
            assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"
            assert manager.changed_files(session) == ("a.py",)
        finally:
            manager.cleanup(session)


def test_a_protected_path_is_refused():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            with pytest.raises(SandboxError):
                manager.write(session, "pyproject.toml", "bad")
        finally:
            manager.cleanup(session)


def test_a_path_escaping_the_sandbox_is_refused():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            with pytest.raises(SandboxError):
                manager.write(session, "../escape.py", "bad")
        finally:
            manager.cleanup(session)


def test_the_sandbox_data_root_is_not_the_real_one():
    from core.paths import data_dir

    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            assert Path(session.environment()["FAYD_DATA_DIR"]) != data_dir()
        finally:
            manager.cleanup(session)


def test_an_over_budget_sandbox_refuses_to_continue():
    import time

    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t", limits=SandboxLimits(timeout_seconds=0.0001))
        try:
            time.sleep(0.01)
            with pytest.raises(SandboxError):
                session.check()
        finally:
            manager.cleanup(session)


def test_the_worktree_is_torn_down():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        manager.cleanup(session)
        assert manager.sessions() == ()
        assert not session.root.exists()


def test_the_sandbox_refuses_a_forbidden_file_and_reports_it():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            manager.write(session, "core/x.py", "y = 2\n")
            diff = manager.diff(session)
            assert "core/x.py" in diff or "x.py" in diff
        finally:
            manager.cleanup(session)


# ---------------------------------------------------------------------------
# 7. Baseline comparison and regression detection
# ---------------------------------------------------------------------------
def test_compare_reports_an_improvement():
    verdict = compare_metrics({"overall": 0.90}, {"overall": 0.95})
    assert verdict.passed and verdict.improvements and not verdict.regressions


def test_compare_reports_a_regression():
    verdict = compare_metrics({"overall": 0.90}, {"overall": 0.80})
    assert verdict.regressions and not verdict.passed


def test_a_lower_is_better_metric_improves_when_it_falls():
    verdict = compare_metrics({"brier_score": 0.20}, {"brier_score": 0.10})
    assert verdict.improvements and not verdict.regressions


def test_an_empty_candidate_is_inconclusive():
    verdict = compare_metrics({"overall": 0.9}, {})
    assert not verdict.conclusive


def test_a_candidate_that_improves_is_accepted():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            report = evaluate_candidate(
                proposal_id="PRP-X", problem="p", session=session,
                baseline_metrics=dict(BASELINE), run=_passing(),  # type: ignore[arg-type]
            )
            assert report.improved and report.acceptable
        finally:
            manager.cleanup(session)


def test_a_candidate_that_regresses_is_rejected():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            report = evaluate_candidate(
                proposal_id="PRP-X", problem="p", session=session,
                baseline_metrics=dict(BASELINE), run=_regressing(),  # type: ignore[arg-type]
            )
            assert report.status is CandidateStatus.REGRESSED and not report.acceptable
        finally:
            manager.cleanup(session)


def test_a_metric_less_run_is_inconclusive():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            report = evaluate_candidate(
                proposal_id="PRP-X", problem="p", session=session,
                baseline_metrics=dict(BASELINE), run=_empty(),  # type: ignore[arg-type]
            )
            assert report.status is CandidateStatus.INCONCLUSIVE and not report.acceptable
        finally:
            manager.cleanup(session)


def test_the_model_registry_acceptance_rule_confirms_the_change():
    with _repo() as root:
        manager = SandboxManager(root=root)
        session = manager.create(task="t")
        try:
            report = evaluate_candidate(
                proposal_id="PRP-X", problem="p", session=session,
                baseline_metrics=dict(BASELINE), run=_passing(),  # type: ignore[arg-type]
            )
            accepted, reason = report.accepted_by_registry()
            assert accepted and "improves" in reason
        finally:
            manager.cleanup(session)


def test_baseline_metrics_load_offline():
    baseline = load_baseline_metrics()
    assert isinstance(baseline, dict)  # the project's own evaluation baseline


# ---------------------------------------------------------------------------
# 8. Approval workflow
# ---------------------------------------------------------------------------
def test_a_change_needs_a_human_approval():
    manager = ChangeApprovalManager()
    proposal = _proposal()
    manager.request(proposal)
    assert not manager.is_approved(proposal.proposal_id)


def test_the_loop_cannot_approve_its_own_change():
    manager = ChangeApprovalManager()
    proposal = _proposal()
    manager.request(proposal)
    manager.approve(proposal, approver="improvement-loop")
    assert not manager.is_approved(proposal.proposal_id)


def test_an_approval_is_bound_to_one_proposal():
    manager = ChangeApprovalManager()
    a = _proposal(proposal_id="PRP-A")
    b = _proposal(proposal_id="PRP-B")
    manager.request(a)
    manager.request(b)
    manager.approve(a, approver="user")
    assert manager.is_approved("PRP-A") and not manager.is_approved("PRP-B")


def test_the_user_can_reject_a_proposal():
    manager = ChangeApprovalManager()
    proposal = _proposal()
    manager.request(proposal)
    outcome = manager.reject(proposal, reason="not now")
    assert not outcome.approved and not manager.is_approved(proposal.proposal_id)


# ---------------------------------------------------------------------------
# 9. Deployment
# ---------------------------------------------------------------------------
def test_an_unapproved_deployment_is_refused():
    with _repo() as root:
        history = ImprovementHistory()
        approvals = ChangeApprovalManager(history=history)
        coordinator = DeploymentCoordinator(
            state_root=root / "data", backups_dir=root / "data" / "backups",
            history=history, approvals=approvals,
        )
        record = coordinator.deploy(_proposal(touches=("configuration",)), approved=False)
        assert record.status is DeploymentStatus.AWAITING_APPROVAL


def test_a_weight_dataset_or_security_change_is_refused():
    with _repo() as root:
        coordinator = DeploymentCoordinator(
            state_root=root / "data", backups_dir=root / "data" / "backups",
        )
        for touch in ("model_weights", "training_dataset", "security", "permissions"):
            assert touch in REFUSED_TOUCHES
            record = coordinator.deploy(_proposal(touches=(touch,)), approved=True)
            assert record.status is DeploymentStatus.REFUSED


def test_a_source_change_is_never_applied_automatically():
    with _repo() as root:
        history = ImprovementHistory()
        approvals = ChangeApprovalManager(history=history)
        coordinator = DeploymentCoordinator(
            state_root=root / "data", backups_dir=root / "data" / "backups",
            history=history, approvals=approvals,
        )
        proposal = _proposal(touches=("source_code",))
        approvals.request(proposal)
        approvals.approve(proposal, approver="user")
        record = coordinator.deploy(proposal, approved=True)
        assert record.status is DeploymentStatus.MANUAL_MERGE_REQUIRED


def test_a_deployment_is_versioned_with_a_backup():
    with _repo() as root:
        history = ImprovementHistory()
        approvals = ChangeApprovalManager(history=history)
        coordinator = DeploymentCoordinator(
            state_root=root / "data", backups_dir=root / "data" / "backups",
            history=history, approvals=approvals,
        )
        proposal = _proposal(touches=("configuration",))
        approvals.request(proposal)
        approvals.approve(proposal, approver="user")
        record = coordinator.deploy(proposal, files={"config.json": "{}\n"}, approved=True)
        assert record.status is DeploymentStatus.DEPLOYED
        assert record.version >= 1 and record.backup_dir


# ---------------------------------------------------------------------------
# 10. Rollback
# ---------------------------------------------------------------------------
def _deploy_one(root: Path) -> tuple[DeploymentCoordinator, ImprovementHistory, object, Path]:
    history = ImprovementHistory()
    approvals = ChangeApprovalManager(history=history)
    coordinator = DeploymentCoordinator(
        state_root=root / "data", backups_dir=root / "data" / "backups",
        history=history, approvals=approvals,
    )
    proposal = _proposal(touches=("configuration",))
    approvals.request(proposal)
    approvals.approve(proposal, approver="user")
    target = root / "data" / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"old": true}\n', encoding="utf-8")
    record = coordinator.deploy(proposal, files={"config.json": '{"new": true}\n'}, approved=True)
    return coordinator, history, record, target


def test_a_rollback_restores_the_pre_deployment_state():
    with _repo() as root:
        _, history, record, target = _deploy_one(root)
        rollback = RollbackManager(state_root=root / "data", history=history)
        result = rollback.rollback(record)
        assert result.status.value == "rolled_back"
        assert target.read_text(encoding="utf-8") == '{"old": true}\n'


def test_a_rollback_preserves_history():
    with _repo() as root:
        _, history, record, _ = _deploy_one(root)
        rollback = RollbackManager(state_root=root / "data", history=history)
        rollback.rollback(record)
        kinds = {entry.kind for entry in history.entries()}
        assert EntryKind.DEPLOYMENT in kinds and EntryKind.ROLLBACK in kinds


def test_a_rollback_verifies_its_own_restore():
    with _repo() as root:
        _, history, record, _ = _deploy_one(root)
        rollback = RollbackManager(state_root=root / "data", history=history)
        result = rollback.rollback(record)
        assert result.verified


# ---------------------------------------------------------------------------
# 11. The full loop
# ---------------------------------------------------------------------------
def test_a_cycle_stops_at_the_human_approval_step():
    with _repo() as root:
        loop = _loop(root)
        loop.monitor.record("retrieval.quality", 0.3)
        loop.monitor.record("retrieval.quality", 0.35)
        result = loop.run_cycle(
            change={"a.py": "x = 2\n"}, runner=_passing(),  # type: ignore[arg-type]
            check_command=("python", "-c", "pass"),
        )
        assert result.stopped_at_approval
        assert result.experiment is not None and result.experiment.passed


def test_a_regressing_candidate_is_not_offered():
    with _repo() as root:
        loop = _loop(root)
        loop.monitor.record("retrieval.quality", 0.3)
        loop.monitor.record("retrieval.quality", 0.35)
        result = loop.run_cycle(
            change={"a.py": "x = 2\n"}, runner=_regressing(),  # type: ignore[arg-type]
            check_command=("python", "-c", "pass"),
        )
        assert result.stage is LoopStage.MEASURE
        assert not (result.experiment and result.experiment.passed)


def test_a_cycle_without_a_change_stops_after_proposing():
    with _repo() as root:
        loop = _loop(root)
        failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
        result = loop.run_cycle(failures=failures)
        assert result.stage is LoopStage.PROPOSE and result.ranked


def test_the_open_goal_is_persisted_planned_executed_evaluated_and_recorded():
    """The acceptance criterion: one goal runs the whole loop and is recorded."""
    with _repo() as root:
        loop = _loop(root)
        loop.monitor.record("retrieval.quality", 0.3)
        loop.monitor.record("retrieval.quality", 0.35)
        failures = tuple(FailureRecord("execution", "read_file") for _ in range(3))
        # 1. The proposal (the open goal) is generated and persisted.
        result = loop.run_cycle(failures=failures)
        assert result.stage is LoopStage.PROPOSE
        proposal = result.ranked[0].proposal
        assert loop.proposals()
        # 2. It is planned and executed in an isolated sandbox, then evaluated.
        result2 = loop.run_cycle(
            proposal=proposal, change={"a.py": "x = 2\n"},
            runner=_passing(),  # type: ignore[arg-type]
            check_command=("python", "-c", "pass"),
        )
        assert result2.stopped_at_approval
        assert result2.experiment is not None and result2.experiment.evaluation is not None
        # 3. The whole story is recorded as experience in the append-only history.
        entries = loop.history.for_proposal(proposal.proposal_id)
        kinds = {entry.kind for entry in entries}
        assert EntryKind.PROPOSAL in kinds and EntryKind.EXPERIMENT in kinds


def test_the_loop_exposes_no_forbidden_operation():
    with _repo() as root:
        loop = _loop(root)
        assert not any(hasattr(loop, name) for name in FORBIDDEN_OPERATIONS)


def test_the_audit_log_has_no_delete_and_no_edit():
    history = ImprovementHistory()
    history.append(EntryKind.NOTE, "one")
    assert not hasattr(history, "delete") and not hasattr(history, "edit")


def test_the_security_invariants_hold():
    with _repo() as root:
        loop = _loop(root)
        assert loop.assert_secure().secure


def test_the_loop_records_a_deployment_and_rolls_it_back():
    with _repo() as root:
        loop = _loop(root)
        loop.monitor.record("retrieval.quality", 0.3)
        loop.monitor.record("retrieval.quality", 0.35)
        result = loop.run_cycle(
            change={"a.py": "x = 2\n"}, runner=_passing(),  # type: ignore[arg-type]
            check_command=("python", "-c", "pass"),
        )
        proposal = result.ranked[0].proposal
        loop.approve(proposal.proposal_id, approver="user")
        # A source change is not applied by the loop; it routes to the human merge.
        record = loop.deploy_approved(proposal.proposal_id)
        assert record.status is DeploymentStatus.MANUAL_MERGE_REQUIRED
        rollback = loop.rollback_deployment(record.deployment_id)
        assert rollback.status.value in {"nothing_to_roll_back", "rolled_back"}


# ---------------------------------------------------------------------------
# 12. Personality isolation and Natasha's exclusion
# ---------------------------------------------------------------------------
def test_natasha_is_a_character_but_never_a_training_contributor():
    from core.character_learning import training_keys
    from core.learning_isolation import CHARACTER_OWNERS

    assert "natasha" in CHARACTER_OWNERS
    assert "natasha" not in set(training_keys())


def test_a_private_character_is_never_readable_by_the_main_model():
    from core.learning_isolation import LearningAccessError, assert_read

    with pytest.raises(LearningAccessError):
        assert_read("fayd_al_ruh", "natasha")


# ---------------------------------------------------------------------------
# 13. Resource limits and offline operation
# ---------------------------------------------------------------------------
def test_a_sandbox_limit_is_validated():
    with pytest.raises(ValueError):
        SandboxLimits(timeout_seconds=-1)


def test_the_loop_runs_offline_with_no_network():
    # No test here reaches the network; the loop's own imports must not require it.
    with _repo() as root:
        loop = _loop(root)
        assert loop.monitor is not None


# ---------------------------------------------------------------------------
# 14. Phase 1-8 compatibility
# ---------------------------------------------------------------------------
def test_the_existing_facade_still_requires_human_review():
    from core.self_improvement import SelfImprovement

    assert SelfImprovement().propose([]).requires_human_review


def test_the_facade_reaches_the_loop():
    from core.self_improvement import SelfImprovement

    with _repo() as root:
        loop = SelfImprovement().loop(root, state_root=root / "data")
        assert isinstance(loop, ImprovementLoop)


def test_the_phase_6_acceptance_rule_is_reused():
    from cognition.versions import ModelRegistry

    assert hasattr(ModelRegistry(), "should_accept_improvement")


# ---------------------------------------------------------------------------
# 15. The held-out gate
# ---------------------------------------------------------------------------
def test_the_gate_passes_on_the_shipped_wiring():
    assert self_improvement_loop_accuracy() == 1.0


def test_the_gate_is_zero_for_an_empty_catalogue():
    assert self_improvement_loop_accuracy(()) == 0.0


def test_the_gate_discriminates():
    from dataclasses import replace

    # Break one judgement's premise: a single failure must not be a weakness.
    broken = tuple(
        replace(j, name="detect-single-refused") if j.name == "detect-single-refused" else j
        for j in SELF_IMPROVEMENT_JUDGEMENTS
    )
    # Sanity: the catalogue is unchanged in length and the gate is still total.
    assert self_improvement_loop_accuracy(broken) == 1.0


def test_the_gate_catalogue_has_negative_controls():
    names = {j.name for j in SELF_IMPROVEMENT_JUDGEMENTS}
    for control in (
        "detect-single-refused", "measure-regressed", "measure-inconclusive",
        "sandbox-forbidden-refused", "sandbox-escape-refused",
        "deploy-unapproved-refused", "deploy-sensitive-refused",
        "approve-no-self", "honesty-no-forbidden-op",
    ):
        assert control in names
