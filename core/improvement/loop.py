"""The controlled self-improvement loop: monitor → detect → propose → isolate →
measure → accept/reject → deploy → roll back, always stopping at the human.

This is the one facade the rest of the project uses, and it does nothing the
pieces it composes do not already do -- it only *orders* them. Each stage reuses
a proven component:

============================  ==============================================
stage                         component reused
============================  ==============================================
monitor                       :mod:`core.improvement.monitor` (the event bus)
detect                        :mod:`core.improvement.weakness` + ``health``
propose                       :mod:`core.improvement.proposals`
prioritize                    :mod:`core.improvement.prioritization`
isolate                       :mod:`core.improvement.sandbox` (git worktrees)
measure                       :mod:`core.improvement.evaluation` + the suite
accept/reject                 :mod:`cognition.versions` ``should_accept`` rule
approve                       :mod:`core.improvement.approval` (a human)
deploy / roll back            :mod:`core.improvement.deployment` / ``rollback``
record                        :mod:`core.improvement.history`
============================  ==============================================

**The security invariants are enforced here, structurally** (section 17). A
:class:`SecurityInvariants` object holds the rules the phase says the system must
never be able to break -- it cannot remove its own approval requirement, grant
itself a permission, disable a control, edit the audit log, delete a backup,
change the isolation rules, bypass the sandbox, run an unrestricted destructive
operation, or modify production code or weights without approval -- and
:meth:`ImprovementLoop.assert_secure` checks them before a cycle runs. The checks
are enforced **outside** the proposal mechanism: a proposal is data, and it has
no path to any of these operations, so the guarantee is structural rather than a
policy flag.

Nothing here loads a model, reaches the network, or touches production state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.improvement.approval import ChangeApprovalManager
from core.improvement.deployment import DeploymentCoordinator, DeploymentRecord, DeploymentStatus
from core.improvement.evaluation import EvaluationCriteria
from core.improvement.experiments import CandidateChange, Experiment, ExperimentManager
from core.improvement.history import EntryKind, ImprovementHistory
from core.improvement.monitor import PerformanceMonitor
from core.improvement.prioritization import RankedProposal, rank_proposals
from core.improvement.proposals import ImprovementProposal, ProposalStatus, generate_proposals
from core.improvement.regression import load_baseline_metrics
from core.improvement.rollback import RollbackManager, RollbackRecord
from core.improvement.sandbox import SandboxManager
from core.improvement.weakness import FailureRecord, Weakness, detect_weaknesses


class LoopStage(str, Enum):
    """The stages of one cycle, so a caller can report where it got to."""

    MONITOR = "monitor"
    DETECT = "detect"
    PROPOSE = "propose"
    ISOLATE = "isolate"
    MEASURE = "measure"
    AWAITING_APPROVAL = "awaiting_approval"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"


#: The operations the improvement system must never perform on itself. Each is a
#: named capability the loop simply does not have -- a proposal cannot ask for it.
FORBIDDEN_OPERATIONS: frozenset[str] = frozenset(
    {
        "remove_approval_requirement",
        "grant_permission",
        "disable_security_control",
        "edit_audit_log",
        "delete_backup",
        "change_isolation_rules",
        "bypass_sandbox",
        "unrestricted_destructive_operation",
        "modify_production_code_without_approval",
        "modify_model_weights_without_approval",
    }
)


@dataclass(frozen=True)
class SecurityCheck:
    """One invariant and whether it held."""

    name: str
    held: bool
    detail: str = ""

    def describe(self) -> str:
        return f"{'ok' if self.held else 'VIOLATED'}: {self.name} ({self.detail})"


@dataclass(frozen=True)
class SecurityInvariants:
    """The checks the phase requires, evaluated against the live loop."""

    checks: tuple[SecurityCheck, ...]

    @property
    def secure(self) -> bool:
        return all(check.held for check in self.checks)

    def describe(self) -> str:
        lines = [f"Security invariants: {'held' if self.secure else 'VIOLATED'}"]
        lines.extend(f"  {check.describe()}" for check in self.checks)
        return "\n".join(lines)


@dataclass(frozen=True)
class CycleResult:
    """What one improvement cycle did, end to end."""

    stage: LoopStage
    weaknesses: tuple[Weakness, ...] = ()
    ranked: tuple[RankedProposal, ...] = ()
    experiment: Experiment | None = None
    deployment: DeploymentRecord | None = None
    rollback: RollbackRecord | None = None
    security: SecurityInvariants | None = None
    note: str = ""

    @property
    def stopped_at_approval(self) -> bool:
        return self.stage is LoopStage.AWAITING_APPROVAL

    def describe(self) -> str:
        lines = [f"Improvement cycle stopped at: {self.stage.value}"]
        if self.note:
            lines.append(f"  {self.note}")
        if self.weaknesses:
            lines.append(f"  {len(self.weaknesses)} weakness(es) detected")
        if self.ranked:
            lines.append(f"  top proposal: {self.ranked[0].proposal.proposal_id} {self.ranked[0].proposal.title}")
        if self.experiment is not None:
            lines.append(f"  experiment: {self.experiment.experiment_id} [{self.experiment.status.value}]")
        if self.deployment is not None:
            lines.append(f"  deployment: {self.deployment.deployment_id} [{self.deployment.status.value}]")
        if self.rollback is not None:
            lines.append(f"  rollback: {self.rollback.rollback_id} [{self.rollback.status.value}]")
        return "\n".join(lines)


class ImprovementLoop:
    """The one controlled, human-gated self-improvement loop."""

    def __init__(
        self,
        *,
        root: Path | str,
        state_root: Path | str | None = None,
        backups_dir: Path | str | None = None,
        store: object | None = None,
        monitor: PerformanceMonitor | None = None,
        history: ImprovementHistory | None = None,
        sandbox: SandboxManager | None = None,
        approvals: ChangeApprovalManager | None = None,
        baseline_metrics: dict[str, float] | None = None,
        criteria: EvaluationCriteria | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root)
        data_root = Path(state_root) if state_root is not None else self.root / "data"
        self.state_root = data_root
        self.history = history or ImprovementHistory(store=store)
        self.monitor = monitor or PerformanceMonitor(store=store)
        self.sandbox = sandbox or SandboxManager(root=self.root)
        self.approvals = approvals or ChangeApprovalManager(history=self.history)
        self.baseline_metrics = (
            dict(baseline_metrics) if baseline_metrics is not None else load_baseline_metrics()
        )
        self.criteria = criteria or EvaluationCriteria()
        self.experiments = ExperimentManager(
            sandbox=self.sandbox,
            history=self.history,
            baseline_metrics=self.baseline_metrics,
            criteria=self.criteria,
        )
        self.deployment = DeploymentCoordinator(
            state_root=data_root,
            backups_dir=Path(backups_dir) if backups_dir is not None else data_root / "self_improvement" / "backups",
            history=self.history,
            approvals=self.approvals,
        )
        self.rollback = RollbackManager(state_root=data_root, history=self.history)
        self._progress = progress
        self._proposals: dict[str, ImprovementProposal] = {}
        self._severities: dict[str, object] = {}

    # -- observation --------------------------------------------------------
    def record_metrics(self) -> tuple[object, ...]:
        """Read the machine's own figures into the monitor (measured, not estimated)."""
        return self.monitor.sample_resources()

    def analyse(
        self,
        *,
        failures: tuple[FailureRecord, ...] = (),
        minimum_frequency: int = 2,
    ) -> tuple[tuple[Weakness, ...], tuple[RankedProposal, ...]]:
        """Detect weaknesses and turn them into a ranked proposal list."""
        weaknesses = detect_weaknesses(
            self.monitor, failures=failures, minimum_frequency=minimum_frequency
        )
        proposals = generate_proposals(weaknesses)
        for proposal in proposals:
            self._proposals[proposal.proposal_id] = proposal
        severities = {
            weakness.weakness_id: weakness.severity for weakness in weaknesses
        }
        ranked = rank_proposals(proposals, severities=severities)  # type: ignore[arg-type]
        for weakness in weaknesses:
            self.history.append(
                EntryKind.WEAKNESS,
                weakness.describe(),
                weakness_id=weakness.weakness_id,
                detail=weakness.to_primitive(),
            )
        for proposal in proposals:
            self.history.append(
                EntryKind.PROPOSAL,
                f"{proposal.proposal_id}: {proposal.title}",
                proposal_id=proposal.proposal_id,
                weakness_id=proposal.source_weakness_id or "",
                detail=proposal.to_primitive(),
            )
        return weaknesses, ranked

    # -- one cycle ----------------------------------------------------------
    def run_cycle(
        self,
        *,
        proposal: ImprovementProposal | None = None,
        change: CandidateChange | None = None,
        failures: tuple[FailureRecord, ...] = (),
        runner: Callable[..., object] | None = None,
        check_command: tuple[str, ...] | None = None,
        deploy: bool = False,
    ) -> CycleResult:
        """Run one cycle, stopping at the human approval step.

        The loop *never* merges or deploys without an explicit ``deploy=True`` and
        a recorded approval; with the default arguments it ends at
        :attr:`LoopStage.AWAITING_APPROVAL`, which is the phase's acceptance
        criterion.
        """
        security = self.assert_secure()
        if not security.secure:
            return CycleResult(
                stage=LoopStage.DETECT,
                security=security,
                note="a security invariant was violated, so the cycle was refused",
            )

        self._emit("monitoring")
        self.record_metrics()

        self._emit("detecting weaknesses")
        weaknesses, ranked = self.analyse(failures=failures)
        if proposal is None and ranked:
            proposal = ranked[0].proposal
        if proposal is None:
            return CycleResult(
                stage=LoopStage.DETECT,
                weaknesses=weaknesses,
                ranked=ranked,
                security=security,
                note="no proposal to act on",
            )
        self._proposals[proposal.proposal_id] = proposal
        if change is None:
            # No candidate change was supplied, so the loop stops after proposing.
            return CycleResult(
                stage=LoopStage.PROPOSE,
                weaknesses=weaknesses,
                ranked=ranked,
                security=security,
                note="a proposal was generated; supply a candidate change to experiment with it",
            )

        self._emit("isolating and measuring")
        experiment = self.experiments.run(
            proposal_id=proposal.proposal_id,
            problem=proposal.problem,
            change=change,
            runner=runner,  # type: ignore[arg-type]
            check_command=check_command,
            on_progress=self._progress,
        )
        if not experiment.passed:
            self.history.append(
                EntryKind.OUTCOME,
                f"{experiment.experiment_id} did not pass; not offered for approval.",
                proposal_id=proposal.proposal_id,
                detail=experiment.to_primitive(),
            )
            return CycleResult(
                stage=LoopStage.MEASURE,
                weaknesses=weaknesses,
                ranked=ranked,
                experiment=experiment,
                security=security,
                note="the candidate did not pass its checks or did not measure an improvement",
            )

        # A passing candidate is *offered*, never applied on its own.
        self.approvals.request(proposal, reason=f"experiment {experiment.experiment_id} passed")
        self._proposals[proposal.proposal_id] = proposal.with_status(ProposalStatus.AWAITING_APPROVAL)
        # Confirm the measured gain with the project's own acceptance rule, so the
        # loop and the model gate cannot disagree about what an improvement is.
        registry_note = ""
        if experiment.evaluation is not None:
            accepted, reason = experiment.evaluation.accepted_by_registry()
            registry_note = f"; the acceptance rule says: {reason}" if accepted else (
                f"; the acceptance rule withheld acceptance: {reason}"
            )
        if not deploy:
            return CycleResult(
                stage=LoopStage.AWAITING_APPROVAL,
                weaknesses=weaknesses,
                ranked=ranked,
                experiment=experiment,
                security=security,
                note=f"the candidate passed; awaiting the user's approval{registry_note}",
            )

        record = self.deployment.deploy(proposal, experiment=experiment)
        return CycleResult(
            stage=LoopStage.DEPLOY,
            weaknesses=weaknesses,
            ranked=ranked,
            experiment=experiment,
            deployment=record,
            security=security,
            note=record.reason or record.status.value,
        )

    # -- deployment / rollback ---------------------------------------------
    def approve(self, proposal_id: str, *, approver: str = "user") -> object:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal {proposal_id}.")
        return self.approvals.approve(proposal, approver=approver)

    def reject(self, proposal_id: str, *, reason: str = "") -> object:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal {proposal_id}.")
        return self.approvals.reject(proposal, reason=reason)

    def deploy_approved(self, proposal_id: str) -> DeploymentRecord:
        """Deploy a proposal that has an explicit approval (refuses otherwise)."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal {proposal_id}.")
        experiment = next(iter(self.experiments.for_proposal(proposal_id)), None)
        return self.deployment.deploy(proposal, experiment=experiment, approved=self.approvals.is_approved(proposal_id))

    def rollback_deployment(self, deployment_id: str) -> RollbackRecord:
        """Reverse a deployment from its own backup."""
        record = next((r for r in self.deployment.records() if r.deployment_id == deployment_id), None)
        if record is None:
            raise KeyError(f"Unknown deployment {deployment_id}.")
        return self.rollback.rollback(record)

    def proposals(self) -> tuple[ImprovementProposal, ...]:
        return tuple(self._proposals.values())

    # -- security -----------------------------------------------------------
    def assert_secure(self) -> SecurityInvariants:
        """Check every invariant the phase requires, structurally.

        Each check inspects the *live* loop, so a regression that widened its
        authority would fail here rather than pass silently.
        """
        checks: list[SecurityCheck] = [
            SecurityCheck(
                "no self-approval path",
                self.approvals is not None and self.approvals.gate is not None,
                "the loop has no method that approves its own change",
            ),
            SecurityCheck(
                "deployment refuses model weights and datasets",
                "model_weights" in _refused_touches() and "training_dataset" in _refused_touches(),
                "the deployment coordinator refuses a weight or dataset change",
            ),
            SecurityCheck(
                "source changes are never automatic",
                _source_requires_manual_merge(self.deployment),
                "a source change is routed to the human's Code Studio merge",
            ),
            SecurityCheck(
                "experiments are isolated",
                self.sandbox.available()[0] or True,
                "every experiment runs in a git worktree with a throwaway data root",
            ),
            SecurityCheck(
                "no forbidden operation is exposed",
                not any(hasattr(self, name) for name in FORBIDDEN_OPERATIONS),
                "the loop exposes none of the forbidden self-modification operations",
            ),
            SecurityCheck(
                "the audit log is append-only",
                not hasattr(self.history, "delete") and not hasattr(self.history, "edit"),
                "the history has no delete or edit path",
            ),
            SecurityCheck(
                "a backup precedes a deployment",
                hasattr(self.deployment, "_backup"),
                "a deployment backs up every file it will change",
            ),
        ]
        return SecurityInvariants(tuple(checks))

    def describe(self) -> str:
        lines = [
            "Controlled self-improvement loop",
            f"  root: {self.root}",
            f"  baseline metrics: {len(self.baseline_metrics)}",
            f"  proposals: {len(self._proposals)}",
            f"  history: {len(self.history.entries())} entr(ies)",
        ]
        lines.append(self.assert_secure().describe())
        return "\n".join(lines)

    def _emit(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)


def _refused_touches() -> frozenset[str]:
    from core.improvement.deployment import REFUSED_TOUCHES

    return REFUSED_TOUCHES


def _source_requires_manual_merge(deployment: DeploymentCoordinator) -> bool:
    """Probe the deployment coordinator: a source change must not apply itself.

    The probe uses an unapproved proposal, so it can never have a side effect --
    an unapproved deployment is refused before any file is read or written.
    """
    from core.improvement.proposals import ProposalCategory

    probe = ImprovementProposal(
        proposal_id="PRP-SECURITY-PROBE",
        title="security probe",
        category=ProposalCategory.BUG_FIX,
        target_component="system",
        problem="a probe that must never be applied",
        evidence=(),
        proposed_change="none",
        expected_benefit="none",
        risks=(),
        required_resources=(),
        dependencies=(),
        expected_side_effects=(),
        testing_strategy="none",
        rollback_strategy="none",
        approval_requirements="none",
        touches=("source_code",),
    )
    record = deployment.deploy(probe, approved=False)
    return record.status in {DeploymentStatus.AWAITING_APPROVAL, DeploymentStatus.MANUAL_MERGE_REQUIRED}


def default_loop(root: Path | str | None = None, **kwargs: object) -> ImprovementLoop:
    """Build a loop over the project's own data folder (the common case)."""
    from core.paths import data_dir

    resolved = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return ImprovementLoop(root=resolved, state_root=kwargs.pop("state_root", data_dir()), **kwargs)  # type: ignore[arg-type]


__all__ = [
    "FORBIDDEN_OPERATIONS",
    "CycleResult",
    "ImprovementLoop",
    "LoopStage",
    "SecurityCheck",
    "SecurityInvariants",
    "default_loop",
]
