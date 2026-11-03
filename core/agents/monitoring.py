"""Agent evaluation and monitoring: what each personality has actually done.

The task asks for per-personality "evaluation and monitoring" -- and it is careful
about honesty: it says not to assume a personality is "objectively superior
without evaluation". So this module *measures* rather than asserts. It reads the
task records the runtime and the goal manager already keep and reports, per
personality, what it has done: tasks run, succeeded, failed, refused, timed out,
cancelled, its success rate, and the capabilities it exercised. Nothing here is a
score of *quality* -- it is a factual tally of outcomes, which is what a
monitoring surface should show and what an evaluation can build on.

It is dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.agents.capabilities import AgentCapability, CapabilityGateway
from core.agents.goals import GoalManager, TaskStatus
from core.agents.registry import PersonalityRegistry
from core.learning_isolation import owner_for


@dataclass(frozen=True)
class PersonalityActivity:
    """A factual tally of one personality's tasks -- never a quality score."""

    owner: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    refused: int = 0
    timed_out: int = 0
    cancelled: int = 0
    pending: int = 0
    running: int = 0
    delegated: int = 0

    @property
    def success_rate(self) -> float:
        """Succeeded / (succeeded + failed + timed_out) -- ``0.0`` when none ran.

        Refusals and cancellations are excluded from the denominator: a refused
        task is the *policy* working, not the personality failing, and counting it
        as a failure would misreport the policy as a defect.
        """
        attempted = self.succeeded + self.failed + self.timed_out
        if attempted == 0:
            return 0.0
        return self.succeeded / attempted

    def describe(self) -> str:
        return (
            f"{self.owner}: {self.total} task(s), {self.succeeded} ok, "
            f"{self.failed} failed, {self.refused} refused, "
            f"{self.timed_out} timed out, {self.cancelled} cancelled, "
            f"{self.delegated} delegated (success {self.success_rate:.0%})"
        )


@dataclass(frozen=True)
class MonitoringReport:
    """The whole system's activity: per personality, plus the refusals."""

    activities: tuple[PersonalityActivity, ...]
    refusals: tuple[str, ...] = ()

    @property
    def total_tasks(self) -> int:
        return sum(activity.total for activity in self.activities)

    def for_owner(self, owner: str) -> PersonalityActivity:
        """One personality's activity (a zero tally for an owner with none)."""
        resolved = owner_for(owner)
        for activity in self.activities:
            if activity.owner == resolved:
                return activity
        return PersonalityActivity(resolved)

    def describe(self) -> str:
        lines = [f"agent monitoring: {self.total_tasks} task(s) across {len(self.activities)} personality(ies)"]
        lines.extend(f"  {activity.describe()}" for activity in self.activities)
        if self.refusals:
            lines.append(f"  refusals: {len(self.refusals)}")
        return "\n".join(lines)


@dataclass
class AgentMonitor:
    """Reads the runtime's own records into a per-personality report.

    ``goals`` is the goal manager the runtime writes to, and ``gateway`` the
    capability gate whose refusals it counts. Both are injectable, so the monitor
    reads whatever the caller's runtime uses and never a second copy.
    """

    registry: PersonalityRegistry = field(default_factory=PersonalityRegistry)
    goals: GoalManager = field(default_factory=GoalManager)
    gateway: CapabilityGateway = field(default_factory=CapabilityGateway)

    def activity_for(self, owner: str) -> PersonalityActivity:
        """The tally for one personality, from its task records."""
        resolved = owner_for(owner)
        tasks = self.goals.tasks_for(resolved)
        counts: dict[TaskStatus, int] = {}
        delegated = 0
        for task in tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
            if task.delegated_by:
                delegated += 1
        return PersonalityActivity(
            owner=resolved,
            total=len(tasks),
            succeeded=counts.get(TaskStatus.SUCCEEDED, 0),
            failed=counts.get(TaskStatus.FAILED, 0),
            refused=counts.get(TaskStatus.REFUSED, 0),
            timed_out=counts.get(TaskStatus.TIMED_OUT, 0),
            cancelled=counts.get(TaskStatus.CANCELLED, 0),
            pending=counts.get(TaskStatus.PENDING, 0),
            running=counts.get(TaskStatus.RUNNING, 0),
            delegated=delegated,
        )

    def report(self) -> MonitoringReport:
        """Every personality's activity, plus the capability refusals."""
        activities = tuple(
            self.activity_for(owner) for owner in self.registry.ids()
        )
        refusals = tuple(
            decision.describe() for decision in self.gateway.refusals()
        )
        return MonitoringReport(activities=activities, refusals=refusals)

    def capabilities_used(self, owner: str) -> tuple[AgentCapability, ...]:
        """The capabilities ``owner`` has actually exercised (from the gate log)."""
        resolved = owner_for(owner)
        used = {
            decision.capability
            for decision in self.gateway.decisions
            if decision.owner == resolved and decision.allowed
        }
        return tuple(sorted(used, key=lambda capability: capability.value))

    def describe(self) -> str:
        return self.report().describe()


__all__ = [
    "AgentMonitor",
    "MonitoringReport",
    "PersonalityActivity",
]
