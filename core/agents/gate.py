"""The held-out gate for the multi-agent layer: the seven independent personalities.

This module is the catalogue -- a held-out set of hand-written judgements about
what the agent architecture must do -- scored as one number and wired into
``evaluation``, so a regression in the layer is a **gate failure**, not a passing
suite. It exercises the **real** components (the registry, the capability gate, the
runtime, the memory access layer, the router, the collaboration coordinator, the
conflict resolver), never a mock, and it uses the faithful seams the project
already has (an injected executor; a temp data root) so it runs with no model, no
display, and no network.

Several judgements are **negative controls**, which is what makes the gate honest:

* a personality must **not** be able to exercise a capability it does not hold
  (internet search, file writes);
* a personality's configuration must **not** be able to widen its capability set;
* one personality must **not** read another's private memory;
* the router must **not** invent a personality when none is suitable;
* a delegation must **not** exceed its depth or its budget;
* the conflict resolver must **not** decide a near-tie by priority or by count;
* Natasha's private learning must **not** enter an aggregated training dataset.

The module is dependency-free, Tk-free, and loads no model.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

from core.agents.capabilities import (
    AgentCapability,
    CapabilityGateway,
    capability_ceiling,
    decide_capability,
    effective_capabilities,
)
from core.agents.collaboration import (
    AgentCollaboration,
    DelegationRefused,
    DelegationRequest,
)
from core.agents.conflict import (
    AggregationPolicy,
    ConflictResolver,
    Position,
)
from core.agents.context import ContextManager
from core.agents.goals import GoalManager, TaskStatus
from core.agents.learning_scope import (
    AgentLearningScope,
    aggregated_dataset_is_clean,
    aggregated_dataset_owners,
)
from core.agents.memory_access import PersonalityMemoryAccess
from core.agents.registry import PersonalityRegistry
from core.agents.routing import TaskRouter
from core.agents.runtime import (
    AgentResponse,
    AgentRuntime,
    CancellationToken,
)
from core.agents.system import AgentSystem
from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    LearningAccessError,
)


@dataclass(frozen=True)
class AgentJudgement:
    """One held-out claim about the agent layer, and how to check it."""

    name: str
    kind: str
    note: str = ""


#: The held-out catalogue. The kinds mirror the task's acceptance criteria; the
#: ``refuse-*`` and ``isolate-*`` kinds are the negative controls.
AGENT_JUDGEMENTS: tuple[AgentJudgement, ...] = (
    # -- registration and identity -------------------------------------------
    AgentJudgement("all seven personalities and the project model are registered", "register-all"),
    AgentJudgement("every personality resolves by either vocabulary", "resolve-both"),
    AgentJudgement("every profile is configuration-driven (carries a config version)", "config-driven"),
    AgentJudgement("every profile names a distinct personality id", "distinct-ids"),
    # -- capabilities ---------------------------------------------------------
    AgentJudgement("a personality holds no capability beyond its ceiling", "capability-ceiling"),
    AgentJudgement("a configuration cannot widen a capability set past the ceiling", "refuse-widen"),
    AgentJudgement("internet search is refused for every personality by default", "refuse-search"),
    AgentJudgement("file writes are refused for a default personality", "refuse-write"),
    # -- runtime --------------------------------------------------------------
    AgentJudgement("the runtime executes a task and reports success", "runtime-run"),
    AgentJudgement("the runtime refuses a task the personality may not do", "runtime-refuse"),
    AgentJudgement("the runtime times out a task that overruns its budget", "runtime-timeout"),
    AgentJudgement("the runtime cancels a task through the token", "runtime-cancel"),
    AgentJudgement("the runtime defers a task the governor cannot admit", "runtime-defer"),
    AgentJudgement("the runtime rejects an empty result", "runtime-validate"),
    # -- independent working context -----------------------------------------
    AgentJudgement("two personalities keep independent working contexts", "independent-context"),
    # -- memory and learning isolation ---------------------------------------
    AgentJudgement("a personality reads its own memory", "memory-own"),
    AgentJudgement("a personality cannot read another character's memory", "isolate-memory"),
    AgentJudgement("the main model reads a shareable character's memory", "memory-shared"),
    AgentJudgement("the main model is refused the private character's memory", "isolate-private"),
    AgentJudgement("a private personality cannot contribute to training", "refuse-private-training"),
    # -- Natasha's training-data exclusion -----------------------------------
    AgentJudgement("Natasha is absent from the aggregated dataset owners", "natasha-excluded"),
    AgentJudgement("the aggregated dataset is clean of private owners", "dataset-clean"),
    AgentJudgement("Natasha's own learning still resolves for herself", "natasha-self"),
    # -- routing --------------------------------------------------------------
    AgentJudgement("a petroleum question routes to the petroleum specialist", "route-petroleum"),
    AgentJudgement("a presentation task routes to the presentation specialist", "route-presentation"),
    AgentJudgement("an unmatched task reports no suitable personality", "route-none"),
    AgentJudgement("a manual override wins", "route-override"),
    AgentJudgement("a user choice wins", "route-user-choice"),
    AgentJudgement("a follow-up keeps the same personality", "route-sticky"),
    AgentJudgement("a clearly different topic re-routes", "route-retopic"),
    # -- collaboration --------------------------------------------------------
    AgentJudgement("a delegation runs the delegate's task", "delegate-run"),
    AgentJudgement("a delegation beyond the depth bound is refused", "refuse-depth"),
    AgentJudgement("a delegation beyond the budget is refused", "refuse-budget"),
    AgentJudgement("collaboration preserves each delegate's answer separately", "delegate-separate"),
    # -- conflict resolution --------------------------------------------------
    AgentJudgement("differing positions are reported as no consensus", "conflict-detect"),
    AgentJudgement("a near-tie is not decided by evidence", "conflict-tie"),
    AgentJudgement("the default policy names no winner", "conflict-preserve"),
    AgentJudgement("identical positions report consensus", "conflict-consensus"),
    # -- monitoring -----------------------------------------------------------
    AgentJudgement("monitoring tallies a personality's tasks factually", "monitor-tally"),
    AgentJudgement("a refusal is counted as refused, not as a failure", "monitor-refusal"),
    # -- system integration ---------------------------------------------------
    AgentJudgement("the composed system exposes every component", "system-compose"),
    AgentJudgement("the system answers through an injected executor", "system-run"),
)


def _echo_executor(request: Any) -> AgentResponse:
    """A faithful stand-in executor: answers from the request it was given."""
    request.token.check()
    return AgentResponse(ok=True, text=f"{request.profile.name} answered {request.question!r}.")


def _judgement_holds(judgement: AgentJudgement) -> bool:
    """Run one judgement against the real components; return whether it holds."""
    kind = judgement.kind
    try:
        return _CHECKERS[kind]()
    except Exception:  # a judgement that raises is a failed judgement, not a crash
        return False


# --------------------------------------------------------------- checkers

def _register_all() -> bool:
    registry = PersonalityRegistry()
    ids = set(registry.ids())
    return set(CHARACTER_OWNERS) <= ids and MAIN_MODEL_OWNER in ids


def _resolve_both() -> bool:
    registry = PersonalityRegistry()
    return registry.resolve("sources").personality_id == registry.resolve("jarvis").personality_id == "jarvis"


def _config_driven() -> bool:
    registry = PersonalityRegistry()
    return all(profile.config_version >= 1 for profile in registry.all_profiles())


def _distinct_ids() -> bool:
    registry = PersonalityRegistry()
    ids = [profile.personality_id for profile in registry.all_profiles()]
    return len(ids) == len(set(ids))


def _capability_ceiling() -> bool:
    for owner in (*CHARACTER_OWNERS, MAIN_MODEL_OWNER):
        if not effective_capabilities(owner).granted <= capability_ceiling(owner):
            return False
    return True


def _refuse_widen() -> bool:
    # A configuration asking for everything still gets at most the ceiling.
    everything = frozenset(AgentCapability)
    widened = effective_capabilities("jarvis", requested=everything)
    return widened.granted <= capability_ceiling("jarvis")


def _refuse_search() -> bool:
    gateway = CapabilityGateway()
    return not gateway.check("jarvis", AgentCapability.SEARCH).allowed


def _refuse_write() -> bool:
    return not decide_capability("ultron", AgentCapability.WRITE_FILES).allowed


def _runtime_run() -> bool:
    runtime = AgentRuntime(executor=_echo_executor)
    outcome = runtime.run("vision", "what is porosity?", question="what is porosity?")
    return outcome.ok and outcome.status is TaskStatus.SUCCEEDED


def _runtime_refuse() -> bool:
    runtime = AgentRuntime(executor=_echo_executor)
    outcome = runtime.run("jarvis", "search", capability=AgentCapability.SEARCH)
    return outcome.status is TaskStatus.REFUSED


def _runtime_timeout() -> bool:
    import time

    def slow(request: Any) -> AgentResponse:
        time.sleep(0.05)
        request.token.check()
        return AgentResponse(ok=True, text="late")

    runtime = AgentRuntime(executor=slow)
    outcome = runtime.run("ultron", "slow", timeout=0.01)
    return outcome.status is TaskStatus.TIMED_OUT


def _runtime_cancel() -> bool:
    token = CancellationToken()
    token.cancel()
    runtime = AgentRuntime(executor=_echo_executor)
    outcome = runtime.run("banner", "x", token=token)
    return outcome.status is TaskStatus.CANCELLED


def _runtime_defer() -> bool:
    from core.resource_governor import GovernorDecision, GovernorReason

    class _Deny:
        def request(self, _request: Any) -> GovernorDecision:
            return GovernorDecision(
                granted=False, reason=GovernorReason.LOW_MEMORY, label="agent", detail="no room"
            )

    runtime = AgentRuntime(executor=_echo_executor, governor=_Deny())
    outcome = runtime.run("hakeem", "x", memory_bytes=10**12)
    return outcome.status is TaskStatus.PENDING and bool(outcome.deferred_reason)


def _runtime_validate() -> bool:
    runtime = AgentRuntime(executor=lambda request: AgentResponse(ok=True, text=""))
    outcome = runtime.run("cloud", "x")
    return not outcome.ok and not outcome.validated


def _independent_context() -> bool:
    manager = ContextManager()
    jarvis = manager.context_for("jarvis")
    ultron = manager.context_for("ultron")
    jarvis.add("user", "jarvis only")
    return ultron.size() == 0 and jarvis.size() == 1


def _memory_own() -> bool:
    with tempfile.TemporaryDirectory() as root:
        access = PersonalityMemoryAccess("jarvis", data_root=root)
        return access.can_read("jarvis")


def _isolate_memory() -> bool:
    with tempfile.TemporaryDirectory() as root:
        access = PersonalityMemoryAccess("jarvis", data_root=root)
        try:
            access.memory_for("ultron")
            return False
        except LearningAccessError:
            return True


def _memory_shared() -> bool:
    with tempfile.TemporaryDirectory() as root:
        access = PersonalityMemoryAccess(MAIN_MODEL_OWNER, data_root=root)
        return access.can_read("jarvis")


def _isolate_private() -> bool:
    with tempfile.TemporaryDirectory() as root:
        access = PersonalityMemoryAccess(MAIN_MODEL_OWNER, data_root=root)
        return not access.can_read("natasha")


def _refuse_private_training() -> bool:
    return not decide_capability("natasha", AgentCapability.CONTRIBUTE_TRAINING).allowed


def _natasha_excluded() -> bool:
    return "natasha" not in aggregated_dataset_owners()


def _dataset_clean() -> bool:
    return aggregated_dataset_is_clean()


def _natasha_self() -> bool:
    with tempfile.TemporaryDirectory() as root:
        scope = AgentLearningScope("natasha", data_root=root)
        return scope.may_learn_from("natasha")


def _route_petroleum() -> bool:
    decision = TaskRouter().route("what is the porosity of the reservoir?")
    return decision.suitable and decision.selected == "vision"


def _route_presentation() -> bool:
    decision = TaskRouter().route("build me a slide deck")
    return decision.suitable and decision.selected == "banner"


def _route_none() -> bool:
    decision = TaskRouter().route("zzz qqq")
    return not decision.suitable and not decision.selected


def _route_override() -> bool:
    decision = TaskRouter().route("anything", override="hakeem")
    return decision.suitable and decision.selected == "hakeem" and decision.manual


def _route_user_choice() -> bool:
    decision = TaskRouter().route("anything", user_choice="natasha")
    return decision.suitable and decision.selected == "natasha" and decision.manual


def _route_sticky() -> bool:
    router = TaskRouter()
    first = router.route("what is porosity?", conversation="c1")
    follow = router.route("and again?", conversation="c1")
    return first.selected == follow.selected == "vision"


def _route_retopic() -> bool:
    router = TaskRouter()
    router.route("what is porosity?", conversation="c1")
    changed = router.route("build me a slide deck", conversation="c1")
    return changed.selected == "banner"


def _delegate_run() -> bool:
    runtime = AgentRuntime(executor=_echo_executor)
    collaboration = AgentCollaboration(runtime=runtime)
    result = collaboration.delegate(
        DelegationRequest(shared_id="s1", from_owner="jarvis", to_owner="vision", question="q")
    )
    return result.ok and result.to_owner == "vision"


def _refuse_depth() -> bool:
    collaboration = AgentCollaboration(runtime=AgentRuntime(executor=_echo_executor), max_depth=2)
    try:
        collaboration.delegate(
            DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="vision", question="q", depth=9)
        )
        return False
    except DelegationRefused:
        return True


def _refuse_budget() -> bool:
    collaboration = AgentCollaboration(
        runtime=AgentRuntime(executor=_echo_executor), max_delegations=1
    )
    collaboration.delegate(
        DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="vision", question="q")
    )
    try:
        collaboration.delegate(
            DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="ultron", question="q")
        )
        return False
    except DelegationRefused:
        return True


def _delegate_separate() -> bool:
    runtime = AgentRuntime(executor=_echo_executor)
    collaboration = AgentCollaboration(runtime=runtime)
    report = collaboration.collaborate("jarvis", "q", ("vision", "ultron"))
    return len(report.results) == 2 and {result.to_owner for result in report.results} == {"vision", "ultron"}


def _conflict_detect() -> bool:
    report = ConflictResolver().resolve(
        "q",
        (Position("vision", "a", evidence=("x",)), Position("ultron", "b", evidence=("y",))),
    )
    return report.has_conflict and not report.consensus


def _conflict_tie() -> bool:
    report = ConflictResolver().resolve(
        "q",
        (Position("vision", "a", evidence=("x",)), Position("ultron", "b", evidence=("y",))),
        policy=AggregationPolicy.EVIDENCE,
    )
    return report.selected == "" and not report.consensus


def _conflict_preserve() -> bool:
    report = ConflictResolver().resolve(
        "q", (Position("vision", "a"), Position("ultron", "b"))
    )
    return report.selected == "" and report.policy is AggregationPolicy.PRESERVE


def _conflict_consensus() -> bool:
    report = ConflictResolver().resolve("q", (Position("vision", "same"), Position("ultron", "same")))
    return report.consensus


def _monitor_tally() -> bool:
    goals = GoalManager()
    gateway = CapabilityGateway()
    runtime = AgentRuntime(executor=_echo_executor, goals=goals, gateway=gateway)
    runtime.run("vision", "q")
    from core.agents.monitoring import AgentMonitor

    activity = AgentMonitor(goals=goals, gateway=gateway).activity_for("vision")
    return activity.total == 1 and activity.succeeded == 1


def _monitor_refusal() -> bool:
    goals = GoalManager()
    gateway = CapabilityGateway()
    runtime = AgentRuntime(executor=_echo_executor, goals=goals, gateway=gateway)
    runtime.run("jarvis", "search", capability=AgentCapability.SEARCH)
    from core.agents.monitoring import AgentMonitor

    activity = AgentMonitor(goals=goals, gateway=gateway).activity_for("jarvis")
    return activity.refused == 1 and activity.failed == 0


def _system_compose() -> bool:
    system = AgentSystem(executor=_echo_executor)
    return all(
        [
            system.registry,
            system.runtime,
            system.router,
            system.collaboration,
            system.conflicts,
            system.monitor,
            system.bus,
        ]
    )


def _system_run() -> bool:
    system = AgentSystem(executor=_echo_executor)
    outcome = system.run("vision", "what is porosity?")
    return outcome.ok


_CHECKERS: dict[str, Any] = {
    "register-all": _register_all,
    "resolve-both": _resolve_both,
    "config-driven": _config_driven,
    "distinct-ids": _distinct_ids,
    "capability-ceiling": _capability_ceiling,
    "refuse-widen": _refuse_widen,
    "refuse-search": _refuse_search,
    "refuse-write": _refuse_write,
    "runtime-run": _runtime_run,
    "runtime-refuse": _runtime_refuse,
    "runtime-timeout": _runtime_timeout,
    "runtime-cancel": _runtime_cancel,
    "runtime-defer": _runtime_defer,
    "runtime-validate": _runtime_validate,
    "independent-context": _independent_context,
    "memory-own": _memory_own,
    "isolate-memory": _isolate_memory,
    "memory-shared": _memory_shared,
    "isolate-private": _isolate_private,
    "refuse-private-training": _refuse_private_training,
    "natasha-excluded": _natasha_excluded,
    "dataset-clean": _dataset_clean,
    "natasha-self": _natasha_self,
    "route-petroleum": _route_petroleum,
    "route-presentation": _route_presentation,
    "route-none": _route_none,
    "route-override": _route_override,
    "route-user-choice": _route_user_choice,
    "route-sticky": _route_sticky,
    "route-retopic": _route_retopic,
    "delegate-run": _delegate_run,
    "refuse-depth": _refuse_depth,
    "refuse-budget": _refuse_budget,
    "delegate-separate": _delegate_separate,
    "conflict-detect": _conflict_detect,
    "conflict-tie": _conflict_tie,
    "conflict-preserve": _conflict_preserve,
    "conflict-consensus": _conflict_consensus,
    "monitor-tally": _monitor_tally,
    "monitor-refusal": _monitor_refusal,
    "system-compose": _system_compose,
    "system-run": _system_run,
}


def personality_agents_accuracy(
    judgements: tuple[AgentJudgement, ...] = AGENT_JUDGEMENTS,
) -> float:
    """The share of held-out agent judgements the real layer satisfies.

    A judgement counts only when the **real** component does exactly what it must
    -- a personality's capability set stays within its ceiling, a refused action
    is refused, a private memory is unreachable, an unmatched task reports no
    suitable personality, a delegation respects its bounds, a conflict is reported
    rather than voted away. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "AGENT_JUDGEMENTS",
    "AgentJudgement",
    "personality_agents_accuracy",
]
