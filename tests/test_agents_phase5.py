"""Phase 5: the seven personalities as independent agents in one system.

Covers the personality registry, the capability model and its ceiling, the runtime
(execution, refusal, timeout, cancellation, deferral, validation), the independent
working context, the memory access layer and the learning scope, the task router,
bounded collaboration, conflict resolution, monitoring, the composed system, and --
critically -- **Natasha's isolation and training-data exclusion**.

Everything here runs with an injected executor and a temp data root, so no model,
display, or network is needed.
"""

from __future__ import annotations

import time

import pytest
from core.agents.capabilities import (
    AgentCapability,
    CapabilityDenied,
    CapabilityGateway,
    CapabilitySet,
    capability_ceiling,
    decide_capability,
    effective_capabilities,
)
from core.agents.collaboration import (
    AgentCollaboration,
    CollaborationReport,
    DelegationRefused,
    DelegationRequest,
)
from core.agents.communication import BROADCAST, AgentMessage, MessageBus, MessageRefused
from core.agents.conflict import (
    AggregationPolicy,
    ConflictResolver,
    Position,
    compare_positions,
)
from core.agents.context import ContextManager, WorkingContext
from core.agents.goals import GoalManager, TaskStatus
from core.agents.learning_scope import (
    AgentLearningScope,
    aggregated_dataset_is_clean,
    aggregated_dataset_owners,
    learning_scope,
)
from core.agents.memory_access import PersonalityMemoryAccess, memory_access
from core.agents.monitoring import AgentMonitor
from core.agents.registry import PersonalityRegistry, profile_for
from core.agents.routing import TaskRouter, TaskType, classify_task
from core.agents.runtime import (
    AgentResponse,
    AgentRuntime,
    CancellationToken,
    default_validator,
)
from core.agents.system import AgentSystem
from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    SHAREABLE_OWNERS,
    LearningAccessError,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _echo(request):
    request.token.check()
    return AgentResponse(ok=True, text=f"{request.profile.name} answered.")


# --- the personality registry ---------------------------------------------

def test_registry_lists_the_seven_and_the_project_model() -> None:
    registry = PersonalityRegistry()
    assert set(CHARACTER_OWNERS) <= set(registry.ids())
    assert MAIN_MODEL_OWNER in registry.ids()


def test_registry_resolves_both_vocabularies_to_one_profile() -> None:
    registry = PersonalityRegistry()
    assert registry.resolve("sources").personality_id == registry.resolve("jarvis").personality_id
    assert registry.resolve("model").personality_id == MAIN_MODEL_OWNER


def test_profile_is_configuration_driven() -> None:
    registry = PersonalityRegistry()
    profile = registry.resolve("vision")
    assert profile.config_version >= 1
    assert profile.role
    assert profile.capabilities.granted


def test_unknown_key_falls_back_to_the_project_model() -> None:
    assert profile_for("no-such-mouth").personality_id == MAIN_MODEL_OWNER


def test_profiles_are_distinct() -> None:
    registry = PersonalityRegistry()
    ids = [profile.personality_id for profile in registry.all_profiles()]
    assert len(ids) == len(set(ids))


def test_reload_drops_the_cache() -> None:
    registry = PersonalityRegistry()
    first = registry.resolve("jarvis")
    registry.reload("jarvis")
    assert registry.resolve("jarvis") == first  # rebuilt, same facts


# --- capabilities ----------------------------------------------------------

def test_every_personality_stays_within_its_ceiling() -> None:
    for owner in (*CHARACTER_OWNERS, MAIN_MODEL_OWNER):
        assert effective_capabilities(owner).granted <= capability_ceiling(owner)


def test_a_configuration_cannot_widen_a_capability_set() -> None:
    everything = frozenset(AgentCapability)
    widened = effective_capabilities("jarvis", requested=everything)
    assert widened.granted <= capability_ceiling("jarvis")
    assert AgentCapability.SEARCH not in widened.granted
    assert AgentCapability.WRITE_FILES not in widened.granted


def test_search_is_refused_by_default() -> None:
    assert not decide_capability("jarvis", AgentCapability.SEARCH).allowed


def test_file_writes_are_refused_by_default() -> None:
    assert not decide_capability("ultron", AgentCapability.WRITE_FILES).allowed


def test_private_character_cannot_contribute_to_training() -> None:
    assert not decide_capability("natasha", AgentCapability.CONTRIBUTE_TRAINING).allowed
    assert AgentCapability.CONTRIBUTE_TRAINING not in capability_ceiling("natasha")


def test_unknown_owner_gets_an_empty_ceiling() -> None:
    assert capability_ceiling("nobody") == frozenset()


def test_gateway_records_every_decision() -> None:
    gateway = CapabilityGateway()
    gateway.check("jarvis", AgentCapability.ANSWER)
    gateway.check("jarvis", AgentCapability.SEARCH)
    assert len(gateway.decisions) == 2
    assert len(gateway.refusals()) == 1


def test_gateway_require_raises() -> None:
    with pytest.raises(CapabilityDenied):
        CapabilityGateway().require("jarvis", AgentCapability.SEARCH)


def test_capability_set_describe_and_primitive() -> None:
    capabilities = CapabilitySet("jarvis", frozenset({AgentCapability.ANSWER}))
    assert "answer" in capabilities.describe()
    assert capabilities.to_primitive()["granted"] == ["answer"]


# --- the runtime -----------------------------------------------------------

def test_runtime_runs_a_task() -> None:
    outcome = AgentRuntime(executor=_echo).run("vision", "q", question="q")
    assert outcome.ok and outcome.status is TaskStatus.SUCCEEDED


def test_runtime_refuses_a_capability_it_does_not_hold() -> None:
    outcome = AgentRuntime(executor=_echo).run("jarvis", "q", capability=AgentCapability.SEARCH)
    assert outcome.status is TaskStatus.REFUSED
    assert not outcome.ok


def test_runtime_times_out() -> None:
    def slow(request):
        time.sleep(0.05)
        request.token.check()
        return AgentResponse(ok=True, text="late")

    outcome = AgentRuntime(executor=slow).run("ultron", "q", timeout=0.01)
    assert outcome.status is TaskStatus.TIMED_OUT


def test_runtime_cancels() -> None:
    token = CancellationToken()
    token.cancel()
    outcome = AgentRuntime(executor=_echo).run("banner", "q", token=token)
    assert outcome.status is TaskStatus.CANCELLED


def test_runtime_defers_when_the_governor_declines() -> None:
    from core.resource_governor import GovernorDecision, GovernorReason

    class _Deny:
        def request(self, _request):
            return GovernorDecision(
                granted=False, reason=GovernorReason.LOW_MEMORY, label="agent", detail="no room"
            )

    outcome = AgentRuntime(executor=_echo, governor=_Deny()).run("hakeem", "q", memory_bytes=10**12)
    assert outcome.status is TaskStatus.PENDING
    assert outcome.deferred_reason


def test_runtime_rejects_an_empty_result() -> None:
    outcome = AgentRuntime(executor=lambda request: AgentResponse(ok=True, text="")).run("cloud", "q")
    assert not outcome.ok and not outcome.validated


def test_runtime_reports_a_crash_as_a_failed_task() -> None:
    def boom(request):
        raise RuntimeError("kaboom")

    outcome = AgentRuntime(executor=boom).run("vision", "q")
    assert outcome.status is TaskStatus.FAILED
    assert "kaboom" in outcome.response.error


def test_runtime_without_an_executor_fails_honestly() -> None:
    outcome = AgentRuntime(executor=None).run("vision", "q")
    assert outcome.status is TaskStatus.FAILED


def test_runtime_records_a_task() -> None:
    runtime = AgentRuntime(executor=_echo)
    outcome = runtime.run("vision", "q")
    assert runtime.goals.get_task(outcome.task.task_id).status is TaskStatus.SUCCEEDED


def test_runtime_publishes_events() -> None:
    seen: list[str] = []

    class _Bus:
        def publish(self, name, data, source=""):
            seen.append(name)

    AgentRuntime(executor=_echo, events=_Bus()).run("vision", "q")
    assert "agent.task.started" in seen and "agent.task.finished" in seen


def test_default_validator_rejects_an_empty_success() -> None:
    assert not default_validator(AgentResponse(ok=True, text="")).ok


# --- independent working context ------------------------------------------

def test_two_personalities_keep_independent_contexts() -> None:
    manager = ContextManager()
    jarvis = manager.context_for("jarvis")
    ultron = manager.context_for("ultron")
    jarvis.add("user", "only jarvis")
    assert jarvis.size() == 1 and ultron.size() == 0


def test_context_is_bounded() -> None:
    context = WorkingContext("jarvis", limit=3)
    for index in range(10):
        context.add("user", str(index))
    assert context.size() == 3
    assert context.recent()[-1].text == "9"


def test_context_owner_is_normalised() -> None:
    assert ContextManager().context_for("sources").owner == "jarvis"


def test_context_rejects_an_unknown_owner() -> None:
    with pytest.raises(ValueError):
        ContextManager().context_for("nobody")


def test_context_reset_clears_it() -> None:
    manager = ContextManager()
    context = manager.context_for("jarvis")
    context.add("user", "x")
    manager.reset("jarvis")
    assert context.size() == 0


# --- goals and tasks -------------------------------------------------------

def test_goal_manager_tracks_a_goal_and_its_tasks() -> None:
    goals = GoalManager()
    goal = goals.create_goal("vision", "answer well")
    task = goals.create_task("vision", "do it", goal_id=goal.goal_id)
    assert task in goals.tasks_for("vision")
    assert goals.goals_for("vision") == (goal,)


def test_goal_manager_refuses_a_deep_delegation() -> None:
    with pytest.raises(ValueError):
        GoalManager(max_depth=1).create_task("vision", "deep", depth=9)


def test_task_status_terminal() -> None:
    assert TaskStatus.SUCCEEDED.terminal and not TaskStatus.RUNNING.terminal


def test_task_tree_children() -> None:
    goals = GoalManager()
    parent = goals.create_task("jarvis", "parent")
    child = goals.create_task("vision", "child", parent_task=parent.task_id, delegated_by="jarvis")
    assert child in goals.children_of(parent.task_id)


# --- memory access and learning scope -------------------------------------

def test_a_personality_reads_its_own_memory(data_root) -> None:
    assert PersonalityMemoryAccess("jarvis", data_root=data_root).can_read("jarvis")


def test_a_personality_cannot_read_another_characters_memory(data_root) -> None:
    access = PersonalityMemoryAccess("jarvis", data_root=data_root)
    assert not access.can_read("ultron")
    with pytest.raises(LearningAccessError):
        access.memory_for("ultron")


def test_the_main_model_reads_a_shareable_character(data_root) -> None:
    access = PersonalityMemoryAccess(MAIN_MODEL_OWNER, data_root=data_root)
    assert access.can_read("jarvis")


def test_the_main_model_is_refused_the_private_character(data_root) -> None:
    access = PersonalityMemoryAccess(MAIN_MODEL_OWNER, data_root=data_root)
    assert not access.can_read("natasha")
    with pytest.raises(LearningAccessError):
        access.learning_for("natasha")


def test_memory_access_rejects_an_unknown_reader(data_root) -> None:
    with pytest.raises(LearningAccessError):
        PersonalityMemoryAccess("nobody", data_root=data_root)


def test_memory_access_helper(data_root) -> None:
    assert memory_access("sources", data_root=data_root).reader == "jarvis"


def test_learning_scope_of_a_character_is_its_own() -> None:
    scope = learning_scope("jarvis")
    assert scope.learns_from == ("jarvis",) and scope.contributes


def test_learning_scope_of_the_main_model_includes_shareable_characters() -> None:
    scope = learning_scope(MAIN_MODEL_OWNER)
    assert set(SHAREABLE_OWNERS) <= set(scope.learns_from)
    assert "natasha" not in scope.learns_from


def test_learning_scope_of_a_private_character_does_not_contribute() -> None:
    scope = learning_scope("natasha")
    assert not scope.contributes
    assert "private" in scope.reason


def test_agent_learning_scope_refuses_a_cross_character_pull(data_root) -> None:
    scope = AgentLearningScope("jarvis", data_root=data_root)
    with pytest.raises(LearningAccessError):
        scope.learn_from("natasha")


# --- Natasha's isolation and training-data exclusion ----------------------

def test_natasha_is_absent_from_the_aggregated_dataset() -> None:
    assert "natasha" not in aggregated_dataset_owners()
    assert aggregated_dataset_is_clean()


def test_natasha_cannot_reach_the_aggregated_dataset_even_when_asked() -> None:
    # Even a caller that explicitly asks for private owners cannot make her
    # *contribute*: the capability ceiling refuses it.
    assert not decide_capability("natasha", AgentCapability.CONTRIBUTE_TRAINING).allowed


def test_natasha_learning_never_enters_the_training_corpus(data_root) -> None:
    from core.character_learning import (
        CharacterLearning,
        character_learning_sentences,
        training_keys,
    )

    CharacterLearning("jarvis", root=data_root).add_fact("a jarvis fact")
    CharacterLearning("natasha", root=data_root).add_fact("a private fact")

    sentences = character_learning_sentences(root=data_root)
    assert any("a jarvis fact" in sentence for sentence in sentences)
    assert not any("private fact" in sentence for sentence in sentences)
    assert "natasha" not in training_keys()


def test_natasha_learning_units_exclude_her(data_root) -> None:
    from core.character_learning import CharacterLearning, character_learning_units

    CharacterLearning("natasha", root=data_root).add_fact("a private fact")
    assert not any("private fact" in unit for unit in character_learning_units(root=data_root))


def test_natasha_still_reads_her_own_learning(data_root) -> None:
    scope = AgentLearningScope("natasha", data_root=data_root)
    assert scope.may_learn_from("natasha")


def test_the_corpus_build_omits_natasha(data_root) -> None:
    from core.character_learning import CharacterLearning
    from ml.llm import corpus

    CharacterLearning("natasha", root=data_root).add_fact("a private fact")
    text = corpus.build_corpus(include_characters=True)
    assert "a private fact" not in text


# --- routing ---------------------------------------------------------------

def test_classify_task_reads_a_hint() -> None:
    task_type, hint = classify_task("what is the porosity of the reservoir?")
    assert task_type is TaskType.PETROLEUM and "porosity" in hint


def test_route_petroleum_to_vision() -> None:
    decision = TaskRouter().route("what is the porosity of the reservoir?")
    assert decision.suitable and decision.selected == "vision"


def test_route_presentation_to_banner() -> None:
    assert TaskRouter().route("build me a slide deck").selected == "banner"


def test_route_religion_to_hakeem() -> None:
    assert TaskRouter().route("tell me about the history of religion").selected == "hakeem"


def test_route_reports_no_suitable_personality() -> None:
    decision = TaskRouter().route("zzz qqq")
    assert not decision.suitable and decision.selected == ""


def test_route_manual_override_wins() -> None:
    decision = TaskRouter().route("anything", override="hakeem")
    assert decision.suitable and decision.selected == "hakeem" and decision.manual


def test_route_user_choice_wins() -> None:
    decision = TaskRouter().route("anything", user_choice="natasha")
    assert decision.selected == "natasha" and decision.manual


def test_route_override_of_a_non_character_is_refused() -> None:
    decision = TaskRouter().route("anything", override=MAIN_MODEL_OWNER)
    assert not decision.suitable and decision.manual


def test_route_keeps_the_personality_for_a_follow_up() -> None:
    router = TaskRouter()
    first = router.route("what is porosity?", conversation="c1")
    follow = router.route("and again?", conversation="c1")
    assert first.selected == follow.selected == "vision"
    assert "kept" in follow.reason


def test_route_changes_on_a_clearly_different_topic() -> None:
    router = TaskRouter()
    router.route("what is porosity?", conversation="c1")
    assert router.route("build me a slide deck", conversation="c1").selected == "banner"


def test_router_records_its_decisions() -> None:
    router = TaskRouter()
    router.route("what is porosity?")
    assert len(router.decisions()) == 1


def test_router_decision_primitive() -> None:
    decision = TaskRouter().route("what is porosity?")
    assert decision.to_primitive()["selected"] == "vision"


# --- collaboration ---------------------------------------------------------

def test_delegation_runs_the_delegate() -> None:
    collaboration = AgentCollaboration(runtime=AgentRuntime(executor=_echo))
    result = collaboration.delegate(
        DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="vision", question="q")
    )
    assert result.ok and result.to_owner == "vision"


def test_delegation_refuses_an_excessive_depth() -> None:
    collaboration = AgentCollaboration(runtime=AgentRuntime(executor=_echo), max_depth=2)
    with pytest.raises(DelegationRefused):
        collaboration.delegate(
            DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="vision", question="q", depth=9)
        )


def test_delegation_refuses_a_non_character_target() -> None:
    collaboration = AgentCollaboration(runtime=AgentRuntime(executor=_echo))
    with pytest.raises(DelegationRefused):
        collaboration.delegate(
            DelegationRequest(shared_id="s", from_owner="jarvis", to_owner=MAIN_MODEL_OWNER, question="q")
        )


def test_delegation_budget_is_bounded() -> None:
    collaboration = AgentCollaboration(runtime=AgentRuntime(executor=_echo), max_delegations=1)
    collaboration.delegate(
        DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="vision", question="q")
    )
    with pytest.raises(DelegationRefused):
        collaboration.delegate(
            DelegationRequest(shared_id="s", from_owner="jarvis", to_owner="ultron", question="q")
        )
    assert collaboration.budget_used("s") == 1


def test_collaboration_keeps_each_answer_separate() -> None:
    report = AgentCollaboration(runtime=AgentRuntime(executor=_echo)).collaborate(
        "jarvis", "q", ("vision", "ultron")
    )
    assert isinstance(report, CollaborationReport)
    assert {result.to_owner for result in report.results} == {"vision", "ultron"}


# --- conflict resolution ---------------------------------------------------

def test_conflict_detects_a_disagreement() -> None:
    report = ConflictResolver().resolve(
        "q", (Position("vision", "a", evidence=("x",)), Position("ultron", "b", evidence=("y",)))
    )
    assert report.has_conflict and not report.consensus
    assert report.differences


def test_default_policy_names_no_winner() -> None:
    report = ConflictResolver().resolve("q", (Position("vision", "a"), Position("ultron", "b")))
    assert report.selected == "" and report.policy is AggregationPolicy.PRESERVE


def test_evidence_policy_does_not_decide_a_near_tie() -> None:
    report = ConflictResolver().resolve(
        "q",
        (Position("vision", "a", evidence=("x",)), Position("ultron", "b", evidence=("y",))),
        policy=AggregationPolicy.EVIDENCE,
    )
    assert report.selected == ""


def test_evidence_policy_selects_the_stronger_case() -> None:
    report = ConflictResolver().resolve(
        "q",
        (Position("vision", "a", evidence=("x", "y", "z")), Position("ultron", "b", evidence=("x",))),
        policy=AggregationPolicy.EVIDENCE,
    )
    assert report.selected == "vision"


def test_conflict_reports_consensus() -> None:
    report = ConflictResolver().resolve("q", (Position("vision", "same"), Position("ultron", "same")))
    assert report.consensus


def test_conflict_never_discards_a_position() -> None:
    positions = (Position("vision", "a"), Position("ultron", "b"))
    report = ConflictResolver().resolve("q", positions)
    assert report.positions == positions


def test_compare_positions_names_the_dimensions() -> None:
    differences = compare_positions(
        (Position("vision", "a", evidence=("x",)), Position("ultron", "b"))
    )
    dimensions = {difference.dimension for difference in differences}
    assert "answer" in dimensions and "evidence" in dimensions


# --- communication ---------------------------------------------------------

def test_message_bus_delivers_and_logs() -> None:
    bus = MessageBus()
    bus.send("jarvis", "vision", "note", "hello")
    assert bus.inbox("vision")[0].body == "hello"


def test_message_bus_broadcast_reaches_everyone() -> None:
    bus = MessageBus()
    bus.send("jarvis", BROADCAST, "note", "hello all")
    assert bus.inbox("ultron")


def test_message_bus_refuses_an_unknown_recipient() -> None:
    with pytest.raises(MessageRefused):
        MessageBus().send("jarvis", "nobody", "note", "hello")


def test_message_needs_a_body() -> None:
    with pytest.raises(ValueError):
        AgentMessage(sender="jarvis", recipient="vision", kind="note", body="  ")


def test_message_thread() -> None:
    bus = MessageBus()
    bus.send("jarvis", "vision", "note", "one", shared_id="t1")
    bus.send("vision", "jarvis", "reply", "two", shared_id="t1")
    assert len(bus.thread("t1")) == 2


# --- monitoring ------------------------------------------------------------

def test_monitoring_tallies_activity() -> None:
    goals = GoalManager()
    gateway = CapabilityGateway()
    runtime = AgentRuntime(executor=_echo, goals=goals, gateway=gateway)
    runtime.run("vision", "q")
    activity = AgentMonitor(goals=goals, gateway=gateway).activity_for("vision")
    assert activity.total == 1 and activity.succeeded == 1 and activity.success_rate == 1.0


def test_monitoring_counts_a_refusal_as_refused_not_failed() -> None:
    goals = GoalManager()
    gateway = CapabilityGateway()
    runtime = AgentRuntime(executor=_echo, goals=goals, gateway=gateway)
    runtime.run("jarvis", "q", capability=AgentCapability.SEARCH)
    activity = AgentMonitor(goals=goals, gateway=gateway).activity_for("jarvis")
    assert activity.refused == 1 and activity.failed == 0
    assert activity.success_rate == 0.0  # nothing attempted


def test_monitoring_report_covers_every_personality() -> None:
    report = AgentMonitor().report()
    assert len(report.activities) == len(PersonalityRegistry().ids())


def test_monitoring_for_an_owner_with_no_tasks_is_a_zero_tally() -> None:
    assert AgentMonitor().report().for_owner("vision").total == 0


# --- the composed system ---------------------------------------------------

def test_system_exposes_every_component(data_root) -> None:
    system = AgentSystem(data_root=data_root, executor=_echo)
    assert all(
        [system.registry, system.runtime, system.router, system.collaboration,
         system.conflicts, system.monitor, system.bus]
    )


def test_system_runs_through_an_injected_executor(data_root) -> None:
    system = AgentSystem(data_root=data_root, executor=_echo)
    assert system.run("vision", "what is porosity?").ok


def test_system_memory_access_is_isolated(data_root) -> None:
    system = AgentSystem(data_root=data_root, executor=_echo)
    assert system.memory_access("jarvis").can_read("jarvis")
    assert not system.memory_access("jarvis").can_read("natasha")


def test_system_status_is_readable(data_root) -> None:
    system = AgentSystem(data_root=data_root, executor=_echo)
    assert "Agent runtime" in system.status()


def test_system_profiles(data_root) -> None:
    system = AgentSystem(data_root=data_root, executor=_echo)
    assert system.profile("vision").name == "Vision"


# --- the held-out gate -----------------------------------------------------

def test_gate_is_one_on_the_shipped_wiring() -> None:
    from core.agents.gate import personality_agents_accuracy

    assert personality_agents_accuracy() == 1.0


def test_gate_is_zero_for_an_empty_catalogue() -> None:
    from core.agents.gate import personality_agents_accuracy

    assert personality_agents_accuracy(()) == 0.0


def test_gate_discriminates_when_a_guard_is_removed(monkeypatch) -> None:
    """A regression that let a private character contribute to training fails the gate."""
    import core.agents.gate as gate
    from core.agents.capabilities import CapabilitySet

    real = gate.effective_capabilities

    def leaky(owner, requested=None):
        result = real(owner, requested=requested)
        return CapabilitySet(owner, result.granted | {AgentCapability.CONTRIBUTE_TRAINING})

    monkeypatch.setattr(gate, "effective_capabilities", leaky)
    assert gate.personality_agents_accuracy() < 1.0
