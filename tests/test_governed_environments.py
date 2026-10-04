import pytest

from core.action import Action
from core.audit_log import AuditLog
from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy
from core.progressive_environment import EnvironmentTier, ProgressiveEnvironment
from core.sandbox_environment import SandboxEnvironment
from core.sandbox_runtime import GovernedRuntime
from core.sandbox_tasks import task_by_name
from core.symbolic_simulation import SymbolicSimulation


def _water() -> Action:
    return Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))


def test_default_policy_denies_external_tools():
    policy = PermissionPolicy()
    assert policy.allows(Capability.SYMBOLIC_SIMULATION)
    assert policy.allows(Capability.SANDBOX_TASK)
    assert not policy.allows(Capability.EXTERNAL_TOOL, "shell")


def test_external_tool_needs_allow_list_entry():
    policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}), allowed_external_tools=frozenset({"search"})
    )
    assert policy.allows(Capability.EXTERNAL_TOOL, "search")
    assert not policy.allows(Capability.EXTERNAL_TOOL, "shell")


def test_runtime_audits_symbolic_and_sandbox_runs():
    runtime = GovernedRuntime()
    symbolic = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=2)
    runtime.run_symbolic(symbolic, [_water()])

    environment = SandboxEnvironment(task_by_name("wet-garden"))
    runtime.run_sandbox(environment, [_water()], lambda state: "wet_soil" in state.facts)

    assert runtime.audit.size == 2
    assert all(record.outcome.value == "allowed" for record in runtime.audit.records)


def test_runtime_refuses_external_tool_without_policy_and_approval():
    runtime = GovernedRuntime()
    calls: list[str] = []

    outcome = runtime.run_external("shell", lambda: calls.append("ran"))

    assert not outcome.allowed
    assert calls == []
    assert runtime.audit.denials()
    assert "denied" in outcome.reason.lower()


def test_external_tool_requires_human_approval_even_when_policy_allows():
    policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}), allowed_external_tools=frozenset({"search"})
    )
    runtime = GovernedRuntime(policy=policy)

    blocked = runtime.run_external("search", lambda: "result")
    assert not blocked.allowed
    assert "approval" in blocked.reason.lower()

    request = runtime.approvals.request("search", requested_by="agent", justification="need facts")
    request.approve("human-reviewer")

    allowed = runtime.run_external("search", lambda: "result")
    assert allowed.allowed
    assert allowed.result == "result"


def test_requester_cannot_self_approve():
    gate = HumanApprovalGate()
    request = gate.request("search", requested_by="agent")

    with pytest.raises(ValueError):
        request.approve("agent")


def test_rejected_approval_blocks_tool():
    policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}), allowed_external_tools=frozenset({"search"})
    )
    runtime = GovernedRuntime(policy=policy)
    request = runtime.approvals.request("search", requested_by="agent")
    request.reject("human-reviewer")

    outcome = runtime.run_external("search", lambda: "result")
    assert not outcome.allowed


def test_expired_approval_blocks_tool():
    policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}), allowed_external_tools=frozenset({"search"})
    )
    runtime = GovernedRuntime(policy=policy)
    request = runtime.approvals.request("search", requested_by="agent", ttl_seconds=0.0001)
    request.approve("human-reviewer")

    import time

    time.sleep(0.01)
    outcome = runtime.run_external("search", lambda: "result")
    assert not outcome.allowed


def test_audit_log_is_append_only_and_reports_denials():
    log = AuditLog()
    policy = PermissionPolicy()
    log.record_decision(policy.decide(Capability.EXTERNAL_TOOL, "shell"))
    log.record_decision(policy.decide(Capability.SYMBOLIC_SIMULATION))

    assert log.size == 2
    assert len(log.denials()) == 1
    assert len(log.for_capability(Capability.SYMBOLIC_SIMULATION)) == 1


def test_progressive_ladder_unlocks_sandbox_after_symbolic_pass():
    ladder = ProgressiveEnvironment()
    assert not ladder.readiness(EnvironmentTier.SANDBOX).available

    ladder.mark_symbolic_passed()

    assert ladder.readiness(EnvironmentTier.SANDBOX).available
    assert ladder.current_tier is EnvironmentTier.SANDBOX


def test_progressive_ladder_gates_external_on_all_requirements():
    ladder = ProgressiveEnvironment()
    assert not ladder.can_run_external()

    ladder.mark_symbolic_passed()
    ladder.mark_sandbox_passed()
    assert not ladder.can_run_external()  # still no policy capability + approval

    request = ladder.approvals.request("search", requested_by="agent")
    request.approve("human-reviewer")
    assert not ladder.can_run_external()  # policy still denies external

    ladder.policy = PermissionPolicy(allowed=frozenset({Capability.EXTERNAL_TOOL}))
    assert ladder.can_run_external()
