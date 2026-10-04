"""Integration checks for the progressive environment ladder.

These verify the safety contract end to end: the agent is promoted from a
replayable symbolic simulation to a task-scoped sandbox, and external tool
execution stays impossible until a permission policy, an audit log, and a human
approval are all in place.
"""

import pytest

from core.action import Action
from core.permissions import Capability, PermissionPolicy
from core.progressive_environment import EnvironmentTier, ProgressiveEnvironment
from core.sandbox_environment import SandboxEnvironment
from core.sandbox_tasks import task_by_name
from core.symbolic_simulation import SymbolicSimulation

pytestmark = pytest.mark.integration


def _water() -> Action:
    return Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))


def test_ladder_runs_symbolic_then_sandbox_then_gates_external():
    ladder = ProgressiveEnvironment()
    runtime = ladder.runtime

    # Tier 1: replayable symbolic simulation.
    symbolic = SymbolicSimulation(frozenset({"dry_soil"}), max_steps=2)
    outcome = runtime.run_symbolic(symbolic, [_water()])
    assert outcome.allowed
    assert SymbolicSimulation.replay(outcome.result.trace).final_state.facts == frozenset(
        {"dry_soil", "wet_soil"}
    )
    ladder.mark_symbolic_passed()

    # Tier 2: task-scoped sandbox.
    environment = SandboxEnvironment(task_by_name("wet-garden"))
    sandbox = runtime.run_sandbox(environment, [_water()], lambda state: "wet_soil" in state.facts)
    assert sandbox.allowed
    assert sandbox.result.success
    ladder.mark_sandbox_passed()

    # Tier 3: external is still refused without policy + approval.
    external = runtime.run_external("shell", lambda: "should not run")
    assert not external.allowed
    assert not ladder.can_run_external()
    assert ladder.readiness(EnvironmentTier.EXTERNAL).available is False


def test_external_execution_impossible_until_every_gate_is_satisfied():
    ladder = ProgressiveEnvironment()
    ladder.mark_symbolic_passed()
    ladder.mark_sandbox_passed()

    assert not ladder.can_run_external()

    approval = ladder.approvals.request("search", requested_by="agent")
    approval.approve("human-reviewer")
    assert not ladder.can_run_external()  # policy missing

    ladder.policy = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}),
        allowed_external_tools=frozenset({"search"}),
    )
    assert ladder.can_run_external()

    runtime = ladder.runtime
    allowed = runtime.run_external("search", lambda: "facts")
    assert allowed.allowed
    assert allowed.result == "facts"
    # The decision is on the shared, append-only audit trail.
    assert runtime.audit.size == 1
    assert runtime.audit.records[0].outcome.value == "allowed"
