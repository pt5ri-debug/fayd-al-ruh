"""Tests for the sandboxed tool layer and agent coordination."""

from __future__ import annotations

import pytest

from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy
from core.tool_agent import ToolAgent, ToolCall, ToolPlan, plan_from_calls
from core.tools import ToolError, ToolRegistry, ToolResult, ToolSpec, build_sandbox


def _sandbox(tmp_path, **kwargs):
    return build_sandbox(tmp_path, **kwargs)


# --- registry -------------------------------------------------------------


def test_registry_registers_and_rejects_duplicates():
    registry = ToolRegistry()
    spec = ToolSpec("echo", "echo text", lambda text: text)
    registry.register(spec)
    assert registry.names() == ("echo",)
    assert registry.get("echo") is spec
    with pytest.raises(ValueError):
        registry.register(spec)
    with pytest.raises(ValueError):
        ToolSpec("", "x", lambda: "")
    with pytest.raises(ValueError):
        ToolSpec("x", "", lambda: "")


# --- file tools -----------------------------------------------------------


def test_read_and_write_inside_the_root(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=False)
    write = sandbox.run("write_file", relative="notes/a.txt", content="مرحبا")
    assert write.ok
    read = sandbox.run("read_file", relative="notes/a.txt")
    assert read.ok and read.output == "مرحبا"


def test_read_only_sandbox_refuses_mutation(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=True)
    result = sandbox.run("write_file", relative="a.txt", content="x")
    assert not result.ok
    assert "read-only" in result.error


def test_list_and_search(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("النفاذية مهمة\n", encoding="utf-8")
    sandbox = _sandbox(tmp_path, read_only=True)

    listing = sandbox.run("list_dir", relative="data")
    assert listing.ok and "a.txt" in listing.output

    found = sandbox.run("search", query="النفاذية", relative=".")
    assert found.ok and "a.txt:1" in found.output

    missing = sandbox.run("search", query="لا يوجد", relative=".")
    assert missing.ok and missing.output == "(no matches)"


def test_path_escape_is_refused(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=True)
    for relative in ("../secret.txt", "a/../../outside.txt", "/etc/passwd"):
        result = sandbox.run("read_file", relative=relative)
        assert not result.ok


def test_resolve_path_rejects_empty(tmp_path):
    sandbox = _sandbox(tmp_path)
    with pytest.raises(ToolError):
        sandbox.resolve_path("   ")


def test_suffix_allow_list(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    sandbox = _sandbox(tmp_path, read_only=True, allowed_suffixes=frozenset({".txt"}))
    assert sandbox.run("read_file", relative="a.txt").ok
    assert not sandbox.run("read_file", relative="b.bin").ok


def test_file_size_limit(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
    sandbox = build_sandbox(tmp_path, read_only=True)
    sandbox.policy = type(sandbox.policy)(root=tmp_path, allowed_tools=sandbox.policy.allowed_tools, max_file_bytes=10)
    assert not sandbox.run("read_file", relative="big.txt").ok


# --- policy gates ---------------------------------------------------------


def test_tool_not_on_allow_list_is_refused(tmp_path):
    sandbox = _sandbox(tmp_path, allowed_tools={"calculator"})
    assert sandbox.run("calculator", expression="1+1").ok
    assert not sandbox.run("read_file", relative="a.txt").ok


def test_unknown_tool_is_refused(tmp_path):
    assert not _sandbox(tmp_path).run("nope").ok


def test_permission_policy_denies_capability(tmp_path):
    permissions = PermissionPolicy(allowed=frozenset({Capability.SANDBOX_TASK}))
    sandbox = _sandbox(tmp_path, permissions=permissions)
    # calculator uses SYMBOLIC_SIMULATION, which this policy does not grant.
    assert not sandbox.run("calculator", expression="1+1").ok


def test_external_tool_needs_human_approval(tmp_path):
    registry = ToolRegistry()
    registry.register(
        ToolSpec("fetch", "fetch a URL", lambda url: "ok", Capability.EXTERNAL_TOOL)
    )
    from core.tools import SandboxPolicy

    policy = SandboxPolicy(root=tmp_path, allowed_tools=frozenset({"fetch"}), read_only=False)
    permissions = PermissionPolicy(
        allowed=frozenset({Capability.EXTERNAL_TOOL}),
        allowed_external_tools=frozenset({"fetch"}),
    )
    approvals = HumanApprovalGate()
    from core.tools import ToolSandbox

    sandbox = ToolSandbox(registry, policy, permissions=permissions, approvals=approvals)
    assert not sandbox.run("fetch", url="https://example.com").ok

    request = approvals.request("fetch", "agent")
    request.approve("human")
    assert sandbox.run("fetch", url="https://example.com").ok


def test_calculator_evaluates_only_arithmetic(tmp_path):
    sandbox = _sandbox(tmp_path)
    assert sandbox.run("calculator", expression="2 + 3 * 4").output == "14"
    assert sandbox.run("calculator", expression="2 ** 3").output == "8"
    assert not sandbox.run("calculator", expression="__import__('os')").ok


def test_output_is_truncated(tmp_path):
    (tmp_path / "a.txt").write_text("x" * 100, encoding="utf-8")
    sandbox = build_sandbox(tmp_path, read_only=True)
    from core.tools import SandboxPolicy

    sandbox.policy = SandboxPolicy(root=tmp_path, allowed_tools=sandbox.policy.allowed_tools, max_output_bytes=10)
    result = sandbox.run("read_file", relative="a.txt")
    assert result.ok and result.metadata.get("truncated") == "true"


def test_tool_failure_is_returned_not_raised(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=True)
    result = sandbox.run("read_file", relative="missing.txt")
    assert isinstance(result, ToolResult)
    assert not result.ok


# --- agent coordination ---------------------------------------------------


def test_agent_runs_a_plan_and_records_steps(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=False)
    plan = plan_from_calls(
        "write then read",
        (
            ToolCall("write_file", {"relative": "a.txt", "content": "نص"}),
            ToolCall("read_file", {"relative": "a.txt"}),
        ),
        reasons=("create", "verify"),
    )
    run = ToolAgent(sandbox).run(plan)
    assert run.succeeded
    assert run.outputs == ("Wrote 4 bytes to a.txt", "نص")
    assert run.steps[0].reason == "create"


def test_agent_stops_early_on_error(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=False)
    plan = plan_from_calls(
        "fail then skip",
        (
            ToolCall("read_file", {"relative": "missing.txt"}),
            ToolCall("calculator", {"expression": "1+1"}),
        ),
    )
    run = ToolAgent(sandbox, stop_on_error=True).run(plan)
    assert not run.succeeded
    assert run.stopped_early is True
    assert len(run.steps) == 1


def test_agent_can_continue_past_errors(tmp_path):
    sandbox = _sandbox(tmp_path, read_only=False)
    plan = plan_from_calls(
        "continue",
        (
            ToolCall("read_file", {"relative": "missing.txt"}),
            ToolCall("calculator", {"expression": "1+1"}),
        ),
    )
    run = ToolAgent(sandbox, stop_on_error=False).run(plan)
    assert len(run.steps) == 2
    assert run.steps[1].result.ok


def test_plan_validates_shape():
    with pytest.raises(ValueError):
        ToolPlan(goal="", calls=(ToolCall("calculator"),))
    with pytest.raises(ValueError):
        ToolPlan(goal="g", calls=())
    with pytest.raises(ValueError):
        ToolPlan(goal="g", calls=(ToolCall("calculator"),), reasons=("a", "b"))
    with pytest.raises(ValueError):
        ToolCall("  ")
