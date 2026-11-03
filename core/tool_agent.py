"""Agent coordination: plan a task into tool steps, then run them in a sandbox.

A single tool call is not an agent. This module adds the coordination an agent
needs while keeping every guarantee of :mod:`core.tools`:

* a :class:`ToolCall` is a proposed tool plus its arguments;
* a :class:`ToolPlan` is an ordered list of calls with a reason per step;
* the :class:`ToolAgent` executes a plan through a :class:`core.tools.ToolSandbox`,
  stopping at the first failed step (or continuing, on request) and recording a
  full, inspectable trace.

Coordination is deliberately modest: a plan is *declared* by the caller or by a
planner function, not invented by the model. The model may propose a plan, but the
agent will only ever run calls the sandbox permits, so a proposal cannot widen its
own authority.

The :class:`ToolChooser` closes the one gap that left: a caller could not ask the
agent to *choose* a tool for an open task, only replay a plan it already had. The
chooser accepts a **proposal** (from a model, a heuristic, or a caller) and filters
it through the sandbox's own allow-list and policy -- the model proposes, the
sandbox disposes -- so a tool the sandbox would refuse is never even offered. It
adds no authority: a chooser can only ever return a tool the sandbox already
permits, and a proposal for anything else is reported as refused rather than
silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core.tools import ToolResult, ToolSandbox

Planner = Callable[[str], "ToolPlan"]


@dataclass(frozen=True)
class ToolCall:
    """A proposed tool call."""

    tool: str
    arguments: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("Tool call needs a tool name.")


@dataclass(frozen=True)
class ToolPlan:
    """An ordered sequence of calls with a goal and a reason per step."""

    goal: str
    calls: tuple[ToolCall, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("Plan goal cannot be empty.")
        if not self.calls:
            raise ValueError("Plan needs at least one call.")
        if self.reasons and len(self.reasons) != len(self.calls):
            raise ValueError("Plan reasons must match calls one-to-one.")


@dataclass(frozen=True)
class PlanStep:
    """One executed step, with its result."""

    call: ToolCall
    result: ToolResult
    reason: str = ""


@dataclass(frozen=True)
class AgentRun:
    """The full, inspectable record of a plan's execution."""

    goal: str
    steps: tuple[PlanStep, ...]
    stopped_early: bool = False

    @property
    def succeeded(self) -> bool:
        return bool(self.steps) and all(step.result.ok for step in self.steps)

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(step.result.output for step in self.steps if step.result.ok)


class ToolAgent:
    """Runs a tool plan through a sandbox, recording every step."""

    def __init__(self, sandbox: ToolSandbox, *, stop_on_error: bool = True) -> None:
        self.sandbox = sandbox
        self.stop_on_error = stop_on_error

    def run(self, plan: ToolPlan) -> AgentRun:
        steps: list[PlanStep] = []
        stopped_early = False
        for index, call in enumerate(plan.calls):
            reason = plan.reasons[index] if plan.reasons else ""
            result = self.sandbox.run(call.tool, **call.arguments)
            steps.append(PlanStep(call, result, reason))
            if not result.ok and self.stop_on_error:
                stopped_early = index < len(plan.calls) - 1
                break
        return AgentRun(goal=plan.goal, steps=tuple(steps), stopped_early=stopped_early)


def plan_from_calls(goal: str, calls: Iterable[ToolCall], reasons: Iterable[str] = ()) -> ToolPlan:
    """Convenience constructor for a plan."""
    return ToolPlan(goal=goal, calls=tuple(calls), reasons=tuple(reasons))


@dataclass(frozen=True)
class ToolProposal:
    """A proposed tool choice, with why it was proposed.

    A proposal is *untrusted*: it may come from a model, a heuristic, or a caller.
    The :class:`ToolChooser` decides whether the sandbox would permit it, so a
    proposal never runs anything by itself.
    """

    tool: str
    reason: str = ""
    arguments: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("A tool proposal needs a tool name.")


@dataclass(frozen=True)
class ChosenTool:
    """The outcome of choosing: the tool the sandbox permits, or the refusal reason."""

    proposal: ToolProposal
    allowed: bool
    reason: str = ""

    def describe(self) -> str:
        if self.allowed:
            return f"{self.proposal.tool} (chosen: {self.reason})"
        return f"{self.proposal.tool} (refused: {self.reason})"


class ToolChooser:
    """Chooses a tool for an open task, bounded by the sandbox's own allow-list.

    The chooser never widens authority: :meth:`choose` returns a :class:`ChosenTool`
    whose ``allowed`` is true only when the sandbox's policy names the tool **and**
    the tool's capability is permitted, exactly as :meth:`ToolSandbox.run` would
    decide. So the model proposes and the sandbox disposes -- a proposal for a tool
    outside the allow-list is reported as refused, never offered.
    """

    def __init__(self, sandbox: ToolSandbox) -> None:
        self.sandbox = sandbox

    def _permitted(self, tool: str) -> tuple[bool, str]:
        """Whether the sandbox would permit ``tool``, and the reason if not.

        The decision mirrors :meth:`ToolSandbox.run`'s own gates (allow-list, then
        capability) *without* executing anything, so a choice can be made before a
        call is made.
        """
        spec = self.sandbox.registry.get(tool)
        if spec is None:
            return False, f"unknown tool '{tool}'"
        if tool not in self.sandbox.policy.allowed_tools:
            return False, f"'{tool}' is not on the sandbox allow-list"
        if spec.mutates and self.sandbox.policy.read_only:
            return False, f"'{tool}' mutates and the sandbox is read-only"
        decision = self.sandbox.permissions.decide(spec.capability, tool)
        if not decision.allowed:
            return False, decision.reason
        return True, "permitted by the sandbox policy"

    def choose(self, proposals: Iterable[ToolProposal]) -> tuple[ChosenTool, ...]:
        """Every proposal judged against the sandbox, in order, nothing dropped.

        A refused proposal is returned with ``allowed=False`` and its reason, so a
        caller sees what was proposed and why it was not chosen rather than a silent
        gap.
        """
        chosen: list[ChosenTool] = []
        for proposal in proposals:
            allowed, reason = self._permitted(proposal.tool)
            chosen.append(ChosenTool(proposal=proposal, allowed=allowed, reason=reason))
        return tuple(chosen)

    def first_allowed(self, proposals: Iterable[ToolProposal]) -> ChosenTool | None:
        """The first proposal the sandbox permits, or ``None`` when none is."""
        for chosen in self.choose(proposals):
            if chosen.allowed:
                return chosen
        return None

    def plan(self, goal: str, proposals: Iterable[ToolProposal]) -> ToolPlan:
        """A runnable :class:`ToolPlan` from the proposals the sandbox permits.

        Refused proposals are dropped from the plan (the agent never runs them), and
        the plan carries the chosen tools' own reasons. Raises ``ValueError`` when no
        proposal is permitted, so a caller cannot build an empty plan by accident.
        """
        chosen = [item for item in self.choose(proposals) if item.allowed]
        if not chosen:
            raise ValueError("No proposed tool is permitted by the sandbox policy.")
        calls = tuple(
            ToolCall(item.proposal.tool, dict(item.proposal.arguments)) for item in chosen
        )
        reasons = tuple(item.proposal.reason or item.reason for item in chosen)
        return ToolPlan(goal=goal, calls=calls, reasons=reasons)


#: The held-out tool-selection catalogue: a proposal list and whether the chooser
#: must find a permitted tool. Negative controls (only a refused tool proposed, an
#: unknown tool) must yield no choice, so a chooser that ignores the allow-list
#: fails the gate as surely as one that finds nothing.
TOOL_SELECTION_JUDGEMENTS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("permitted-first", ("calculator", "read_file"), True),
    ("skip-refused", ("__not_a_tool__", "calculator"), True),
    ("all-refused", ("__not_a_tool__", "__also_not__"), False),
    ("empty", (), False),
)


def tool_selection_accuracy(
    judgements: tuple[tuple[str, tuple[str, ...], bool], ...] = TOOL_SELECTION_JUDGEMENTS,
) -> float:
    """The share of held-out judgements the bounded chooser gets right.

    Each judgement proposes tools and states whether the chooser must find one the
    sandbox permits. The negative controls (only refused/unknown tools proposed, an
    empty proposal) must yield **no** choice, so a chooser that returns a tool
    outside the allow-list fails the gate. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    import tempfile
    from pathlib import Path

    from core.tools import build_sandbox

    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        correct = 0
        for _name, tools, expect in catalogue:
            proposals = tuple(ToolProposal(tool) for tool in tools)
            chosen = chooser.first_allowed(proposals)
            if (chosen is not None) == expect:
                correct += 1
        return correct / len(catalogue)


__all__ = [
    "TOOL_SELECTION_JUDGEMENTS",
    "AgentRun",
    "ChosenTool",
    "PlanStep",
    "Planner",
    "ToolAgent",
    "ToolCall",
    "ToolChooser",
    "ToolPlan",
    "ToolProposal",
    "plan_from_calls",
    "tool_selection_accuracy",
]
