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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

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


__all__ = [
    "AgentRun",
    "PlanStep",
    "Planner",
    "ToolAgent",
    "ToolCall",
    "ToolPlan",
    "plan_from_calls",
]
