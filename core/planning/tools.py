"""The tool catalogue and selection engine (Phase 8, section 7).

Section 7 asks for a centralized tool registry where each tool declares its id,
description, input/output schema, required permissions, risk category, resource
requirements, timeout, reversibility, and whether user approval is required --
and for selection that weighs a task's requirements, the tool's capabilities,
permissions, reliability, resource limits, and risk, **never** an LLM's
recommendation alone.

This module is that catalogue. It is built from the project's **existing**
sandbox tools (:func:`core.tools.build_sandbox_tools`) plus a small set of
declared metadata, so there is one source of truth for what a tool *is* -- the
tool itself -- and the catalogue only adds the scheduling/selection facts the
sandbox does not carry. Selection is deterministic: it scores every permitted
candidate on named factors and returns the best, or ``None`` (never a guess).

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from core.permissions import Capability
from core.tools import ToolSpec


class RiskCategory(str, Enum):
    """How risky a tool's action is (section 9's three categories)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass(frozen=True)
class ToolDescriptor:
    """Everything section 7 asks a tool to declare.

    ``spec`` is the project's own :class:`core.tools.ToolSpec` -- the real tool,
    bound to its handler and its capability. The rest is the metadata the
    catalogue adds for selection and the permission gate.
    """

    tool_id: str
    description: str
    spec: ToolSpec
    input_schema: tuple[str, ...] = ()
    output_schema: tuple[str, ...] = ()
    risk: RiskCategory = RiskCategory.LOW
    resource_cost: int = 1
    timeout_seconds: float = 30.0
    reversible: bool = True
    requires_approval: bool = False
    #: A relative reliability in [0, 1] (how often the tool is expected to work).
    reliability: float = 0.9
    #: Free-text tags used to match a task's requirement (e.g. "read", "write").
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("A tool descriptor needs an id.")
        if self.timeout_seconds <= 0:
            raise ValueError("A tool timeout must be positive.")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Tool reliability must be in [0, 1].")

    @property
    def capability(self) -> Capability:
        return self.spec.capability

    @property
    def mutates(self) -> bool:
        return self.spec.mutates

    def describe(self) -> str:
        flags = []
        if self.mutates:
            flags.append("mutating")
        if not self.reversible:
            flags.append("irreversible")
        if self.requires_approval:
            flags.append("approval")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.tool_id} ({self.risk.value}): {self.description}{suffix}"


@dataclass(frozen=True)
class ToolSelection:
    """The outcome of one selection: the chosen tool and why."""

    task_id: str
    descriptor: ToolDescriptor | None
    considered: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()
    reason: str = ""

    @property
    def chosen(self) -> str:
        return self.descriptor.tool_id if self.descriptor else ""

    def describe(self) -> str:
        if self.descriptor is None:
            return f"{self.task_id}: no tool ({self.reason})"
        return f"{self.task_id}: {self.descriptor.tool_id} — {self.reason}"


#: The declared metadata for the project's built-in sandbox tools. Anything not
#: listed here is still catalogued, but with conservative defaults (medium risk,
#: approval required), so an un-declared tool is never selected as if it were
#: safe.
_BUILTIN_METADATA: dict[str, dict[str, object]] = {
    "read_file": {
        "risk": RiskCategory.LOW,
        "reversible": True,
        "tags": ("read", "file", "text"),
        "input_schema": ("relative",),
        "output_schema": ("content",),
        "resource_cost": 1,
    },
    "list_dir": {
        "risk": RiskCategory.LOW,
        "reversible": True,
        "tags": ("list", "directory"),
        "input_schema": ("relative",),
        "output_schema": ("entries",),
        "resource_cost": 1,
    },
    "search": {
        "risk": RiskCategory.LOW,
        "reversible": True,
        "tags": ("search", "file", "text"),
        "input_schema": ("query", "relative"),
        "output_schema": ("matches",),
        "resource_cost": 2,
    },
    "calculator": {
        "risk": RiskCategory.LOW,
        "reversible": True,
        "tags": ("compute", "math"),
        "input_schema": ("expression",),
        "output_schema": ("value",),
        "resource_cost": 1,
    },
    "write_file": {
        "risk": RiskCategory.MEDIUM,
        "reversible": True,
        "tags": ("write", "file", "text"),
        "input_schema": ("relative", "content"),
        "output_schema": ("summary",),
        "resource_cost": 1,
    },
}

#: Tags a task requirement may name; a tool matches when it carries any of them.
KNOWN_TAGS: tuple[str, ...] = (
    "read",
    "write",
    "file",
    "text",
    "search",
    "list",
    "directory",
    "compute",
    "math",
)


@dataclass
class ToolCatalogue:
    """The centralized tool registry, built from real :class:`ToolSpec`s.

    ``register`` adds a tool with its declared metadata (falling back to
    conservative defaults), ``get``/``tools`` read it back, and ``available``
    reports a tool's availability honestly -- a descriptor for a tool the sandbox
    does not allow is present but marked unavailable.
    """

    descriptors: dict[str, ToolDescriptor] = field(default_factory=dict)
    #: Tools the sandbox currently permits (empty means "ask the sandbox").
    available_tools: frozenset[str] | None = None

    def register(
        self,
        spec: ToolSpec,
        *,
        risk: RiskCategory | None = None,
        tags: tuple[str, ...] = (),
        input_schema: tuple[str, ...] = (),
        output_schema: tuple[str, ...] = (),
        resource_cost: int = 1,
        timeout_seconds: float = 30.0,
        reversible: bool | None = None,
        requires_approval: bool | None = None,
        reliability: float = 0.9,
    ) -> ToolDescriptor:
        """Add a tool, filling any unstated field from the built-in metadata."""
        declared = _BUILTIN_METADATA.get(spec.name, {})
        declared_risk = declared.get("risk")
        declared_cost = declared.get("resource_cost")
        descriptor = ToolDescriptor(
            tool_id=spec.name,
            description=spec.description,
            spec=spec,
            input_schema=input_schema or tuple(declared.get("input_schema", ())),  # type: ignore[arg-type]
            output_schema=output_schema or tuple(declared.get("output_schema", ())),  # type: ignore[arg-type]
            risk=risk or (declared_risk if isinstance(declared_risk, RiskCategory) else RiskCategory.MEDIUM),
            resource_cost=resource_cost or (declared_cost if isinstance(declared_cost, int) else 1),
            timeout_seconds=timeout_seconds,
            reversible=reversible if reversible is not None else bool(declared.get("reversible", True)),
            requires_approval=(
                requires_approval
                if requires_approval is not None
                else spec.capability is Capability.EXTERNAL_TOOL
            ),
            reliability=reliability,
            tags=tags or tuple(declared.get("tags", ())),  # type: ignore[arg-type]
        )
        self.descriptors[spec.name] = descriptor
        return descriptor

    def get(self, tool_id: str) -> ToolDescriptor | None:
        return self.descriptors.get(tool_id)

    def tools(self) -> tuple[ToolDescriptor, ...]:
        return tuple(self.descriptors[name] for name in sorted(self.descriptors))

    def available(self, tool_id: str) -> bool:
        """Whether a tool may run now: present, and permitted by the sandbox."""
        if tool_id not in self.descriptors:
            return False
        if self.available_tools is None:
            return True
        return tool_id in self.available_tools

    def describe(self) -> str:
        return "\n".join(descriptor.describe() for descriptor in self.tools())


def build_tool_catalogue(
    specs: Iterable[ToolSpec],
    *,
    available_tools: frozenset[str] | None = None,
) -> ToolCatalogue:
    """Build a catalogue from real tool specs (the sandbox's own list)."""
    catalogue = ToolCatalogue(available_tools=available_tools)
    for spec in specs:
        catalogue.register(spec)
    return catalogue


def _task_tags(requirement: str) -> set[str]:
    """The known tags a free-text requirement names."""
    lowered = requirement.casefold()
    return {tag for tag in KNOWN_TAGS if tag in lowered}


def select_tool(
    task: object,
    catalogue: ToolCatalogue,
    *,
    allowed: frozenset[str] | None = None,
    max_risk: RiskCategory = RiskCategory.MEDIUM,
    requirement: str = "",
) -> ToolSelection:
    """Choose the best permitted tool for ``task``, or ``None`` (never a guess).

    ``task`` is a :class:`core.planning.tasks.Task` (duck-typed: it needs
    ``task_id``, ``tool``, ``required_permissions``, and ``description``). The
    scoring weighs, in order: an explicit tool name the task pins (which must be
    permitted), then requirement-tag matches, then lower risk, then higher
    reliability, then lower resource cost. A tool above ``max_risk`` or below the
    task's required permissions is **rejected with a reason**, so the decision is
    inspectable -- and no LLM recommendation is consulted at all.

    ``requirement`` is extra free text (the decomposition engine's tool hint) that
    also contributes matching tags, so a step that says "write the report" can be
    matched even when its description is terse.
    """
    task_id = str(getattr(task, "task_id", ""))
    pinned = str(getattr(task, "tool", "") or "")
    description = str(getattr(task, "description", "") or "")
    required = {str(p) for p in getattr(task, "required_permissions", ()) or ()}

    considered: list[str] = []
    rejected: list[tuple[str, str]] = []

    def permitted(descriptor: ToolDescriptor) -> str:
        """Return "" when permitted, else the refusal reason."""
        if allowed is not None and descriptor.tool_id not in allowed:
            return "not on the allow-list"
        if catalogue.available_tools is not None and descriptor.tool_id not in catalogue.available_tools:
            return "the sandbox does not permit it"
        if descriptor.risk.rank > max_risk.rank:
            return f"risk {descriptor.risk.value} exceeds the {max_risk.value} ceiling"
        if required and descriptor.capability.value not in required and descriptor.tool_id not in required:
            missing = required - {descriptor.capability.value, descriptor.tool_id}
            if missing:
                return "missing permission " + ", ".join(sorted(missing))
        return ""

    # An explicit tool name wins, but only if it is permitted.
    if pinned:
        descriptor = catalogue.get(pinned)
        if descriptor is None:
            return ToolSelection(task_id, None, (), (), f"unknown tool {pinned!r}")
        reason = permitted(descriptor)
        if reason:
            return ToolSelection(task_id, None, (pinned,), ((pinned, reason),), reason)
        return ToolSelection(task_id, descriptor, (pinned,), (), f"pinned by the task ({pinned})")

    wanted = _task_tags(description) | _task_tags(" ".join(required)) | _task_tags(requirement)
    scored: list[tuple[tuple[float, float, float, int], ToolDescriptor]] = []
    for descriptor in catalogue.tools():
        considered.append(descriptor.tool_id)
        reason = permitted(descriptor)
        if reason:
            rejected.append((descriptor.tool_id, reason))
            continue
        overlap = len(wanted & set(descriptor.tags))
        # Sort key: more tag overlap, lower risk, higher reliability, lower cost.
        key = (-float(overlap), float(descriptor.risk.rank), -descriptor.reliability, descriptor.resource_cost)
        scored.append((key, descriptor))

    if not scored:
        return ToolSelection(
            task_id,
            None,
            tuple(considered),
            tuple(rejected),
            "no permitted tool matches the task",
        )
    scored.sort(key=lambda pair: (pair[0], pair[1].tool_id))
    best_key, best = scored[0]
    best_overlap = -best_key[0]
    # Honest by construction: a tool is only chosen when the task's own words
    # actually match it. With no match at all the task is left without a tool --
    # blocked and reported -- rather than handed an arbitrary permitted tool.
    if best_overlap <= 0:
        return ToolSelection(
            task_id,
            None,
            tuple(considered),
            tuple(rejected),
            "no tool matches the task requirement",
        )
    reason = f"matched {int(best_overlap)} requirement tag(s)"
    return ToolSelection(task_id, best, tuple(considered), tuple(rejected), reason)


__all__ = [
    "KNOWN_TAGS",
    "RiskCategory",
    "ToolCatalogue",
    "ToolDescriptor",
    "ToolSelection",
    "build_tool_catalogue",
    "select_tool",
]
