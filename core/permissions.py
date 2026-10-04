"""Capability/permission policy: the gate every environment tier must pass.

A ``PermissionPolicy`` is an allow-list. Nothing is granted by default, and
external-tool capabilities are denied unless the policy explicitly opts in. The
runtime consults this policy before it will run anything that leaves the
symbolic/sandbox tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    """What a component is asking permission to do."""

    SYMBOLIC_SIMULATION = "symbolic_simulation"
    SANDBOX_TASK = "sandbox_task"
    EXTERNAL_TOOL = "external_tool"


@dataclass(frozen=True)
class PermissionDecision:
    capability: Capability
    target: str
    allowed: bool
    reason: str


@dataclass
class PermissionPolicy:
    """Allow-list policy. Empty by default; external tools are always opt-in."""

    name: str = "default"
    allowed: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.SYMBOLIC_SIMULATION, Capability.SANDBOX_TASK})
    )
    allowed_external_tools: frozenset[str] = frozenset()
    deny_external_by_default: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Permission policy name cannot be empty.")

    def allows(self, capability: Capability, target: str = "") -> bool:
        return self.decide(capability, target).allowed

    @property
    def allows_external(self) -> bool:
        """True only when the policy explicitly grants the external capability."""
        return Capability.EXTERNAL_TOOL in self.allowed

    def decide(self, capability: Capability, target: str = "") -> PermissionDecision:
        if capability is Capability.EXTERNAL_TOOL:
            if self.deny_external_by_default and capability not in self.allowed:
                return PermissionDecision(
                    capability, target, False, "External tools are denied by policy."
                )
            if target not in self.allowed_external_tools:
                return PermissionDecision(
                    capability, target, False, f"External tool '{target}' is not on the allow-list."
                )
            return PermissionDecision(capability, target, True, "External tool explicitly allowed.")

        if capability not in self.allowed:
            return PermissionDecision(
                capability, target, False, f"Capability '{capability.value}' is not granted."
            )
        return PermissionDecision(capability, target, True, "Capability granted.")
