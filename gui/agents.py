"""The Personalities (agents) service: the engine behind the agents window.

The whole multi-agent subsystem (:mod:`core.agents`) was reachable only from the
CLI (``python -m cli agents``), so a user could not *see* the seven personalities
as independent agents, what each may and may not do, which one would answer a
question, or what each has actually done. This module is the Tk-free service the
window drives, so the window and the CLI read **one** system -- no logic is
reimplemented in either front end.

It wraps the objects that already exist:

* :class:`core.agents.system.AgentSystem` -- the composed registry, capability
  gate, runtime, router, collaboration, conflict resolver, and monitor;
* :class:`core.learning_isolation` -- who exists and where their records live.

Honest by construction: every value is read from the system the caller already
has (never invented), a capability the personality does not hold is reported as
*denied* with the gate's own reason, and a routing decision that finds no
suitable personality is reported as such rather than forced. Nothing here runs a
model or reaches the network: ``route``/``monitoring``/``isolation`` are pure
reads, and a personality's *activity* is the runtime's own tally, never a
fabricated score. The module imports no Tk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.agents.capabilities import ALL_CAPABILITIES
from core.agents.monitoring import PersonalityActivity
from core.agents.registry import PersonalityProfile
from core.agents.routing import TaskType
from core.agents.system import AgentSystem
from core.learning_isolation import PRIVATE_OWNERS, SHAREABLE_OWNERS


@dataclass(frozen=True)
class PersonalityRow:
    """One personality as the window lists it: the facts a user reads at a glance."""

    owner: str
    name: str
    role: str
    model: str
    private: bool
    contributes_training: bool
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityLine:
    """One capability and whether a personality holds it, with the gate's reason."""

    capability: str
    allowed: bool
    reason: str


@dataclass(frozen=True)
class RouteReport:
    """The router's answer as the window shows it: who, why, and the candidates."""

    question: str
    task_type: str
    selected: str
    suitable: bool
    manual: bool
    reason: str
    candidates: tuple[tuple[str, float], ...] = ()
    refused: tuple[str, ...] = ()

    def describe(self) -> str:
        if not self.suitable:
            return (
                f"No personality is suitable for a {self.task_type} task "
                f"({self.reason})."
            )
        how = "chosen manually" if self.manual else "routed"
        lines = [f"{how}: {self.selected} for a {self.task_type} task — {self.reason}"]
        if self.refused:
            lines.append(f"Could not serve: {', '.join(self.refused)}")
        for owner, score in self.candidates[:4]:
            lines.append(f"  {owner}: {score:.3f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class IsolationReport:
    """The memory and learning isolation policy, as it stands."""

    private: tuple[str, ...]
    shareable: tuple[str, ...]
    dataset_owners: tuple[str, ...]
    dataset_is_clean: bool

    def describe(self) -> str:
        lines = [
            f"Private characters (records never shared): {', '.join(self.private) or 'none'}",
            f"Shareable characters: {', '.join(self.shareable) or 'none'}",
            f"Aggregated training dataset owners: {', '.join(self.dataset_owners) or 'none'}",
            "Dataset is clean of private owners: " + ("yes" if self.dataset_is_clean else "NO"),
        ]
        return "\n".join(lines)


class AgentsService:
    """The Tk-free engine the agents window and the CLI both drive.

    ``system`` is injectable, so the whole surface is exercised with no model and
    no display: a test passes a system whose executor is a stub, exactly as the
    CLI's own ``--json`` tests do.
    """

    def __init__(self, *, system: AgentSystem | None = None, data_root: str | Path | None = None) -> None:
        self._system = system
        self._data_root = data_root

    # -- the system --------------------------------------------------------

    @property
    def system(self) -> AgentSystem:
        """The shared agent system (built once, on first use)."""
        if self._system is None:
            self._system = AgentSystem(data_root=self._data_root)
        return self._system

    # -- the personalities -------------------------------------------------

    def profiles(self) -> tuple[PersonalityProfile, ...]:
        """Every personality's full profile, in the project's own order."""
        return self.system.profiles()

    def rows(self) -> tuple[PersonalityRow, ...]:
        """The personalities as the window's list rows (a summary each)."""
        rows: list[PersonalityRow] = []
        for profile in self.profiles():
            rows.append(
                PersonalityRow(
                    owner=profile.personality_id,
                    name=profile.name,
                    role=profile.specialization or profile.role,
                    model=profile.model_preference or "(mouth default)",
                    private=profile.private,
                    contributes_training=profile.contributes_training,
                    capabilities=tuple(
                        sorted(capability.value for capability in profile.capabilities.granted)
                    ),
                )
            )
        return tuple(rows)

    def profile(self, key: str) -> PersonalityProfile:
        """One personality's full profile (an unknown key resolves to the model)."""
        return self.system.profile(key)

    def describe(self, key: str) -> str:
        """One personality's full profile as readable text."""
        return self.system.profile(key).describe()

    def capabilities(self, key: str) -> tuple[CapabilityLine, ...]:
        """Every capability and whether ``key`` holds it -- the whole vocabulary.

        The *whole* vocabulary is reported, not only what is granted, so a reader
        can see what a personality is *denied* as well as what it may do -- which
        is the point of a capability ceiling.
        """
        profile = self.system.profile(key)
        return tuple(
            CapabilityLine(
                capability=capability.value,
                allowed=profile.holds(capability),
                reason=(
                    "granted within this personality's ceiling"
                    if profile.holds(capability)
                    else "outside this personality's ceiling (denied by default)"
                ),
            )
            for capability in ALL_CAPABILITIES
        )

    def capability_owners(self) -> dict[str, tuple[str, ...]]:
        """Which personalities hold each capability, for the window's matrix."""
        result: dict[str, list[str]] = {capability.value: [] for capability in ALL_CAPABILITIES}
        for profile in self.profiles():
            for capability in ALL_CAPABILITIES:
                if profile.holds(capability):
                    result[capability.value].append(profile.personality_id)
        return {capability: tuple(owners) for capability, owners in result.items()}

    # -- routing -----------------------------------------------------------

    def route(
        self,
        question: str,
        *,
        conversation: str = "",
        task_type: str = "",
        override: str = "",
        user_choice: str = "",
    ) -> RouteReport:
        """Which personality should answer ``question``, and the traceable reason.

        ``task_type`` is the readable name of a :class:`TaskType` (or empty to let
        the router classify); an unknown name is ignored rather than raising, so a
        stale choice never breaks the surface.
        """
        declared: TaskType | None = None
        if task_type:
            try:
                declared = TaskType(task_type)
            except ValueError:
                declared = None
        decision = self.system.route(
            question,
            conversation=conversation,
            task_type=declared,
            override=override,
            user_choice=user_choice,
        )
        return RouteReport(
            question=question,
            task_type=decision.task_type.value,
            selected=decision.selected,
            suitable=decision.suitable,
            manual=decision.manual,
            reason=decision.reason,
            candidates=tuple((candidate.owner, candidate.score) for candidate in decision.candidates),
            refused=tuple(decision.refused),
        )

    def task_types(self) -> tuple[str, ...]:
        """The task types a caller may pin, in a fixed order."""
        return tuple(task_type.value for task_type in TaskType)

    # -- monitoring --------------------------------------------------------

    def activity(self) -> tuple[PersonalityActivity, ...]:
        """Each personality's measured activity (a factual tally, never a score)."""
        return self.system.monitoring().activities

    def activity_for(self, owner: str) -> PersonalityActivity:
        """One personality's activity (a zero tally for an owner with none)."""
        return self.system.monitoring().for_owner(owner)

    # -- isolation ---------------------------------------------------------

    def isolation(self) -> IsolationReport:
        """The memory and learning isolation policy, read from the real modules."""
        from core.agents.learning_scope import (
            aggregated_dataset_is_clean,
            aggregated_dataset_owners,
        )

        return IsolationReport(
            private=tuple(sorted(PRIVATE_OWNERS)),
            shareable=tuple(sorted(SHAREABLE_OWNERS)),
            dataset_owners=tuple(aggregated_dataset_owners()),
            dataset_is_clean=bool(aggregated_dataset_is_clean()),
        )


def default_service(*, data_root: str | Path | None = None) -> AgentsService:
    """Build the default :class:`AgentsService`."""
    return AgentsService(data_root=data_root)


__all__ = [
    "AgentsService",
    "CapabilityLine",
    "IsolationReport",
    "PersonalityRow",
    "RouteReport",
    "default_service",
]
