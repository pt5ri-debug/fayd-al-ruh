"""Capability permissions for the seven personalities -- deny-by-default.

The task is explicit that isolation and permissions must be enforced by *trusted
system components*, never by a prompt or by the personality's own configuration:
"a personality's configuration or generated text must not be able to grant itself
additional permissions". This module is that trusted component.

Two ideas, kept apart on purpose:

* A :class:`CapabilitySet` is an **immutable** allow-list of what an agent may do.
  Nothing is granted unless it is present, and a set is built only by trusted
  code -- an agent cannot hand itself one.
* A **ceiling** (:func:`capability_ceiling`) is the most any personality of a
  given kind may ever hold, hard-coded here. A personality's *configuration* may
  only **narrow** the ceiling (:func:`effective_capabilities`), never widen it, so
  editing a config file or convincing a model to "grant me tools" changes nothing.

Every check returns a :class:`CapabilityDecision` with a readable reason, and the
gateway (:class:`CapabilityGateway`) records every decision -- granted or refused
-- so an attempt to exceed a personality's authority is *auditable*, not silent.

This module is dependency-free and Tk-free; it loads no model and touches no
network.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    is_private,
    owner_for,
)


class AgentCapability(str, Enum):
    """What an agent may be permitted to do. One closed vocabulary."""

    #: Answer a user's question in its own voice.
    ANSWER = "answer"
    #: Read the shared semantic knowledge / retrieve from the project's sources.
    RETRIEVE = "retrieve"
    #: Use the shared reasoning engine (the Phase-4 facade and its engines).
    REASON = "reason"
    #: Run a sandboxed tool (the project's own deny-by-default tool sandbox).
    RUN_TOOLS = "run_tools"
    #: Write files through the sandbox (a *mutating* tool).
    WRITE_FILES = "write_files"
    #: Search the internet (an external action; the project denies it by default).
    SEARCH = "search"
    #: Contribute its learning to an aggregated training dataset.
    CONTRIBUTE_TRAINING = "contribute_training"
    #: Delegate a task to another personality.
    DELEGATE = "delegate"
    #: Accept a delegated task from another personality.
    RECEIVE_DELEGATION = "receive_delegation"
    #: Manage its own goals and tasks.
    MANAGE_GOALS = "manage_goals"


#: The whole vocabulary, so a report or a config can iterate it deterministically.
ALL_CAPABILITIES: tuple[AgentCapability, ...] = tuple(AgentCapability)

#: The capabilities that reach outside the machine or change durable state -- the
#: ones a personality must never be able to grant itself. Kept in one place so a
#: report can name them.
SENSITIVE_CAPABILITIES: frozenset[AgentCapability] = frozenset(
    {
        AgentCapability.WRITE_FILES,
        AgentCapability.SEARCH,
        AgentCapability.DELEGATE,
        AgentCapability.RECEIVE_DELEGATION,
        AgentCapability.CONTRIBUTE_TRAINING,
    }
)

#: What a plain **character** may hold at most: it answers, retrieves, reasons,
#: manages its own goals, and -- when it is *shareable* -- contributes its learning
#: to an aggregated training dataset. It may use read-only tools when the sandbox
#: permits, but it may not write files or search the internet. A **private**
#: character (Natasha) additionally loses ``CONTRIBUTE_TRAINING`` from its ceiling,
#: so its exclusion from the aggregated dataset is structural rather than a flag
#: a configuration could flip (see :func:`capability_ceiling`).
_CHARACTER_CEILING: frozenset[AgentCapability] = frozenset(
    {
        AgentCapability.ANSWER,
        AgentCapability.RETRIEVE,
        AgentCapability.REASON,
        AgentCapability.RUN_TOOLS,
        AgentCapability.MANAGE_GOALS,
        AgentCapability.RECEIVE_DELEGATION,
        AgentCapability.CONTRIBUTE_TRAINING,
    }
)

#: What the **project's own model** may hold at most: the same as a character,
#: plus the training contribution it exists to make. It is *not* a character and
#: has no chat mouth of its own, so it answers only through its own surfaces.
_MAIN_MODEL_CEILING: frozenset[AgentCapability] = frozenset(
    {
        AgentCapability.ANSWER,
        AgentCapability.RETRIEVE,
        AgentCapability.REASON,
        AgentCapability.MANAGE_GOALS,
        AgentCapability.CONTRIBUTE_TRAINING,
    }
)

#: The default capability set a character holds out of the box -- deliberately
#: **within** the ceiling, and deliberately *not* including the sensitive
#: capabilities (no file writes, no internet search), so a fresh install grants no
#: external or mutating authority.
_DEFAULT_CHARACTER: frozenset[AgentCapability] = frozenset(
    {
        AgentCapability.ANSWER,
        AgentCapability.RETRIEVE,
        AgentCapability.REASON,
        AgentCapability.RUN_TOOLS,
        AgentCapability.MANAGE_GOALS,
        AgentCapability.RECEIVE_DELEGATION,
        AgentCapability.CONTRIBUTE_TRAINING,
    }
)

#: The default set the project's own model holds.
_DEFAULT_MAIN_MODEL: frozenset[AgentCapability] = frozenset(
    {
        AgentCapability.ANSWER,
        AgentCapability.RETRIEVE,
        AgentCapability.REASON,
        AgentCapability.MANAGE_GOALS,
        AgentCapability.CONTRIBUTE_TRAINING,
    }
)


class CapabilityDenied(PermissionError):
    """Raised when an agent attempts an action its capability set does not grant.

    It subclasses :class:`PermissionError`, so a caller that already handles a
    refused permission catches it too, while a caller that wants to be specific
    can catch this exact type.
    """


@dataclass(frozen=True)
class CapabilityDecision:
    """Whether ``owner`` may exercise ``capability``, and why."""

    owner: str
    capability: AgentCapability
    allowed: bool
    reason: str

    def describe(self) -> str:
        verdict = "allowed" if self.allowed else "denied"
        return f"{self.owner} may {self.capability.value}: {verdict} ({self.reason})"


@dataclass(frozen=True)
class CapabilitySet:
    """An immutable, deny-by-default allow-list for one personality.

    A set is created only through :func:`capability_ceiling`,
    :func:`default_capabilities`, or :func:`effective_capabilities` -- trusted
    code -- so an agent can never construct one that exceeds its ceiling. The
    mutators return a **new** set (the object is frozen), which keeps a decision
    taken against a set stable.
    """

    owner: str
    granted: frozenset[AgentCapability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("A capability set needs an owner.")

    def allows(self, capability: AgentCapability) -> bool:
        """True when ``capability`` is present in the set (deny by default)."""
        return capability in self.granted

    def with_capability(self, capability: AgentCapability) -> CapabilitySet:
        """A copy with ``capability`` added -- used only by trusted code."""
        return replace(self, granted=self.granted | {capability})

    def without_capability(self, capability: AgentCapability) -> CapabilitySet:
        """A copy with ``capability`` removed."""
        return replace(self, granted=self.granted - {capability})

    def intersect(self, ceiling: frozenset[AgentCapability]) -> CapabilitySet:
        """A copy holding only what is also in ``ceiling`` -- the narrowing rule."""
        return replace(self, granted=self.granted & ceiling)

    def to_primitive(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "granted": sorted(capability.value for capability in self.granted),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> CapabilitySet:
        """Rebuild a set, dropping any name the vocabulary does not know.

        An unknown capability name is ignored rather than trusted, so a
        hand-edited file can never introduce an authority this build does not
        define.
        """
        known = {capability.value: capability for capability in AgentCapability}
        granted = frozenset(
            known[name]
            for name in _as_names(data.get("granted"))
            if name in known
        )
        return cls(owner=str(data.get("owner") or ""), granted=granted)

    def describe(self) -> str:
        if not self.granted:
            return f"{self.owner}: no capabilities (deny by default)"
        names = ", ".join(sorted(capability.value for capability in self.granted))
        return f"{self.owner}: {names}"


def _as_names(value: object) -> tuple[str, ...]:
    """Coerce a stored ``granted`` field into a tuple of names, tolerantly."""
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def capability_ceiling(owner: str) -> frozenset[AgentCapability]:
    """The most ``owner`` may ever hold -- the hard, trusted ceiling.

    The project's own model has its own ceiling; every character has the
    character ceiling, **minus** ``CONTRIBUTE_TRAINING`` when the character is
    private (Natasha), so its exclusion from the aggregated training dataset is
    structural. An unknown owner gets an **empty** ceiling (nothing is granted to
    something this build does not know), which is the safe direction.
    """
    resolved = owner_for(owner)
    if resolved == MAIN_MODEL_OWNER:
        return _MAIN_MODEL_CEILING
    if resolved in CHARACTER_OWNERS:
        if is_private(resolved):
            return _CHARACTER_CEILING - {AgentCapability.CONTRIBUTE_TRAINING}
        return _CHARACTER_CEILING
    return frozenset()


def default_capabilities(owner: str) -> CapabilitySet:
    """The capability set ``owner`` holds out of the box (within its ceiling)."""
    resolved = owner_for(owner)
    if resolved == MAIN_MODEL_OWNER:
        return CapabilitySet(resolved, _DEFAULT_MAIN_MODEL)
    if resolved in CHARACTER_OWNERS:
        return CapabilitySet(resolved, _DEFAULT_CHARACTER)
    return CapabilitySet(resolved, frozenset())


def effective_capabilities(
    owner: str,
    *,
    requested: frozenset[AgentCapability] | None = None,
) -> CapabilitySet:
    """The set ``owner`` really holds: its default (or ``requested``) **∩ ceiling**.

    ``requested`` is what a *configuration* asks for. The intersection is the
    whole point: a configuration may narrow the default set but can never widen it
    past the ceiling, so no edit -- and no text a model generates -- can grant a
    personality authority the trusted policy withholds. Passing ``None`` means
    "the default", so an unconfigured character behaves exactly as before.
    """
    base = default_capabilities(owner) if requested is None else CapabilitySet(
        owner_for(owner), frozenset(requested)
    )
    return base.intersect(capability_ceiling(owner))


def decide_capability(owner: str, capability: AgentCapability) -> CapabilityDecision:
    """Whether ``owner`` may exercise ``capability``, with a readable reason.

    The check is against the **effective** set (default ∩ ceiling), so it is the
    same answer every trusted surface would give.
    """
    resolved = owner_for(owner)
    effective = effective_capabilities(resolved)
    if capability not in ALL_CAPABILITIES:
        return CapabilityDecision(
            resolved, capability, False, "that is not a capability this build defines."
        )
    if capability in effective.granted:
        return CapabilityDecision(resolved, capability, True, "granted within the ceiling.")
    if capability in capability_ceiling(resolved):
        return CapabilityDecision(
            resolved,
            capability,
            False,
            "permitted by the ceiling but not granted by this personality's set.",
        )
    return CapabilityDecision(
        resolved, capability, False, "not permitted for this personality at all."
    )


def assert_capability(owner: str, capability: AgentCapability) -> None:
    """Raise :class:`CapabilityDenied` when ``owner`` may not exercise ``capability``."""
    decision = decide_capability(owner, capability)
    if not decision.allowed:
        raise CapabilityDenied(decision.reason)


def required_for_tool(*, mutates: bool, external: bool) -> tuple[AgentCapability, ...]:
    """The capabilities a tool of the given shape requires.

    A read-only sandbox tool needs :attr:`~AgentCapability.RUN_TOOLS`; a mutating
    one needs :attr:`~AgentCapability.WRITE_FILES` as well; an external tool needs
    :attr:`~AgentCapability.SEARCH`. Written in one place so the runtime and the
    tool sandbox agree about what a tool costs a personality.
    """
    needed = [AgentCapability.RUN_TOOLS]
    if mutates:
        needed.append(AgentCapability.WRITE_FILES)
    if external:
        needed.append(AgentCapability.SEARCH)
    return tuple(needed)


@dataclass
class CapabilityGateway:
    """The one audited gate every personality action passes.

    A gateway holds a policy (whether the machine permits a capability at all,
    e.g. internet search) and records every decision it is asked for, so a refused
    attempt to exceed a personality's authority leaves a trace rather than being
    silently dropped. It is deliberately *not* configurable by a personality.
    """

    #: Capabilities the *machine* permits at all (deny-by-default, like the rest
    #: of the project: internet search is off unless a trusted caller enables it).
    machine_permitted: frozenset[AgentCapability] = field(
        default_factory=lambda: frozenset(ALL_CAPABILITIES) - {AgentCapability.SEARCH}
    )
    decisions: list[CapabilityDecision] = field(default_factory=list)
    on_decision: object | None = None

    def check(self, owner: str, capability: AgentCapability) -> CapabilityDecision:
        """Decide, record, and return whether ``owner`` may exercise ``capability``.

        A capability the machine does not permit is refused for everyone, whatever
        the personality's own set says -- the two gates are independent, and the
        stricter one wins.
        """
        if capability not in self.machine_permitted:
            decision = CapabilityDecision(
                owner_for(owner),
                capability,
                False,
                "the machine does not permit this capability (e.g. internet access).",
            )
        else:
            decision = decide_capability(owner, capability)
        self.decisions.append(decision)
        if callable(self.on_decision):
            from core.best_effort import best_effort

            with best_effort("capability gateway observer"):
                self.on_decision(decision)  # type: ignore[misc]
        return decision

    def require(self, owner: str, capability: AgentCapability) -> None:
        """Like :meth:`check`, but raises :class:`CapabilityDenied` when refused."""
        decision = self.check(owner, capability)
        if not decision.allowed:
            raise CapabilityDenied(decision.reason)

    def require_all(self, owner: str, capabilities: tuple[AgentCapability, ...]) -> None:
        """Require every capability in ``capabilities`` (used for a tool's shape)."""
        for capability in capabilities:
            self.require(owner, capability)

    def refusals(self) -> tuple[CapabilityDecision, ...]:
        """Every refused decision recorded so far, for a report or a test."""
        return tuple(decision for decision in self.decisions if not decision.allowed)

    def permit(self, capability: AgentCapability) -> None:
        """Allow a capability machine-wide (a trusted caller's explicit opt-in)."""
        self.machine_permitted = self.machine_permitted | {capability}


def private_owner_capabilities(owner: str) -> CapabilitySet:
    """The effective set of a **private** personality, with sharing withheld.

    Natasha is the one private owner. This helper states, in one place, that a
    private personality never holds :attr:`~AgentCapability.CONTRIBUTE_TRAINING`:
    its learning is excluded from every aggregated dataset by the isolation
    policy, and its capability set says so too, so the two cannot disagree.
    """
    resolved = owner_for(owner)
    effective = effective_capabilities(resolved)
    if is_private(resolved):
        return effective.without_capability(AgentCapability.CONTRIBUTE_TRAINING)
    return effective


__all__ = [
    "ALL_CAPABILITIES",
    "SENSITIVE_CAPABILITIES",
    "AgentCapability",
    "CapabilityDecision",
    "CapabilityDenied",
    "CapabilityGateway",
    "CapabilitySet",
    "assert_capability",
    "capability_ceiling",
    "decide_capability",
    "default_capabilities",
    "effective_capabilities",
    "private_owner_capabilities",
    "required_for_tool",
]
