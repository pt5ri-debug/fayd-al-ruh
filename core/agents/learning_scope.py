"""The agent learning scope: what a personality may learn from and contribute to.

A personality's learning has two directions, and both are governed:

* **in** -- what it may learn *from* (its own experience, the shared semantic
  knowledge, another shareable personality's learning). Governed by the project's
  isolation policy: a personality reads its own records, the main model reads a
  shareable character's, and Natasha's are private.
* **out** -- what it may *contribute* to (an aggregated training dataset). Governed
  by the :mod:`core.agents.capabilities` ceiling, which removes
  ``CONTRIBUTE_TRAINING`` from a private personality, so Natasha's learning is
  excluded structurally.

This module is the one place both directions are answered for an agent, over the
Phase-3 :class:`core.learning_coordinator.LearningCoordinator` (which owns the
actual knowledge/experience/feedback/pattern pipelines) and the isolation policy.
It never re-implements learning; it *scopes* it.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.agents.capabilities import AgentCapability, decide_capability
from core.learning_isolation import (
    MAIN_MODEL_OWNER,
    PRIVATE_OWNERS,
    SHAREABLE_OWNERS,
    is_owner,
    is_private,
    owner_for,
)


@dataclass(frozen=True)
class LearningScope:
    """What one personality may learn from, and whether it may contribute.

    ``learns_from`` is the owners whose learning this personality may read (its
    own first, then whatever the policy permits -- for a character that is only
    itself; for the main model, the shareable characters). ``contributes`` is
    whether its own learning may enter an aggregated dataset, and ``reason``
    explains a refusal so a report can state it rather than assert it.
    """

    owner: str
    learns_from: tuple[str, ...]
    contributes: bool
    reason: str

    def may_learn_from(self, other: str) -> bool:
        """True when this personality may read ``other``'s learning."""
        return owner_for(other) in self.learns_from

    def describe(self) -> str:
        froms = ", ".join(self.learns_from)
        verdict = "contributes to training" if self.contributes else "does not contribute"
        return f"{self.owner}: learns from {froms}; {verdict} ({self.reason})"


def learning_scope(owner: str) -> LearningScope:
    """The learning scope of ``owner``, computed from policy + capabilities."""
    resolved = owner_for(owner)
    if not is_owner(resolved):
        return LearningScope(resolved, (), False, f"{owner!r} is not a known personality.")
    readable = _readable_owners(resolved)
    contributes = decide_capability(resolved, AgentCapability.CONTRIBUTE_TRAINING).allowed
    if is_private(resolved):
        reason = "private personality: its learning is excluded from every aggregated dataset."
    elif resolved == MAIN_MODEL_OWNER:
        reason = "the project's own model contributes the aggregated dataset it owns."
    else:
        reason = "shareable character: its learning may enter the aggregated dataset."
    return LearningScope(resolved, readable, contributes, reason)


def _readable_owners(reader: str) -> tuple[str, ...]:
    """Whose learning ``reader`` may read, its own first."""
    from core.learning_isolation import OWNERS, decide

    ordered = [reader]
    for other in OWNERS:
        if other == reader:
            continue
        if decide(reader, other).allowed:
            ordered.append(other)
    return tuple(ordered)


@dataclass
class AgentLearningScope:
    """The learning scope for one personality, with its own learning coordinator.

    ``coordinator`` is the Phase-3 :class:`LearningCoordinator` (built lazily for
    the owner); this class only *scopes* it -- it exposes the owner's own
    coordinator and refuses a cross-owner one, so a personality learns through the
    governed pipeline rather than a private one.
    """

    owner: str
    data_root: str | Path | None = None
    _coordinator: Any | None = None

    def __post_init__(self) -> None:
        self.owner = owner_for(self.owner)
        if not is_owner(self.owner):
            raise ValueError(f"{self.owner!r} is not a known personality.")

    @property
    def scope(self) -> LearningScope:
        return learning_scope(self.owner)

    def coordinator(self) -> Any:
        """This personality's own learning coordinator (built on first use)."""
        if self._coordinator is None:
            from core.learning_coordinator import LearningCoordinator

            self._coordinator = LearningCoordinator(self.owner, data_root=self.data_root)
        return self._coordinator

    def may_learn_from(self, other: str) -> bool:
        """Whether this personality may read ``other``'s learning."""
        return self.scope.may_learn_from(other)

    def learn_from(self, source_owner: str) -> tuple[Any, ...]:
        """Pull permitted knowledge from ``source_owner`` into this personality.

        Refuses a source the policy withholds **before** the coordinator is
        touched, so a refused pull never opens the source's store. Delegates to
        the coordinator's own governed ``learn_from`` (which re-checks), so the
        two cannot disagree.
        """
        if not self.may_learn_from(source_owner):
            from core.learning_isolation import LearningAccessError

            raise LearningAccessError(
                f"{self.owner} may not learn from {owner_for(source_owner)}."
            )
        return tuple(self.coordinator().learn_from(owner_for(source_owner), reader=self.owner))

    def contributes_training(self) -> bool:
        """Whether this personality's learning may enter an aggregated dataset."""
        return self.scope.contributes

    def status(self) -> str:
        """A readable status line."""
        return self.scope.describe()


def aggregated_dataset_owners(*, include_private: bool = False) -> tuple[str, ...]:
    """The owners whose learning may enter the **primary model's** dataset.

    The main model plus the shareable characters; Natasha is excluded unless a
    trusted caller explicitly passes ``include_private=True`` (and even then the
    capability ceiling refuses her contribution, so the exclusion holds).
    """
    owners = [MAIN_MODEL_OWNER, *SHAREABLE_OWNERS]
    if include_private:
        owners.extend(sorted(PRIVATE_OWNERS))
    return tuple(owners)


def aggregated_dataset_is_clean() -> bool:
    """True when no private owner is in the aggregated dataset (a self-check)."""
    return not (set(aggregated_dataset_owners()) & set(PRIVATE_OWNERS))


__all__ = [
    "AgentLearningScope",
    "LearningScope",
    "aggregated_dataset_is_clean",
    "aggregated_dataset_owners",
    "learning_scope",
]
