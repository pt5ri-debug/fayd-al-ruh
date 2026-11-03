"""The personality memory access layer: per-personality memory, policy-enforced.

Each personality has its **own** durable memory (the Phase-2 long-term memory, one
SQLite database per owner) and its own learning record (the file-backed
:class:`core.character_learning.CharacterLearning` and the append-only
:class:`core.character_learning_store.CharacterLearningStore`). This module is the
**one** read/write interface over them, so a personality can never reach another's
private memory by opening a path: the path is resolved from the owner through the
isolation policy, and every cross-owner read is checked **before** a path exists.

It is deliberately thin -- it reuses :mod:`core.learning_isolation` for the policy,
:mod:`core.memory_management`'s ``PersonalityMemoryRegistry`` for the durable
memory, and :mod:`core.character_learning_store` for the records -- so there is
one store per personality and one policy, not a second of either.

Dependency-free and Tk-free: importing it loads no model and touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.learning_isolation import (
    OWNERS,
    AccessDecision,
    LearningAccessError,
    decide,
    is_owner,
    owner_for,
)


@dataclass(frozen=True)
class MemoryAccessDecision:
    """Whether ``reader`` may read ``owner``'s memory, and why."""

    reader: str
    owner: str
    allowed: bool
    reason: str

    def describe(self) -> str:
        verdict = "allowed" if self.allowed else "denied"
        return f"{self.reader} -> {self.owner}: {verdict} ({self.reason})"


class PersonalityMemoryAccess:
    """The memory access layer for one reader (a personality or the main model).

    ``reader`` is fixed at construction, so the layer can only ever act *as* that
    personality: there is no method that takes another reader's identity, which is
    what makes "a personality cannot read another's private memory" a property of
    the type rather than a promise. ``data_root`` lets a test keep everything under
    a temp folder; the layout beneath it is the isolation policy's own.
    """

    def __init__(self, reader: str, *, data_root: str | Path | None = None) -> None:
        resolved = owner_for(reader)
        if not is_owner(resolved):
            raise LearningAccessError(f"{reader!r} is not a known memory reader.")
        self.reader = resolved
        self._data_root = Path(data_root) if data_root is not None else None
        self._memory_registry: Any | None = None

    # -------------------------------------------------------------- policy

    def can_read(self, owner: str) -> bool:
        """Whether this reader may read ``owner``'s memory (a pure policy check)."""
        return decide(self.reader, owner).allowed

    def check(self, owner: str) -> MemoryAccessDecision:
        """The decision for reading ``owner``'s memory, with a readable reason."""
        decision: AccessDecision = decide(self.reader, owner)
        return MemoryAccessDecision(
            decision.reader, decision.owner, decision.allowed, decision.reason
        )

    def readable_owners(self) -> tuple[str, ...]:
        """Every owner whose memory this reader may read (its own first)."""
        return tuple(owner for owner in OWNERS if self.can_read(owner))

    def assert_can_read(self, owner: str) -> None:
        """Raise :class:`LearningAccessError` when the read is not permitted.

        Called *before* any path is resolved, so a refused read never even opens a
        directory -- the same ordering the project's own stores use.
        """
        decision = decide(self.reader, owner)
        if not decision.allowed:
            raise LearningAccessError(decision.reason)

    # ------------------------------------------------------------ own store

    def own_memory(self) -> Any:
        """This reader's **own** long-term memory (no check -- it is its own)."""
        return self._registry().owner_memory(self.reader)

    def memory_for(self, owner: str) -> Any:
        """``owner``'s long-term memory **as read by this reader**, or refuse."""
        self.assert_can_read(owner)
        return self._registry().owner_memory(owner_for(owner))

    # -------------------------------------------------------- learning store

    def own_learning(self) -> Any:
        """This reader's own durable learning record (facts + exchanges)."""
        from core.character_learning import CharacterLearning

        return CharacterLearning(self.reader, root=self._root())

    def own_records(self) -> Any:
        """This reader's own append-only learning store (provenance-carrying)."""
        from core.character_learning_store import CharacterLearningStore

        return CharacterLearningStore(self.reader, root=self._root())

    def learning_for(self, owner: str) -> Any:
        """``owner``'s learning record **as read by this reader**, or refuse.

        A cross-owner read is checked first; the record is then opened read-only
        (``CharacterLearning`` only ever reads until asked to write, and this
        layer exposes no write here), so reading a shareable character's learning
        never mutates it.
        """
        self.assert_can_read(owner)
        from core.character_learning import CharacterLearning

        return CharacterLearning(owner_for(owner), root=self._root())

    # --------------------------------------------------------------- helpers

    def _root(self) -> Path | None:
        return self._data_root

    def _registry(self) -> Any:
        if self._memory_registry is None:
            from core.memory_management import PersonalityMemoryRegistry

            self._memory_registry = PersonalityMemoryRegistry(data_root=self._data_root)
        return self._memory_registry

    def describe(self) -> str:
        readable = ", ".join(self.readable_owners())
        return f"{self.reader} may read memory of: {readable}"


def memory_access(reader: str, *, data_root: str | Path | None = None) -> PersonalityMemoryAccess:
    """The memory access layer for ``reader`` (a convenience constructor)."""
    return PersonalityMemoryAccess(reader, data_root=data_root)


__all__ = [
    "MemoryAccessDecision",
    "PersonalityMemoryAccess",
    "memory_access",
]
