"""A learned action-selection protocol usable by the agent loop.

The agent loop should not depend on any particular learning algorithm. It only
needs something that, given the current facts and the available actions, returns
the name of the action it prefers (or ``None`` to fall back to the existing
declarative decision engine).

This keeps the learned policy swappable and keeps ``core`` free of numpy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ActionPolicy(Protocol):
    """Selects one action name from the facts and the available actions."""

    def select(self, facts: frozenset[str], actions: tuple[str, ...]) -> str | None: ...
