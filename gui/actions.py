"""Sidebar actions for the GUI, reusing the CLI menu's own functions.

Each action is one of the ``run_*`` functions the numbered menu already calls,
so the GUI and the CLI share behaviour instead of duplicating it. The GUI runs
them on a worker thread and streams their printed output into the transcript.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cli.menu import (
    run_curriculum,
    run_demo,
    run_evaluation,
    run_learn,
    run_llm,
    run_ml,
    run_release,
    run_selfcheck,
)

Command = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class GuiAction:
    """A named, ready-to-run command shown as a sidebar button."""

    key: str
    label: str
    command: Command
    argv: tuple[str, ...] = ()


# Ordered for the sidebar. Long-running training and evaluation stay available
# here even though they are hidden from the numbered CLI menu.
ACTIONS: tuple[GuiAction, ...] = (
    GuiAction("release", "Train everything for a release", run_release, ()),
    GuiAction("evaluation", "Run the evaluation suite", run_evaluation, ()),
    GuiAction("llm", "Train the language model", run_llm, ("--steps", "200", "--kb-max-chunks", "2000")),
    GuiAction("ml", "Train the ML pipeline", run_ml, ("--episodes", "60")),
    GuiAction("learn", "Continuous learning and regression gate", run_learn, ()),
    GuiAction("curriculum", "Show the petroleum-engineering curriculum", run_curriculum, ()),
    GuiAction("demo", "Run the local cognitive demo", run_demo, ()),
    GuiAction("selfcheck", "Self-check the install", run_selfcheck, ()),
)


def action_by_key(key: str) -> GuiAction:
    """Look up one action by its stable key. Raises ``KeyError`` if unknown."""
    for action in ACTIONS:
        if action.key == key:
            return action
    raise KeyError(key)


__all__ = ["ACTIONS", "Command", "GuiAction", "action_by_key"]
