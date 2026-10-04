"""A small catalog of clearly-specified sandbox tasks.

Each task fixes its own allow-list and success predicate, so the agent always
operates against an explicit contract rather than an open environment.
"""

from __future__ import annotations

from core.sandbox_environment import SandboxTask
from core.symbolic_simulation import ActionSpec


def _wet_garden_task() -> SandboxTask:
    return SandboxTask(
        name="wet-garden",
        description="Wet the soil using only the water action.",
        allowed_actions=(
            ActionSpec("water", frozenset({"dry_soil"}), frozenset({"wet_soil"})),
            ActionSpec("wait", frozenset({"dry_soil"}), frozenset()),
        ),
        initial_facts=frozenset({"dry_soil"}),
        max_steps=4,
    )


def _stack_blocks_task() -> SandboxTask:
    return SandboxTask(
        name="stack-blocks",
        description="Stack a block on a table to reach a two-block tower.",
        allowed_actions=(
            ActionSpec("pick", frozenset({"block_on_table"}), frozenset({"block_held"}),
                       frozenset({"block_on_table"})),
            ActionSpec("place", frozenset({"block_held", "table_clear"}), frozenset({"tower_built"}),
                       frozenset({"block_held"})),
        ),
        initial_facts=frozenset({"block_on_table", "table_clear"}),
        max_steps=4,
    )


def _light_candle_task() -> SandboxTask:
    return SandboxTask(
        name="light-candle",
        description="Light a candle by first striking a match, then applying it.",
        allowed_actions=(
            ActionSpec("strike", frozenset({"match_available"}), frozenset({"match_lit"})),
            ActionSpec("light", frozenset({"unlit_candle", "match_lit"}),
                       frozenset({"lit_candle"}), frozenset({"unlit_candle"})),
        ),
        initial_facts=frozenset({"unlit_candle", "match_available"}),
        max_steps=4,
    )


TASKS: tuple[SandboxTask, ...] = (_wet_garden_task(), _stack_blocks_task())

# Tasks the learning agents never train on; used to measure generalisation.
HELD_OUT_TASKS: tuple[SandboxTask, ...] = (_light_candle_task(),)


def task_by_name(name: str) -> SandboxTask:
    for task in (*TASKS, *HELD_OUT_TASKS):
        if task.name == name:
            return task
    raise KeyError(f"Unknown sandbox task: {name}")
