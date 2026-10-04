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


def _plant_seed_task() -> SandboxTask:
    """A longer chain: prepare the soil, plant, water, then wait for a sprout."""
    return SandboxTask(
        name="plant-seed",
        description="Grow a sprout: till, plant a seed, water it, and wait.",
        allowed_actions=(
            ActionSpec("till", frozenset({"hard_soil"}), frozenset({"tilled_soil"}),
                       frozenset({"hard_soil"})),
            ActionSpec("plant", frozenset({"tilled_soil", "seed_available"}),
                       frozenset({"seed_planted"}), frozenset({"seed_available"})),
            ActionSpec("water", frozenset({"seed_planted"}), frozenset({"seed_watered"})),
            ActionSpec("wait", frozenset({"seed_watered"}), frozenset({"sprout"}),
                       frozenset({"seed_watered"})),
        ),
        initial_facts=frozenset({"hard_soil", "seed_available"}),
        max_steps=6,
    )


def _boil_water_task() -> SandboxTask:
    """A branch with a detour: fill and heat a pot before the tea can steep."""
    return SandboxTask(
        name="boil-water",
        description="Make tea: fill the pot, heat the water, then steep the tea.",
        allowed_actions=(
            ActionSpec("fill", frozenset({"empty_pot", "water_tap"}),
                       frozenset({"pot_filled"}), frozenset({"empty_pot"})),
            ActionSpec("heat", frozenset({"pot_filled", "stove_on"}),
                       frozenset({"water_boiling"}), frozenset({"pot_filled"})),
            ActionSpec("steep", frozenset({"water_boiling", "tea_available"}),
                       frozenset({"tea_ready"})),
            ActionSpec("stir", frozenset({"pot_filled"}), frozenset()),
        ),
        initial_facts=frozenset({"empty_pot", "water_tap", "stove_on", "tea_available"}),
        max_steps=6,
    )


def _recharge_lamp_task() -> SandboxTask:
    """An off-path held-out task: open a panel, swap the cell, then switch on."""
    return SandboxTask(
        name="recharge-lamp",
        description="Recharge a lamp: open the panel, replace the cell, switch it on.",
        allowed_actions=(
            ActionSpec("open", frozenset({"panel_closed"}), frozenset({"panel_open"}),
                       frozenset({"panel_closed"})),
            ActionSpec("swap", frozenset({"panel_open", "spare_cell"}),
                       frozenset({"cell_replaced"}), frozenset({"spare_cell"})),
            ActionSpec("switch", frozenset({"cell_replaced"}),
                       frozenset({"lamp_on"}), frozenset({"cell_replaced"})),
            ActionSpec("shake", frozenset({"panel_closed"}), frozenset()),
        ),
        initial_facts=frozenset({"panel_closed", "spare_cell"}),
        max_steps=6,
    )


TASKS: tuple[SandboxTask, ...] = (
    _wet_garden_task(),
    _stack_blocks_task(),
    _plant_seed_task(),
    _boil_water_task(),
)

# Tasks the learning agents never train on; used to measure generalisation.
HELD_OUT_TASKS: tuple[SandboxTask, ...] = (_light_candle_task(), _recharge_lamp_task())


def task_by_name(name: str) -> SandboxTask:
    for task in (*TASKS, *HELD_OUT_TASKS):
        if task.name == name:
            return task
    raise KeyError(f"Unknown sandbox task: {name}")
