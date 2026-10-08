"""Procedural tasks derived from the project's own knowledge.

The hand-written petroleum catalog in :mod:`core.petroleum_tasks` gives each
discipline a task to learn. This module widens that: every procedure recorded in
the petroleum corpus (and, through the corpus, the curriculum that shares its
concepts) becomes a short ordered action chain, so the learners practice the
*procedures* the project knows, not only the hand-written ones.

A derived task is deliberately simple and inspectable: the step order is the
procedure's own order, and success is reaching the final step. It is not a
claim about the real world; it is a faithful translation of recorded know-how
into the same safe symbolic sandbox the rest of the learning stack uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.petroleum_corpus import entries
from core.sandbox_environment import SandboxTask
from core.state import State
from core.symbolic_simulation import ActionSpec

_SUCCESS_FACT = "procedure_done"


@dataclass(frozen=True)
class ProceduralTask:
    """A derived task together with the knowledge it was built from."""

    task: SandboxTask
    entry_key: str
    discipline: str
    steps: int


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.casefold()).strip("_")


def _build_task(entry_key: str, discipline: str, title: str, procedures: tuple[str, ...]) -> SandboxTask:
    step_facts = [f"p_{_slug(entry_key)}_{index}" for index in range(len(procedures))]
    actions: list[ActionSpec] = []
    last = len(step_facts) - 1
    for index, fact in enumerate(step_facts):
        prerequisite = "procedure_ready" if index == 0 else step_facts[index - 1]
        effects = frozenset({fact}) | (frozenset({_SUCCESS_FACT}) if index == last else frozenset())
        actions.append(ActionSpec(f"step_{index + 1}", frozenset({prerequisite}), effects))
    # Two actions that are applicable from the start but lead nowhere, so the
    # learner must discover the ordered chain rather than act arbitrarily.
    actions.append(ActionSpec("inspect", frozenset({"procedure_ready"}), frozenset({"inspected"})))
    actions.append(ActionSpec("review", frozenset({"procedure_ready"}), frozenset({"reviewed"})))
    return SandboxTask(
        name=f"procedure-{entry_key}",
        description=f"Follow the recorded procedure for {title}.",
        discipline=discipline,
        allowed_actions=tuple(actions),
        initial_facts=frozenset({"procedure_ready"}),
        max_steps=len(procedures) + 3,
    )


def procedural_tasks() -> tuple[ProceduralTask, ...]:
    """One derived task per corpus entry that records at least one procedure."""
    tasks: list[ProceduralTask] = []
    for entry in entries():
        procedures = tuple(step for step in entry.procedures if step.strip())
        if not procedures:
            continue
        tasks.append(
            ProceduralTask(
                task=_build_task(entry.key, entry.discipline, entry.title, procedures),
                entry_key=entry.key,
                discipline=entry.discipline,
                steps=len(procedures),
            )
        )
    return tuple(tasks)


def procedural_success(_task_name: str):
    """The success predicate shared by every derived task: the last step is reached."""
    return lambda state: _SUCCESS_FACT in state.facts


def procedural_sandbox_tasks() -> tuple[SandboxTask, ...]:
    return tuple(item.task for item in procedural_tasks())


__all__ = [
    "ProceduralTask",
    "procedural_sandbox_tasks",
    "procedural_success",
    "procedural_tasks",
]
