"""Petroleum-engineering learning across every discipline.

The same two learning tiers used by ``python -m ml`` -- a symbolic Q-learning
policy and the predictive-coding cognitive brain -- are trained here on the
petroleum task catalog (``core.petroleum_tasks``). Each task belongs to a
discipline, so the report says how well the machinery learns reservoir,
drilling, production, petrophysics, and the rest, and whether it generalises to
disciplines it never trained on.

Nothing here is petroleum-specific machinery: the point is that one learner,
given discipline-shaped tasks, covers the field.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.domain_knowledge import discipline_labels
from core.petroleum_tasks import (
    PETROLEUM_HELD_OUT_TASKS,
    PETROLEUM_TASKS,
    success_fact_for,
)
from core.progress import ProgressReporter
from core.sandbox_environment import SandboxTask
from core.state import State

from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher
from ml.learner import evaluate_policy, train_policy
from ml.procedural_tasks import procedural_sandbox_tasks, procedural_success


def petroleum_success(task_name: str) -> Callable[[State], bool]:
    """The success predicate for a petroleum task: its milestone fact is present."""
    if task_name.startswith("procedure-"):
        return procedural_success(task_name)
    fact = success_fact_for(task_name)
    return lambda state: fact in state.facts


# Episodes per task used by the evaluation gate. Enough for every discipline to
# converge while keeping the suite fast (a couple of seconds).
PETROLEUM_LEARNING_EPISODES = 80


def petroleum_learning_accuracy(episodes: int = PETROLEUM_LEARNING_EPISODES) -> float:
    """Mean held-out success over every trained discipline.

    A single number the evaluation suite can track as a regression metric: it
    falls if any discipline stops being learnable.
    """
    report = train_petroleum(episodes=episodes, include_held_out=False)
    if not report.discipline_success:
        return 0.0
    return sum(report.discipline_success.values()) / len(report.discipline_success)


def petroleum_generalisation_accuracy(episodes: int = PETROLEUM_LEARNING_EPISODES) -> float:
    """Mean success over the disciplines the learner never trained on.

    ``petroleum_learning_accuracy`` measures disciplines whose tasks *were* in
    the training set, so a memorising learner scores well on it. This one trains
    the same machinery on the trained catalog and scores the **held-out**
    disciplines, so it is the honest half of the question "did it learn the
    field, or the tasks?". It falls to 0.0 when the held-out set is empty, so a
    caller can tell "no check" from "a perfect check".
    """
    report = train_petroleum(episodes=episodes, include_held_out=True)
    if not report.generalisation_success:
        return 0.0
    return sum(report.generalisation_success.values()) / len(report.generalisation_success)


@dataclass(frozen=True)
class TaskOutcome:
    """How one petroleum task learned."""

    name: str
    discipline: str
    train_success: float
    held_out_success: float
    brain_initial_error: float
    brain_final_error: float
    brain_policy_success: float


@dataclass(frozen=True)
class PetroleumReport:
    """Per-task and per-discipline learning results.

    ``discipline_success`` covers the disciplines the agents trained on;
    ``generalisation_success`` covers the held-out disciplines they never saw.
    """

    outcomes: tuple[TaskOutcome, ...]
    discipline_success: dict[str, float]
    generalisation_success: dict[str, float]
    discipline_labels: dict[str, str]
    procedural_covered: int = 0
    procedural_total: int = 0

    @property
    def overall_success(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(outcome.held_out_success for outcome in self.outcomes) / len(self.outcomes)

    @property
    def disciplines_covered(self) -> tuple[str, ...]:
        return tuple(self.discipline_success)

    @property
    def procedural_coverage(self) -> float:
        """Fraction of the derived procedural tasks that were trained on."""
        if self.procedural_total <= 0:
            return 0.0
        return self.procedural_covered / self.procedural_total

    def as_dict(self) -> dict[str, object]:
        return {
            "overall_success": round(self.overall_success, 6),
            "procedural_coverage": round(self.procedural_coverage, 6),
            "procedural_tasks": {"covered": self.procedural_covered, "total": self.procedural_total},
            "discipline_success": {
                key: round(value, 6) for key, value in self.discipline_success.items()
            },
            "generalisation_success": {
                key: round(value, 6) for key, value in self.generalisation_success.items()
            },
            "tasks": [
                {
                    "name": outcome.name,
                    "discipline": outcome.discipline,
                    "train_success": round(outcome.train_success, 6),
                    "held_out_success": round(outcome.held_out_success, 6),
                    "brain_initial_error": round(outcome.brain_initial_error, 6),
                    "brain_final_error": round(outcome.brain_final_error, 6),
                    "brain_policy_success": round(outcome.brain_policy_success, 6),
                }
                for outcome in self.outcomes
            ],
        }


def _train_task(
    task: SandboxTask,
    *,
    episodes: int,
    hasher: FeatureHasher,
    seed: int,
    replay_batch: int,
    save_dir: Path | None,
    verbose: bool = False,
) -> TaskOutcome:
    success = petroleum_success(task.name)
    policy_progress = ProgressReporter(f"policy {task.name}", episodes, enabled=verbose)
    brain_progress = ProgressReporter(f"brain {task.name}", episodes, enabled=verbose)

    agent, policy_report = train_policy(
        task, success, episodes=episodes, hasher=hasher, seed=seed, progress=policy_progress
    )
    held_out = evaluate_policy(task, success, agent, episodes=20, hasher=hasher)

    brain = CognitiveBrain(task, success, seed=seed, replay_batch_size=replay_batch)
    brain_report = brain.learn(episodes=episodes, progress=brain_progress)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        agent.save(save_dir / f"policy_{task.name}.npz")
        brain.save(save_dir / f"brain_{task.name}")

    return TaskOutcome(
        name=task.name,
        discipline=task.discipline,
        train_success=policy_report.success_rate_window,
        held_out_success=held_out,
        brain_initial_error=brain_report.initial_error,
        brain_final_error=brain_report.final_error,
        brain_policy_success=brain_report.policy_success_rate,
    )


def train_petroleum(
    *,
    episodes: int = 120,
    seed: int = 0,
    replay_batch: int = 32,
    save_dir: Path | None = None,
    include_held_out: bool = True,
    include_procedural: bool = False,
    max_procedural: int | None = None,
    verbose: bool = False,
) -> PetroleumReport:
    """Train the policy and brain on every petroleum task.

    Held-out disciplines are trained fresh (never reused) so their success is a
    genuine generalisation check rather than memorised performance. When
    ``include_procedural`` is set, the tasks derived from the corpus procedures
    are trained too, so the learners cover the recorded know-how as well as the
    hand-written catalog; ``max_procedural`` bounds how many derived tasks are
    used (``None`` uses all), which keeps the evaluation gate fast. With
    ``verbose`` set, a live progress line is printed for each task's policy and
    brain.
    """
    hasher = FeatureHasher(64)
    trained_tasks = PETROLEUM_TASKS
    held_out_tasks = PETROLEUM_HELD_OUT_TASKS if include_held_out else ()
    procedural = procedural_sandbox_tasks() if include_procedural else ()
    procedural_total = len(procedural)
    if max_procedural is not None:
        procedural = procedural[:max_procedural]
    tasks = (*trained_tasks, *held_out_tasks, *procedural)

    outcomes: list[TaskOutcome] = []
    for index, task in enumerate(tasks, start=1):
        if verbose:
            print(f"[{index}/{len(tasks)}] petroleum task: {task.name}", file=sys.stderr, flush=True)
        outcomes.append(
            _train_task(
                task,
                episodes=episodes,
                hasher=hasher,
                seed=seed,
                replay_batch=replay_batch,
                save_dir=save_dir,
                verbose=verbose,
            )
        )

    def grouped_success(selected: tuple[SandboxTask, ...]) -> dict[str, float]:
        names = {task.name for task in selected}
        buckets: dict[str, list[float]] = {}
        for outcome in outcomes:
            if outcome.name in names:
                buckets.setdefault(outcome.discipline, []).append(outcome.held_out_success)
        return {key: sum(values) / len(values) for key, values in buckets.items()}

    return PetroleumReport(
        outcomes=tuple(outcomes),
        discipline_success=grouped_success((*trained_tasks, *procedural)),
        generalisation_success=grouped_success(tuple(held_out_tasks)),
        discipline_labels=discipline_labels(),
        procedural_covered=len(procedural),
        procedural_total=procedural_total,
    )


def format_report(report: PetroleumReport) -> str:
    """A readable summary of the petroleum learning run."""
    lines = ["Petroleum engineering learning", ""]
    lines.append("Per task (policy + predictive-coding brain):")
    for outcome in report.outcomes:
        lines.append(
            f"  {outcome.name:24s} [{outcome.discipline:15s}] "
            f"train={outcome.train_success:.2f} held-out={outcome.held_out_success:.2f} "
            f"brain-error={outcome.brain_initial_error:.4f}->{outcome.brain_final_error:.4f} "
            f"brain-policy={outcome.brain_policy_success:.2f}"
        )
    lines.append("")
    lines.append("Per discipline (trained):")
    for key, value in report.discipline_success.items():
        label = report.discipline_labels.get(key, key)
        lines.append(f"  {label:28s} success={value:.2f}")
    if report.generalisation_success:
        lines.append("")
        lines.append("Generalisation (disciplines never trained on):")
        for key, value in report.generalisation_success.items():
            label = report.discipline_labels.get(key, key)
            lines.append(f"  {label:28s} success={value:.2f}")
    lines.append("")
    lines.append(
        f"Procedural tasks covered: {report.procedural_covered}/{report.procedural_total} "
        f"({report.procedural_coverage:.0%})"
    )
    lines.append(f"Overall held-out success: {report.overall_success:.2f}")
    return "\n".join(lines)


__all__ = [
    "PETROLEUM_LEARNING_EPISODES",
    "PetroleumReport",
    "TaskOutcome",
    "format_report",
    "petroleum_generalisation_accuracy",
    "petroleum_learning_accuracy",
    "petroleum_success",
    "train_petroleum",
]


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m ml.petroleum [--episodes N] [--json] [--save-dir DIR]``."""
    import argparse
    import json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Train the petroleum-engineering tasks.")
    parser.add_argument("--episodes", type=int, default=120, help="episodes per task")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--replay-batch", type=int, default=32, help="brain replay batch size")
    parser.add_argument("--save-dir", type=_Path, default=None, help="directory for artifacts")
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="Also train the tasks derived from every corpus procedure.",
    )
    parser.add_argument(
        "--max-procedural",
        type=int,
        default=None,
        help="Bound how many derived procedural tasks to train on.",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final report, not the live per-task progress lines.",
    )
    args = parser.parse_args(argv)

    report = train_petroleum(
        episodes=args.episodes,
        seed=args.seed,
        replay_batch=args.replay_batch,
        save_dir=args.save_dir,
        include_procedural=args.procedural,
        max_procedural=args.max_procedural,
        verbose=not (args.json or args.quiet),
    )
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
