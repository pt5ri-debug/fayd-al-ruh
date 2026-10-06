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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.domain_knowledge import discipline_labels
from core.petroleum_tasks import (
    PETROLEUM_HELD_OUT_TASKS,
    PETROLEUM_TASKS,
    success_fact_for,
)
from core.sandbox_environment import SandboxTask
from core.state import State
from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher
from ml.learner import evaluate_policy, train_policy


def petroleum_success(task_name: str) -> Callable[[State], bool]:
    """The success predicate for a petroleum task: its milestone fact is present."""
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

    @property
    def overall_success(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(outcome.held_out_success for outcome in self.outcomes) / len(self.outcomes)

    @property
    def disciplines_covered(self) -> tuple[str, ...]:
        return tuple(self.discipline_success)

    def as_dict(self) -> dict[str, object]:
        return {
            "overall_success": round(self.overall_success, 6),
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
) -> TaskOutcome:
    success = petroleum_success(task.name)

    agent, policy_report = train_policy(
        task, success, episodes=episodes, hasher=hasher, seed=seed
    )
    held_out = evaluate_policy(task, success, agent, episodes=20, hasher=hasher)

    brain = CognitiveBrain(task, success, seed=seed, replay_batch_size=replay_batch)
    brain_report = brain.learn(episodes=episodes)

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
) -> PetroleumReport:
    """Train the policy and brain on every petroleum task.

    Held-out disciplines are trained fresh (never reused) so their success is a
    genuine generalisation check rather than memorised performance.
    """
    hasher = FeatureHasher(64)
    trained_tasks = PETROLEUM_TASKS
    held_out_tasks = PETROLEUM_HELD_OUT_TASKS if include_held_out else ()
    tasks = (*trained_tasks, *held_out_tasks)

    outcomes: list[TaskOutcome] = []
    for task in tasks:
        outcomes.append(
            _train_task(
                task,
                episodes=episodes,
                hasher=hasher,
                seed=seed,
                replay_batch=replay_batch,
                save_dir=save_dir,
            )
        )

    def grouped_success(names: tuple[str, ...]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for outcome in outcomes:
            if outcome.name in names:
                buckets.setdefault(outcome.discipline, []).append(outcome.held_out_success)
        return {key: sum(values) / len(values) for key, values in buckets.items()}

    trained_names = tuple(task.name for task in trained_tasks)
    held_out_names = tuple(task.name for task in held_out_tasks)

    return PetroleumReport(
        outcomes=tuple(outcomes),
        discipline_success=grouped_success(trained_names),
        generalisation_success=grouped_success(held_out_names),
        discipline_labels=discipline_labels(),
    )


def format_report(report: PetroleumReport) -> str:
    """A readable, Arabic-labelled summary of the petroleum learning run."""
    lines = ["تدريب مهام هندسة البترول (Petroleum engineering learning)", ""]
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
    lines.append(f"Overall held-out success: {report.overall_success:.2f}")
    return "\n".join(lines)


__all__ = [
    "PETROLEUM_LEARNING_EPISODES",
    "PetroleumReport",
    "TaskOutcome",
    "format_report",
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
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args(argv)

    report = train_petroleum(
        episodes=args.episodes,
        seed=args.seed,
        replay_batch=args.replay_batch,
        save_dir=args.save_dir,
    )
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
