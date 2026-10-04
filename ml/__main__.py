"""CLI: run the machine-learning pipeline end to end.

    python -m ml                  # train the cognitive brain + policies, print a report
    python -m ml --episodes 400   # longer learning session
    python -m ml --save-dir ml/artifacts
    python -m ml --json           # machine-readable summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from core.sandbox_tasks import TASKS
from core.state import State
from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher, Transition, build_classification_dataset, train_val_split
from ml.learner import evaluate_policy, train_policy
from ml.models import MLP, save_model
from ml.trainer import Trainer


def _success_for(task_name: str) -> Callable[[State], bool]:
    if task_name == "wet-garden":
        return lambda state: "wet_soil" in state.facts
    return lambda state: "tower_built" in state.facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ML learning pipeline.")
    parser.add_argument("--episodes", type=int, default=300, help="brain training episodes")
    parser.add_argument("--save-dir", type=Path, default=None, help="directory for artifacts")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args(argv)

    hasher = FeatureHasher(64)
    policy_summary: dict[str, dict[str, float]] = {}
    brain_summary: dict[str, dict[str, float]] = {}
    experiences: list[Transition] = []

    for task in TASKS:
        success = _success_for(task.name)

        # Tier 1: symbolic reinforcement learning with a linear Q-function.
        agent, policy_report = train_policy(task, success, episodes=args.episodes, hasher=hasher)
        held_out = evaluate_policy(task, success, agent, episodes=20, hasher=hasher)
        policy_summary[task.name] = {
            "train_success_rate": policy_report.success_rate_window,
            "held_out_success_rate": held_out,
            "final_epsilon": policy_report.final_epsilon,
        }

        # Tier 2: predictive-coding brain with memory + metacognition.
        brain = CognitiveBrain(task, success, seed=0)
        brain_report = brain.learn(episodes=args.episodes)
        brain_summary[task.name] = {
            "initial_error": brain_report.initial_error,
            "final_error": brain_report.final_error,
            "error_improvement": brain_report.error_improvement,
            "policy_success_rate": brain_report.policy_success_rate,
            "memory_size": float(brain_report.memory_size),
            "self_tuned_learning_rate": brain_report.learning_rate,
        }
        experiences.extend(brain.memory.items)

    # Supervised classifier trained on the consolidated experience.
    dataset = build_classification_dataset(experiences, hasher)
    train, val = train_val_split(dataset, val_fraction=0.25, seed=1)
    model = MLP(hasher.size, 32, max(2, dataset.num_classes), seed=3)
    trainer = Trainer(learning_rate=0.05, batch_size=8, max_epochs=150, patience=15, seed=2)
    training = trainer.fit(model, train, val)
    supervised: dict[str, float] = {
        "samples": float(len(dataset)),
        "train_accuracy": training.final.train_accuracy,
        "val_accuracy": training.final.val_accuracy,
        "val_brier": training.final.val_brier,
        "epochs": float(len(training.epochs)),
        "stopped_early": float(training.stopped_early),
    }

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, args.save_dir / "experience_classifier.npz")

    if args.json:
        print(
            json.dumps(
                {"policies": policy_summary, "brains": brain_summary, "supervised": supervised},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print("Policy (symbolic RL):")
    for name, stats in policy_summary.items():
        print(
            f"  {name:14s} train={stats['train_success_rate']:.2f} "
            f"held-out={stats['held_out_success_rate']:.2f}"
        )
    print("Cognitive brain (predictive coding):")
    for name, stats in brain_summary.items():
        print(
            f"  {name:14s} error {stats['initial_error']:.4f}->{stats['final_error']:.4f} "
            f"improvement={stats['error_improvement']:+.4f} "
            f"policy={stats['policy_success_rate']:.2f}"
        )
    print(
        f"Supervised classifier: samples={int(supervised['samples'])} "
        f"train_acc={supervised['train_accuracy']:.2f} "
        f"val_acc={supervised['val_accuracy']:.2f} brier={supervised['val_brier']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
