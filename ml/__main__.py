"""CLI: run the machine-learning pipeline end to end.

    python -m ml                  # train the cognitive brain + policies, print a report
    python -m ml --episodes 400   # longer learning session
    python -m ml --save-dir ml/artifacts
    python -m ml --json           # machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from core.interaction_reward import (
    InteractionRewardModel,
    outcomes_from_evaluation_corpus,
)
from core.progress import ProgressReporter
from core.sandbox_tasks import HELD_OUT_TASKS, TASKS
from core.state import State

from ml.checkpoint import TrainingProgress
from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher, Transition, build_classification_dataset, train_val_split
from ml.general_learning import train_relation_outcome
from ml.interaction_learner import (
    evaluate_interaction_policy,
    recorded_action_outcomes,
    train_interaction_policy,
)
from ml.learner import QLearningAgent, evaluate_policy, train_policy
from ml.model_based_planner import train_model_based_policy
from ml.models import MLP, save_model
from ml.trainer import Trainer

_SUCCESS_FACTS = {
    "wet-garden": "wet_soil",
    "stack-blocks": "tower_built",
    "plant-seed": "sprout",
    "boil-water": "tea_ready",
    "light-candle": "lit_candle",
    "recharge-lamp": "lamp_on",
}


def _success_for(task_name: str) -> Callable[[State], bool]:
    fact = _SUCCESS_FACTS[task_name]
    return lambda state: fact in state.facts


def _train_task_with_resume(
    task,
    success: Callable[[State], bool],
    *,
    episodes: int,
    resume: bool,
    save_dir: Path | None,
    hasher: FeatureHasher,
    replay_batch: int,
    progress_stream,
    show_progress: bool,
    progress_state: TrainingProgress | None,
):
    """Train one task's policy and brain, continuing a saved run when asked.

    Returns ``(held_out, policy_stats, brain_stats, brain)``. With ``resume`` a
    task already trained to the budget is loaded from its saved policy and brain
    (no retraining) and one trained part-way is continued for the remaining
    episodes; either way the artifacts and the ledger are written back so the
    next run continues from here.
    """
    policy_path = save_dir / f"policy_{task.name}.npz" if save_dir else None
    brain_dir = save_dir / f"brain_{task.name}" if save_dir else None
    already = progress_state.completed(task.name) if progress_state is not None else 0
    brain_saved = brain_dir is not None and (brain_dir / "brain.json").exists()
    trained = (
        resume
        and already >= episodes
        and policy_path is not None
        and policy_path.exists()
        and brain_saved
    )

    if trained:
        agent = QLearningAgent.load(policy_path)  # type: ignore[arg-type]
        brain = CognitiveBrain.load(brain_dir, task, success)  # type: ignore[arg-type]
        brain.replay_batch_size = replay_batch
        held_out = evaluate_policy(task, success, agent, episodes=20, hasher=hasher)
        recent = brain.reflect().recent_error
        if progress_state is not None:
            progress_state.record(task.name, already, artifact=brain_dir)
        policy_stats = {
            "train_success_rate": held_out,
            "held_out_success_rate": held_out,
            "final_epsilon": agent.epsilon,
        }
        brain_stats = {
            "initial_error": recent,
            "final_error": recent,
            "error_improvement": 0.0,
            "policy_success_rate": held_out,
            "memory_size": float(len(brain.memory)),
            "self_tuned_learning_rate": brain.learning_rate,
            "episodes_trained": float(brain.episodes_trained),
        }
        return held_out, policy_stats, brain_stats, brain

    start = already if (resume and already < episodes) else 0
    remaining = max(1, episodes - start)

    loaded_agent: QLearningAgent | None = None
    if start > 0 and policy_path is not None and policy_path.exists():
        loaded_agent = QLearningAgent.load(policy_path)
    agent, policy_report = train_policy(
        task,
        success,
        episodes=remaining,
        hasher=hasher,
        agent=loaded_agent,
        progress=ProgressReporter(
            f"policy {task.name}", remaining, stream=progress_stream, enabled=show_progress
        ),
    )
    held_out = evaluate_policy(task, success, agent, episodes=20, hasher=hasher)

    resumed_brain: CognitiveBrain | None = None
    if start > 0 and brain_saved:
        resumed_brain = CognitiveBrain.load(brain_dir, task, success)  # type: ignore[arg-type]
        resumed_brain.replay_batch_size = replay_batch
    brain = resumed_brain if resumed_brain is not None else CognitiveBrain(task, success, seed=0, replay_batch_size=replay_batch)
    brain_report = brain.learn(
        episodes=remaining,
        progress=ProgressReporter(
            f"brain {task.name}", remaining, stream=progress_stream, enabled=show_progress
        ),
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        agent.save(save_dir / f"policy_{task.name}.npz")
        brain.save(save_dir / f"brain_{task.name}")
    if progress_state is not None:
        progress_state.record(task.name, start + remaining, artifact=brain_dir)

    policy_stats = {
        "train_success_rate": policy_report.success_rate_window,
        "held_out_success_rate": held_out,
        "final_epsilon": policy_report.final_epsilon,
    }
    brain_stats = {
        "initial_error": brain_report.initial_error,
        "final_error": brain_report.final_error,
        "error_improvement": brain_report.error_improvement,
        "policy_success_rate": brain_report.policy_success_rate,
        "memory_size": float(brain_report.memory_size),
        "self_tuned_learning_rate": brain_report.learning_rate,
        "episodes_trained": float(brain.episodes_trained),
    }
    return held_out, policy_stats, brain_stats, brain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ML learning pipeline.")
    parser.add_argument("--episodes", type=int, default=300, help="brain training episodes")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="directory for artifacts (default: a portable data folder)",
    )
    parser.add_argument(
        "--replay-batch",
        type=int,
        default=32,
        help="past experiences re-learned per episode (0 disables replay)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Continue from a previously saved run in --save-dir (the default; "
        "kept for compatibility). Use --fresh to start over.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved progress and retrain every task from scratch.",
    )
    parser.add_argument(
        "--real-reward",
        action="store_true",
        help="also train a policy whose reward is learned from observed interaction",
    )
    parser.add_argument(
        "--model-based",
        action="store_true",
        help="also compare model-free vs Dyna-Q (learning from a learned world model)",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=10,
        help="imagined updates per real step for Dyna-Q (--model-based)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="seeds averaged for the model-based comparison (--model-based)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final report, not the live progress lines.",
    )
    parser.add_argument(
        "--install-weights",
        action="store_true",
        help="Install the committed weights pack's policy entries into --save-dir "
        "(or the default artifacts folder) before training, then continue from them. "
        "Each entry is verified first (digest, kind, architecture), so a clean clone "
        "can continue the project's own trained policies without naming a file.",
    )
    args = parser.parse_args(argv)

    if args.install_weights:
        from core.paths import artifacts_dir
        from core.weights_pack import KIND_POLICY, install_pack, load_pack, select_keys

        save_dir = args.save_dir or (artifacts_dir() / "petroleum")
        save_dir.mkdir(parents=True, exist_ok=True)
        args.save_dir = save_dir
        pack = load_pack()
        keys = select_keys(pack, KIND_POLICY)
        if not keys:
            print("The weights pack holds no policy entries to install.")
            return 3
        install_report = install_pack(pack, keys=keys)
        print(install_report.describe())
        if not install_report.ok:
            print("Some pack policies did not install; training was not started.")
            return 4

    # With --json, stdout must stay pure JSON, so live progress goes to stderr.
    show_progress = not args.quiet
    progress_stream = sys.stderr if args.json else None

    def announce(message: str) -> None:
        if show_progress:
            print(message, file=progress_stream or sys.stdout, flush=True)

    # Resuming needs a folder to resume *from*; without one a run is simply fresh,
    # so this is a no-op rather than an error (the folder is what carries state).
    resume = bool(args.resume) and not args.fresh and args.save_dir is not None

    hasher = FeatureHasher(64)
    policy_summary: dict[str, dict[str, float]] = {}
    brain_summary: dict[str, dict[str, float]] = {}
    generalisation_summary: dict[str, dict[str, float]] = {}
    interaction_summary: dict[str, dict[str, float]] = {}
    model_based_summary: dict[str, dict[str, float]] = {}
    experiences: list[Transition] = []

    # Reward learned from observed interaction, kept separate from the sandbox
    # proxy. It is built once from recorded outcomes and shared by every task.
    reward_model = InteractionRewardModel()
    reward_model.record_many(outcomes_from_evaluation_corpus())
    for task in (*TASKS, *HELD_OUT_TASKS):
        reward_model.record_many(recorded_action_outcomes(task))

    # The ledger records how far each task has come, so a second run continues
    # rather than restarts. Its fingerprint excludes the episode budget on
    # purpose: raising --episodes tops a task up instead of invalidating it.
    progress_state: TrainingProgress | None = None
    if resume and args.save_dir is not None:
        progress_state = TrainingProgress(
            args.save_dir,
            scope="ml",
            config={
                "tasks": [t.name for t in (*TASKS, *HELD_OUT_TASKS)],
                "replay_batch": args.replay_batch,
                "real_reward": args.real_reward,
            },
        )
        if progress_state.stale:
            announce("Progress recorded for a different run; starting fresh.")
        elif progress_state.available:
            announce(f"Resuming: {progress_state.describe()}")

    for task_index, task in enumerate(TASKS, start=1):
        success = _success_for(task.name)
        announce(f"[{task_index}/{len(TASKS)}] task: {task.name}")
        # Tier 1 (symbolic Q-learning) and tier 2 (predictive-coding brain with
        # memory + metacognition) are trained together by the resumable helper,
        # which loads a finished task instead of retraining it.
        _held_out, policy_stats, brain_stats, brain = _train_task_with_resume(
            task,
            success,
            episodes=args.episodes,
            resume=resume,
            save_dir=args.save_dir,
            hasher=hasher,
            replay_batch=args.replay_batch,
            progress_stream=progress_stream,
            show_progress=show_progress,
            progress_state=progress_state,
        )
        policy_summary[task.name] = policy_stats
        brain_summary[task.name] = brain_stats
        experiences.extend(brain.memory.items)

    # Generalisation: train a fresh brain on the novel, held-out task and check
    # that the same learning machinery works without task-specific tuning.
    for task in HELD_OUT_TASKS:
        success = _success_for(task.name)
        announce(f"generalisation task: {task.name}")
        _held_out, _policy_stats, gen_stats, brain = _train_task_with_resume(
            task,
            success,
            episodes=args.episodes,
            resume=resume,
            save_dir=args.save_dir,
            hasher=hasher,
            replay_batch=args.replay_batch,
            progress_stream=progress_stream,
            show_progress=show_progress,
            progress_state=progress_state,
        )
        generalisation_summary[task.name] = {
            "initial_error": gen_stats["initial_error"],
            "final_error": gen_stats["final_error"],
            "error_improvement": gen_stats["error_improvement"],
            "policy_success_rate": gen_stats["policy_success_rate"],
        }
        _ = brain

    if progress_state is not None:
        progress_state.save()

    # Reward from observed interaction: train the same Q-learning agent on every
    # task, but let the reward come from the learned outcome model rather than
    # the sandbox success predicate. Success is then measured against the real
    # goal, so a weaker reward source shows up honestly.
    if args.real_reward:
        for task in (*TASKS, *HELD_OUT_TASKS):
            goal = _success_for(task.name)
            agent, report = train_interaction_policy(
                task,
                reward_model,
                goal,
                episodes=args.episodes,
                hasher=hasher,
                progress=ProgressReporter(f"interaction {task.name}", args.episodes, stream=progress_stream, enabled=show_progress),
            )
            held_out = evaluate_interaction_policy(
                task, reward_model, goal, agent, episodes=20, hasher=hasher
            )
            interaction_summary[task.name] = {
                "success_rate": report.success_rate,
                "held_out_success_rate": held_out,
                "final_epsilon": report.final_epsilon,
            }
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                agent.save(args.save_dir / f"interaction_policy_{task.name}.npz")

    # Model-based planning: compare model-free Q-learning with Dyna-Q, which
    # also learns from imagined transitions in a *learned* world model. The
    # budget is kept small on purpose -- that is where a model should pay off.
    if args.model_based:
        comparison_episodes = max(1, min(args.episodes, 40))
        seeds = max(1, args.seeds)
        for task in (*TASKS, *HELD_OUT_TASKS):
            goal = _success_for(task.name)
            free_total = 0.0
            dyna_total = 0.0
            dyna_first = 0
            planned: tuple[str, ...] = ()
            for seed in range(seeds):
                _, free_report = train_policy(
                    task, goal, episodes=comparison_episodes, hasher=hasher, seed=seed
                )
                _, _, dyna_report = train_model_based_policy(
                    task,
                    goal,
                    episodes=comparison_episodes,
                    planning_steps=args.planning_steps,
                    seed=seed,
                    progress=ProgressReporter(
                        f"dyna-q {task.name}",
                        comparison_episodes,
                        stream=progress_stream,
                        enabled=show_progress,
                    ),
                )
                free_total += free_report.success_rate_window
                dyna_total += dyna_report.success_rate
                dyna_first += dyna_report.first_success_episode
                planned = dyna_report.planned_solution
            model_based_summary[task.name] = {
                "episodes": float(comparison_episodes),
                "model_free_success": round(free_total / seeds, 6),
                "dyna_q_success": round(dyna_total / seeds, 6),
                "dyna_q_first_success_episode": round(dyna_first / seeds, 3),
                "planned_solution_steps": float(len(planned)),
            }

    # Supervised classifier trained on the consolidated experience.
    dataset = build_classification_dataset(experiences, hasher)
    train, val = train_val_split(dataset, val_fraction=0.25, seed=1)
    model = MLP(hasher.size, 32, max(2, dataset.num_classes), seed=3)
    trainer = Trainer(learning_rate=0.05, batch_size=8, max_epochs=150, patience=15, seed=2)
    announce("Supervised classifier on consolidated experience ...")
    training = trainer.fit(
        model,
        train,
        val,
        progress=ProgressReporter("classifier", trainer.max_epochs, stream=progress_stream, enabled=show_progress),
    )
    supervised: dict[str, float] = {
        "samples": float(len(dataset)),
        "train_accuracy": training.final.train_accuracy,
        "val_accuracy": training.final.val_accuracy,
        "val_brier": training.final.val_brier,
        "epochs": float(len(training.epochs)),
        "stopped_early": float(training.stopped_early),
    }

    # General knowledge: learn the whole relation graph, not just the sandboxes.
    # The model predicts held-out relations from concept vocabulary alone, so the
    # reported Hit@1 measures real generalisation against the random baseline.
    relation_model, relation_report = train_relation_outcome(
        hasher=hasher,
        save_to=(args.save_dir / "relation_model.npz") if args.save_dir else None,
        progress=ProgressReporter("relation model", 300, stream=progress_stream, enabled=show_progress),
    )
    general_knowledge = relation_report.as_dict()
    _ = relation_model

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, args.save_dir / "experience_classifier.npz")

    if args.json:
        print(
            json.dumps(
                {
                    "policies": policy_summary,
                    "brains": brain_summary,
                    "generalisation": generalisation_summary,
                    "interaction_reward_model": reward_model.summary(),
                    "interaction_policies": interaction_summary,
                    "model_based": model_based_summary,
                    "supervised": supervised,
                    "general_knowledge": general_knowledge,
                },
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
    print("Generalisation (novel held-out tasks):")
    for name, stats in generalisation_summary.items():
        print(
            f"  {name:14s} error {stats['initial_error']:.4f}->{stats['final_error']:.4f} "
            f"policy={stats['policy_success_rate']:.2f}"
        )
    print(
        f"Supervised classifier: samples={int(supervised['samples'])} "
        f"train_acc={supervised['train_accuracy']:.2f} "
        f"val_acc={supervised['val_accuracy']:.2f} brier={supervised['val_brier']:.3f}"
    )
    print(
        f"General knowledge (link prediction): relations={general_knowledge['relations']} "
        f"concepts={general_knowledge['concepts']} "
        f"types={general_knowledge['relation_types']} "
        f"held-out Hit@1={general_knowledge['hit_at_1']:.2f} "
        f"(random 0.25) mean-rank={general_knowledge['mean_rank']:.2f}"
    )
    if args.real_reward and interaction_summary:
        diagnostics = reward_model.summary()
        print("Interaction reward (learned from observed outcomes, no sandbox predicate):")
        print(
            f"  outcomes={int(diagnostics['outcomes_recorded'])} "
            f"forecasts={int(diagnostics['forecasts_recorded'])} "
            f"contexts={int(diagnostics['contexts'])} "
            f"ece={diagnostics['expected_calibration_error']:.3f} "
            f"brier={diagnostics['brier_score']:.3f}"
        )
        for name, stats in interaction_summary.items():
            print(
                f"  {name:14s} train={stats['success_rate']:.2f} "
                f"held-out={stats['held_out_success_rate']:.2f}"
            )
    if args.model_based and model_based_summary:
        print("Model-based planning (Dyna-Q learns from a learned world model):")
        for name, stats in model_based_summary.items():
            print(
                f"  {name:14s} episodes={int(stats['episodes'])} "
                f"model-free={stats['model_free_success']:.2f} "
                f"dyna-q={stats['dyna_q_success']:.2f} "
                f"first-success={stats['dyna_q_first_success_episode']:.1f} "
                f"planned-steps={int(stats['planned_solution_steps'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
