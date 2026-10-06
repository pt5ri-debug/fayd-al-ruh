"""Train the whole project and persist every training artifact for release.

This is the single entry point that makes a release "ready to launch": it runs
every learning layer the project has -- the base cognitive pipeline, the
petroleum-engineering tasks, and the in-project language model -- and writes the
resulting artifacts plus a manifest that records exactly what was trained, on
which knowledge, with which metrics, and when.

Everything is stored inside the project (``ml/artifacts`` and
``ml/llm/checkpoints`` by default) so a release is self-contained: clone, build,
and the trained state is already there. The run is deterministic for a fixed
seed, so a manifest can be re-verified.

    python -m ml.release                      # train everything, write a manifest
    python -m ml.release --episodes 120 --llm-steps 4000
    python -m ml.release --manifest-only      # re-verify without retraining
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.curriculum_knowledge import (
    course_keys,
    courses,
    curriculum_coverage_score,
    seed_curriculum,
)
from core.domain_knowledge import discipline_keys, disciplines, seed_ontology
from core.ontology import Ontology
from core.paths import artifacts_dir, checkpoints_dir, default_checkpoint
from core.version import __version__
from ml.llm.corpus import build_corpus, curriculum_sentences, structured_examples
from ml.llm.__main__ import train_and_save
from ml.petroleum import train_petroleum

RELEASE_MANIFEST_NAME = "release_manifest.json"
DEFAULT_EPISODES = 120
DEFAULT_LLM_STEPS = 4000

Manifest = dict[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    """Inputs that determine a release run; recorded in the manifest."""

    episodes: int = DEFAULT_EPISODES
    llm_steps: int = DEFAULT_LLM_STEPS
    llm_repeats: int = 40
    llm_structured_repeats: int = 60
    seed: int = 0
    replay_batch: int = 32
    include_base: bool = True


def _git_revision() -> str:
    """Best-effort short commit hash; empty when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def knowledge_summary() -> Manifest:
    """The seeded knowledge a release was trained on, counted from the source."""
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return {
        "disciplines": len(discipline_keys()),
        "courses": len(course_keys()),
        "discipline_concepts": sum(len(d.concepts) for d in disciplines()),
        "course_concepts": sum(len(c.concepts) for c in courses()),
        "discipline_relations": sum(len(d.relations) for d in disciplines()),
        "course_relations": sum(len(c.relations) for c in courses()),
        "ontology_concepts": len(ontology.concepts),
        "ontology_relations": len(ontology.relations),
        "curriculum_coverage": round(curriculum_coverage_score(ontology), 6),
    }


def corpus_summary(config: TrainingConfig) -> Manifest:
    """The language-model training text, counted from the generator."""
    text = build_corpus(
        repeats=config.llm_repeats, structured_repeats=config.llm_structured_repeats
    )
    return {
        "curriculum_sentences": len(curriculum_sentences()),
        "structured_examples": len(structured_examples()),
        "characters": len(text),
        "repeats": config.llm_repeats,
        "structured_repeats": config.llm_structured_repeats,
    }


def _base_training(config: TrainingConfig, save_dir: Path) -> Manifest:
    """Run the base ML pipeline (policies, brain, classifier) via its CLI."""
    command = [
        sys.executable,
        "-m",
        "ml",
        "--episodes",
        str(config.episodes),
        "--replay-batch",
        str(config.replay_batch),
        "--save-dir",
        str(save_dir),
        "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def train_release(
    config: TrainingConfig | None = None,
    *,
    save_dir: Path | None = None,
    checkpoint: Path | None = None,
    write_manifest: bool = True,
) -> Manifest:
    """Train every layer and (optionally) write the release manifest."""
    config = config or TrainingConfig()
    save_dir = save_dir or artifacts_dir()
    checkpoint = checkpoint or checkpoints_dir() / "fayd-llm.npz"
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest: Manifest = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "config": {
            "episodes": config.episodes,
            "llm_steps": config.llm_steps,
            "llm_repeats": config.llm_repeats,
            "llm_structured_repeats": config.llm_structured_repeats,
            "seed": config.seed,
            "replay_batch": config.replay_batch,
            "include_base": config.include_base,
        },
        "knowledge": knowledge_summary(),
        "corpus": corpus_summary(config),
        "artifacts": {
            "save_dir": str(save_dir),
            "checkpoint": str(checkpoint),
        },
    }

    if config.include_base:
        print("Training base cognitive pipeline ...")
        manifest["base"] = _base_training(config, save_dir)

    print("Training petroleum-engineering tasks ...")
    petroleum = train_petroleum(
        episodes=config.episodes,
        seed=config.seed,
        replay_batch=config.replay_batch,
        save_dir=save_dir / "petroleum",
    )
    manifest["petroleum"] = petroleum.as_dict()

    print(f"Training language model ({config.llm_steps} steps) ...")
    model = train_and_save(
        steps=config.llm_steps,
        repeats=config.llm_repeats,
        structured_repeats=config.llm_structured_repeats,
        seed=config.seed,
        checkpoint=checkpoint,
    )
    manifest["language_model"] = {
        "steps": config.llm_steps,
        "parameters": int(sum(p.size for p in model.parameters())),
        "checkpoint": str(checkpoint),
    }

    if write_manifest:
        manifest_path = save_dir / RELEASE_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(manifest_path)
        print(f"Wrote release manifest: {manifest_path}")

    return manifest


def format_manifest(manifest: Manifest) -> str:
    """A short, Arabic-friendly summary of a release manifest."""
    knowledge = manifest["knowledge"]
    config = manifest["config"]
    lines = [
        f"إصدار Fayd Al-Ruh {manifest['version']}",
        f"  وقت التدريب: {manifest['trained_at']}",
        f"  المراجعة: {manifest.get('git_revision') or '—'}",
        f"  التخصصات: {knowledge['disciplines']}  المقررات: {knowledge['courses']}",
        f"  مفاهيم المنهج: {knowledge['course_concepts']}  "
        f"علاقات المنهج: {knowledge['course_relations']}",
        f"  تغطية المنهج: {knowledge['curriculum_coverage']:.4f}",
        f"  حلقات التدريب: {config['episodes']}  خطوات اللغة: {config['llm_steps']}",
    ]
    petroleum = manifest.get("petroleum")
    if isinstance(petroleum, dict):
        lines.append(f"  نجاح المهام البترولية: {petroleum['overall_success']:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the whole project for release.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--llm-steps", type=int, default=DEFAULT_LLM_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-batch", type=int, default=32)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-base", action="store_true", help="skip the base ML pipeline")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="do not retrain; only rewrite the manifest from current knowledge",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = TrainingConfig(
        episodes=args.episodes,
        llm_steps=args.llm_steps,
        seed=args.seed,
        replay_batch=args.replay_batch,
        include_base=not args.skip_base,
    )

    if args.manifest_only:
        save_dir = args.save_dir or artifacts_dir()
        manifest: Manifest = {
            "version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "config": {
                "episodes": config.episodes,
                "llm_steps": config.llm_steps,
                "llm_repeats": config.llm_repeats,
                "llm_structured_repeats": config.llm_structured_repeats,
                "seed": config.seed,
                "replay_batch": config.replay_batch,
                "include_base": config.include_base,
            },
            "knowledge": knowledge_summary(),
            "corpus": corpus_summary(config),
            "artifacts": {
                "save_dir": str(save_dir),
                "checkpoint": str(args.checkpoint or default_checkpoint()),
            },
        }
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / RELEASE_MANIFEST_NAME
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(path)
    else:
        manifest = train_release(
            config, save_dir=args.save_dir, checkpoint=args.checkpoint
        )

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_manifest(manifest))
    return 0


__all__ = [
    "DEFAULT_EPISODES",
    "DEFAULT_LLM_STEPS",
    "RELEASE_MANIFEST_NAME",
    "TrainingConfig",
    "corpus_summary",
    "format_manifest",
    "knowledge_summary",
    "main",
    "train_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
