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
from core.petroleum_corpus import corpus_coverage_score, entries as corpus_entries, seed_petroleum_corpus
from core.ontology import Ontology
from core.paths import artifacts_dir, checkpoints_dir, default_checkpoint
from core.version import __version__
from ml.knowledge_corpus import (
    authored_documents,
    knowledge_sample,
    repository_key,
)
from ml.llm.corpus import (
    build_corpus,
    corpus_sentences,
    corpus_units,
    curriculum_sentences,
    knowledge_base_sentences,
    repository_sentences,
    structured_examples,
)
from ml.llm.__main__ import train_and_save
from ml.llm.evaluate import document_split, evaluate_model
from ml.llm.tokenizer import TOKENIZER_KINDS, load_tokenizer
from ml.llm.transformer import PRESETS, TransformerConfig, TransformerLM
from ml.petroleum import train_petroleum

RELEASE_MANIFEST_NAME = "release_manifest.json"
DEFAULT_EPISODES = 120
DEFAULT_LLM_STEPS = 4000
# The relevance-filtered bulk sample a release trains its language model on.
DEFAULT_KB_CHUNKS = 15000

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
    llm_preset: str = "release-1"
    tokenizer_kind: str = "byte"
    tokenizer_vocab_size: int = 1024
    kb_max_chunks: int = DEFAULT_KB_CHUNKS
    kb_repeats: int = 1
    include_runtime: bool = True
    include_procedural: bool = True
    max_procedural: int | None = None
    measure_perplexity: bool = False


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
    """The knowledge a release was trained on, counted from every repository."""
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    seed_petroleum_corpus(ontology)
    authored = authored_documents()
    return {
        "disciplines": len(discipline_keys()),
        "courses": len(course_keys()),
        "discipline_concepts": sum(len(d.concepts) for d in disciplines()),
        "course_concepts": sum(len(c.concepts) for c in courses()),
        "discipline_relations": sum(len(d.relations) for d in disciplines()),
        "course_relations": sum(len(c.relations) for c in courses()),
        "corpus_entries": len(corpus_entries()),
        "ontology_concepts": len(ontology.concepts),
        "ontology_relations": len(ontology.relations),
        "curriculum_coverage": round(curriculum_coverage_score(ontology), 6),
        "corpus_coverage": round(corpus_coverage_score(ontology), 6),
        "authored_documents": len(authored),
        "authored_by_repository": _count_by(authored),
    }


def _count_by(documents: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        key = repository_key(document)
        counts[key] = counts.get(key, 0) + 1
    return counts


def corpus_summary(config: TrainingConfig) -> Manifest:
    """The language-model training text, counted from the generator."""
    text = build_corpus(
        repeats=config.llm_repeats,
        structured_repeats=config.llm_structured_repeats,
        kb_max_chunks=config.kb_max_chunks,
        kb_repeats=config.kb_repeats,
        include_runtime=config.include_runtime,
    )
    sample = knowledge_sample(
        kb_max_chunks=config.kb_max_chunks, include_runtime=config.include_runtime
    )
    return {
        "curriculum_sentences": len(curriculum_sentences()),
        "corpus_sentences": len(corpus_sentences()),
        "repository_sentences": len(repository_sentences(include_runtime=config.include_runtime)),
        "knowledge_base_sentences": len(knowledge_base_sentences(kb_max_chunks=config.kb_max_chunks)),
        "structured_examples": len(structured_examples()),
        "characters": len(text),
        "repeats": config.llm_repeats,
        "kb_max_chunks": config.kb_max_chunks,
        "kb_repeats": config.kb_repeats,
        "include_runtime": config.include_runtime,
        "structured_repeats": config.llm_structured_repeats,
        "tokenizer_kind": config.tokenizer_kind,
        "tokenizer_vocab_size": config.tokenizer_vocab_size,
        "document_counts": sample.counts,
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


def _match_preset(config: TransformerConfig) -> str | None:
    """The capacity tier a config matches, so a manifest names the real size."""
    for name, spec in PRESETS.items():
        if all(getattr(config, key) == value for key, value in spec.items()):
            return name
    return None


def _tokenizer_kind(checkpoint: Path, default: str) -> str:
    """The tokenizer a checkpoint was saved with, read from its sidecar file."""
    sidecar = checkpoint.with_suffix(".tokenizer.json")
    if sidecar.exists():
        try:
            return str(json.loads(sidecar.read_text(encoding="utf-8")).get("kind", default))
        except (ValueError, OSError):
            return default
    return default


def describe_checkpoint(
    checkpoint: str | Path,
    *,
    steps: int | None = None,
    preset: str | None = None,
    tokenizer_kind: str = "byte",
    measure_perplexity: bool = False,
    seed: int = 0,
    model: TransformerLM | None = None,
) -> Manifest:
    """Describe a saved language-model checkpoint for the manifest.

    Reads the checkpoint when a model is not passed, so a manifest can be
    rewritten with `--manifest-only` for the checkpoint that is actually
    committed -- the honest way to record the model a release ships.
    """
    checkpoint = Path(checkpoint)
    if model is None and checkpoint.exists():
        model = TransformerLM.load(checkpoint)
    parameters = int(sum(p.size for p in model.parameters())) if model is not None else None
    config = model.config if model is not None else None
    if config is not None:
        preset = _match_preset(config) or preset
        tokenizer_kind = _tokenizer_kind(checkpoint, tokenizer_kind)
    info: Manifest = {
        "steps": steps,
        "parameters": parameters,
        "checkpoint": str(checkpoint),
        "preset": preset,
        "tokenizer": tokenizer_kind,
    }
    if config is not None:
        info["architecture"] = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "dropout": config.dropout,
            "tie_weights": config.tie_weights,
        }
    if measure_perplexity and model is not None:
        split = document_split(corpus_units(), val_fraction=0.1, test_fraction=0.1, seed=seed)
        tokenizer = load_tokenizer(checkpoint.with_suffix(".tokenizer.json"))
        model.eval()
        report = evaluate_model(model, tokenizer, split, seed=seed)
        info["perplexity"] = {
            "train": round(report.train_ppl, 6),
            "validation": round(report.validation_ppl, 6),
            "test": round(report.test_ppl, 6),
            "test_tokens": report.test_tokens,
            "bits_per_byte": round(report.test_bits_per_byte, 6),
        }
    return info


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
            "llm_preset": config.llm_preset,
            "tokenizer_kind": config.tokenizer_kind,
            "tokenizer_vocab_size": config.tokenizer_vocab_size,
            "kb_max_chunks": config.kb_max_chunks,
            "kb_repeats": config.kb_repeats,
            "include_runtime": config.include_runtime,
            "include_procedural": config.include_procedural,
            "max_procedural": config.max_procedural,
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
        include_procedural=config.include_procedural,
        max_procedural=config.max_procedural,
    )
    manifest["petroleum"] = petroleum.as_dict()

    print(f"Training language model ({config.llm_steps} steps) ...")
    model = train_and_save(
        steps=config.llm_steps,
        repeats=config.llm_repeats,
        structured_repeats=config.llm_structured_repeats,
        seed=config.seed,
        checkpoint=checkpoint,
        tokenizer_kind=config.tokenizer_kind,
        vocab_size=config.tokenizer_vocab_size,
        kb_max_chunks=config.kb_max_chunks,
        kb_repeats=config.kb_repeats,
        include_runtime=config.include_runtime,
        preset=config.llm_preset,
    )
    manifest["language_model"] = describe_checkpoint(
        checkpoint,
        steps=config.llm_steps,
        preset=config.llm_preset,
        tokenizer_kind=config.tokenizer_kind,
        measure_perplexity=config.measure_perplexity,
        seed=config.seed,
        model=model,
    )

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
    """A short summary of a release manifest."""
    knowledge = manifest["knowledge"]
    config = manifest["config"]
    lines = [
        f"Fayd Al-Ruh release {manifest['version']}",
        f"  Trained at: {manifest['trained_at']}",
        f"  Revision: {manifest.get('git_revision') or '—'}",
        f"  Disciplines: {knowledge['disciplines']}  Courses: {knowledge['courses']}",
        f"  Curriculum concepts: {knowledge['course_concepts']}  "
        f"Curriculum relations: {knowledge['course_relations']}",
        f"  Curriculum coverage: {knowledge['curriculum_coverage']:.4f}",
        f"  Training episodes: {config['episodes']}  Language steps: {config['llm_steps']}",
    ]
    petroleum = manifest.get("petroleum")
    if isinstance(petroleum, dict):
        lines.append(f"  Petroleum task success: {petroleum['overall_success']:.4f}")
        if "procedural_tasks" in petroleum:
            procedural = petroleum["procedural_tasks"]
            lines.append(
                f"  Procedural tasks: {procedural['covered']}/{procedural['total']} "
                f"({petroleum.get('procedural_coverage', 0.0):.0%})"
            )
    corpus = manifest.get("corpus")
    if isinstance(corpus, dict) and "knowledge_base_sentences" in corpus:
        lines.append(
            f"  Corpus: {corpus['characters']} chars "
            f"(repositories={corpus['repository_sentences']} "
            f"kb={corpus['knowledge_base_sentences']})"
        )
    language_model = manifest.get("language_model")
    if isinstance(language_model, dict):
        parameters = language_model.get("parameters")
        preset = language_model.get("preset") or "—"
        size = f"{parameters:,}" if isinstance(parameters, int) else "—"
        lines.append(f"  Language model: {size} params  preset {preset}")
        perplexity = language_model.get("perplexity")
        if isinstance(perplexity, dict):
            lines.append(
                f"  Language perplexity: validation {perplexity['validation']:.3f}  "
                f"test {perplexity['test']:.3f}  "
                f"bits/byte {perplexity.get('bits_per_byte', float('nan')):.3f}"
            )
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
        "--tokenizer-kind",
        choices=TOKENIZER_KINDS,
        default="byte",
        help="language-model tokenizer: byte (linear, used for the full corpus) or "
        "subword BPE (better compression, but slow to train on a large corpus)",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--preset",
        choices=("small", "release-1", "capable"),
        default="release-1",
        help="language-model capacity tier",
    )
    parser.add_argument(
        "--measure-perplexity",
        action="store_true",
        help="after training, record held-out perplexity in the manifest "
        "(exact, and cheap only when the KB sample is small)",
    )
    parser.add_argument(
        "--kb-max-chunks",
        type=int,
        default=DEFAULT_KB_CHUNKS,
        help="relevance-filtered knowledge-base sample size for language training",
    )
    parser.add_argument("--kb-repeats", type=int, default=1)
    parser.add_argument(
        "--no-runtime-docs",
        action="store_true",
        help="skip documents written at runtime when training the language model",
    )
    parser.add_argument(
        "--no-procedural",
        action="store_true",
        help="skip the procedural tasks derived from the corpus",
    )
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
        llm_preset=args.preset,
        tokenizer_kind=args.tokenizer_kind,
        tokenizer_vocab_size=args.vocab_size,
        kb_max_chunks=args.kb_max_chunks,
        kb_repeats=args.kb_repeats,
        include_runtime=not args.no_runtime_docs,
        include_procedural=not args.no_procedural,
        measure_perplexity=args.measure_perplexity,
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
                "llm_preset": config.llm_preset,
                "tokenizer_kind": config.tokenizer_kind,
                "tokenizer_vocab_size": config.tokenizer_vocab_size,
                "kb_max_chunks": config.kb_max_chunks,
                "kb_repeats": config.kb_repeats,
                "include_runtime": config.include_runtime,
                "include_procedural": config.include_procedural,
                "max_procedural": config.max_procedural,
            },
            "knowledge": knowledge_summary(),
            "corpus": corpus_summary(config),
            "language_model": describe_checkpoint(
                args.checkpoint or default_checkpoint(),
                steps=config.llm_steps,
                preset=config.llm_preset,
                tokenizer_kind=config.tokenizer_kind,
                measure_perplexity=args.measure_perplexity,
                seed=config.seed,
            ),
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
    "describe_checkpoint",
    "format_manifest",
    "knowledge_summary",
    "main",
    "train_release",
]


if __name__ == "__main__":
    raise SystemExit(main())
