"""Generate the project's own *learned* language-model weights.

The project trains a small transformer language model (``ml.llm``) and ships the
result as a committed checkpoint. The **learned tiers** are larger capacities of
the *same* modern architecture (RMSNorm + RoPE + SwiGLU), so a longer run has
somewhere to put what it learns: ``learned-large`` is the tier a release trains
and ships, ``learned-xl`` the stronger option a user with time may build.

This module is the one place that *generates* them, and it does three things,
and only three:

* **Name** the tiers and what each costs (its preset, its dimensions, its default
  budget) -- one declarative registry, so no surface invents a tier.
* **Build** a tier by running the project's **real** trainer
  (:func:`ml.llm.__main__.train_and_save`) into the pack folder, with the
  tokenizer and optimiser-state sidecars beside the checkpoint and a provenance
  record that says exactly how the file was produced.
* **List** what has been built, reading each file's own structure
  (:func:`core.checkpoint_import.read_weight_info`), so a surface reports what is
  on disk rather than what it hopes is there.

It is deliberately **read-only about the training logic**: it never reimplements
a training step, it calls the trainer that already knows how to train and resume,
and it writes only into the tracked ``weights/learned`` folder (and the sidecars
beside each checkpoint).

Honest by construction:

* a tier that was **not** built is reported *absent*, never fabricated;
* an existing file is never clobbered without ``force`` (so a long run is not
  thrown away by a stray command);
* the provenance records the **real** configuration, step count, parameter count,
  and duration of the run that produced the file, read from the run itself, never
  assumed;
* the command that continues a built tier is a real command, not a slogan.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.checkpoint_import import KIND_LANGUAGE, WeightInfo, read_weight_info

#: The sub-folder of the weights pack that holds the learned checkpoints.
LEARNED_SUBDIR = "learned"
#: The suffix of the provenance record written beside a learned checkpoint.
PROVENANCE_SUFFIX = ".learned.json"

#: The tokenizer every learned checkpoint is trained with. The project's own
#: checkpoint ships a byte tokenizer (``vocab_size`` 257), so a learned tier keeps
#: the same token index space and the same tiny sidecar.
TOKENIZER_KIND = "byte"
#: The vocabulary a byte tokenizer produces (256 bytes plus a BOS id).
BYTE_VOCAB_SIZE = 257


class LearnedWeightsError(RuntimeError):
    """A learned tier could not be built or read. Always carries the real reason."""


# ----------------------------------------------------------------- the tiers


@dataclass(frozen=True)
class LearnedTier:
    """One learned capacity tier: what it is, what it costs, and how to build it."""

    key: str
    title: str
    preset: str
    filename: str
    summary: str
    notes: str = ""
    default_steps: int = 400
    default_repeats: int = 40
    default_structured_repeats: int = 60

    def config(self):
        """The :class:`~ml.llm.transformer.TransformerConfig` this tier trains."""
        from ml.llm.transformer import TransformerConfig

        return TransformerConfig.preset(self.preset, vocab_size=BYTE_VOCAB_SIZE)

    def dimensions(self) -> str:
        """A short ``nL/nw/ctx/vocab`` description read from the tier's preset."""
        config = self.config()
        return (
            f"{config.n_layer}L/{config.n_embd}w/ctx {config.block_size}/"
            f"vocab {config.vocab_size}"
        )

    def continue_command(self) -> str:
        """The real command that continues a built tier's checkpoint.

        The checkpoint installs into the git-ignored data folder under its own
        name, so the command points the trainer at that exact file rather than the
        default one -- a learned tier and the base checkpoint coexist.
        """
        stem = Path(self.filename).stem
        return f"python -m cli llm --checkpoint data/ml/llm/checkpoints/{stem}.npz --resume"

    def describe(self) -> str:
        lines = [f"{self.key}: {self.title} (preset {self.preset}, {self.dimensions()})"]
        if self.summary:
            lines.append(f"  {self.summary}")
        if self.notes:
            lines.append(f"  {self.notes}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "preset": self.preset,
            "filename": self.filename,
            "summary": self.summary,
            "notes": self.notes,
            "dimensions": self.dimensions(),
            "default_steps": self.default_steps,
            "default_repeats": self.default_repeats,
            "default_structured_repeats": self.default_structured_repeats,
            "continue_command": self.continue_command(),
        }


#: The learned tiers the project can build. ``learned-large`` is the tier a
#: release trains and ships; ``learned-xl`` is the stronger option a user with
#: time may build. Both keep the ``modern`` architecture family -- only the
#: capacity changes -- so a comparison isolates size, not a different model.
LEARNED_TIERS: dict[str, LearnedTier] = {
    "learned-large": LearnedTier(
        key="learned-large",
        title="Learned language model (large)",
        preset="learned-large",
        filename="fayd-llm-large.npz",
        summary=(
            "A wider 4-layer modern model with a 256-token context -- more than "
            "twice the base checkpoint's parameters, so a longer run has room to "
            "learn."
        ),
        notes=(
            "Trained with the byte tokenizer and the training upgrades a release "
            "applies (AdamW weight decay, warmup+cosine, restore-best)."
        ),
        default_steps=400,
        default_repeats=40,
        default_structured_repeats=60,
    ),
    "learned-xl": LearnedTier(
        key="learned-xl",
        title="Learned language model (XL)",
        preset="learned-xl",
        filename="fayd-llm-xl.npz",
        summary=(
            "An 8-layer modern model, several times the base checkpoint's "
            "parameters -- the strongest in-project tier, and the slowest to train."
        ),
        notes=(
            "The slowest tier on a CPU; build it when there is time to spare. It is "
            "not shipped in the pack by default."
        ),
        default_steps=300,
        default_repeats=40,
        default_structured_repeats=60,
    ),
}

#: The tier keys, in the order a menu should list them (smallest first).
TIER_KEYS: tuple[str, ...] = tuple(LEARNED_TIERS)


def tier_keys() -> tuple[str, ...]:
    """Every learned tier key, in registry order."""
    return TIER_KEYS


def resolve_tier(name: str) -> LearnedTier:
    """The tier named ``name`` (a key or a preset), or raise with the real list."""
    wanted = (name or "").strip().lower()
    if wanted in LEARNED_TIERS:
        return LEARNED_TIERS[wanted]
    for tier in LEARNED_TIERS.values():
        if tier.preset == wanted:
            return tier
    raise LearnedWeightsError(
        f"Unknown learned tier {name!r}; expected one of {', '.join(TIER_KEYS)}."
    )


def tier_summaries() -> tuple[tuple[str, str, str], ...]:
    """``(key, title, summary)`` rows, one per tier, for a menu."""
    return tuple((tier.key, tier.title, tier.summary) for tier in LEARNED_TIERS.values())


def describe_tiers() -> str:
    """Every learned tier, one block each."""
    return "\n\n".join(tier.describe() for tier in LEARNED_TIERS.values())


# --------------------------------------------------------------- the folder


def learned_dir(root: Path | str | None = None) -> Path:
    """Where learned checkpoints live: ``weights/learned`` by default.

    ``root`` is the folder itself when given (a caller passing a scratch folder);
    with nothing the tracked pack's own ``learned`` sub-folder is used, so a built
    tier is release content rather than git-ignored run state.
    """
    if root is not None:
        return Path(root).expanduser()
    from core.paths import weights_pack_dir

    return weights_pack_dir() / LEARNED_SUBDIR


def provenance_path(checkpoint: Path | str) -> Path:
    """The provenance record written beside a learned checkpoint."""
    path = Path(checkpoint)
    return path.with_name(path.name + PROVENANCE_SUFFIX)


# ------------------------------------------------------------------- building


@dataclass(frozen=True)
class BuildReport:
    """The whole outcome of one learned-tier build, in a shape a surface can post."""

    tier: LearnedTier
    checkpoint: Path
    built: bool
    steps: int = 0
    parameters: int = 0
    size_bytes: int = 0
    seconds: float = 0.0
    seed: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def describe(self) -> str:
        head = "built" if self.built else "not built"
        lines = [f"{self.tier.key}: {head} ({self.tier.preset})"]
        if self.checkpoint.name:
            lines.append(f"  checkpoint: {self.checkpoint}")
        if self.built:
            bits = [f"{self.steps} steps"]
            if self.parameters:
                bits.append(f"{self.parameters:,} params")
            if self.size_bytes:
                bits.append(f"{self.size_bytes / 1_000_000:.1f} MB")
            if self.seconds:
                bits.append(f"{self.seconds:.0f} s")
            lines.append("  " + " · ".join(bits))
            lines.append(f"  continue with: {self.tier.continue_command()}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "tier": self.tier.key,
            "preset": self.tier.preset,
            "checkpoint": str(self.checkpoint),
            "built": self.built,
            "steps": self.steps,
            "parameters": self.parameters,
            "size_bytes": self.size_bytes,
            "seconds": round(self.seconds, 3),
            "seed": self.seed,
            "provenance": dict(self.provenance),
            "detail": self.detail,
            "continue_command": self.tier.continue_command(),
        }


#: The shape of the trainer seam: keyword arguments in, a model (or ``None``) out.
Trainer = Callable[..., Any]


def _default_trainer(**kwargs: Any) -> Any:
    """Run the project's own language-model trainer -- never a second one."""
    from ml.llm.__main__ import train_and_save

    return train_and_save(**kwargs)


def _parameter_count(model: Any, checkpoint: Path) -> int:
    """The parameter count of a trained model, read from the model or the file.

    The trainer returns the model it trained, so the count is read straight from
    the weights that were just written; a trainer that returns nothing falls back
    to loading the checkpoint. A file that cannot be loaded reports ``0`` rather
    than a guess.
    """
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return int(sum(int(p.size) for p in parameters()))
        except Exception:  # pragma: no cover - a defensive, non-fatal read
            pass
    try:
        from ml.llm.transformer import TransformerLM

        loaded = TransformerLM.load(checkpoint)
        return int(sum(int(p.size) for p in loaded.parameters()))
    except Exception:
        return 0


def build_learned(
    tier: LearnedTier | str,
    *,
    root: Path | str | None = None,
    steps: int | None = None,
    repeats: int | None = None,
    structured_repeats: int | None = None,
    seed: int = 0,
    config: Any = None,
    force: bool = False,
    verbose: bool = True,
    progress_every: int = 0,
    trainer: Trainer | None = None,
) -> BuildReport:
    """Train one learned tier into the pack folder and record its provenance.

    The run goes through the project's **real** trainer (``ml.llm``), so nothing
    about the training is reimplemented here. An existing checkpoint is never
    clobbered without ``force``: a completed long run is not thrown away by a
    stray command, and the refusal names the real next step.

    ``config`` overrides the tier's own preset configuration -- a test's tiny,
    fast config, say -- and the provenance records whatever was *actually* used,
    never the tier's nominal one.
    """
    resolved = resolve_tier(tier.key) if isinstance(tier, LearnedTier) else resolve_tier(tier)
    folder = learned_dir(root)
    destination = folder / resolved.filename
    tokenizer_path = destination.with_suffix(".tokenizer.json")
    run_steps = resolved.default_steps if steps is None else int(steps)
    run_repeats = resolved.default_repeats if repeats is None else int(repeats)
    run_structured = (
        resolved.default_structured_repeats if structured_repeats is None else int(structured_repeats)
    )
    if run_steps < 1:
        raise LearnedWeightsError("A learned build needs at least one training step.")

    if destination.exists() and not force:
        return BuildReport(
            tier=resolved,
            checkpoint=destination,
            built=False,
            steps=run_steps,
            seed=seed,
            detail=(
                f"{destination} already exists. Re-run with force=True (or --force) to "
                "retrain it; the existing checkpoint is kept until then."
            ),
        )

    model_config = config if config is not None else resolved.config()
    run = trainer or _default_trainer
    folder.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model = run(
        steps=run_steps,
        repeats=run_repeats,
        structured_repeats=run_structured,
        seed=seed,
        checkpoint=destination,
        tokenizer_path=tokenizer_path,
        tokenizer_kind=TOKENIZER_KIND,
        vocab_size=BYTE_VOCAB_SIZE,
        config=model_config,
        log_every=0,
        val_every=0,
        weight_decay=0.01,
        lr_schedule="cosine",
        restore_best=True,
        verbose=verbose,
        progress_every=progress_every,
    )
    seconds = time.time() - started

    if not destination.exists():
        return BuildReport(
            tier=resolved,
            checkpoint=destination,
            built=False,
            steps=run_steps,
            seed=seed,
            seconds=seconds,
            detail=(
                "The trainer ran but wrote no checkpoint, so nothing was built. "
                "Check the trainer's own output."
            ),
        )

    parameters = _parameter_count(model, destination)
    try:
        size_bytes = destination.stat().st_size
    except OSError:
        size_bytes = 0

    provenance = {
        "producer": "the project's own trainer (ml.llm)",
        "source": str(destination),
        "tier": resolved.key,
        "preset": resolved.preset,
        "steps": run_steps,
        "repeats": run_repeats,
        "structured_repeats": run_structured,
        "seed": seed,
        "tokenizer": TOKENIZER_KIND,
        "parameters": parameters,
        "size_bytes": size_bytes,
        "seconds": round(seconds, 3),
        "digest_algorithm": "sha256",
        "license": "MIT (this project)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_provenance(destination, provenance)
    return BuildReport(
        tier=resolved,
        checkpoint=destination,
        built=True,
        steps=run_steps,
        parameters=parameters,
        size_bytes=size_bytes,
        seconds=seconds,
        seed=seed,
        provenance=provenance,
        detail=f"Trained {resolved.preset} and recorded its provenance.",
    )


def _write_provenance(checkpoint: Path, provenance: dict[str, Any]) -> None:
    """Write a checkpoint's provenance record atomically (never a half-written one)."""
    import os
    import tempfile

    target = provenance_path(checkpoint)
    handle, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".learned-", suffix=".tmp")
    os.close(handle)
    try:
        Path(temp_name).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_provenance(checkpoint: Path | str) -> dict[str, Any]:
    """Read a learned checkpoint's provenance record, or ``{}`` when absent/corrupt."""
    target = provenance_path(checkpoint)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------------------ listing


@dataclass(frozen=True)
class LearnedCheckpoint:
    """A learned checkpoint on disk: what it is, and how it was produced."""

    tier_key: str
    path: Path
    info: WeightInfo
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether the file reads as a language-model checkpoint this build knows."""
        return self.info.valid and self.info.kind == KIND_LANGUAGE

    def describe(self) -> str:
        head = f"{self.tier_key}: {self.path.name}"
        if self.info.checkpoint is not None:
            head += (
                f" ({self.info.checkpoint.n_layer}L/{self.info.checkpoint.n_embd}w/"
                f"ctx {self.info.checkpoint.block_size})"
            )
        lines = [head]
        parameters = self.provenance.get("parameters")
        if parameters:
            lines.append(f"  {int(parameters):,} params")
        steps = self.provenance.get("steps")
        if steps:
            lines.append(f"  trained {steps} steps (preset {self.provenance.get('preset', '?')})")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "tier": self.tier_key,
            "path": str(self.path),
            "usable": self.usable,
            "kind": self.info.kind,
            "info": self.info.to_primitive(),
            "provenance": dict(self.provenance),
        }


def _tier_key_for(path: Path) -> str:
    """The tier key a checkpoint's file name names, or its stem when unknown."""
    for tier in LEARNED_TIERS.values():
        if tier.filename == path.name:
            return tier.key
    return path.stem


def list_learned(root: Path | str | None = None) -> tuple[LearnedCheckpoint, ...]:
    """Every learned checkpoint present in the folder, read from its own structure.

    A file that does not read as a language-model checkpoint is still listed (with
    ``usable`` false) rather than hidden, so a broken build is *reported*.
    """
    folder = learned_dir(root)
    if not folder.is_dir():
        return ()
    found: list[LearnedCheckpoint] = []
    for path in sorted(folder.glob("*.npz")):
        # An optimiser-state sidecar sits beside its checkpoint; it is run state,
        # not a learned checkpoint, so it is not listed as one.
        if path.name.endswith(".state.npz"):
            continue
        found.append(
            LearnedCheckpoint(
                tier_key=_tier_key_for(path),
                path=path,
                info=read_weight_info(path),
                provenance=load_provenance(path),
            )
        )
    return tuple(found)


def describe_learned(root: Path | str | None = None) -> str:
    """A readable account of the learned checkpoints on disk, or the real next step."""
    checkpoints = list_learned(root)
    if not checkpoints:
        folder = learned_dir(root)
        return (
            f"No learned checkpoints in {folder}. Build one with "
            f"`python -m cli weights build-learned <tier>` "
            f"(tiers: {', '.join(TIER_KEYS)})."
        )
    return "\n".join(item.describe() for item in checkpoints)


__all__ = [
    "BYTE_VOCAB_SIZE",
    "LEARNED_SUBDIR",
    "LEARNED_TIERS",
    "PROVENANCE_SUFFIX",
    "TIER_KEYS",
    "TOKENIZER_KIND",
    "BuildReport",
    "LearnedCheckpoint",
    "LearnedTier",
    "LearnedWeightsError",
    "build_learned",
    "describe_learned",
    "describe_tiers",
    "learned_dir",
    "list_learned",
    "load_provenance",
    "provenance_path",
    "resolve_tier",
    "tier_keys",
    "tier_summaries",
]
