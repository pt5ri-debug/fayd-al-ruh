"""The held-out gate for the learned capacity tiers and their pack integration.

Every claim the learned-weights layer makes is checked here on the **real**
engine, with **negative controls** so an over-eager or broken implementation
fails as surely as a missing one. The gate builds tiny real checkpoints inside a
scratch folder, so it stays fast, and it measures the properties that matter:

1. each learned tier names a real transformer preset and a real architecture, and
   its parameters are **measured** from the built model rather than asserted --
   the larger tier really is larger;
2. building a tier writes the checkpoint, its tokenizer sidecar, and a provenance
   file beside it, and the provenance records the tier, the preset, and the step
   count -- never a fabricated field;
3. a rebuilt tier is **not** retrained over an existing checkpoint unless
   ``force`` is set (the negative control that stops a long run being thrown away
   by a stray command);
4. listing the learned checkpoints reads them from their own structure, so a file
   that is not a learned checkpoint is **not** listed as one;
5. the pack packs a learned tier under its own key and installs it **beside** the
   base checkpoint (never over it), and the entry verifies against its own digest;
6. an unknown tier is **refused** with the reason, never silently built as
   something else.

:func:`learned_weights_accuracy` is the metric the evaluation wires in.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from core.learned_weights import (
    LEARNED_TIERS,
    TIER_KEYS,
    BuildReport,
    LearnedCheckpoint,
    LearnedTier,
    LearnedWeightsError,
    build_learned,
    list_learned,
    load_provenance,
    provenance_path,
    resolve_tier,
)
from core.weights_pack import generate_manifest, installed_destination, load_pack, verify_entry

#: The tiny architecture the gate trains with, so the checks are fast. It shares
#: the real tiers' tokenizer and preset shape, only smaller.
_GATE_CONFIG = {
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "block_size": 32,
    "tie_weights": True,
}


@dataclass(frozen=True)
class LearnedJudgement:
    """One gate judgement: a name, whether it passed, and why."""

    name: str
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _tiny_config():
    from ml.llm.transformer import TransformerConfig

    return TransformerConfig(**_GATE_CONFIG, vocab_size=257)


def _build(root: Path, tier: str = "learned-large", *, force: bool = False) -> BuildReport:
    with contextlib.redirect_stdout(io.StringIO()):
        return build_learned(
            tier,
            root=root,
            steps=2,
            repeats=1,
            structured_repeats=1,
            config=_tiny_config(),
            force=force,
            verbose=False,
        )


def judgements() -> tuple[LearnedJudgement, ...]:
    """Run the whole gate and return one judgement per check."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with mock.patch.dict(os.environ, {"FAYD_DATA_DIR": str(root / "data")}):
            return _run_checks(root)


def _run_checks(root: Path) -> tuple[LearnedJudgement, ...]:
    out: list[LearnedJudgement] = []

    # 1. Each tier names a real preset, and the larger tier really is larger.
    preset_ok = True
    sizes: dict[str, int] = {}
    from ml.llm.transformer import TransformerConfig, TransformerLM

    for key, tier in LEARNED_TIERS.items():
        try:
            config = TransformerConfig.preset(tier.preset, vocab_size=257)
            model = TransformerLM(config, seed=0)
        except Exception:
            preset_ok = False
            continue
        sizes[key] = sum(int(p.size) for p in model.parameters())
    larger_ok = (
        "learned-large" in sizes
        and "learned-xl" in sizes
        and sizes["learned-xl"] > sizes["learned-large"]
    )
    out.append(
        LearnedJudgement(
            "each_tier_names_a_real_preset_and_the_larger_tier_is_larger",
            preset_ok and larger_ok,
            f"measured {sizes}; the XL tier is the larger one"
            if preset_ok and larger_ok
            else "a tier named an unknown preset or the XL tier was not larger",
        )
    )

    # 2. Building writes the checkpoint, its tokenizer, and honest provenance.
    build_root = root / "built"
    report = _build(build_root)
    checkpoint = report.checkpoint
    provenance = load_provenance(checkpoint)
    artifacts_ok = (
        report.built
        and checkpoint.exists()
        and checkpoint.with_suffix(".tokenizer.json").exists()
        and provenance_path(checkpoint).exists()
    )
    provenance_ok = (
        provenance.get("tier") == "learned-large"
        and provenance.get("preset") == LEARNED_TIERS["learned-large"].preset
        and provenance.get("steps") == 2
        and bool(provenance.get("created_at"))
    )
    out.append(
        LearnedJudgement(
            "building_writes_the_checkpoint_sidecar_and_honest_provenance",
            artifacts_ok and provenance_ok,
            "the checkpoint, tokenizer, and provenance were written with the tier's own fields"
            if artifacts_ok and provenance_ok
            else "an artifact or a provenance field was missing",
        )
    )

    # 3. A second build does not retrain over the existing checkpoint (negative
    #    control): the recorded mtime is unchanged, and force does retrain.
    before_mtime = checkpoint.stat().st_mtime_ns
    again = _build(build_root)
    untouched = (not again.built) and checkpoint.stat().st_mtime_ns == before_mtime
    forced = _build(build_root, force=True)
    retrained = forced.built
    out.append(
        LearnedJudgement(
            "an_existing_checkpoint_is_not_retrained_without_force",
            untouched and retrained,
            "a second build left the checkpoint alone and --force rebuilt it"
            if untouched and retrained
            else "a stray build retrained an existing checkpoint",
        )
    )

    # 4. Listing reads the checkpoints from their own structure; a foreign .npz is
    #    reported as *not usable* rather than mistaken for a learned checkpoint.
    foreign = build_root / "not-a-checkpoint.npz"
    foreign.write_bytes(b"not an npz")
    listed = list_learned(build_root)
    by_name = {item.path.name: item for item in listed}
    learned = by_name.get("fayd-llm-large.npz")
    other = by_name.get("not-a-checkpoint.npz")
    listing_ok = (
        learned is not None
        and learned.usable
        and learned.provenance.get("tier") == "learned-large"
        and other is not None
        and not other.usable
    )
    out.append(
        LearnedJudgement(
            "listing_reads_learned_checkpoints_and_reports_a_foreign_file",
            listing_ok,
            "the learned checkpoint was listed and usable and a foreign file was reported unusable"
            if listing_ok
            else "listing misread the folder",
        )
    )

    # 5. The pack packs the tier under its own key and installs it beside the base.
    pack_root = root / "pack"
    generate_manifest(
        root=pack_root,
        checkpoint=root / "absent-base.npz",
        learned_dir=build_root,
        include_policies=False,
    )
    pack = load_pack(pack_root)
    entry = pack.get("learned-large")
    destination = installed_destination(entry) if entry is not None else None
    pack_ok = (
        pack.available
        and entry is not None
        and entry.kind == "language-model"
        and entry.destination_name == "fayd-llm-large.npz"
        and destination is not None
        and destination.name == "fayd-llm-large.npz"
        and verify_entry(entry).ok
    )
    out.append(
        LearnedJudgement(
            "the_pack_packs_a_learned_tier_under_its_key_beside_the_base_checkpoint",
            pack_ok,
            "the tier packed under its own key and installs beside the base checkpoint"
            if pack_ok
            else "the learned tier was not packed or would overwrite the base checkpoint",
        )
    )

    # 6. An unknown tier is refused with the reason (negative control).
    try:
        resolve_tier("enormous")
        refused = False
    except LearnedWeightsError as error:
        refused = "Unknown learned tier" in str(error)
    out.append(
        LearnedJudgement(
            "an_unknown_tier_is_refused_with_the_reason",
            refused,
            "an unknown tier was refused by name"
            if refused
            else "an unknown tier was accepted",
        )
    )

    return tuple(out)


def learned_weights_accuracy() -> float:
    """The share of the gate's judgements that passed."""
    results = judgements()
    if not results:
        return 0.0
    return sum(1 for item in results if item.passed) / len(results)


__all__ = [
    "TIER_KEYS",
    "LearnedCheckpoint",
    "LearnedJudgement",
    "LearnedTier",
    "judgements",
    "learned_weights_accuracy",
]
