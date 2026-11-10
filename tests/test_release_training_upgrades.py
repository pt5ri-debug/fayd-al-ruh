"""Phase 6: the release pipeline applies the training upgrades.

Phases 3-5 added AdamW decay, an LR schedule, best-checkpoint restore, and
gradient accumulation to ``ml.llm``, but a *release* never used them: the
committed checkpoint was trained with the plain loop. These tests pin that a
release now (a) turns the upgrades on by default, (b) records them in the
manifest, and (c) exposes them on the CLI.
"""

from __future__ import annotations

import json

from ml.release import (
    RELEASE_MANIFEST_NAME,
    TrainingConfig,
    _config_manifest,
    main,
    train_release,
)


def _fast_config(**overrides) -> TrainingConfig:
    base = {
        "episodes": 3,
        "llm_steps": 5,
        "llm_repeats": 1,
        "llm_structured_repeats": 1,
        "include_base": False,
    }
    return TrainingConfig(**{**base, **overrides})


def test_release_config_enables_the_training_upgrades_by_default():
    config = TrainingConfig()
    assert config.weight_decay == 0.01
    assert config.lr_schedule == "cosine"
    assert config.restore_best is True
    assert config.accum_steps == 1
    assert config.warmup == 40
    assert config.max_grad_norm == 1.0


def test_manifest_records_the_training_upgrades():
    config = _fast_config(weight_decay=0.02, lr_schedule="linear", accum_steps=2)
    recorded = _config_manifest(config)
    assert recorded["weight_decay"] == 0.02
    assert recorded["lr_schedule"] == "linear"
    assert recorded["restore_best"] is True
    assert recorded["accum_steps"] == 2
    assert recorded["warmup"] == 40


def test_train_release_writes_the_upgrades_into_the_manifest(tmp_path):
    config = _fast_config(weight_decay=0.02, accum_steps=2)
    checkpoint = tmp_path / "llm" / "fayd-llm.npz"
    manifest = train_release(config, save_dir=tmp_path, checkpoint=checkpoint)

    stored = json.loads((tmp_path / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["config"]["weight_decay"] == 0.02
    assert stored["config"]["accum_steps"] == 2
    assert stored["config"]["restore_best"] is True
    assert manifest["config"]["lr_schedule"] == "cosine"


def test_train_release_applies_the_upgrades_without_error(tmp_path):
    """A run with every upgrade on (decay, schedule, best, accumulation) works."""
    config = _fast_config(
        weight_decay=0.01,
        lr_schedule="cosine",
        restore_best=True,
        accum_steps=2,
        warmup=2,
        max_grad_norm=1.0,
    )
    checkpoint = tmp_path / "llm" / "fayd-llm.npz"
    manifest = train_release(config, save_dir=tmp_path, checkpoint=checkpoint)
    assert checkpoint.exists()
    assert manifest["language_model"]["steps"] == 5
    assert manifest["config"]["restore_best"] is True


def test_release_cli_accepts_the_training_flags(tmp_path, capsys):
    checkpoint = tmp_path / "llm" / "fayd-llm.npz"
    code = main(
        [
            "--episodes", "3",
            "--llm-steps", "5",
            "--skip-base",
            "--save-dir", str(tmp_path),
            "--checkpoint", str(checkpoint),
            "--weight-decay", "0.02",
            "--lr-schedule", "linear",
            "--accum-steps", "2",
            "--warmup", "2",
            "--quiet",
        ]
    )
    assert code == 0
    stored = json.loads((tmp_path / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["config"]["weight_decay"] == 0.02
    assert stored["config"]["lr_schedule"] == "linear"
    assert stored["config"]["accum_steps"] == 2
    assert stored["config"]["restore_best"] is True


def test_release_cli_no_restore_best_is_recorded(tmp_path):
    checkpoint = tmp_path / "llm" / "fayd-llm.npz"
    code = main(
        [
            "--episodes", "3",
            "--llm-steps", "5",
            "--skip-base",
            "--save-dir", str(tmp_path),
            "--checkpoint", str(checkpoint),
            "--no-restore-best",
            "--quiet",
        ]
    )
    assert code == 0
    stored = json.loads((tmp_path / RELEASE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["config"]["restore_best"] is False
