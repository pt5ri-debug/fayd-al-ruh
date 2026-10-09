"""Tests for resuming language-model training.

The model weights are the committed checkpoint; the Adam moments and step count
are a run-state sidecar. These tests pin the contract: a resume continues from
the saved weights and moments, falls back to a fresh run when there is nothing
to resume, and refuses a sidecar that does not match the architecture.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.llm.__main__ import main, train_and_save
from ml.llm.state import load_training_state, save_training_state, state_path
from ml.llm.tokenizer import ByteTokenizer
from ml.llm.transformer import TransformerConfig, TransformerLM, _lr_at, train_language_model
from ml.models import Adam

_SMALL = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=16)


def _stream(size: int = 400) -> list[int]:
    text = ("water makes the soil wet and the reservoir holds oil. " * 20)[:size]
    return ByteTokenizer().encode(text, bos=False)


def test_training_state_round_trips_moments_and_step(tmp_path):
    model = TransformerLM(_SMALL, seed=0)
    optimizer = Adam(model.parameters())
    train_language_model(model, _stream(), steps=3, batch_size=4, optimizer=optimizer)

    path = state_path(tmp_path / "fayd-llm.npz")
    assert path.name == "fayd-llm.state.npz"
    save_training_state(optimizer, 3, path)

    loaded = load_training_state(path, model)
    assert loaded is not None
    restored, steps = loaded
    assert steps == 3
    assert restored.state.t == 3
    for before, after in zip(optimizer.state.m, restored.state.m):
        assert np.allclose(before, after)
    for before, after in zip(optimizer.state.v, restored.state.v):
        assert np.allclose(before, after)


def test_missing_training_state_is_absent_not_an_error(tmp_path):
    model = TransformerLM(_SMALL, seed=0)
    assert load_training_state(tmp_path / "absent.state.npz", model) is None


def test_training_state_from_a_different_architecture_is_rejected(tmp_path):
    small = TransformerLM(_SMALL, seed=0)
    optimizer = Adam(small.parameters())
    train_language_model(small, _stream(), steps=2, batch_size=4, optimizer=optimizer)
    path = tmp_path / "state.npz"
    save_training_state(optimizer, 2, path)

    bigger = TransformerLM(TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=16), seed=0)
    with pytest.raises(ValueError):
        load_training_state(path, bigger)


def test_supplied_optimizer_keeps_its_moments_across_a_resume():
    model = TransformerLM(_SMALL, seed=0)
    optimizer = Adam(model.parameters())
    train_language_model(model, _stream(), steps=3, batch_size=4, optimizer=optimizer)
    moments_after_first = [m.copy() for m in optimizer.state.m]

    # A second call with the same optimiser must keep learning from those moments,
    # not reset them to zero.
    train_language_model(model, _stream(), steps=3, batch_size=4, optimizer=optimizer, start_step=3)
    assert optimizer.state.t == 6
    assert any(not np.allclose(before, after) for before, after in zip(moments_after_first, optimizer.state.m))


def test_lr_schedule_spans_the_extended_horizon_when_resuming():
    # Resuming at step 3 for 3 more steps must use the schedule for 6 total steps,
    # so the first resumed step is _lr_at(4, steps=6) rather than a frozen value.
    model = TransformerLM(_SMALL, seed=0)
    report = train_language_model(
        model, _stream(), steps=3, batch_size=4, warmup=1, start_step=3
    )
    assert report.learning_rates[0] == pytest.approx(_lr_at(4, steps=6, base_lr=3e-3, warmup=1))


def test_resume_continues_from_the_saved_checkpoint(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=4, repeats=1, structured_repeats=1, checkpoint=checkpoint, log_every=0, val_every=0
    )

    resumed = train_and_save(
        steps=4,
        repeats=1,
        structured_repeats=1,
        checkpoint=checkpoint,
        log_every=0,
        val_every=0,
        resume=True,
    )
    fresh = train_and_save(
        steps=8,
        repeats=1,
        structured_repeats=1,
        checkpoint=tmp_path / "fresh.npz",
        log_every=0,
        val_every=0,
    )

    state = np.load(state_path(checkpoint))
    assert int(state["steps"][0]) == 8
    assert int(state["t"][0]) == 8
    # Continuing from 4 steps must not reproduce a run trained from scratch for 8.
    assert not np.allclose(resumed.token_embedding, fresh.token_embedding)


def test_resume_without_a_checkpoint_starts_fresh(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    model = train_and_save(
        steps=3,
        repeats=1,
        structured_repeats=1,
        checkpoint=checkpoint,
        log_every=0,
        val_every=0,
        resume=True,
    )
    assert checkpoint.exists()
    assert (tmp_path / "fayd-llm.state.npz").exists()
    assert int(np.load(state_path(checkpoint))["steps"][0]) == 3
    assert model.config.vocab_size == 257


def test_resume_does_not_retrain_the_saved_tokenizer(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint, log_every=0, val_every=0
    )
    # Mark the tokenizer file so we can prove a resume leaves it untouched.
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    marker = '{"kind": "byte", "merges": [], "sentinel": true}'
    tokenizer_path.write_text(marker, encoding="utf-8")

    train_and_save(
        steps=2,
        repeats=1,
        structured_repeats=1,
        checkpoint=checkpoint,
        log_every=0,
        val_every=0,
        resume=True,
    )
    assert tokenizer_path.read_text(encoding="utf-8") == marker


def test_cli_resume_flag_is_wired(tmp_path, capsys):
    checkpoint = tmp_path / "fayd-llm.npz"
    code = main(
        [
            "--resume",
            "--checkpoint",
            str(checkpoint),
            "--steps",
            "2",
            "--repeats",
            "1",
            "--structured-repeats",
            "1",
            "--val-every",
            "0",
        ]
    )
    assert code == 0
    assert checkpoint.exists()
    assert "fresh run" in capsys.readouterr().out
