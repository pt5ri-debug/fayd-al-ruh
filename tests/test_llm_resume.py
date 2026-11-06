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
from ml.llm.state import load_step_count, load_training_state, save_training_state, state_path
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
    for before, after in zip(optimizer.state.m, restored.state.m, strict=False):
        assert np.allclose(before, after)
    for before, after in zip(optimizer.state.v, restored.state.v, strict=False):
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
    assert any(not np.allclose(before, after) for before, after in zip(moments_after_first, optimizer.state.m, strict=False))


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


def test_menu_llm_and_release_entries_resume_by_default():
    """The numbered menu's training entries must ask to resume, or they restart."""
    from cli import menu

    by_label = {label: argv for label, _action, argv in menu.MENU}
    assert "--resume" in by_label["Train the language model from all repositories"]
    assert "--resume" in by_label["Train everything for a release"]


def test_gui_llm_and_release_actions_resume_by_default():
    """The GUI sidebar shares the CLI menu's arguments, so it must resume too."""
    from gui import actions

    by_key = {action.key: action.argv for action in actions.ACTIONS}
    assert "--resume" in by_key["llm"]
    assert "--resume" in by_key["release"]


def test_describe_checkpoint_reports_cumulative_steps_when_resumed(tmp_path):
    """A resumed manifest must not claim the checkpoint holds only the new budget."""
    from ml.release import describe_checkpoint

    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint, log_every=0, val_every=0
    )
    info = describe_checkpoint(
        checkpoint, steps=4, total_steps=load_step_count(state_path(checkpoint)), resumed=True
    )
    assert info["resumed"] is True
    assert info["steps"] == 4
    assert info["total_steps"] == 2


def test_describe_checkpoint_defaults_total_steps_to_the_budget(tmp_path):
    """A fresh run's manifest has no separate cumulative figure to report."""
    from ml.release import describe_checkpoint

    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint, log_every=0, val_every=0
    )
    info = describe_checkpoint(checkpoint, steps=7)
    assert info["resumed"] is False
    assert info["steps"] == 7
    assert info["total_steps"] == 7


def test_dropout_rng_is_persisted_so_a_resume_repeats_the_masks(tmp_path):
    """A saved model restores the dropout RNG, not a fresh one.

    Without this, a dropout model resumed from disk drew *different* masks and
    diverged from an uninterrupted run -- "resume" would quietly mean "retrain".
    """
    config = TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=16, dropout=0.3)
    model = TransformerLM(config, seed=0)
    path = tmp_path / "m.npz"
    model.save(path)
    restored = TransformerLM.load(path)
    assert restored._rng.random() == model._rng.random()


def test_older_checkpoint_without_rng_still_loads(tmp_path):
    """A checkpoint written before the RNG field loads, with a fresh RNG."""
    config = TransformerConfig(n_layer=1, n_head=2, n_embd=16, block_size=8)
    model = TransformerLM(config, seed=0)
    path = tmp_path / "old.npz"
    # Emulate the old on-disk shape: parameters + a 7-entry config, no "rng".
    arrays = {f"p{i}": p for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(
        [config.vocab_size, config.n_layer, config.n_head, config.n_embd, config.block_size, 0.0, 0]
    )
    np.savez_compressed(path, **arrays)
    restored = TransformerLM.load(path)
    assert restored.config.n_embd == 16


def test_resumed_batch_sequence_continues_the_uninterrupted_run():
    """A resumed run draws the batch an uninterrupted run would have drawn.

    Re-seeding at ``seed + start_step`` used to hand back a different batch order,
    so continuing a run sampled different data than never stopping would have.
    """
    block, batch_size, start_step = 16, 4, 3
    stream = list(range(3000))
    max_start = len(stream) - block - 1

    def draws_from(advance: int) -> np.ndarray:
        rng = np.random.default_rng(7)
        for _ in range(advance):
            rng.integers(0, max_start, size=batch_size)
        return rng.integers(0, max_start, size=batch_size)

    # A fresh run's 4th draw (index 3) must equal the resumed run's 1st draw.
    assert np.array_equal(draws_from(3), draws_from(start_step))


def test_resume_matches_an_uninterrupted_run_when_lr_is_constant(tmp_path):
    """Save/load preserves every piece of state a continued run needs.

    The learning-rate schedule is horizon-relative on purpose (a later ``--resume``
    extends the horizon), so the *schedule* is not reproduced here. This pins the
    other half of the contract at the level below it: given a constant rate, the
    weights, Adam moments, and the dropout/batch RNG all survive a save/load, so
    three steps then three resumed steps land exactly where six straight steps do.
    """
    from ml.models import Adam

    config = TransformerConfig(n_layer=2, n_head=2, n_embd=16, block_size=16)
    stream = list(np.random.default_rng(0).integers(0, 257, size=3000))
    block, batch = config.block_size, 4
    max_start = len(stream) - block - 1

    def step_once(model, optimizer, rng):
        starts = rng.integers(0, max_start, size=batch)
        x = np.stack([stream[s : s + block] for s in starts])
        y = np.stack([stream[s + 1 : s + block + 1] for s in starts])
        model.train(True)
        model.loss(model.forward(x), y)
        optimizer.step(model.backward(y))

    uninterrupted = TransformerLM(config, seed=0)
    opt_a = Adam(uninterrupted.parameters(), lr=0.01)
    rng_a = np.random.default_rng(5)
    for _ in range(6):
        step_once(uninterrupted, opt_a, rng_a)

    resumed = TransformerLM(config, seed=0)
    opt_b = Adam(resumed.parameters(), lr=0.01)
    rng_b = np.random.default_rng(5)
    for _ in range(3):
        step_once(resumed, opt_b, rng_b)
    checkpoint = tmp_path / "m.npz"
    resumed.save(checkpoint)
    loaded = TransformerLM.load(checkpoint)
    opt_c = Adam(loaded.parameters(), lr=0.01)
    opt_c.state.m = [m.copy() for m in opt_b.state.m]
    opt_c.state.v = [v.copy() for v in opt_b.state.v]
    opt_c.state.t = opt_b.state.t
    rng_c = np.random.default_rng(5)
    for _ in range(3):  # advance to the resume point, as the trainer does
        rng_c.integers(0, max_start, size=batch)
    for _ in range(3):
        step_once(loaded, opt_c, rng_c)

    for a, b in zip(uninterrupted.parameters(), loaded.parameters(), strict=False):
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12)
