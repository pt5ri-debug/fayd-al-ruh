"""Tests for the sharded, resumable language-model schedule.

The point of the schedule is honest: instead of replaying the whole corpus on
every resume, it walks a fixed partition of the document universe one shard per
run and records which shards it consumed, so a later `--resume` reaches material
the model has not trained on yet. These tests pin the partition, the cycling,
the sidecar contract, and the wiring into `train_and_save`, the CLI, and the
front ends.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.llm.__main__ import main, train_and_save
from ml.llm.corpus import build_corpus, corpus_units
from ml.llm.schedule import (
    DEFAULT_SHARD_COUNT,
    ScheduleState,
    ShardPlan,
    describe_schedule,
    load_schedule,
    save_schedule,
    schedule_path,
    select_shard,
)


def _units() -> list[str]:
    return corpus_units()


def test_shards_partition_every_document_exactly_once():
    units = _units()
    shards = ShardPlan(4).shards(units, seed=0)
    assert len(shards) == 4
    merged = [unit for shard in shards for unit in shard]
    assert sorted(merged) == sorted(unit for unit in units if unit.strip())
    # Balanced: no shard is more than one document larger than another.
    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shards_are_deterministic_for_a_seed():
    units = _units()
    assert ShardPlan(3).shards(units, seed=7) == ShardPlan(3).shards(units, seed=7)
    assert ShardPlan(3).shards(units, seed=7) != ShardPlan(3).shards(units, seed=8)


def test_next_index_cycles_after_every_shard_is_consumed():
    plan = ShardPlan(DEFAULT_SHARD_COUNT)
    assert plan.next_index([]) == 0
    assert plan.next_index([0, 1, 2]) == 3
    assert plan.next_index([0, 1, 2, 3, 4]) == 0


def test_select_shard_returns_the_next_unconsumed_shard():
    units = _units()
    plan = ShardPlan(4)
    state = ScheduleState(count=4, seed=0, consumed=[0])
    documents, index = select_shard(units, plan, state, seed=0)
    assert index == 1
    assert documents == plan.shards(units, seed=0)[1]


def test_schedule_state_round_trips_through_the_sidecar(tmp_path):
    path = schedule_path(tmp_path / "fayd-llm.npz")
    assert path.name == "fayd-llm.schedule.json"
    state = ScheduleState(count=5, seed=0, consumed=[0, 1, 2], units=120)
    save_schedule(state, path)

    loaded = load_schedule(path)
    assert loaded is not None
    assert loaded.as_dict() == state.as_dict()
    assert loaded.matches(ShardPlan(5), seed=0)
    assert not loaded.matches(ShardPlan(4), seed=0)


def test_missing_schedule_sidecar_is_absent_not_an_error(tmp_path):
    assert load_schedule(tmp_path / "absent.schedule.json") is None
    assert describe_schedule(tmp_path / "absent.schedule.json") is None


def test_describe_schedule_reports_the_next_shard(tmp_path):
    path = schedule_path(tmp_path / "fayd-llm.npz")
    save_schedule(ScheduleState(count=4, seed=0, consumed=[0]), path)
    info = describe_schedule(path)
    assert info == {"count": 4, "seed": 0, "consumed": [0], "next": 1, "units": 0}


def test_build_corpus_include_keeps_only_the_chosen_documents():
    units = _units()
    shard = ShardPlan(5).shards(units, seed=0)[0]
    text = build_corpus(repeats=1, structured_repeats=1, include=set(shard))
    whole = build_corpus(repeats=1, structured_repeats=1)
    # A single shard is a strict subset of the whole corpus, and every chosen
    # document (that is not held out) appears in it.
    assert len(text) < len(whole)
    for unit in shard:
        assert unit in text


def test_train_and_save_records_each_consumed_shard(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    for run in range(4):
        train_and_save(
            steps=2,
            repeats=1,
            structured_repeats=1,
            checkpoint=checkpoint,
            log_every=0,
            val_every=0,
            resume=run > 0,
            shard_count=4,
            verbose=False,
        )
        state = load_schedule(schedule_path(checkpoint))
        assert state is not None
        assert state.consumed == list(range(run + 1))
    # A fifth run cycles back to the first shard rather than running off the end.
    train_and_save(
        steps=2,
        repeats=1,
        structured_repeats=1,
        checkpoint=checkpoint,
        log_every=0,
        val_every=0,
        resume=True,
        shard_count=4,
        verbose=False,
    )
    state = load_schedule(schedule_path(checkpoint))
    assert state is not None
    assert state.consumed == [0, 1, 2, 3, 0]


def _load_params(checkpoint):
    from ml.llm.transformer import TransformerLM

    return TransformerLM.load(checkpoint).parameters()


def test_a_resume_reaches_new_material_rather_than_replaying(tmp_path):
    """Shard 1 differs from shard 0, and a resume actually trains shard 1.

    A checkpoint trained twice with the sharded schedule has consumed shards 0
    then 1. If the second run had replayed shard 0 (the old whole-corpus
    behaviour), its weights would match a model that trained shard 0 twice.
    """
    twice = tmp_path / "twice.npz"  # shard 0, then resume -> shard 1
    only_zero = tmp_path / "only-zero.npz"  # shard 0, then resume -> shard 0 again (forced)

    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=twice,
        log_every=0, val_every=0, shard_count=4, verbose=False,
    )
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=twice,
        log_every=0, val_every=0, resume=True, shard_count=4, verbose=False,
    )
    # Force a replay of shard 0 by seeding the sidecar as already consumed [0,1,2,3].
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=only_zero,
        log_every=0, val_every=0, shard_count=4, verbose=False,
    )
    save_schedule(ScheduleState(count=4, seed=0, consumed=[0, 1, 2, 3]), schedule_path(only_zero))
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=only_zero,
        log_every=0, val_every=0, resume=True, shard_count=4, verbose=False,
    )

    assert load_schedule(schedule_path(twice)).consumed == [0, 1]  # type: ignore[union-attr]
    assert load_schedule(schedule_path(only_zero)).consumed == [0, 1, 2, 3, 0]  # type: ignore[union-attr]
    # The two final models trained on different documents, so they must differ.
    assert any(
        not np.allclose(left, right)
        for left, right in zip(_load_params(twice), _load_params(only_zero), strict=False)
    )


def test_train_and_save_rejects_a_changed_shard_count(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint,
        log_every=0, val_every=0, shard_count=4, verbose=False,
    )
    with pytest.raises(ValueError, match="shards"):
        train_and_save(
            steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint,
            log_every=0, val_every=0, resume=True, shard_count=5, verbose=False,
        )


def test_no_shards_keeps_the_whole_corpus_and_writes_no_sidecar(tmp_path):
    checkpoint = tmp_path / "fayd-llm.npz"
    train_and_save(
        steps=2, repeats=1, structured_repeats=1, checkpoint=checkpoint,
        log_every=0, val_every=0, verbose=False,
    )
    assert not schedule_path(checkpoint).exists()


def test_cli_llm_shards_flag_is_wired(tmp_path, capsys):
    checkpoint = tmp_path / "fayd-llm.npz"
    code = main(
        [
            "--llm-shards",
            "4",
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
    assert schedule_path(checkpoint).exists()
    out = capsys.readouterr().out
    assert "shard 1/4" in out


def test_menu_and_gui_training_entries_use_the_sharded_schedule():
    from cli import menu
    from gui import actions

    by_label = {label: argv for label, _action, argv in menu.MENU}
    assert "--llm-shards" in by_label["Train the language model from all repositories"]
    assert "--llm-shards" in by_label["Train everything for a release"]

    by_key = {action.key: action.argv for action in actions.ACTIONS}
    assert "--llm-shards" in by_key["llm"]
    assert "--llm-shards" in by_key["release"]


def test_schedule_state_rejects_a_nonpositive_count():
    with pytest.raises(ValueError):
        ShardPlan(0)
    with pytest.raises(ValueError):
        ScheduleState.from_dict({"count": 0, "seed": 0, "consumed": []})
