"""Phase 6: controlled model-weight training -- dataset governance and isolation.

These tests exercise the **real** dataset pipeline: the record contract, the
validator, the deterministic split, the versioned snapshot, and the personality
isolation rules (including Natasha's exclusion). No model is loaded and nothing
outside a temporary directory is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.training.datasets import (
    DatasetBuilder,
    DatasetRecord,
    DatasetSplit,
    DatasetValidationError,
    DatasetValidator,
    TrainingExample,
    deterministic_split,
)
from ml.training.isolation import (
    audit_dataset,
    can_learn_from,
    dataset_owners,
    main_model_dataset_is_clean,
    main_model_dataset_owners,
)


def _examples(count: int, owner: str = "jarvis") -> tuple[TrainingExample, ...]:
    return tuple(
        TrainingExample(
            text=f"Q: question number {i} about porosity?\nA: answer {i} explains porosity.",
            owner=owner,
            source="test",
            license="project-authored",
        )
        for i in range(count)
    )


@pytest.fixture()
def builder(tmp_path: Path) -> DatasetBuilder:
    return DatasetBuilder(root=tmp_path / "datasets")


# ------------------------------------------------------------- the example


def test_example_id_is_stable_and_owner_scoped():
    first = TrainingExample(text="porosity is the void space", owner="jarvis")
    again = TrainingExample(text="porosity is the void space", owner="jarvis")
    other = TrainingExample(text="porosity is the void space", owner="ultron")
    assert first.example_id == again.example_id
    assert first.example_id != other.example_id


def test_example_verified_requires_source_and_license():
    assert not TrainingExample(text="x" * 30).verified
    assert not TrainingExample(text="x" * 30, source="s").verified
    assert TrainingExample(text="x" * 30, source="s", license="l").verified


def test_example_round_trips_through_a_primitive():
    example = TrainingExample(text="a fact", owner="jarvis", source="s", license="l", tags=("t",))
    restored = TrainingExample.from_primitive(example.to_primitive())
    assert restored == example


# ------------------------------------------------------------- the validator


def test_validator_rejects_blank_short_and_contentless():
    validator = DatasetValidator()
    assert validator.check_example(TrainingExample(text="   ")).kind == "empty"
    assert validator.check_example(TrainingExample(text="abc")).kind == "too_short"
    assert validator.check_example(TrainingExample(text="! ! ! ! ! ! ! ! ! ! ! !")).kind == "no_content"
    assert validator.check_example(TrainingExample(text="a real sentence here")) is None


def test_validator_finds_exact_and_near_duplicates():
    validator = DatasetValidator()
    # The example id is a content hash, so an exact duplicate shares its id.
    first = TrainingExample(text="porosity measures the void space in the rock", source="s", license="l")
    exact = TrainingExample(text="porosity measures the void space in the rock", source="s", license="l")
    near = TrainingExample(text="Porosity measures the void space in the rock.", source="s", license="l")
    assert first.example_id == exact.example_id
    dropped, issues = validator.find_duplicates([first, exact, near])
    # The exact duplicate and the near duplicate are both dropped; the first is kept.
    assert len(dropped) == 2
    assert {i.kind for i in issues} == {"duplicate", "near_duplicate"}
    kept, _ = validator.find_duplicates([first, near])
    assert kept == (near.example_id,)


def test_validator_detects_leakage_between_splits():
    validator = DatasetValidator()
    shared = TrainingExample(text="the same example on both sides", source="s", license="l")
    split = DatasetSplit(train=(shared,), test=(shared,))
    leaks, issues = validator.find_leakage(split)
    assert shared.example_id in leaks
    assert issues and issues[0].kind == "leakage"


def test_quality_is_zero_for_an_empty_split():
    validator = DatasetValidator()
    report = validator.validate(DatasetSplit())
    assert validator.quality_score(DatasetSplit(), report) == 0.0


# ------------------------------------------------------------- the split


def test_split_is_deterministic_and_separates():
    examples = _examples(20)
    first = deterministic_split(examples, seed=3)
    second = deterministic_split(examples, seed=3)
    assert first == second
    assert first.train and first.validation and first.test
    ids = {e.example_id for e in first.train} & {e.example_id for e in first.test}
    assert not ids


def test_split_survives_a_tiny_dataset():
    split = deterministic_split(_examples(2), seed=0)
    assert len(split.all_examples()) == 2


# ------------------------------------------------------------- the builder


def test_build_refuses_an_empty_dataset(builder: DatasetBuilder):
    with pytest.raises(DatasetValidationError):
        builder.build("empty", [])


def test_build_refuses_a_bad_id(builder: DatasetBuilder):
    with pytest.raises(DatasetValidationError):
        builder.build("Bad Id!", _examples(6))


def test_build_refuses_an_example_outside_the_scope(builder: DatasetBuilder):
    with pytest.raises(DatasetValidationError):
        builder.build("mix", _examples(6, owner="ultron"), personality_scope=("jarvis",))


def test_build_refuses_an_unverified_example_by_default(builder: DatasetBuilder):
    with pytest.raises(DatasetValidationError):
        builder.build(
            "unver", [TrainingExample(text="a fact with no source at all", owner="jarvis")],
            personality_scope=("jarvis",),
        )


def test_build_refuses_a_private_scope(builder: DatasetBuilder):
    with pytest.raises(DatasetValidationError):
        builder.build("priv", _examples(6, owner="natasha"), personality_scope=("natasha",))


def test_build_records_the_task_fields(builder: DatasetBuilder):
    built = builder.build(
        "gov",
        _examples(12),
        personality_scope=("jarvis",),
        source="unit-test",
        provenance="hand-written",
        license="project-authored",
        intended_model="fayd_al_ruh",
        intended_task="chat",
        exclusion_rules=("natasha",),
    )
    record = built.record
    assert record.dataset_id == "gov"
    assert record.version == 1
    assert record.source == "unit-test"
    assert record.license == "project-authored"
    assert record.intended_model == "fayd_al_ruh"
    assert record.personality_scope == ("jarvis",)
    assert record.exclusion_rules == ("natasha",)
    assert record.eligible
    assert record.digest


def test_build_is_immutable_and_versioned(builder: DatasetBuilder):
    first = builder.build("ver", _examples(10), personality_scope=("jarvis",))
    second = builder.build("ver", _examples(14), personality_scope=("jarvis",))
    assert second.record.version == first.record.version + 1
    with pytest.raises(DatasetValidationError):
        builder.save(first)


def test_snapshot_round_trips_with_provenance(builder: DatasetBuilder):
    built = builder.build(
        "rt", _examples(15), personality_scope=("jarvis",), source="s", license="l"
    )
    loaded = builder.load("rt", built.record.version)
    assert loaded.record.digest == built.record.digest
    assert loaded.record.source == "s"
    assert loaded.split.counts() == built.split.counts()
    assert {e.example_id for e in loaded.split.all_examples()} == {
        e.example_id for e in built.split.all_examples()
    }


def test_list_datasets_reports_every_version(builder: DatasetBuilder):
    builder.build("a", _examples(8), personality_scope=("jarvis",))
    builder.build("b", _examples(8), personality_scope=("jarvis",))
    records = builder.list_datasets()
    assert {r.dataset_id for r in records} == {"a", "b"}


def test_dataset_record_round_trips():
    record = DatasetRecord(dataset_id="x", version=2, personality_scope=("jarvis",), eligible=True)
    assert DatasetRecord.from_primitive(record.to_primitive()) == record


# ------------------------------------------------------------- isolation


def test_dataset_owners_lists_distinct_owners():
    split = DatasetSplit(train=_examples(2, "jarvis") + _examples(2, "ultron"))
    assert set(dataset_owners(split)) == {"jarvis", "ultron"}


def test_audit_allows_a_single_owner_dataset():
    split = DatasetSplit(train=_examples(3, "jarvis"))
    assert audit_dataset(split, declared_scope=("jarvis",)).ok


def test_audit_refuses_an_unknown_owner():
    split = DatasetSplit(train=(TrainingExample(text="x" * 20, owner="ghost", source="s", license="l"),))
    verdict = audit_dataset(split)
    assert not verdict.ok and "unknown owner" in verdict.reason


def test_audit_refuses_mixing_two_characters_without_a_scope():
    split = DatasetSplit(train=_examples(2, "jarvis") + _examples(2, "ultron"))
    assert not audit_dataset(split).ok


def test_audit_refuses_natasha_in_the_main_model_aggregate():
    split = DatasetSplit(train=(TrainingExample(text="x" * 20, owner="natasha", source="s", license="l"),))
    verdict = audit_dataset(split, for_main_model=True)
    assert not verdict.ok and "private" in verdict.reason


def test_main_model_aggregate_is_clean_and_excludes_natasha():
    assert main_model_dataset_is_clean()
    owners = main_model_dataset_owners()
    assert "natasha" not in owners
    assert "jarvis" in owners


def test_can_learn_from_respects_the_policy():
    assert can_learn_from("jarvis", "jarvis")
    assert can_learn_from("fayd_al_ruh", "jarvis")
    assert not can_learn_from("fayd_al_ruh", "natasha")
    assert not can_learn_from("jarvis", "ultron")
    assert not can_learn_from("ghost", "jarvis")


# ------------------------------------------------------------- aggregation


def test_aggregate_examples_excludes_natasha_by_default():
    from ml.training.datasets import aggregate_examples

    examples = aggregate_examples()
    assert all(example.owner != "natasha" for example in examples)
