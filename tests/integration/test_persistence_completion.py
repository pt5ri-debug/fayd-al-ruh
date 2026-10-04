"""Completing persistence: safe rebuild, snapshot migrations, and retention.

Proves that a snapshot can be loaded back into fully reconstructed objects,
that a pre-versioning (v1) record is upgraded through the migration registry,
that a corrupt record is rejected rather than partially loaded, and that a
long-term retention policy prunes aged, weak memory.
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.evidence_processor import EvidencePolarity
from core.observation import Observation
from core.retention_policy import RetentionPolicy
from core.snapshot import SNAPSHOT_SCHEMA_VERSION
from core.snapshot_loader import SnapshotLoader
from core.snapshot_migrations import SnapshotVersionError
from storage.json_store import JsonStore


pytestmark = pytest.mark.integration


def _run_episode(controller: CognitiveController):
    belief = Belief("Rain makes the ground wet.", 0.5)
    controller.process(
        Observation("Rain falls on the ground.", "weather_sensor", 0.9),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    cycle = controller.process(
        Observation("Rain falls on the ground.", "weather_sensor", 0.9),
        belief,
        EvidencePolarity.SUPPORTS,
    )
    return belief, cycle


def test_snapshot_loads_into_fully_rebuilt_objects(tmp_path):
    controller = CognitiveController()
    belief, cycle = _run_episode(controller)

    store = JsonStore(tmp_path)
    controller.create_snapshot("episode", store)
    record = store.load("snapshots", "episode")

    assert record["version"] == SNAPSHOT_SCHEMA_VERSION
    restored = SnapshotLoader().restore(record)
    assert restored.version == SNAPSHOT_SCHEMA_VERSION
    assert restored.migrations_applied == ()

    # Every object type is reconstructed, not just referenced.
    assert set(restored.state.patterns) == set(controller.state.patterns)
    assert set(restored.state.hypotheses) == set(controller.state.hypotheses)
    assert set(restored.state.predictions) == set(controller.state.predictions)
    assert set(restored.state.traces) == set(controller.state.traces)
    for prediction_id, prediction in restored.state.predictions.items():
        assert prediction.statement == controller.state.predictions[prediction_id].statement
        assert prediction.hypothesis_id == controller.state.predictions[prediction_id].hypothesis_id
    for hypothesis_id, hypothesis in restored.state.hypotheses.items():
        assert hypothesis.source_pattern == controller.state.hypotheses[hypothesis_id].source_pattern


def test_controller_restore_rebuilds_state_and_provenance(tmp_path):
    controller = CognitiveController()
    belief, cycle = _run_episode(controller)
    store = JsonStore(tmp_path)
    controller.create_snapshot("episode", store)

    restored = CognitiveController()
    assert restored.restore_snapshot("episode", store) is True

    assert set(restored.state.patterns) == set(controller.state.patterns)
    assert set(restored.state.hypotheses) == set(controller.state.hypotheses)
    assert restored.provenance.dump() == controller.provenance.dump()
    assert restored.provenance_for(belief.id) is not None
    assert restored.validate_contexts() == []


def test_v1_record_is_migrated_on_load(tmp_path):
    controller = CognitiveController()
    _run_episode(controller)
    store = JsonStore(tmp_path)
    controller.create_snapshot("episode", store)

    # Simulate a record written before provenance/versioning existed.
    record = store.load("snapshots", "episode")
    record.pop("version")
    record["payload"].pop("provenance")
    from core.snapshot import Snapshot as Snap

    fresh = Snap.create("episode", record["payload"])
    record["payload"] = fresh.payload
    record["checksum"] = fresh.checksum

    restored = SnapshotLoader().restore(record)

    assert restored.version == SNAPSHOT_SCHEMA_VERSION
    assert restored.migrations_applied == (1,)
    assert restored.provenance.dump() == {}


def test_future_version_is_rejected(tmp_path):
    controller = CognitiveController()
    _run_episode(controller)
    store = JsonStore(tmp_path)
    controller.create_snapshot("episode", store)
    record = store.load("snapshots", "episode")
    record["version"] = SNAPSHOT_SCHEMA_VERSION + 1

    with pytest.raises(SnapshotVersionError):
        SnapshotLoader().restore(record)


def test_corrupt_section_is_rejected_not_partially_loaded(tmp_path):
    controller = CognitiveController()
    _run_episode(controller)
    store = JsonStore(tmp_path)
    controller.create_snapshot("episode", store)
    record = store.load("snapshots", "episode")

    record["payload"]["graph"] = ["not", "an", "object"]
    from core.snapshot import Snapshot as Snap

    fresh = Snap.create("episode", record["payload"])
    record["payload"] = fresh.payload
    record["checksum"] = fresh.checksum

    with pytest.raises(ValueError):
        SnapshotLoader().restore(record)


def test_retention_prunes_long_term_memory():
    controller = CognitiveController()
    belief = Belief("Rain makes the ground wet.", 0.5)
    controller.process(
        Observation("Rain falls on the ground.", "weather_sensor", 0.9),
        belief,
        EvidencePolarity.SUPPORTS,
    )

    # Age and weaken a stored item so it falls outside the retention window.
    aged = next(iter(controller.memory._items.values()))
    aged.created_at = datetime.now(timezone.utc) - timedelta(days=90)
    aged.importance = 0.02
    survivor_id = next(
        item.item_id
        for item in controller.memory._items.values()
        if item.item_id != aged.item_id
    )

    report = controller.prune_memory(RetentionPolicy(max_age_days=30, minimum_importance=0.1))

    assert aged.item_id in report.removed
    assert controller.memory.get(aged.item_id) is None
    assert controller.memory.get(survivor_id) is not None
