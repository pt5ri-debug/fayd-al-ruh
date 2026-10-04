import pytest

from core.snapshot import SNAPSHOT_SCHEMA_VERSION, Snapshot
from core.snapshot_migrations import (
    SnapshotVersionError,
    migrate_payload,
)


def test_current_snapshot_records_schema_version():
    snapshot = Snapshot.create("state", {"beliefs": 1})
    assert snapshot.version == SNAPSHOT_SCHEMA_VERSION


def test_migrate_v1_payload_adds_provenance_section():
    migrated = migrate_payload({"graph": {}, "memory": {}}, 1)
    assert "provenance" in migrated
    assert migrated["provenance"] == {}


def test_migrate_does_not_mutate_the_original_payload():
    original = {"graph": {}}
    migrate_payload(original, 1)
    assert "provenance" not in original


def test_migrate_current_version_is_a_no_op():
    payload = {"provenance": {"BL-1": []}}
    assert migrate_payload(payload, SNAPSHOT_SCHEMA_VERSION) == payload


def test_migrate_rejects_downgrade():
    with pytest.raises(SnapshotVersionError):
        migrate_payload({}, SNAPSHOT_SCHEMA_VERSION, 1)


def test_migrate_rejects_future_version():
    with pytest.raises(SnapshotVersionError):
        migrate_payload({}, SNAPSHOT_SCHEMA_VERSION + 1)


def test_migrate_rejects_missing_migration_step():
    with pytest.raises(SnapshotVersionError):
        migrate_payload({}, 1, SNAPSHOT_SCHEMA_VERSION + 1)


def test_migrate_rejects_invalid_version():
    with pytest.raises(SnapshotVersionError):
        migrate_payload({}, 0)
