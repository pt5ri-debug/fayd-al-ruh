from __future__ import annotations

from collections.abc import Callable

from core.snapshot import SNAPSHOT_SCHEMA_VERSION

# A migration upgrades a payload from version N to N+1. Migrations are applied
# in order and must be side-effect free so a failed upgrade can be retried.
Migration = Callable[[dict], dict]


def _migrate_1_to_2(payload: dict) -> dict:
    """Version 2 introduced the central provenance ledger.

    Older payloads have no ledger, so we add an empty one rather than failing.
    """
    payload.setdefault("provenance", {})
    return payload


MIGRATIONS: dict[int, Migration] = {
    1: _migrate_1_to_2,
}


class SnapshotVersionError(ValueError):
    """Raised when a snapshot cannot be migrated to the current schema."""


def migrate_payload(payload: dict, from_version: int, to_version: int = SNAPSHOT_SCHEMA_VERSION) -> dict:
    """Return a payload upgraded from ``from_version`` to ``to_version``.

    Migrations are applied one step at a time so intermediate versions are
    never skipped. The original payload is not mutated.
    """
    if from_version < 1:
        raise SnapshotVersionError(f"Invalid snapshot version: {from_version}.")
    if to_version < from_version:
        raise SnapshotVersionError(
            f"Cannot downgrade snapshot from version {from_version} to {to_version}."
        )
    if from_version > SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotVersionError(
            f"Snapshot version {from_version} is newer than supported "
            f"version {SNAPSHOT_SCHEMA_VERSION}."
        )

    migrated = dict(payload)
    version = from_version
    while version < to_version:
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise SnapshotVersionError(
                f"No migration registered from version {version} to {version + 1}."
            )
        migrated = migration(migrated)
        version += 1
    return migrated
