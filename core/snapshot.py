from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256

from core.serializer import Serializer

# Bump when the snapshot payload layout changes in a way older readers cannot
# understand. Each bump must ship a matching migration in snapshot_migrations.
SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Snapshot:
    name: str
    payload: dict
    checksum: str
    version: int = SNAPSHOT_SCHEMA_VERSION
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        name: str,
        payload: dict,
        serializer: Serializer | None = None,
        version: int = SNAPSHOT_SCHEMA_VERSION,
    ) -> Snapshot:
        if not name.strip():
            raise ValueError("Snapshot name cannot be empty.")
        if version < 1:
            raise ValueError("Snapshot version must be at least 1.")
        encoded = (serializer or Serializer()).dumps(payload).encode("utf-8")
        return cls(name, payload, sha256(encoded).hexdigest(), version)

    def verify(self, serializer: Serializer | None = None) -> bool:
        encoded = (serializer or Serializer()).dumps(self.payload).encode("utf-8")
        return sha256(encoded).hexdigest() == self.checksum
