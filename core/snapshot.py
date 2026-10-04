from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256

from core.serializer import Serializer


@dataclass(frozen=True)
class Snapshot:
    name: str
    payload: dict
    checksum: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(cls, name: str, payload: dict, serializer: Serializer | None = None) -> "Snapshot":
        if not name.strip():
            raise ValueError("Snapshot name cannot be empty.")
        encoded = (serializer or Serializer()).dumps(payload).encode("utf-8")
        return cls(name, payload, sha256(encoded).hexdigest())

    def verify(self, serializer: Serializer | None = None) -> bool:
        encoded = (serializer or Serializer()).dumps(self.payload).encode("utf-8")
        return sha256(encoded).hexdigest() == self.checksum
