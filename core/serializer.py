from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
from typing import Any


class Serializer:
    """Serializes cognitive data to JSON-safe primitives; it never deserializes executable objects."""
    def to_primitive(self, value: Any) -> Any:
        if is_dataclass(value):
            return self.to_primitive(asdict(value))
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(key): self.to_primitive(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.to_primitive(item) for item in value]
        if hasattr(value, "to_dict"):
            return self.to_primitive(value.to_dict())
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def dumps(self, value: Any) -> str:
        return json.dumps(self.to_primitive(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def loads(self, payload: str) -> Any:
        return json.loads(payload)
