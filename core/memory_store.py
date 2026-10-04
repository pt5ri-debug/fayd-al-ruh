from __future__ import annotations

from core.memory import CognitiveMemory
from core.serializer import Serializer


class MemoryStore:
    """Exports memory items only; runtime services and callbacks are not persisted."""
    def __init__(self, serializer: Serializer | None = None) -> None:
        self.serializer = serializer or Serializer()

    def dump(self, memory: CognitiveMemory) -> dict:
        return self.serializer.to_primitive({"items": memory._items})
