from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkingMemoryItem:
    key: str
    content: Any
    activation: float = 0.5


class WorkingMemory:
    """Small, capacity-bounded context for the current cognitive task."""
    def __init__(self, capacity: int = 7) -> None:
        if capacity < 1:
            raise ValueError("Working-memory capacity must be at least 1.")
        self.capacity = capacity
        self._items: OrderedDict[str, WorkingMemoryItem] = OrderedDict()

    def put(self, key: str, content: Any, activation: float = 0.5) -> WorkingMemoryItem:
        if not 0.0 <= activation <= 1.0:
            raise ValueError("Activation must be between 0 and 1.")
        item = WorkingMemoryItem(key, content, activation)
        self._items[key] = item
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return item

    def get(self, key: str) -> WorkingMemoryItem | None:
        item = self._items.get(key)
        if item is not None:
            item.activation = min(1.0, item.activation + 0.1)
            self._items.move_to_end(key)
        return item

    def items(self) -> list[WorkingMemoryItem]:
        return list(self._items.values())

    def remove(self, key: str) -> None:
        self._items.pop(key, None)
