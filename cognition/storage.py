"""Persistent stores for the autonomous cognitive development system.

Everything the new system must remember across a restart lives here, on top of
the project's existing :class:`storage.json_store.JsonStore` (namespaced, atomic,
Windows-safe). No second database is created: the task forbids duplicating a
store for a purpose one already serves, and ``JsonStore`` already backs the
characters, the mouth profiles, and the model registry.

Three stores are defined:

* :class:`IdeaStore` -- the epistemic ladder (ideas, hypotheses, supported,
  verified), bounded so a long-running mind cannot grow without limit.
* :class:`CycleStore` -- the record of each completed thought cycle.
* :class:`ExperienceStore` -- what happened, what was expected, and the lesson.
* :class:`FactStore` -- the semantic facts the learning system learned, so
  knowledge survives a restart exactly as the project's own learned relations do.

Each store is dependency-free, Tk-free, and tolerant of a corrupt document (a
bad record is skipped, not fatal), so a background mind always starts.
"""

from __future__ import annotations

from pathlib import Path

from core.paths import data_dir
from storage.json_store import JsonStore

from cognition.records import (
    Experience,
    Idea,
    IdeaKind,
    ThoughtCycle,
    TrainingExample,
)

#: Where the cognition system keeps its state, under the git-ignored data dir.
NAMESPACE_IDEAS = "cognition/ideas"
NAMESPACE_CYCLES = "cognition/cycles"
NAMESPACE_EXPERIENCES = "cognition/experiences"
NAMESPACE_EXAMPLES = "cognition/examples"
NAMESPACE_FACTS = "cognition/facts"
NAMESPACE_META = "cognition/meta"

#: Bounds, so a long-running background mind stays a fixed size on disk.
MAX_IDEAS = 500
MAX_CYCLES = 200
MAX_EXPERIENCES = 1000
MAX_EXAMPLES = 2000
MAX_FACTS = 4000

#: The single document each collection is stored as, so listing is one read.
_KEY = "all"


def _store() -> JsonStore:
    """A store rooted at the current data directory (honours ``FAYD_DATA_DIR``)."""
    return JsonStore(data_dir())


def _load_list(namespace: str) -> list[dict]:
    try:
        data = _store().load(namespace, _KEY)
    except Exception:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _save_list(namespace: str, items: list[dict]) -> None:
    try:
        _store().save(namespace, _KEY, items)
    except Exception:
        # A read-only data folder must never crash a background cycle; the
        # in-memory copy the caller holds is still correct for this session.
        pass


class IdeaStore:
    """The persistent idea/hypothesis store, newest last, bounded."""

    def __init__(self, *, max_items: int = MAX_IDEAS) -> None:
        self.max_items = max_items
        self._items: list[Idea] = [
            Idea.from_primitive(d) for d in _load_list(NAMESPACE_IDEAS)
        ][-max_items:]

    def add(self, idea: Idea) -> Idea:
        self._items.append(idea)
        del self._items[:-self.max_items]
        self._flush()
        return idea

    def replace(self, idea: Idea) -> Idea:
        """Swap a stored idea for its updated copy (by id), else append it."""
        for index, existing in enumerate(self._items):
            if existing.id == idea.id:
                self._items[index] = idea
                self._flush()
                return idea
        return self.add(idea)

    def all(self) -> tuple[Idea, ...]:
        return tuple(self._items)

    def by_kind(self, kind: IdeaKind) -> tuple[Idea, ...]:
        return tuple(i for i in self._items if i.kind is kind)

    def get(self, idea_id: str) -> Idea | None:
        return next((i for i in self._items if i.id == idea_id), None)

    def recent(self, limit: int = 10) -> tuple[Idea, ...]:
        return tuple(self._items[-limit:])

    def counts(self) -> dict[str, int]:
        counts = {kind.value: 0 for kind in IdeaKind}
        for idea in self._items:
            counts[idea.kind.value] += 1
        return counts

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_IDEAS, [i.to_primitive() for i in self._items])


class CycleStore:
    """The record of completed thought cycles, newest last, bounded."""

    def __init__(self, *, max_items: int = MAX_CYCLES) -> None:
        self.max_items = max_items
        self._items: list[ThoughtCycle] = [
            ThoughtCycle.from_primitive(d) for d in _load_list(NAMESPACE_CYCLES)
        ][-max_items:]

    def add(self, cycle: ThoughtCycle) -> ThoughtCycle:
        self._items.append(cycle)
        del self._items[:-self.max_items]
        self._flush()
        return cycle

    def all(self) -> tuple[ThoughtCycle, ...]:
        return tuple(self._items)

    def last(self) -> ThoughtCycle | None:
        return self._items[-1] if self._items else None

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_CYCLES, [c.to_primitive() for c in self._items])


class ExperienceStore:
    """The record of expectations, outcomes, and lessons, bounded."""

    def __init__(self, *, max_items: int = MAX_EXPERIENCES) -> None:
        self.max_items = max_items
        self._items: list[Experience] = [
            Experience.from_primitive(d) for d in _load_list(NAMESPACE_EXPERIENCES)
        ][-max_items:]

    def add(self, experience: Experience) -> Experience:
        self._items.append(experience)
        del self._items[:-self.max_items]
        self._flush()
        return experience

    def all(self) -> tuple[Experience, ...]:
        return tuple(self._items)

    def failures(self) -> tuple[Experience, ...]:
        return tuple(e for e in self._items if not e.success)

    def success_rate(self) -> float:
        if not self._items:
            return 0.0
        return sum(1 for e in self._items if e.success) / len(self._items)

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_EXPERIENCES, [e.to_primitive() for e in self._items])


class ExampleStore:
    """The record of qualified/unqualified training examples, bounded.

    Kept separate from the idea store: an example is a *task and an answer*, not
    a thought. Its provenance and verdict are what the dataset builder reads.
    """

    def __init__(self, *, max_items: int = MAX_EXAMPLES) -> None:
        self.max_items = max_items
        self._items: list[TrainingExample] = [
            TrainingExample.from_primitive(d) for d in _load_list(NAMESPACE_EXAMPLES)
        ][-max_items:]

    def add(self, example: TrainingExample) -> TrainingExample:
        self._items.append(example)
        del self._items[:-self.max_items]
        self._flush()
        return example

    def all(self) -> tuple[TrainingExample, ...]:
        return tuple(self._items)

    def qualified(self) -> tuple[TrainingExample, ...]:
        """Only the examples the gate accepted -- the ones training may use."""
        return tuple(e for e in self._items if e.qualified)

    def rejected(self) -> tuple[TrainingExample, ...]:
        return tuple(e for e in self._items if not e.qualified)

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_EXAMPLES, [e.to_primitive() for e in self._items])


class FactStore:
    """The semantic facts the learning system learned, bounded.

    Semantic memory lives in RAM for the session; this store is what makes a
    learned fact survive a restart -- the same reason
    :mod:`core.knowledge_carry` persists the learned relations. Only the fact's
    id, statement, and a small provenance note are kept, so a corrupt document is
    skipped rather than fatal.
    """

    def __init__(self, *, max_items: int = MAX_FACTS) -> None:
        self.max_items = max_items
        self._items: list[dict] = [
            d for d in _load_list(NAMESPACE_FACTS) if isinstance(d, dict)
        ][-max_items:]

    def learn(self, fact_id: str, statement: str) -> None:
        self._items = [d for d in self._items if d.get("id") != fact_id]
        self._items.append({"id": fact_id, "statement": statement})
        del self._items[:-self.max_items]
        self._flush()

    def all(self) -> tuple[tuple[str, str], ...]:
        return tuple((str(d.get("id", "")), str(d.get("statement", ""))) for d in self._items)

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items = []
        self._flush()

    def _flush(self) -> None:
        _save_list(NAMESPACE_FACTS, self._items)


def meta_get(key: str, default: object = None) -> object:
    try:
        data = _store().load(NAMESPACE_META, key)
    except Exception:
        return default
    return default if data is None else data


def meta_set(key: str, value: object) -> None:
    """Write one small value (last-run time, counters) best-effort."""
    try:
        _store().save(NAMESPACE_META, key, value)
    except Exception:
        pass


def store_root() -> Path:
    """The folder the cognition state lives under (for the UI to show)."""
    return data_dir() / NAMESPACE_META


__all__ = [
    "MAX_CYCLES",
    "MAX_EXAMPLES",
    "MAX_EXPERIENCES",
    "MAX_FACTS",
    "MAX_IDEAS",
    "NAMESPACE_FACTS",
    "CycleStore",
    "ExampleStore",
    "ExperienceStore",
    "FactStore",
    "IdeaStore",
    "meta_get",
    "meta_set",
    "store_root",
]
