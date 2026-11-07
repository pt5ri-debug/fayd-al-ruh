"""Question-and-answer learning with an explicit review status.

The Training Center can gather answers from many sources, but the project's hard
rule is that **generated data is never trained on without review**. This module is
where that rule lives: every gathered answer becomes a
:class:`~core.training_center.records.LearningExample` with a status, a reviewer
moves it through *pending → approved / rejected / corrected / needs review*, and
only an approved example may be exported to a dataset.

The store is durable (through the project's own
:class:`storage.json_store.JsonStore`) and corruption-tolerant: a broken file
reads as empty rather than crashing a window. It is dependency-free and Tk-free,
so a window, a CLI, and a test share one review queue.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from storage.json_store import JsonStore

from core.paths import data_dir
from core.training_center.records import (
    APPROVED_STATUSES,
    LEARNING_STATUSES,
    STATUS_APPROVED,
    STATUS_EXPORTED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_USED_IN_TRAINING,
    LearningExample,
)

#: The JsonStore namespace and key the review queue lives under.
NAMESPACE = "training_center"
KEY = "learning_examples"


@dataclass(frozen=True)
class ReviewSummary:
    """A count per status, for a status line."""

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def approved(self) -> int:
        return sum(self.counts.get(status, 0) for status in APPROVED_STATUSES)

    def describe(self) -> str:
        if not self.total:
            return "No learning examples yet."
        parts = [f"{count} {status}" for status, count in sorted(self.counts.items()) if count]
        return f"{self.total} example(s): " + ", ".join(parts)


class LearningReviewQueue:
    """The durable queue of gathered answers and their review status."""

    def __init__(self, *, root: Path | str | None = None, store: JsonStore | None = None) -> None:
        self.root = Path(root) if root is not None else data_dir()
        self._store = store or JsonStore(self.root)

    # -- reading -----------------------------------------------------------
    def all(self) -> tuple[LearningExample, ...]:
        """Every stored example, newest first. A corrupt file reads as empty."""
        raw = self._store.load(NAMESPACE, KEY)
        if not isinstance(raw, list):
            return ()
        examples: list[LearningExample] = []
        for item in raw:
            try:
                examples.append(LearningExample.from_primitive(item))
            except (ValueError, TypeError):
                continue
        examples.sort(key=lambda e: e.created_at, reverse=True)
        return tuple(examples)

    def by_status(self, status: str) -> tuple[LearningExample, ...]:
        return tuple(example for example in self.all() if example.status == status)

    def approved(self) -> tuple[LearningExample, ...]:
        return tuple(example for example in self.all() if example.approved)

    def get(self, example_id: str) -> LearningExample | None:
        for example in self.all():
            if example.example_id == example_id:
                return example
        return None

    def summary(self) -> ReviewSummary:
        counts: dict[str, int] = {}
        for example in self.all():
            counts[example.status] = counts.get(example.status, 0) + 1
        return ReviewSummary(counts=counts)

    def search(self, query: str) -> tuple[LearningExample, ...]:
        """Examples whose question, answer, or topic contains ``query``."""
        needle = query.strip().lower()
        if not needle:
            return self.all()
        return tuple(
            example
            for example in self.all()
            if needle in example.question.lower()
            or needle in example.answer.lower()
            or needle in example.topic.lower()
        )

    # -- writing -----------------------------------------------------------
    def add(self, example: LearningExample) -> LearningExample:
        """Store one example (de-duplicated by id, the newer status winning)."""
        examples = list(self.all())
        for index, existing in enumerate(examples):
            if existing.example_id == example.example_id:
                examples[index] = example
                break
        else:
            examples.append(example)
        self._save(examples)
        return example

    def add_many(self, examples: Iterable[LearningExample]) -> tuple[LearningExample, ...]:
        stored: list[LearningExample] = []
        for example in examples:
            stored.append(self.add(example))
        return tuple(stored)

    def set_status(self, example_id: str, status: str, *, note: str = "") -> LearningExample:
        """Move one example to a new status (refuses an unknown status)."""
        if status not in LEARNING_STATUSES:
            raise ValueError(f"Unknown status {status!r}; expected one of {LEARNING_STATUSES}.")
        example = self.get(example_id)
        if example is None:
            raise KeyError(f"No learning example {example_id!r} was found.")
        updated = example.with_status(status, note=note)
        return self.add(updated)

    def approve(self, example_id: str, *, note: str = "") -> LearningExample:
        return self.set_status(example_id, STATUS_APPROVED, note=note)

    def reject(self, example_id: str, *, note: str = "") -> LearningExample:
        return self.set_status(example_id, STATUS_REJECTED, note=note)

    def correct(self, example_id: str, answer: str, *, note: str = "") -> LearningExample:
        """Store a corrected answer; the corrected example counts as approved."""
        example = self.get(example_id)
        if example is None:
            raise KeyError(f"No learning example {example_id!r} was found.")
        return self.add(example.with_correction(answer, note=note))

    def mark_exported(self, example_ids: Iterable[str]) -> None:
        for example_id in example_ids:
            example = self.get(example_id)
            if example is None:
                continue
            self.add(example.with_status(STATUS_EXPORTED))

    def mark_used_in_training(self, example_ids: Iterable[str]) -> None:
        for example_id in example_ids:
            example = self.get(example_id)
            if example is None:
                continue
            self.add(example.with_status(STATUS_USED_IN_TRAINING))

    def remove(self, example_id: str) -> bool:
        examples = [example for example in self.all() if example.example_id != example_id]
        if len(examples) == len(self.all()):
            return False
        self._save(examples)
        return True

    def clear(self, *, status: str | None = None) -> int:
        """Delete every example (or every example with one status). Returns the count."""
        if status is None:
            removed = len(self.all())
            self._save([])
            return removed
        kept = [example for example in self.all() if example.status != status]
        removed = len(self.all()) - len(kept)
        self._save(kept)
        return removed

    # -- internals ---------------------------------------------------------
    def _save(self, examples: Iterable[LearningExample]) -> None:
        ordered = sorted(examples, key=lambda e: e.created_at)
        self._store.save(NAMESPACE, KEY, [example.to_primitive() for example in ordered])


def pending_examples(root: Path | str | None = None) -> tuple[LearningExample, ...]:
    """Every pending example, for a status line or a CLI listing."""
    return LearningReviewQueue(root=root).by_status(STATUS_PENDING)


__all__ = [
    "KEY",
    "NAMESPACE",
    "LearningReviewQueue",
    "ReviewSummary",
    "pending_examples",
]
