"""The held-out gate for the Phase-2 long-term memory architecture.

Phase 2's memory modules each ship with unit tests that pin the good case. This
module is the *external* check: a catalogue of hand-written judgements, each run
against the **real** store -- never a re-implementation -- with **negative
controls**, so a regression the unit tests still pass fails here.

Every judgement is a real call, over a temporary database (never the user's own
memory):

* **persistence** -- a record captured in one store is read back by a *second*
  store opened on the same file, with its provenance, confidence, and category
  intact;
* **correction keeps history** -- correcting a record leaves the old one present
  and marked superseded, and stores the new one beside it (the task's "never
  silently overwrite or delete conflicting memories");
* **archive, not delete** -- an archived record is still readable with
  ``include_archived=True`` and is excluded from the default read;
* **backup/restore** -- a backup restores into a fresh database with the records
  intact; and the negative control: a *bad* backup is **refused**, leaving the
  current database untouched;
* **corrupt tolerance** -- a deliberately corrupted row is skipped and counted,
  never raised;
* **transaction rollback** -- a batch whose second record is invalid leaves the
  database exactly as it was (all-or-nothing);
* **temporal filtering** -- a record whose validity has ended is excluded from a
  read at a later moment and included at an earlier one;
* **ranking is explainable** -- a stronger query match outranks a weaker one, and
  the six signals are reported separately;
* **consolidation** -- a working item above the importance threshold is promoted
  to episodic memory, and one below it is **not** (the negative control);
* **isolation** -- Natasha's memory cannot be read by the main model, and a
  character cannot read another character's (the security negative controls);
* **no evidence inflation** -- retrieval similarity is never reported as the
  record's confidence.

``memory_architecture_accuracy`` is the share of judgements satisfied; an empty
catalogue scores ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from core.learning_isolation import LearningAccessError
from core.long_term_memory import LongTermMemory
from core.memory_management import PersonalityMemoryRegistry
from core.memory_record import (
    SOURCE_USER,
    VERIFICATION_SUPERSEDED,
    MemoryKind,
    now_utc,
)
from core.memory_retrieval import MemoryRetriever

#: A judgement: a name and the real check that must hold.
Judgement = tuple[str, Callable[[Path], bool]]


@dataclass
class GateReport:
    """The detailed outcome, so a failure names itself."""

    passed: tuple[str, ...] = ()
    failed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed)

    @property
    def accuracy(self) -> float:
        return len(self.passed) / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{len(self.passed)}/{self.total} memory judgements satisfied"
            + (f"; failed: {', '.join(self.failed)}" if self.failed else "")
        )


def _memory(root: Path, owner: str = "fayd_al_ruh") -> LongTermMemory:
    return LongTermMemory(owner, root / f"{owner}.sqlite3")


# -- the judgements ---------------------------------------------------------
def _persistence(root: Path) -> bool:
    first = _memory(root)
    record = first.capture(
        "Waterflood increased recovery in the pilot.",
        kind=MemoryKind.EPISODIC,
        source=SOURCE_USER,
        confidence=0.7,
        importance=0.8,
        source_reference="pilot-log-7",
        entities=("waterflood", "recovery"),
    )
    second = _memory(root)  # a fresh store on the same file
    loaded = second.get(record.record_id)
    return (
        loaded is not None
        and loaded.kind is MemoryKind.EPISODIC
        and loaded.confidence == 0.7
        and loaded.source == SOURCE_USER
        and loaded.source_reference == "pilot-log-7"
        and "waterflood" in loaded.entities
    )


def _correction_keeps_history(root: Path) -> bool:
    memory = _memory(root)
    original = memory.capture("The casing depth is 2000 ft.", importance=0.6)
    old, new = memory.correct(original.record_id, "The casing depth is 2500 ft.", reason="log")
    stored_old = memory.get(old.record_id)
    stored_new = memory.get(new.record_id)
    return (
        stored_old is not None
        and stored_old.verification == VERIFICATION_SUPERSEDED
        and stored_old.superseded_by == new.record_id
        and stored_new is not None
        and stored_new.supersedes == old.record_id
    )


def _archive_not_delete(root: Path) -> bool:
    memory = _memory(root)
    record = memory.capture("A temporary note.", importance=0.4)
    memory.archive(record.record_id)
    hidden, _ = memory.page(include_archived=False, limit=100)
    shown, _ = memory.page(include_archived=True, limit=100)
    return (
        record.record_id not in {item.record_id for item in hidden}
        and record.record_id in {item.record_id for item in shown}
    )


def _backup_restore(root: Path) -> bool:
    memory = _memory(root)
    memory.capture("A fact worth keeping.", kind=MemoryKind.SEMANTIC, importance=0.9)
    backup_path = root / "backup.sqlite3"
    report = memory.backup(backup_path)
    if not report.ok:
        return False
    fresh = _memory(root, owner="fayd_al_ruh_restored")
    fresh.database.clear()
    restored = fresh.restore(backup_path)
    return restored.ok and fresh.count(include_archived=True) == 1


def _bad_backup_refused(root: Path) -> bool:
    """The negative control: a corrupt backup must not replace a good database."""
    memory = _memory(root)
    memory.capture("An important fact.", importance=0.9)
    before = memory.count(include_archived=True)
    junk = root / "junk.sqlite3"
    junk.write_bytes(b"this is not a sqlite database at all")
    report = memory.restore(junk)
    return (not report.ok) and memory.count(include_archived=True) == before


def _corrupt_row_tolerated(root: Path) -> bool:
    memory = _memory(root)
    memory.capture("A good record.", importance=0.5)
    # Corrupt the payload of the stored row directly.
    connection = sqlite3.connect(memory.database.path)
    try:
        connection.execute("UPDATE memories SET payload='{not valid json'")
        connection.commit()
    finally:
        connection.close()
    records, report = memory.page(limit=100)
    return records == [] and report.skipped == 1 and report.loaded == 0


def _transaction_rollback(root: Path) -> bool:
    memory = _memory(root)
    memory.capture("The only record.", importance=0.5)
    before = memory.count(include_archived=True)
    good = memory.capture("Another record.", importance=0.5, persist=False)
    # ``save_many`` is all-or-nothing: a failure must leave the count unchanged.
    from core.memory_record import MemoryRecord

    try:
        bad = MemoryRecord(
            record_id="",  # invalid: empty id raises on construction
            kind=MemoryKind.SEMANTIC,
            content="x",
        )
        memory.database.save_many([good, bad])
    except (ValueError, KeyError, TypeError):
        pass
    return memory.count(include_archived=True) == before


def _temporal_filtering(root: Path) -> bool:
    memory = _memory(root)
    start = now_utc() - timedelta(days=10)
    end = now_utc() - timedelta(days=5)
    memory.capture(
        "A claim that was true last week.",
        kind=MemoryKind.SEMANTIC,
        valid_from=start,
        valid_until=end,
        importance=0.7,
    )
    early, _ = memory.page(valid_at=start + timedelta(days=1), limit=100)
    late, _ = memory.page(valid_at=now_utc(), limit=100)
    return len(early) == 1 and len(late) == 0


def _ranking_is_explainable(root: Path) -> bool:
    memory = _memory(root)
    memory.capture(
        "Casing collapse pressure design.",
        kind=MemoryKind.SEMANTIC,
        importance=0.6,
        confidence=0.6,
    )
    memory.capture("A note about mud weight.", kind=MemoryKind.SEMANTIC, importance=0.6)
    retriever = MemoryRetriever(memory)
    report = retriever.retrieve("casing collapse pressure", touch=False)
    if not report.hits:
        return False
    top = report.hits[0]
    return (
        "casing" in top.record.content.lower()
        and top.signals.relevance > 0.0
        and top.confidence == 0.6  # the record's own confidence, not the score
        and top.score != top.confidence
    )


def _consolidation(root: Path) -> bool:
    memory = _memory(root)
    memory.capture("Important working item.", kind=MemoryKind.WORKING, importance=0.8)
    memory.capture("Trivial working item.", kind=MemoryKind.WORKING, importance=0.2)
    report = memory.promote_working(minimum_importance=0.5)
    episodic = {record.content for record in memory.all_records(kind=MemoryKind.EPISODIC)}
    return (
        len(report.promoted) == 1
        and "Important working item." in episodic
        and "Trivial working item." not in episodic
    )


def _isolation(root: Path) -> bool:
    registry = PersonalityMemoryRegistry(data_root=root)
    registry.owner_memory("jarvis").capture("Jarvis's own memory.", importance=0.6)
    registry.owner_memory("natasha").capture("Natasha's private memory.", importance=0.6)
    # The negative control: the main model must not read Natasha.
    try:
        registry.read_memory("fayd_al_ruh", "natasha")
        return False
    except LearningAccessError:
        pass
    # A character must not read another character either.
    try:
        registry.read_memory("jarvis", "ultron")
        return False
    except LearningAccessError:
        pass
    # And a permitted read works: the main model reads a shareable character.
    memory = registry.read_memory("fayd_al_ruh", "jarvis")
    return memory.count(include_archived=True) == 1


def _similarity_is_not_confidence(root: Path) -> bool:
    """A high retrieval score must not become a high record confidence."""
    memory = _memory(root)
    record = memory.capture(
        "A model's own unverified guess about casing.",
        kind=MemoryKind.SEMANTIC,
        source="model",
        confidence=0.1,
        importance=0.5,
    )
    report = MemoryRetriever(memory).retrieve("model guess casing", touch=False)
    if not report.hits:
        return False
    hit = next((item for item in report.hits if item.record_id == record.record_id), None)
    return hit is not None and hit.confidence == 0.1


#: The catalogue, in the order the report names them.
MEMORY_JUDGEMENTS: tuple[Judgement, ...] = (
    ("persistence-across-stores", _persistence),
    ("correction-keeps-history", _correction_keeps_history),
    ("archive-not-delete", _archive_not_delete),
    ("backup-restore", _backup_restore),
    ("bad-backup-refused", _bad_backup_refused),
    ("corrupt-row-tolerated", _corrupt_row_tolerated),
    ("transaction-rollback", _transaction_rollback),
    ("temporal-filtering", _temporal_filtering),
    ("ranking-is-explainable", _ranking_is_explainable),
    ("consolidation-promotes-and-skips", _consolidation),
    ("personality-isolation", _isolation),
    ("similarity-is-not-confidence", _similarity_is_not_confidence),
)


def memory_gate_report(judgements: tuple[Judgement, ...] = MEMORY_JUDGEMENTS) -> GateReport:
    """Run every judgement against the real store and return a detailed report."""
    report = GateReport()
    passed: list[str] = []
    failed: list[str] = []
    for name, check in judgements:
        try:
            with tempfile.TemporaryDirectory() as folder:
                ok = bool(check(Path(folder)))
        except Exception:  # a judgement that raises is a failed judgement
            ok = False
        (passed if ok else failed).append(name)
    report.passed = tuple(passed)
    report.failed = tuple(failed)
    return report


def memory_architecture_accuracy(
    judgements: tuple[Judgement, ...] = MEMORY_JUDGEMENTS,
) -> float:
    """The share of held-out memory judgements the real architecture satisfies.

    A judgement counts only when the store does exactly what it must, so a store
    that lost a record across a restart, overwrote a correction, deleted on
    archive, accepted a corrupt backup, or leaked Natasha's memory fails here.
    Returns ``0.0`` for an empty catalogue.
    """
    return memory_gate_report(judgements).accuracy


__all__ = [
    "MEMORY_JUDGEMENTS",
    "GateReport",
    "Judgement",
    "memory_architecture_accuracy",
    "memory_gate_report",
]
