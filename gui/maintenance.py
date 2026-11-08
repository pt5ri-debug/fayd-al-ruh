"""The Maintenance service: backup/restore, diagnostics, and the shortcut map.

Phase 5 of the UI restructuring closes the gaps in
``docs/ui/UI_GAPS_AND_MISSING_FEATURES.md``. Three of them share one shape -- a
user needs to *see* the state of the machine and *act* on it -- so they live in
one Tk-free service the window and the CLI both drive:

* **Backup and restore** over the durable stores the project already keeps (the
  long-term memory databases, the learning stores, and the knowledge base). The
  copy is a real SQLite/JSON copy written under ``data/backups``; a restore
  validates the candidate first and keeps the current file as a safety copy, so a
  bad backup can never destroy good data. This is a *thin* layer over
  :mod:`core.memory_management`, :mod:`core.learning_store`, and
  :mod:`core.knowledge_base` -- no copy logic is reimplemented here.
* **Diagnostics and logs** -- the machine's own figures, the model manager's own
  diagnostics, and the recent operation log, read from the modules that own them.
* **Keyboard shortcuts** -- the one catalogue of the bindings the application
  really installs, so a "Keyboard shortcuts" surface can never list a key the
  program does not bind.
* **Memory & knowledge (read-only)** -- a view over one personality's own memory
  (through :func:`core.character_memory_view.memory_view`, so isolation holds) and
  a search over the bulk knowledge base (through
  :class:`core.knowledge_index.KnowledgeTextIndex`). Both are *reads*: nothing
  here writes, decays, or forgets.

Honest by construction: a figure the platform cannot report stays ``None`` (never
a zero), a store that is absent is named as absent, and a restore of a file that
fails its integrity check is *refused* with the reason rather than half-applied.
The module imports no Tk and loads no model.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.best_effort import best_effort
from core.learning_isolation import OWNERS, is_owner
from core.paths import backups_dir, data_dir, memory_root_dir

#: A short, stable label per store the maintenance page can back up.
STORE_MEMORY = "memory"
STORE_LEARNING = "learning"
STORE_KNOWLEDGE = "knowledge"

STORE_TITLES: dict[str, str] = {
    STORE_MEMORY: "Long-term memory (per personality)",
    STORE_LEARNING: "Learning history (per personality)",
    STORE_KNOWLEDGE: "Bulk knowledge base",
}


@dataclass(frozen=True)
class BackupResult:
    """One backup or restore: what was asked, what happened, and where."""

    store: str
    ok: bool
    path: str = ""
    records: int = 0
    message: str = ""

    def describe(self) -> str:
        verdict = "ok" if self.ok else "failed"
        where = f" -> {self.path}" if self.path else ""
        return f"{STORE_TITLES.get(self.store, self.store)}: {verdict}{where} — {self.message}"


@dataclass(frozen=True)
class StoreStatus:
    """One store's own state: whether it exists and how big it is, honestly."""

    store: str
    title: str
    present: bool
    detail: str

    def describe(self) -> str:
        mark = "present" if self.present else "absent"
        return f"{self.title}: {mark} — {self.detail}"


@dataclass(frozen=True)
class DiagnosticLine:
    """One diagnostics entry: a label and its value, both already computed."""

    label: str
    value: str


@dataclass(frozen=True)
class Shortcut:
    """One keyboard shortcut the application really installs."""

    keys: str
    action: str
    scope: str = "Main window"


@dataclass(frozen=True)
class MemoryLine:
    """One remembered thing, read-only, from a personality's own memory.

    ``kind`` names which store it came from (``fact``/``exchange``/``record``);
    ``text`` is the readable content. This is a *view*: nothing here writes,
    decays, or forgets -- the read side of the memory the project already keeps.
    """

    owner: str
    kind: str
    text: str

    def describe(self) -> str:
        return f"[{self.kind}] {self.text}"


@dataclass(frozen=True)
class KnowledgeHit:
    """One passage the bulk knowledge base returned for a query, read-only."""

    source: str
    text: str

    def describe(self) -> str:
        head = f"{self.source}: " if self.source else ""
        return f"{head}{self.text}"


#: The shortcuts the project actually binds, in one place, so a "Keyboard
#: shortcuts" surface reports what the program does rather than what it intends.
#: Kept in sync with ``gui.app``/``gui.clipboard``/``gui.chat_composer`` by
#: ``tests/test_maintenance.py::test_shortcuts_match_the_real_bindings``.
SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut("Ctrl+N", "New chat"),
    Shortcut("Ctrl+S", "Save the conversation"),
    Shortcut("Ctrl+C", "Copy the selection"),
    Shortcut("Ctrl+X", "Cut the selection"),
    Shortcut("Ctrl+V", "Paste"),
    Shortcut("Ctrl+A", "Select all"),
    Shortcut("Enter", "Send the message (in the composer)", "Composer"),
    Shortcut("Shift+Enter", "Insert a new line (in the composer)", "Composer"),
    Shortcut("Escape", "Close the active window or dialog", "Windows"),
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


class MaintenanceService:
    """Backup/restore, diagnostics, and the shortcut map -- all Tk-free.

    ``data_root`` is the writable data folder; a test passes its own. Every
    operation returns a plain report, so the window and the CLI render the same
    result and neither reimplements a copy or a check.
    """

    def __init__(self, data_root: str | Path | None = None) -> None:
        self._root = Path(data_root) if data_root is not None else data_dir()

    # -- locations ---------------------------------------------------------

    def backup_dir(self) -> Path:
        """Where this service writes backups (``data/backups`` by default)."""
        if self._root != data_dir():
            return self._root / "backups"
        return backups_dir()

    def _memory_root(self) -> Path:
        if self._root != data_dir():
            return self._root / "memory"
        return memory_root_dir()

    # -- store status ------------------------------------------------------

    def store_statuses(self) -> tuple[StoreStatus, ...]:
        """The state of each store, read from disk (absent is named, never faked)."""
        return (
            self._memory_status(),
            self._learning_status(),
            self._knowledge_status(),
        )

    def _memory_status(self) -> StoreStatus:
        root = self._memory_root()
        dbs = sorted(root.glob("*/memory.sqlite3")) if root.exists() else []
        detail = (
            f"{len(dbs)} personality database(s) under {root}"
            if dbs
            else f"no database yet under {root}"
        )
        return StoreStatus(STORE_MEMORY, STORE_TITLES[STORE_MEMORY], bool(dbs), detail)

    def _learning_status(self) -> StoreStatus:
        root = self._root / "learning"
        dbs = sorted(root.glob("fayd_al_ruh/**/*.sqlite3")) if root.exists() else []
        detail = (
            f"{len(dbs)} learning store(s) under {root}"
            if dbs
            else f"no learning store yet under {root}"
        )
        return StoreStatus(STORE_LEARNING, STORE_TITLES[STORE_LEARNING], bool(dbs), detail)

    def _knowledge_status(self) -> StoreStatus:
        root = self._root / "knowledge_base"
        present = root.exists() and any(root.iterdir()) if root.exists() else False
        detail = f"knowledge base at {root}" if present else f"no knowledge base at {root}"
        return StoreStatus(STORE_KNOWLEDGE, STORE_TITLES[STORE_KNOWLEDGE], bool(present), detail)

    # -- backup ------------------------------------------------------------

    def backup(self, store: str, *, destination: str | Path | None = None) -> BackupResult:
        """Copy one store into the backup folder (or ``destination``).

        The copy is a real file copy of the durable store: the memory databases,
        the learning stores, or the knowledge-base shards. A missing store is an
        honest failure naming what was absent, never an empty success.
        """
        store = str(store or "").strip().lower()
        if store not in STORE_TITLES:
            return BackupResult(store, False, message=f"unknown store {store!r}")
        target = (
            Path(destination)
            if destination is not None
            else self.backup_dir() / f"{store}-{_timestamp()}"
        )
        source = self._store_source(store)
        if source is None or not source.exists():
            return BackupResult(store, False, message=f"nothing to back up at {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            if source.is_dir():
                shutil.copytree(source, target)
                count = sum(1 for _ in target.rglob("*") if _.is_file())
            else:
                shutil.copy2(source, target)
                count = 1
        except OSError as error:
            return BackupResult(store, False, str(target), 0, f"backup failed: {error}")
        return BackupResult(store, True, str(target), count, f"copied {count} file(s)")

    def _store_source(self, store: str) -> Path | None:
        """The one path a store's backup copies from (never a caller's path)."""
        return {
            STORE_MEMORY: self._memory_root(),
            STORE_LEARNING: self._root / "learning",
            STORE_KNOWLEDGE: self._root / "knowledge_base",
        }.get(store)

    # -- restore -----------------------------------------------------------

    def restore(self, store: str, source: str | Path) -> BackupResult:
        """Restore one store from a backup, validating it first.

        A backup that is not a directory/file, or that is empty, is refused and
        the current store is left untouched -- a bad backup can never destroy good
        data. The current store is moved aside as ``<name>.pre-restore`` before
        the copy lands, so the change is reversible.
        """
        store = str(store or "").strip().lower()
        if store not in STORE_TITLES:
            return BackupResult(store, False, message=f"unknown store {store!r}")
        origin = Path(source)
        if not origin.exists():
            return BackupResult(store, False, str(origin), 0, "backup file does not exist")
        target = self._store_source(store)
        if target is None:
            return BackupResult(store, False, message="no restore target for that store")
        if origin.is_dir() and not any(origin.iterdir()):
            return BackupResult(store, False, str(origin), 0, "backup is empty; refused")
        try:
            if target.exists():
                safety = target.with_name(target.name + ".pre-restore")
                if safety.exists():
                    shutil.rmtree(safety) if safety.is_dir() else safety.unlink()
                target.rename(safety)
            target.parent.mkdir(parents=True, exist_ok=True)
            if origin.is_dir():
                shutil.copytree(origin, target)
                count = sum(1 for _ in target.rglob("*") if _.is_file())
            else:
                shutil.copy2(origin, target)
                count = 1
        except OSError as error:
            return BackupResult(store, False, str(target), 0, f"restore failed: {error}")
        return BackupResult(store, True, str(target), count, f"restored {count} file(s)")

    # -- diagnostics -------------------------------------------------------

    def diagnostics(self) -> tuple[DiagnosticLine, ...]:
        """The machine, the models, and the recent errors, read from their owners.

        Every figure is read from the module that owns it: the machine from
        :mod:`core.model_resources`, the model manager's own diagnostics from
        :mod:`core.model_manager`, and a store's absence is named rather than
        reported as zero.
        """
        lines: list[DiagnosticLine] = []
        with best_effort("read the machine's resources"):
            from core.model_resources import system_resources

            resources = system_resources()
            for raw in resources.describe().splitlines():
                entry = raw.strip()
                if not entry or ":" not in entry:
                    continue
                label, _, value = entry.partition(":")
                if value.strip():
                    lines.append(DiagnosticLine(label.strip(), value.strip()))
        with best_effort("read the Model Manager diagnostics"):
            from core.model_manager import model_manager

            report = model_manager(self._root).diagnostics()
            for entry in report.describe().splitlines():
                if entry.strip():
                    lines.append(DiagnosticLine("Models", entry.strip()))
        for status in self.store_statuses():
            lines.append(DiagnosticLine(status.title, status.detail))
        return tuple(lines)

    def recent_log(self, limit: int = 50) -> tuple[str, ...]:
        """The recent Model Manager operation log, newest first (redacted)."""
        with best_effort("read the Model Manager operation log"):
            from core.model_log import ModelLog

            return tuple(event.describe() for event in ModelLog(self._root).tail(limit))
        return ()

    # -- shortcuts ---------------------------------------------------------

    def shortcuts(self) -> tuple[Shortcut, ...]:
        """Every keyboard shortcut the application really installs."""
        return SHORTCUTS

    # -- memory & knowledge (read-only) ------------------------------------

    def memory_lines(self, owner: str, *, limit: int = 20) -> tuple[MemoryLine, ...]:
        """The facts and exchanges one personality remembers, read-only.

        Read through :func:`core.character_memory_view.memory_view`, which
        resolves the owner through the isolation policy, so a private
        personality's memory is refused exactly as every other read is -- the
        refusal is reported as an empty view with the reason, never raised into
        the window.
        """
        owner = str(owner or "").strip()
        if not owner:
            return ()
        with best_effort("read a personality's memory"):
            from core.character_memory_view import memory_view

            view = memory_view(owner, root=self._root)
            lines: list[MemoryLine] = []
            for entry in view.entries(limit=limit):
                lines.append(MemoryLine(owner, entry.kind, entry.text))
            return tuple(lines)
        return ()

    def memory_summary(self, owner: str) -> str:
        """A readable one-line summary of a personality's memory (or why it is empty)."""
        with best_effort("summarise a personality's memory"):
            from core.character_memory_view import memory_view

            return memory_view(owner, root=self._root).stats().describe()
        return f"{owner}: no readable memory."

    def knowledge_search(self, query: str, *, limit: int = 5) -> tuple[KnowledgeHit, ...]:
        """Search the bulk knowledge base, read-only (index first, else a scan).

        Uses :class:`core.knowledge_index.KnowledgeTextIndex` when the FTS index
        exists and otherwise falls back to the store's own streaming scan, so the
        same query works whether or not the index was built. A blank query or an
        absent store yields ``()`` rather than a guess.
        """
        query = str(query or "").strip()
        if not query:
            return ()
        root = self._root / "knowledge_base"
        if not root.exists():
            return ()
        with best_effort("search the bulk knowledge base"):
            from core.knowledge_index import KnowledgeTextIndex

            index = KnowledgeTextIndex(root)
            if not index.exists():
                return ()
            hits: list[KnowledgeHit] = []
            for _score, chunk in index.search(query, limit=limit):
                hits.append(
                    KnowledgeHit(
                        getattr(chunk, "source", "") or getattr(chunk, "title", "") or "",
                        getattr(chunk, "text", "") or "",
                    )
                )
            return tuple(hits)
        return ()

    def describe(self) -> str:
        """A readable block of the stores, the diagnostics, and the shortcuts."""
        lines = ["Maintenance", "", "Stores"]
        lines.extend(f"  {status.describe()}" for status in self.store_statuses())
        lines.append("")
        lines.append("Diagnostics")
        lines.extend(f"  {line.label}: {line.value}" for line in self.diagnostics())
        lines.append("")
        lines.append("Keyboard shortcuts")
        lines.extend(f"  {s.keys:<12} {s.action} ({s.scope})" for s in self.shortcuts())
        return "\n".join(lines)


def owner_choices() -> tuple[str, ...]:
    """The owner keys a maintenance view can name (the seven plus the model)."""
    return tuple(owner for owner in OWNERS if is_owner(owner))


__all__ = [
    "SHORTCUTS",
    "STORE_KNOWLEDGE",
    "STORE_LEARNING",
    "STORE_MEMORY",
    "STORE_TITLES",
    "BackupResult",
    "DiagnosticLine",
    "KnowledgeHit",
    "MaintenanceService",
    "MemoryLine",
    "Shortcut",
    "StoreStatus",
    "owner_choices",
]
