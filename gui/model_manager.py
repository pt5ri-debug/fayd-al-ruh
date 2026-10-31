"""The Model Manager service: the engine behind the window and the CLI.

One Tk-free service drives the whole Model Manager surface, so the window,
``python -m cli models``, and any other caller share one registry, one runtime
view, and one operation log -- no listing or state is computed twice.

The service is deliberately a **thin** layer: it does not re-implement discovery,
inspection, compatibility, or runtime work, all of which already live in
:mod:`core.model_manager` and its collaborators. What it adds is the shape a
window and a CLI both want -- rows a table can draw, a saved selection, a
progress callback for a slow operation, and a background-thread seam so a long
load or test never freezes the UI.

Everything here is exercised headless: no display, no GPU, and no model file.
A ``manager`` may be injected (a test's own), so nothing here needs a real
registry or a real runtime.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.model_files import ModelFileError
from core.model_log import ModelEvent
from core.model_manager import (
    STATE_DAMAGED,
    STATE_MISSING,
    STATE_NOT_DOWNLOADED,
    STATE_ORDER,
    STATE_READY,
    STATE_UNKNOWN,
    TASKS,
    CompatibilityReport,
    DiagnosticsReport,
    ModelEntry,
    ModelManager,
    OperationResult,
)
from core.model_manager import (
    model_manager as build_model_manager,
)
from core.model_registry import (
    SOURCE_LOCAL,
    ConfirmationRequired,
    ModelNotFoundError,
    RegistryError,
)
from core.model_runtime import RuntimeUnavailable
from core.model_settings import SettingsError
from core.paths import data_dir
from storage.json_store import JsonStore

#: A one-word marker per state, for a table cell that has little room.
STATE_MARKERS: dict[str, str] = {
    STATE_READY: "✓ ready",
    STATE_NOT_DOWNLOADED: "· not downloaded",
    STATE_MISSING: "! file missing",
    STATE_DAMAGED: "! damaged",
    STATE_UNKNOWN: "? unknown",
}

#: A short, honest sentence per state, so a table's tooltip never overstates.
STATE_BLURBS: dict[str, str] = {
    STATE_READY: "The file is present and usable; this model can be loaded and run.",
    STATE_NOT_DOWNLOADED: "The build knows this model, but its weights are not here yet.",
    STATE_MISSING: "The model was registered, but its file is no longer on disk.",
    STATE_DAMAGED: "The file is present but unreadable or truncated, so it cannot be loaded.",
    STATE_UNKNOWN: "The state could not be determined from what is on disk.",
}

#: The namespace and key the last selection is remembered under.
_NAMESPACE = "models"
_KEY = "manager_selection"


@dataclass(frozen=True)
class ModelRow:
    """One model as a table row: what to draw, and what to sort on."""

    model_id: str
    display_name: str
    state: str
    state_marker: str
    runtime: str
    source: str
    format: str
    path: str
    size_bytes: int
    quantization: str
    ready: bool
    from_catalog: bool
    status_line: str

    def describe(self) -> str:
        """A one-line description, for a status line or a log entry."""
        where = self.path or "no local file"
        return f"{self.display_name} [{self.state_marker}] ({self.runtime}) — {where}"


@dataclass(frozen=True)
class ModelDetail:
    """One model, opened: the row plus everything the registry holds about it."""

    row: ModelRow
    identity: str
    integrity: str
    integrity_detail: str
    architecture: str
    context_length: int | None
    parameter_count: int | None
    license: str
    tags: tuple[str, ...]
    notes: str
    engine: str
    registered_at: str
    updated_at: str
    settings: Any = None
    compatibility: CompatibilityReport | None = None
    download_command: str = ""
    corrupt: bool = False

    @property
    def model_id(self) -> str:
        return self.row.model_id

    @property
    def display_name(self) -> str:
        return self.row.display_name


@dataclass(frozen=True)
class ManagerSnapshot:
    """Everything a window needs to draw itself in one call."""

    rows: tuple[ModelRow, ...] = ()
    registered: int = 0
    catalog: int = 0
    ready: int = 0
    runtime_lines: tuple[str, ...] = ()
    corrupt_records: tuple[str, ...] = ()
    log_events: int = 0
    #: The models currently *loaded* in their runtime (their ids), so a dashboard
    #: can say what is active rather than only what is present.
    active: tuple[str, ...] = ()
    #: Recent failed operations, newest first, ready to show under "Errors".
    errors: tuple[str, ...] = ()
    #: The machine's own figures (a :class:`core.model_resources.SystemResources`),
    #: or ``None`` when they could not be read.
    resources: Any = None

    def summary(self) -> str:
        return (
            f"{self.registered} registered, {self.catalog} in the build's catalog, "
            f"{self.ready} ready to run."
        )


@dataclass(frozen=True)
class OperationReport:
    """The outcome of a background operation, in a shape a window can post."""

    ok: bool
    operation: str
    model_id: str = ""
    detail: str = ""
    result: OperationResult | None = None
    events: tuple[ModelEvent, ...] = field(default=())

    def describe(self) -> str:
        if self.result is not None:
            return self.result.describe()
        return self.detail or ("done" if self.ok else "failed")


class ModelManagerService:
    """The window's and the CLI's view over :class:`core.model_manager.ModelManager`.

    ``root`` is the writable data folder; a test passes its own. A ``manager``
    may be injected instead, so a test can drive the whole service with a fake
    registry and fake runtimes.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        manager: ModelManager | None = None,
        store: JsonStore | None = None,
    ) -> None:
        self.manager = manager if manager is not None else build_model_manager(root)
        self._store = store if store is not None else JsonStore(root or data_dir())
        self._selection: str = self._load_selection()

    # ------------------------------------------------------------- snapshot

    def snapshot(self) -> ManagerSnapshot:
        """Every model plus the subsystem's counts, in one read.

        Includes what a *dashboard* needs beyond a listing: the models currently
        loaded, the recent failures, and the machine's own figures -- each read
        from the module that owns it, so nothing is recomputed here.
        """
        status = self.manager.status()
        rows = tuple(self._row(entry) for entry in self.manager.list_models())
        active = tuple(entry.model_id for entry in self.manager.active_models())
        errors = tuple(
            event.describe() for event in self.manager.log_tail(50) if not event.ok
        )[:10]
        resources = None
        with best_effort("read the machine's resources"):
            from core.model_resources import system_resources

            resources = system_resources()
        return ManagerSnapshot(
            rows=rows,
            registered=status.registered,
            catalog=status.catalog,
            ready=status.ready,
            runtime_lines=tuple(status.runtimes.splitlines()),
            corrupt_records=tuple(status.corrupt_records),
            log_events=status.log_events,
            active=active,
            errors=errors,
            resources=resources,
        )

    def search(
        self,
        query: str = "",
        *,
        state: str = "",
        runtime: str = "",
        rows: tuple[ModelRow, ...] | None = None,
    ) -> tuple[ModelRow, ...]:
        """Filter a listing by free text and by state/runtime.

        A *view* over the rows already read, so the same models are never listed
        twice. The text matches the display name, model id, path, engine, and
        quantization, case-insensitively; an empty filter keeps everything, so a
        blank box shows the whole library.
        """
        source = rows if rows is not None else self.rows()
        needle = str(query or "").strip().casefold()
        wanted_state = str(state or "").strip()
        wanted_runtime = str(runtime or "").strip()
        result: list[ModelRow] = []
        for row in source:
            if wanted_state and row.state != wanted_state:
                continue
            if wanted_runtime and row.runtime != wanted_runtime:
                continue
            if needle and needle not in self._haystack(row):
                continue
            result.append(row)
        return tuple(result)

    @staticmethod
    def _haystack(row: ModelRow) -> str:
        """The lower-cased text a search matches against one row."""
        return " ".join(
            (row.display_name, row.model_id, row.path, row.runtime, row.quantization, row.source)
        ).casefold()

    def state_names(self) -> tuple[str, ...]:
        """Every distinct state a row can carry, in the build's honest order."""
        return STATE_ORDER

    def diagnostics(self) -> DiagnosticsReport:
        """The whole diagnostics body, straight from the core manager."""
        return self.manager.diagnostics()

    def rows(self, *, registered_only: bool = False) -> tuple[ModelRow, ...]:
        """The models as table rows, catalog first filtered if asked."""
        return tuple(
            self._row(entry)
            for entry in self.manager.list_models(include_catalog=not registered_only)
        )

    def _row(self, entry: ModelEntry) -> ModelRow:
        record = entry.record
        return ModelRow(
            model_id=entry.model_id,
            display_name=entry.display_name,
            state=entry.state,
            state_marker=STATE_MARKERS.get(entry.state, entry.state),
            runtime=entry.runtime,
            source=record.source,
            format=record.model_format,
            path=str(record.path) if record.path else "",
            size_bytes=record.size_bytes,
            quantization=record.quantization,
            ready=entry.ready,
            from_catalog=entry.from_catalog,
            status_line=entry.status_line,
        )

    def task_names(self) -> tuple[str, ...]:
        """Every task a model can be chosen for, in the build's order."""
        return tuple(TASKS)

    def models_for_task(self, task: str) -> tuple[ModelRow, ...]:
        """The rows suited to ``task``, best first (see the manager's own rule)."""
        return tuple(self._row(entry) for entry in self.manager.models_for_task(task))

    def select_for_task(self, task: str) -> ModelRow | None:
        """The single best ready row for ``task``, or ``None`` when none can run."""
        entry = self.manager.select_for_task(task)
        return self._row(entry) if entry is not None else None

    # ------------------------------------------------------------ selection

    def selection(self) -> str:
        """The remembered model id, or ``""`` when nothing is chosen."""
        return self._selection

    def select(self, model_id: str) -> str:
        """Remember which model the user is looking at, and return it."""
        key = str(model_id or "").strip()
        self._selection = key
        with best_effort("remember the model selection"):
            self._store.save(_NAMESPACE, _KEY, {"model_id": key})
        return self._selection

    def _load_selection(self) -> str:
        """Read the remembered selection, tolerating an absent or corrupt record."""
        try:
            saved = self._store.load(_NAMESPACE, _KEY)
        except Exception:
            return ""
        if isinstance(saved, dict):
            return str(saved.get("model_id", "") or "")
        return ""

    # ---------------------------------------------------------------- detail

    def detail(self, model_id: str) -> ModelDetail:
        """One model opened: its record, its settings, and its compatibility."""
        entry = self.manager.get(model_id)
        record = entry.record
        settings = None
        compatibility: CompatibilityReport | None = None
        download = ""
        try:
            settings = self.manager.settings(entry.model_id)
        except (ModelNotFoundError, RegistryError, SettingsError):
            settings = None
        try:
            compatibility = self.manager.compatibility(entry.model_id)
        except (ModelNotFoundError, RegistryError, RuntimeUnavailable):
            compatibility = None
        try:
            download = self.manager.download_command(entry.model_id)
        except (ModelNotFoundError, RegistryError):
            download = ""
        return ModelDetail(
            row=self._row(entry),
            identity=record.identity,
            integrity=record.integrity,
            integrity_detail=record.integrity_detail,
            architecture=record.architecture,
            context_length=record.context_length,
            parameter_count=record.parameter_count,
            license=record.license,
            tags=tuple(record.tags),
            notes=record.notes,
            engine=record.engine,
            registered_at=record.registered_at,
            updated_at=record.updated_at,
            settings=settings,
            compatibility=compatibility,
            download_command=download,
        )

    # ----------------------------------------------------------- operations

    def discover(self, roots: Any = None) -> Any:
        """Scan for model files, returning the core report unchanged."""
        return self.manager.discover(roots)

    def register(self, path: str | Path, **kwargs: Any) -> ModelRow:
        """Register a local file and return its row."""
        return self._row(self.manager.register(path, **kwargs))

    def import_custom_models(self, *, replace_existing: bool = False) -> tuple[ModelRow, ...]:
        """Bring in the models added by hand in a mouth window."""
        return tuple(
            self._row(entry)
            for entry in self.manager.import_custom_models(replace_existing=replace_existing)
        )

    def update(self, model_id: str, **changes: Any) -> ModelRow:
        """Edit a record's user-owned fields."""
        return self._row(self.manager.update(model_id, **changes))

    def remove(self, model_id: str, *, confirm: bool = False) -> ModelRow:
        """Forget a record (the file is never touched)."""
        record = self.manager.remove(model_id, confirm=confirm)
        return self._row(ModelEntry(record=record))

    def inspect(self, model_id: str, *, digest: bool = False) -> Any:
        """Re-read a model's file from disk."""
        return self.manager.inspect(model_id, digest=digest)

    def compatibility(self, model_id: str) -> CompatibilityReport:
        """Whether a model can run here, and why not when it cannot."""
        return self.manager.compatibility(model_id)

    def download_command(self, model_id: str) -> str:
        """The command that fetches a model's weights, or ``""`` when none exists."""
        return self.manager.download_command(model_id)

    # -------------------------------------------------------- selection

    def rank_for_task(self, task: str, *, preferred: str = "") -> tuple[Any, ...]:
        """Rank the models for a task, with the reasons behind each score."""
        return self.manager.rank_for_task(
            task, preferred_model_id=preferred or self._selection
        )

    def fallback_plan(self, task: str) -> Any:
        """The bounded ladder of models to try for a task, or why there is none."""
        return self.manager.fallback_plan(task, preferred_model_id=self._selection)

    # ----------------------------------------------------------- sources

    def sources(self) -> tuple[Any, ...]:
        """The trusted model sources this build supports."""
        return self.manager.sources()

    def search_offers(self, query: str, *, source: str = "", limit: int = 10) -> tuple[Any, ...]:
        """Models matching a query across the sources. Downloads nothing."""
        return self.manager.search_offers(query, source=source, limit=limit)

    def download(
        self,
        offer: Any,
        *,
        approved: bool = False,
        should_cancel: Any = None,
        on_progress: Any = None,
        pinned_digests: Any = None,
    ) -> Any:
        """Fetch an offer's files. A governed action; the caller grants approval."""
        return self.manager.download(
            offer, approved=approved, should_cancel=should_cancel,
            on_progress=on_progress, pinned_digests=pinned_digests,
        )

    def resumable_downloads(self) -> tuple[Path, ...]:
        """The partial downloads a later run could resume."""
        return self.manager.resumable_downloads()

    # ----------------------------------------------- resources & processes

    def resources(self) -> Any:
        """This machine's own figures, each ``None`` when the platform cannot say."""
        return self.manager.resources()

    def preflight(self, model_id: str, *, context_length: int | None = None) -> Any:
        """Every check that should pass before a model is loaded, as data."""
        return self.manager.preflight(model_id, context_length=context_length)

    def memory_estimate(self, model_id: str) -> Any:
        """An estimate of a model's memory need, with its uncertainty stated."""
        return self.manager.memory_estimate(model_id)

    def processes(self) -> Any:
        """The model processes this program started."""
        return self.manager.processes()

    def stop_processes(self) -> tuple[Any, ...]:
        """Stop every model process this program started."""
        return self.manager.processes().stop_all()

    # ------------------------------------------------------ trained models

    def trained_artifacts(self) -> tuple[Any, ...]:
        """The project's own trained artifacts, described read-only."""
        return self.manager.trained_artifacts()

    def trained_model(self) -> Any:
        """The project's own trained language model, or ``None``."""
        return self.manager.trained_model()

    def register_trained(self) -> tuple[Any, ...]:
        """Register the project's own trained checkpoint as an ordinary record."""
        return self.manager.register_trained()

    # -------------------------------------------------------------- settings

    def settings(self, model_id: str) -> Any:
        """A model's effective inference settings."""
        return self.manager.settings(model_id)

    def update_settings(self, model_id: str, **changes: Any) -> Any:
        """Change a model's inference settings, validating them first."""
        return self.manager.update_settings(model_id, **changes)

    def reset_settings(self, model_id: str) -> Any:
        """Return a model's settings to the project defaults."""
        return self.manager.reset_settings(model_id)

    # ------------------------------------------------------------- runtimes

    def runtime_lines(self) -> tuple[str, ...]:
        """A readable line per runtime, for a diagnostics panel."""
        return tuple(self.manager.runtime_report().splitlines())

    def runtime_status(self, runtime: str) -> Any:
        """One runtime's honest status."""
        return self.manager.runtime_status(runtime)

    # --------------------------------------------------------------- actions

    def load(self, model_id: str) -> OperationReport:
        """Load a model into its runtime."""
        return self._report(self.manager.load(model_id))

    def unload(self, model_id: str = "") -> OperationReport:
        """Release a loaded model (or every loaded model)."""
        return self._report(self.manager.unload(model_id))

    def test(self, model_id: str, *, prompt: str = "") -> OperationReport:
        """Ask a model a trivial question, to prove it really answers."""
        kwargs: dict[str, Any] = {}
        if prompt:
            kwargs["prompt"] = prompt
        return self._report(self.manager.test(model_id, **kwargs))

    def _report(self, result: OperationResult) -> OperationReport:
        events = self.manager.log_tail(5)
        return OperationReport(
            ok=result.ok,
            operation=result.operation,
            model_id=result.model_id,
            detail=result.detail,
            result=result,
            events=tuple(events),
        )

    def log_tail(self, limit: int = 30) -> tuple[ModelEvent, ...]:
        """The most recent operations, newest first."""
        return self.manager.log_tail(limit)

    def run_in_background(
        self,
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any, Exception | None], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        """Run ``work`` on a worker thread and hand the result to ``on_done``.

        The window's own queue-draining pump is what calls ``on_done``; this only
        keeps the long operation (a load, a test, a scan) off the UI thread. Tk is
        not thread-safe, so nothing here touches a widget.
        """

        def _run() -> None:
            if on_progress is not None:
                with best_effort("report progress"):
                    on_progress("working…")
            try:
                result = work()
            except Exception as error:  # handed back, never swallowed
                on_done(None, error)
                return
            on_done(result, None)

        thread = threading.Thread(target=_run, name="model-manager", daemon=True)
        thread.start()
        return thread


#: The errors a caller of the service may have to show, re-exported so a window
#: and the CLI catch one set rather than importing from four core modules.
MANAGER_ERRORS: tuple[type[BaseException], ...] = (
    ModelFileError,
    ModelNotFoundError,
    RegistryError,
    RuntimeUnavailable,
    SettingsError,
    ConfirmationRequired,
)


def model_manager_service(root: Path | str | None = None, **kwargs: Any) -> ModelManagerService:
    """A service over the project's own data folder (or ``root``)."""
    return ModelManagerService(root, **kwargs)


def human_size(size: int) -> str:
    """A readable file size, or ``—`` when it is not known."""
    if not size:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def detail_text(detail: ModelDetail) -> str:
    """The detail pane's body, honest about what is unknown.

    Tk-free on purpose, so the wording a window shows is testable without a
    display (and a CLI can print the same block).
    """
    row = detail.row
    lines = [
        f"{row.display_name}",
        f"  state:      {row.state_marker} — {STATE_BLURBS.get(row.state, '')}",
        f"  id:         {row.model_id}",
        f"  identity:   {detail.identity or '(unknown)'}",
        f"  runtime:    {row.runtime}",
        f"  format:     {row.format}",
        f"  path:       {row.path or 'no local file'}",
        f"  size:       {human_size(row.size_bytes)}",
        f"  source:     {row.source}"
        + (" (from the build's catalog)" if row.from_catalog else ""),
        f"  engine:     {detail.engine or '(none)'}",
    ]
    if detail.architecture:
        lines.append(f"  arch:       {detail.architecture}")
    if row.quantization:
        lines.append(f"  quant:      {row.quantization}")
    if detail.context_length:
        lines.append(f"  context:    {detail.context_length}")
    if detail.parameter_count:
        lines.append(f"  parameters: {detail.parameter_count:,}")
    if detail.license:
        lines.append(f"  license:    {detail.license}")
    if detail.integrity:
        verdict = detail.integrity
        if detail.integrity_detail:
            verdict += f" ({detail.integrity_detail})"
        lines.append(f"  integrity:  {verdict}")
    if detail.tags:
        lines.append(f"  tags:       {', '.join(detail.tags)}")
    if detail.notes:
        lines.append(f"  notes:      {detail.notes}")
    if detail.registered_at:
        lines.append(f"  registered: {detail.registered_at}")
    if detail.updated_at and detail.updated_at != detail.registered_at:
        lines.append(f"  updated:    {detail.updated_at}")
    if detail.download_command:
        lines.append("")
        lines.append("  To fetch these weights, run:")
        lines.append(f"    {detail.download_command}")
    elif not row.ready and not row.from_catalog:
        lines.append("")
        lines.append("  The weights are not here, and this build has no fetch command for them.")
    return "\n".join(lines)


__all__ = [
    "MANAGER_ERRORS",
    "SOURCE_LOCAL",
    "STATE_BLURBS",
    "STATE_MARKERS",
    "ManagerSnapshot",
    "ModelDetail",
    "ModelManagerService",
    "ModelRow",
    "OperationReport",
    "detail_text",
    "human_size",
    "model_manager_service",
]
