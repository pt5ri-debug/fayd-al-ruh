"""The Model Manager: one API over the registry, the runtimes, and the catalog.

This is the orchestrator the audit called for, and it is deliberately thin. It
owns no facts of its own -- every fact belongs to a module that already had it:

* **which models this build ships, and whether each is present** -- read from
  :mod:`core.language_models`, never copied. That module already merges every
  per-engine catalog, de-duplicates by weights identity, and rebuilds live.
* **which models someone registered** -- :class:`core.model_registry.ModelRegistry`.
* **what a model file actually is** -- :func:`core.model_files.probe_file`.
* **what runs it, and whether it can** -- :mod:`core.model_runtime`.
* **how it should be run** -- :func:`core.model_settings.effective_settings`.
* **what happened** -- :class:`core.model_log.ModelLog`.

What the manager adds is the *join*: one call that answers "show me every model
this program can run, what state each is in, and what the next step is", and one
place that turns a name into a runtime adapter and an operation into a recorded
result. It never loads a model at import time, never downloads anything, and never
writes to the knowledge graph or to a mouth's configuration.

Two honesty rules, carried over from the audit:

* a catalog model that is *not* downloaded is reported as such, with the real next
  step -- never silently omitted and never claimed to be ready;
* an operation that a runtime genuinely cannot perform raises
  :class:`core.model_runtime.UnsupportedCapability` rather than appearing to
  succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.model_files import (
    FORMAT_GGUF,
    FORMAT_UNKNOWN,
    DiscoveryReport,
    FileProbe,
    ModelFileError,
    discover_models,
    probe_file,
)
from core.model_log import (
    OP_DISCOVER,
    OP_FAILURE,
    OP_INSPECT,
    OP_LOAD,
    OP_REGISTER,
    OP_REMOVE,
    OP_TEST,
    OP_UNLOAD,
    OP_UPDATE,
    ModelEvent,
    ModelLog,
)
from core.model_registry import (
    SOURCE_CATALOG,
    SOURCE_LOCAL,
    SOURCE_TRAINED,
    ConfirmationRequired,
    ModelNotFoundError,
    ModelRecord,
    ModelRegistry,
    RegistryError,
    RegistryValidationError,
    model_id_for,
)
from core.model_runtime import (
    RUNTIME_NAMES,
    RUNTIME_UNKNOWN,
    InferenceResult,
    LoadResult,
    RuntimeAdapter,
    RuntimeStatus,
    RuntimeUnavailable,
    UnsupportedCapability,
    adapter_for,
    runtime_for_format,
)
from core.model_settings import ModelSettings, effective_settings
from core.paths import models_dir

#: What a model's overall state is, in one word, for a list a user reads.
STATE_READY = "ready"
STATE_NOT_DOWNLOADED = "not-downloaded"
STATE_MISSING = "missing"
STATE_DAMAGED = "damaged"
STATE_UNKNOWN = "unknown"

#: The answering tasks a caller can ask the manager to select a model for, each
#: mapped to the engine(s) whose models suit it best. The values are the catalog
#: ``engine`` names already carried on :attr:`core.model_registry.ModelRecord.engine`
#: (see :mod:`core.language_models`), so this table is a *view* of existing facts,
#: never a second registry. A task whose engine is not a catalog engine (``chat``,
#: and ``code`` -- the Code Studio's coding model is an Ollama model outside this
#: catalog) prefers nothing, so every ready model is offered in the build's order.
TASKS: dict[str, tuple[str, ...]] = {
    "chat": (),
    "code": (),
    "presentation": ("banner",),
    "petroleum": ("vision",),
    "companion": ("natasha",),
    "religion": ("hakeem",),
}

#: The order states are shown in: what works first, what needs attention after.
STATE_ORDER: tuple[str, ...] = (
    STATE_READY, STATE_NOT_DOWNLOADED, STATE_MISSING, STATE_DAMAGED, STATE_UNKNOWN,
)


@dataclass(frozen=True)
class ModelEntry:
    """One model, as a caller sees it: the record plus its live state.

    ``record`` is what the registry knows; ``state`` and ``status_line`` are what
    is true *right now*. Keeping them apart is what lets a listing say "registered
    but the file is gone" instead of quietly reporting a model that cannot run.
    """

    record: ModelRecord
    state: str = STATE_UNKNOWN
    status_line: str = ""
    #: True when this model came from the build's own catalog rather than the
    #: registry -- a catalog model is described, never stored.
    from_catalog: bool = False

    @property
    def model_id(self) -> str:
        return self.record.model_id

    @property
    def display_name(self) -> str:
        return self.record.display_name

    @property
    def runtime(self) -> str:
        return self.record.runtime

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    def describe(self) -> str:
        marker = {
            STATE_READY: "ready",
            STATE_NOT_DOWNLOADED: "not downloaded",
            STATE_MISSING: "file missing",
            STATE_DAMAGED: "damaged",
            STATE_UNKNOWN: "unknown",
        }.get(self.state, self.state)
        return f"{self.record.describe()} [{marker}]"


@dataclass(frozen=True)
class CompatibilityReport:
    """Whether a model can run here, and why not when it cannot.

    ``memory_estimate_bytes`` is an **estimate** and says so: it is derived from
    the file size (the weights) plus a context term when the architecture is
    known. It is never presented as a measurement, and ``None`` means the estimate
    could not be made rather than that it is zero.
    """

    model_id: str
    runtime: str
    can_run: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    runtime_status: RuntimeStatus | None = None
    memory_estimate_bytes: int | None = None
    context_length: int | None = None
    gpu_offload: bool | None = None

    def describe(self) -> str:
        head = "can run" if self.can_run else "cannot run"
        lines = [f"{head} ({self.runtime})"]
        lines.extend(f"  reason: {reason}" for reason in self.reasons)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        if self.memory_estimate_bytes is not None:
            lines.append(
                f"  estimated memory: ~{self.memory_estimate_bytes / 1e9:.2f} GB "
                "(an estimate from the file size, not a measurement)"
            )
        if self.gpu_offload is False:
            lines.append("  GPU offload: not available; the model runs on the CPU")
        elif self.gpu_offload is None:
            lines.append("  GPU offload: could not be determined; the CPU path is used")
        return "\n".join(lines)


@dataclass(frozen=True)
class OperationResult:
    """The outcome of an operation that touched a runtime. Never raised at a caller."""

    ok: bool
    operation: str
    model_id: str = ""
    detail: str = ""
    load: LoadResult | None = None
    inference: InferenceResult | None = None

    def describe(self) -> str:
        if self.load is not None:
            return self.load.describe()
        if self.inference is not None:
            return self.inference.describe()
        return self.detail or ("done" if self.ok else "failed")


@dataclass(frozen=True)
class ModelManagerStatus:
    """The whole subsystem's state, for a status line or a diagnostics report."""

    registry_path: Path
    registered: int = 0
    catalog: int = 0
    ready: int = 0
    runtimes: str = ""
    corrupt_records: tuple[str, ...] = ()
    log_events: int = 0

    def summary(self) -> str:
        return (
            f"{self.registered} registered, {self.catalog} in the build's catalog, "
            f"{self.ready} ready to run."
        )

    def describe(self) -> str:
        lines = [
            "Model Manager",
            f"  registry: {self.registry_path}",
            f"  {self.summary()}",
            f"  operation log: {self.log_events} event(s)",
            "",
            "Runtimes",
        ]
        lines.extend(f"  {line}" for line in self.runtimes.splitlines())
        if self.corrupt_records:
            lines.append("")
            lines.append("Unreadable registry records")
            lines.extend(f"  {key}" for key in self.corrupt_records)
        return "\n".join(lines)


@dataclass(frozen=True)
class DiagnosticsReport:
    """The dashboard's and the diagnostics surface's whole body, in one read.

    Every field is already-computed data from the module that owns it -- the
    machine's figures from :mod:`core.model_resources`, the runtimes' statuses,
    the loaded models, the failures, and the log tail. Nothing here is measured
    twice or guessed, and an empty tuple means "nothing to report", never "unknown".
    """

    status: ModelManagerStatus
    resources: Any = None
    runtime_lines: tuple[str, ...] = ()
    active: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    recent: tuple[str, ...] = ()

    def describe(self) -> str:
        """A readable block, honest about an empty section."""
        lines = [self.status.describe(), ""]
        if self.resources is not None:
            lines.append(self.resources.describe())
            lines.append("")
        lines.append("Active models (loaded right now)")
        if self.active:
            lines.extend(f"  {line}" for line in self.active)
        else:
            lines.append("  none — no model is loaded.")
        lines.append("")
        lines.append("Recent errors")
        if self.errors:
            lines.extend(f"  {line}" for line in self.errors)
        else:
            lines.append("  none recorded.")
        return "\n".join(lines)


class ModelManager:
    """The one place a caller asks about, and acts on, models.

    ``root`` is the writable data folder; a test passes its own. The two seams --
    ``adapter_factory`` and ``catalog`` -- let a test drive the whole manager with
    fake runtimes and a fake catalog, so nothing here needs a model, a GPU, or a
    network.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        registry: ModelRegistry | None = None,
        log: ModelLog | None = None,
        adapter_factory: Any = None,
        catalog: Any = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.registry = registry if registry is not None else ModelRegistry(self.root)
        self.log = log if log is not None else ModelLog(self.root)
        self._adapter_factory = adapter_factory
        self._catalog = catalog
        # One adapter per runtime, kept so a load followed by a test uses the same
        # loaded model. Never populated at import time.
        self._adapters: dict[str, RuntimeAdapter] = {}

    # ------------------------------------------------------------- catalog

    def _catalog_models(self) -> tuple[Any, ...]:
        """The build's own catalog, or ``()`` when it cannot be read.

        Read, never copied: :mod:`core.language_models` is the one source of truth
        for which models this build ships, and a failure to read it must not stop
        the registry half of the manager from working.
        """
        if self._catalog is not None:
            try:
                return tuple(self._catalog)
            except TypeError:
                # A caller may inject a *callable* catalog; call it, and treat a
                # failure the same as an unreadable one.
                try:
                    return tuple(self._catalog())
                except Exception:
                    return ()
            except Exception:
                return ()
        try:
            from core import language_models

            return tuple(_unwrap_catalog(item) for item in language_models.all_language_models())
        except Exception:
            return ()

    def catalog_entries(self) -> tuple[ModelEntry, ...]:
        """Every model the build knows about, with its live state.

        A catalog model is *described*, never written to the registry -- the
        catalog is already the record of it, and duplicating it here would create
        the second source of truth the audit warned against.
        """
        entries: list[ModelEntry] = []
        for model in self._catalog_models():
            record = _record_from_catalog(model)
            if record is None:
                continue
            entries.append(
                ModelEntry(
                    record=record,
                    state=_state_for_record(record, available=_catalog_available(model)),
                    status_line=_clean(getattr(model, "detail", "")),
                    from_catalog=True,
                )
            )
        return tuple(entries)

    # ------------------------------------------------------------ listing

    def list_models(self, *, include_catalog: bool = True) -> tuple[ModelEntry, ...]:
        """Every model, registry first then catalog, each with its live state.

        A registry entry and a catalog entry for the *same* weights are one line:
        the registered one wins, because it carries what the user chose to say
        about it. This is what keeps the listing honest without duplicating.
        """
        entries = [
            self._entry_for(record) for record in self.registry.list_records()
        ]
        if not include_catalog:
            return tuple(entries)
        known = {entry.record.identity for entry in entries}
        for entry in self.catalog_entries():
            if entry.record.identity in known:
                continue
            entries.append(entry)
            known.add(entry.record.identity)
        entries.sort(key=lambda entry: (STATE_ORDER.index(entry.state), entry.display_name.casefold()))
        return tuple(entries)

    def ready_models(self) -> tuple[ModelEntry, ...]:
        """Only the models that can run right now."""
        return tuple(entry for entry in self.list_models() if entry.ready)

    def models_for_task(self, task: str) -> tuple[ModelEntry, ...]:
        """The models suited to ``task``, best first, from what can run here.

        A *task* is one of :data:`TASKS` (``chat``, ``code``, ``presentation``,
        ``petroleum``, ``companion``, ``religion``). The selection is a **filter
        over the models already known**, never a new fact: it prefers the engine
        the task names, then every other ready model, so a task with a dedicated
        engine still falls back to whatever else can answer rather than offering
        nothing. An unknown task yields nothing rather than raising -- a caller
        that asks about a task this build does not know simply gets no list.
        """
        wanted = _clean(task)
        if wanted not in TASKS:
            return ()
        preferred = TASKS[wanted]
        ready = [entry for entry in self.list_models() if entry.ready]
        preferred_entries = [entry for entry in ready if entry.record.engine in preferred]
        others = [entry for entry in ready if entry.record.engine not in preferred]
        return tuple(preferred_entries + others)

    def select_for_task(self, task: str) -> ModelEntry | None:
        """The single best ready model for ``task``, or ``None`` when none can run.

        ``None`` is the honest answer -- it means "nothing here can answer this
        task", not "use a made-up default".
        """
        models = self.models_for_task(task)
        return models[0] if models else None

    def get(self, model_id: str) -> ModelEntry:
        """One entry by id, from the registry *or* the shipped catalog.

        Raises :class:`ModelNotFoundError` for an id neither knows, rather than
        answering with a made-up entry.
        """
        key = _clean(model_id)
        if key:
            try:
                return self._entry_for(self.registry.get(key))
            except ModelNotFoundError:
                pass
        for entry in self.catalog_entries():
            if entry.model_id == key:
                return entry
        raise ModelNotFoundError(f"No model is known as {model_id!r}.")

    def _entry_for(self, record: ModelRecord) -> ModelEntry:
        """A registry record as an entry, with its state read live."""
        state, line = self._state_of(record)
        return ModelEntry(record=record, state=state, status_line=line)

    def _state_of(self, record: ModelRecord) -> tuple[str, str]:
        """The live state of a registered record, and a readable reason."""
        if record.path is None:
            if record.source == SOURCE_TRAINED:
                return STATE_READY, "The trained checkpoint needs no file to register."
            return STATE_UNKNOWN, "This record has no local file."
        try:
            exists = record.path.is_file()
        except OSError:
            exists = False
        if not exists:
            return STATE_MISSING, f"The file is gone: {record.path}"
        if record.integrity and not record.verified:
            return STATE_DAMAGED, record.integrity_detail or "The file failed its integrity check."
        return STATE_READY, ""

    # ----------------------------------------------------------- registry

    def register(
        self,
        path: Path | str,
        *,
        display_name: str = "",
        source: str = SOURCE_LOCAL,
        tags: object = None,
        notes: str = "",
        engine: str = "",
        identity: str = "",
        digest: bool = False,
        replace_existing: bool = False,
        **kwargs: Any,
    ) -> ModelEntry:
        """Register a local model file, recording the operation.

        A refusal is recorded as a failed operation *and* raised, so a caller sees
        the reason immediately and the log keeps it for later.
        """
        try:
            record = self.registry.register(
                path,
                display_name=display_name,
                source=source,
                tags=tags,
                notes=notes,
                engine=engine,
                identity=identity,
                digest=digest,
                replace_existing=replace_existing,
                **kwargs,
            )
        except (ModelFileError, RegistryError) as error:
            self.log.append(
                OP_FAILURE, display_name=_clean(str(path)), ok=False,
                message=f"register: {error}",
            )
            raise
        self.log.append(
            OP_REGISTER, model_id=record.model_id, display_name=record.display_name,
            runtime=record.runtime, ok=True,
            message=f"Registered {record.path}",
        )
        return self._entry_for(record)

    def update(self, model_id: str, **changes: Any) -> ModelEntry:
        """Update a record's editable fields, recording the operation."""
        try:
            record = self.registry.update(model_id, **changes)
        except RegistryError as error:
            self.log.append(
                OP_FAILURE, model_id=model_id, ok=False, message=f"update: {error}",
            )
            raise
        self.log.append(
            OP_UPDATE, model_id=record.model_id, display_name=record.display_name,
            ok=True, message="Metadata updated.",
        )
        return self._entry_for(record)

    def remove(self, model_id: str, *, confirm: bool = False) -> ModelRecord:
        """Forget a registry entry. **Never** deletes the model file.

        The confirmation is required at this level too, so no caller can reach the
        destructive path by accident, and the operation is recorded either way.
        """
        try:
            record = self.registry.remove(model_id, confirm=confirm)
        except ConfirmationRequired:
            self.log.append(
                OP_FAILURE, model_id=model_id, ok=False,
                message="remove: refused without an explicit confirmation.",
            )
            raise
        except RegistryError as error:
            self.log.append(
                OP_FAILURE, model_id=model_id, ok=False, message=f"remove: {error}",
            )
            raise
        self.log.append(
            OP_REMOVE, model_id=record.model_id, display_name=record.display_name,
            ok=True, message=f"Forgot the registry entry. The file at {record.path} was not touched.",
        )
        return record

    def import_custom_models(self, *, replace_existing: bool = False) -> tuple[ModelEntry, ...]:
        """Bring the mouth picker's own shortlist in as ordinary records."""
        records = self.registry.import_custom_models(replace_existing=replace_existing)
        if records:
            self.log.append(
                OP_REGISTER, ok=True,
                message=f"Imported {len(records)} model(s) from the per-mouth shortlist.",
            )
        return tuple(self._entry_for(record) for record in records)

    # ------------------------------------------------------------ inspect

    def inspect(self, model_id: str, *, digest: bool = False) -> FileProbe:
        """Re-read a registered model's file and return what it actually is.

        Reads the file again rather than trusting the stored record, so a file
        replaced in place is reported as it is now. A model with no local file
        (a remote one, or the trained checkpoint registered without a path) is
        reported as such rather than raising.
        """
        record = self.registry.get(model_id)
        if record.path is None:
            probe = FileProbe(
                path=Path(record.identity), exists=False,
                format=record.model_format,
                detail="This record has no local file to inspect.",
            )
        else:
            probe = probe_file(record.path, digest=digest)
        self.log.append(
            OP_INSPECT, model_id=record.model_id, display_name=record.display_name,
            runtime=record.runtime, ok=probe.usable,
            message=probe.detail or f"{probe.format}, {probe.size_bytes} bytes",
        )
        return probe

    def discover(
        self, roots: object = None, *, limit: int = 200, digest: bool = False
    ) -> DiscoveryReport:
        """Find supported model files under ``roots``.

        With no ``roots`` the search covers the project's own model folders --
        never the whole filesystem, and never a network location.
        """
        targets = roots if roots is not None else self.default_scan_roots()
        report = discover_models(targets, limit=limit, digest=digest)
        self.log.append(
            OP_DISCOVER, ok=True,
            message=f"Scanned {len(report.roots)} folder(s); {report.summary()}",
        )
        return report

    def default_scan_roots(self) -> tuple[Path, ...]:
        """The folders a bare discovery should search.

        The project's own model folders, the writable checkpoints folder, and the
        **bundled** checkpoints folder -- the last so the shipped trained model is
        discoverable without the user having retrained anything. ``default_checkpoint()``
        is deliberately *not* used here: it resolves to whichever copy exists, so
        it would silently drop the bundled folder on a machine that has retrained.

        A user's home directory is never searched: a scan is a filesystem read, and
        reading someone's whole home folder unasked is not something a model
        manager should do.
        """
        from core.paths import bundled_checkpoints_dir, checkpoints_dir

        candidates = [
            models_dir(),
            checkpoints_dir(),
            bundled_checkpoints_dir(),
        ]
        return tuple(dict.fromkeys(root for root in candidates))

    # -------------------------------------------------------- settings

    def settings(self, model_id: str) -> ModelSettings:
        """The effective settings for a model, in the documented order."""
        record = self.get(model_id).record
        return effective_settings(record.identity, root=self.root)

    def update_settings(self, model_id: str, **changes: Any) -> ModelSettings:
        """Validate and store new settings for a model."""
        from core.model_settings import update_settings

        record = self.get(model_id).record
        settings = update_settings(record.identity, root=self.root, **changes)
        self.log.append(
            OP_UPDATE, model_id=record.model_id, display_name=record.display_name,
            ok=True, message=f"Settings: {settings.describe()}",
        )
        return settings

    def reset_settings(self, model_id: str) -> ModelSettings:
        """Return a model's settings to the project defaults."""
        from core.model_settings import reset_settings

        record = self.get(model_id).record
        settings = reset_settings(record.identity, root=self.root)
        self.log.append(
            OP_UPDATE, model_id=record.model_id, display_name=record.display_name,
            ok=True, message="Settings reset to the project defaults.",
        )
        return settings

    # -------------------------------------------------------- runtimes

    def adapter(self, runtime: str) -> RuntimeAdapter:
        """The adapter for ``runtime``, built once and kept.

        Kept so a load followed by a test uses the *same* loaded model: a fresh
        adapter per call would silently load the weights twice and report a test
        that proved nothing.
        """
        if runtime in self._adapters:
            return self._adapters[runtime]
        if self._adapter_factory is not None:
            adapter = self._adapter_factory(runtime)
        else:
            adapter = adapter_for(runtime)
        self._adapters[runtime] = adapter
        return adapter

    def runtime_status(self, runtime: str) -> RuntimeStatus:
        """An honest status for one runtime."""
        try:
            return self.adapter(runtime).status()
        except RuntimeUnavailable as error:
            return RuntimeStatus(runtime=runtime, available=False, detail=str(error))

    def runtime_names(self) -> tuple[str, ...]:
        """Every runtime name this manager can report on, in a stable order."""
        return RUNTIME_NAMES

    def runtime_report(self) -> str:
        """A readable report of every runtime, for a diagnostics surface.

        Each runtime is asked for its *own* status, so the report reflects what
        this machine really has rather than a guess made from the runtime's name.
        """
        lines: list[str] = []
        for name in self.runtime_names():
            status = self.runtime_status(name)
            lines.append(status.summary())
            if not status.available and status.hint:
                lines.append(f"    next step: {status.hint}")
        return "\n".join(lines)

    def compatibility(self, model_id: str) -> CompatibilityReport:
        """Whether a model can run here, with the reasons and the honest estimate."""
        record = self.registry.get(model_id)
        runtime = record.runtime or RUNTIME_UNKNOWN
        reasons: list[str] = []
        warnings: list[str] = []

        status: RuntimeStatus | None = None
        try:
            status = self.adapter(runtime).status()
        except RuntimeUnavailable as error:
            reasons.append(str(error))
        else:
            if not status.available:
                reasons.append(status.detail or f"The {runtime} runtime is not available.")
            elif status.hint:
                warnings.append(status.hint)

        if record.path is None:
            if record.source != SOURCE_TRAINED and record.source != "remote":
                reasons.append("This record has no local file, so there is nothing to run.")
        else:
            probe = probe_file(record.path)
            if not probe.exists:
                reasons.append(f"The file is gone: {record.path}")
            elif probe.format == FORMAT_UNKNOWN:
                reasons.append(probe.detail or "The file is not a recognised model format.")
            elif not probe.usable:
                reasons.append(probe.detail or "The file failed its integrity check.")
            else:
                adapter = None
                try:
                    adapter = self.adapter(runtime)
                except RuntimeUnavailable:
                    adapter = None
                if adapter is not None:
                    ok, reason = adapter.validate(probe)
                    if not ok:
                        reasons.append(reason)

        gpu_offload = status.gpu_offload if status is not None else None
        if gpu_offload is None:
            warnings.append(
                "Whether the installed runtime supports GPU offload could not be "
                "determined, so the model runs on the CPU."
            )
        elif not gpu_offload:
            warnings.append("The installed runtime reports no GPU offload; the CPU is used.")

        estimate = _estimate_memory(record, gpu_layers=0)
        return CompatibilityReport(
            model_id=record.model_id,
            runtime=runtime,
            can_run=not reasons,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            runtime_status=status,
            memory_estimate_bytes=estimate,
            context_length=record.context_length,
            gpu_offload=gpu_offload,
        )

    # ------------------------------------------------------- operations

    def load(self, model_id: str, *, settings: ModelSettings | None = None) -> OperationResult:
        """Load a model into its runtime, recording the outcome.

        A model that is already loaded is **not** silently replaced: loading a
        different model into the same runtime unloads the first explicitly, and
        the log says so. That is the audit's "do not silently replace an active
        model" rule made concrete.
        """
        record = self.registry.get(model_id)
        if record.path is None:
            return self._fail(OP_LOAD, record, "This model has no local file to load.")
        try:
            adapter = self.adapter(record.runtime)
        except RuntimeUnavailable as error:
            return self._fail(OP_LOAD, record, str(error))
        capabilities = adapter.capabilities()
        if not capabilities.can_load:
            return self._fail(
                OP_LOAD, record,
                f"The {record.runtime} runtime cannot load a model here.",
            )
        previous = getattr(adapter, "loaded_path", None)
        if callable(previous):
            current = previous()
            if current is not None and Path(current) != record.path:
                self.log.append(
                    OP_UNLOAD, model_id=record.model_id, display_name=record.display_name,
                    runtime=record.runtime, ok=True,
                    message=f"Unloaded {current} before loading {record.path}.",
                )
                adapter.unload()
        resolved = settings or effective_settings(record.identity, root=self.root)
        try:
            result = adapter.load(record.path, settings=resolved)
        except UnsupportedCapability as error:
            return self._fail(OP_LOAD, record, str(error))
        except Exception as error:  # a runtime failure must not crash the caller
            return self._fail(OP_LOAD, record, f"{type(error).__name__}: {error}")
        self.log.append(
            OP_LOAD, model_id=record.model_id, display_name=record.display_name,
            runtime=record.runtime, ok=result.ok,
            message=result.detail or f"Loaded in {result.seconds:.2f}s",
        )
        return OperationResult(
            ok=result.ok, operation=OP_LOAD, model_id=record.model_id,
            detail=result.detail, load=result,
        )

    def unload(self, model_id: str = "") -> OperationResult:
        """Unload a model, or every loaded model when no id is given."""
        targets = [self.registry.get(model_id)] if model_id else self.registry.list_records()
        last = OperationResult(ok=True, operation=OP_UNLOAD, detail="Nothing was loaded.")
        for record in targets:
            try:
                adapter = self.adapter(record.runtime)
            except RuntimeUnavailable:
                continue
            if not adapter.capabilities().can_unload:
                continue
            try:
                result = adapter.unload()
            except UnsupportedCapability:
                continue
            except Exception as error:
                self.log.append(
                    OP_UNLOAD, model_id=record.model_id, display_name=record.display_name,
                    runtime=record.runtime, ok=False, message=str(error),
                )
                continue
            self.log.append(
                OP_UNLOAD, model_id=record.model_id, display_name=record.display_name,
                runtime=record.runtime, ok=result.ok,
                message=result.detail or "Unloaded.",
            )
            last = OperationResult(
                ok=result.ok, operation=OP_UNLOAD, model_id=record.model_id,
                detail=result.detail, load=result,
            )
        return last

    def test(
        self,
        model_id: str,
        *,
        prompt: str = "Reply with the single word: ready.",
        settings: ModelSettings | None = None,
        load_if_needed: bool = True,
    ) -> OperationResult:
        """Ask a model a trivial question, so a user can prove it really answers.

        The prompt is deliberately tiny: a test that costs a minute is a test
        nobody runs. ``load_if_needed`` loads the model first when nothing is
        loaded, which is what a window's *Test* button needs.
        """
        record = self.registry.get(model_id)
        try:
            adapter = self.adapter(record.runtime)
        except RuntimeUnavailable as error:
            return self._fail(OP_TEST, record, str(error))
        if not adapter.capabilities().can_infer:
            return self._fail(
                OP_TEST, record, f"The {record.runtime} runtime cannot run inference here."
            )
        loaded = getattr(adapter, "loaded", None)
        if load_if_needed and callable(loaded) and not loaded():
            loaded_result = self.load(model_id, settings=settings)
            if not loaded_result.ok:
                return self._fail(OP_TEST, record, loaded_result.detail)
        resolved = settings or effective_settings(record.identity, root=self.root)
        try:
            result = adapter.infer(prompt, settings=resolved)
        except UnsupportedCapability as error:
            return self._fail(OP_TEST, record, str(error))
        except Exception as error:
            return self._fail(OP_TEST, record, f"{type(error).__name__}: {error}")
        self.log.append(
            OP_TEST, model_id=record.model_id, display_name=record.display_name,
            runtime=record.runtime, ok=result.ok,
            message=result.detail or f"Answered in {result.seconds:.2f}s",
        )
        return OperationResult(
            ok=result.ok, operation=OP_TEST, model_id=record.model_id,
            detail=result.detail, inference=result,
        )

    def download_command(self, model_id: str) -> str:
        """The command that fetches a model's weights, or ``""`` when none exists.

        Deliberately a **string**, not an action: fetching weights is a governed
        external operation the existing engines already implement behind their own
        permission and human-approval gates, and this subsystem must not open a
        second path to the network. It reports what to run; the user runs it.
        """
        record = self.get(model_id).record
        for model in self._catalog_models():
            if _identity_of(model) == record.identity:
                return str(getattr(model, "download_command", "") or "")
        return ""

    def _fail(self, operation: str, record: ModelRecord, detail: str) -> OperationResult:
        """Record a failure and return it, so no caller has to catch."""
        self.log.append(
            OP_FAILURE, model_id=record.model_id, display_name=record.display_name,
            runtime=record.runtime, ok=False, message=f"{operation}: {detail}",
        )
        return OperationResult(
            ok=False, operation=operation, model_id=record.model_id, detail=detail,
        )

    # ----------------------------------------------------------- status

    def active_models(self) -> tuple[ModelEntry, ...]:
        """The models currently **loaded** in their runtime, in listing order.

        "Loaded" is the runtime's own answer (its ``loaded()``), never a guess from
        the registry: a model that is registered and ready is not thereby active.
        A runtime that cannot be built (its extra is missing) is skipped, so asking
        this never raises on a machine with no runtime at all.
        """
        active: list[ModelEntry] = []
        for entry in self.list_models():
            record = entry.record
            if record.path is None:
                continue
            try:
                adapter = self.adapter(record.runtime)
            except RuntimeUnavailable:
                continue
            probe = getattr(adapter, "loaded", None)
            if not callable(probe):
                continue
            try:
                if probe():
                    active.append(entry)
            except Exception:  # a broken runtime must not blank the dashboard
                continue
        return tuple(active)

    def diagnostics(self) -> DiagnosticsReport:
        """Everything a diagnostics surface shows, in one honest read.

        Bundles the machine's own figures (through
        :mod:`core.model_resources`), the runtimes' own statuses, the models
        currently loaded, the recent failures, and the operation log's tail --
        each read from the module that owns it, so nothing is recomputed or
        guessed.
        """
        from core.model_resources import system_resources

        errors = tuple(
            event for event in self.log.entries() if not event.ok
        )[-20:]
        return DiagnosticsReport(
            status=self.status(),
            resources=system_resources(),
            runtime_lines=tuple(self.runtime_report().splitlines()),
            active=tuple(entry.describe() for entry in self.active_models()),
            errors=tuple(event.describe() for event in errors),
            recent=tuple(event.describe() for event in self.log_tail(20)),
        )

    def status(self) -> ModelManagerStatus:
        """The whole subsystem's state, for a status line or a diagnostics report."""
        records = self.registry.list_records()
        catalog = self.catalog_entries()
        entries = self.list_models()
        return ModelManagerStatus(
            registry_path=self.registry.root,
            registered=len(records),
            catalog=len(catalog),
            ready=sum(1 for entry in entries if entry.ready),
            runtimes=self.runtime_report(),
            corrupt_records=self.registry.corrupt_keys(),
            log_events=len(self.log.entries()),
        )

    def log_tail(self, limit: int = 30) -> tuple[ModelEvent, ...]:
        """The most recent operations, newest first."""
        return self.log.tail(limit)


def _clean(text: object) -> str:
    return str(text or "").strip()[:512]


def _unwrap_catalog(item: Any) -> Any:
    """A catalog entry, whether the catalog hands back a model or a ``(mouth, model)``.

    :func:`core.language_models.all_language_models` returns ``(mouth, model)``
    pairs, while a caller injecting its own catalog naturally passes plain models.
    Accepting both keeps one code path without guessing at the model's own shape.
    """
    if isinstance(item, tuple) and len(item) == 2 and not hasattr(item, "key"):
        return item[1]
    return item


def _identity_of(model: Any) -> str:
    """The stable identity of a catalog entry, matching what a record stores.

    ``identity`` when the catalog carries one; otherwise the ``repo/filename``
    pair it already uses for de-duplication, or the path.
    """
    identity = getattr(model, "identity", "")
    if isinstance(identity, str) and identity:
        return identity
    repo = getattr(model, "repo", "") or ""
    filename = getattr(model, "filename", "") or ""
    if repo and filename:
        return f"{repo}/{filename}"
    path = getattr(model, "path", None)
    return str(path) if path else ""


def _catalog_available(model: Any) -> bool:
    """Whether a catalog entry's weights are present, read defensively.

    A catalog object that cannot answer is reported as *not* available, which is
    the safe direction: the next step is stated rather than a false "ready".
    """
    value = getattr(model, "available", None)
    if callable(value):
        try:
            return bool(value())
        except Exception:
            return False
    return bool(value)


def _record_from_catalog(model: Any) -> ModelRecord | None:
    """A read-only record view of one catalog entry.

    Built on the fly and never stored: the catalog is already the record, so
    writing one here would create a second source of truth.
    """
    identity = _identity_of(model)
    if not identity:
        return None
    path_value = getattr(model, "path", None)
    path = Path(path_value) if path_value else None
    probe: FileProbe | None = None
    if path is not None:
        try:
            probe = probe_file(path)
        except Exception:
            probe = None
    model_format = probe.format if probe is not None else (
        FORMAT_GGUF if str(getattr(model, "filename", "")).endswith(".gguf") else FORMAT_UNKNOWN
    )
    engine = str(getattr(model, "engine", "") or "")
    runtime = runtime_for_format(model_format)
    source = SOURCE_TRAINED if engine == "model" else SOURCE_CATALOG
    return ModelRecord(
        model_id=model_id_for(identity),
        identity=identity,
        display_name=_clean(getattr(model, "title", "")) or identity,
        model_format=model_format,
        runtime=runtime,
        path=path,
        source=source,
        size_bytes=probe.size_bytes if probe is not None else 0,
        quantization=probe.gguf.quantization if probe is not None and probe.gguf else "",
        architecture=probe.gguf.architecture if probe is not None and probe.gguf else "",
        context_length=probe.gguf.context_length if probe is not None and probe.gguf else None,
        parameter_count=probe.gguf.parameter_count if probe is not None and probe.gguf else None,
        license=probe.gguf.license if probe is not None and probe.gguf else "",
        integrity=probe.integrity if probe is not None else "",
        integrity_detail=probe.detail if probe is not None else "",
        notes=_clean(getattr(model, "summary", "")),
        engine=engine,
    )


def _state_for_record(record: ModelRecord, *, available: bool) -> str:
    """The state word for a catalog record."""
    if available:
        return STATE_READY
    if record.path is not None and record.path.is_file():
        return STATE_DAMAGED
    return STATE_NOT_DOWNLOADED


def _estimate_memory(record: ModelRecord, *, gpu_layers: int = 0) -> int | None:
    """An **estimate** of the memory a model needs, or ``None`` when unknowable.

    The weights are the file size. The KV cache is estimated from the context
    length and the architecture when both are known -- deliberately a band, not a
    number presented as a measurement, and ``None`` when even the file size is
    unknown. ``gpu_layers`` is accepted so a caller can note that an offloaded
    layer would move part of this to VRAM, which this function does **not** model.
    """
    if not record.size_bytes:
        return None
    estimate = record.size_bytes
    context = record.context_length
    if context and record.architecture:
        # A deliberately rough KV term: ~2 bytes per element, ~2 * layers * d_model
        # per token, inferred from the parameter count when it is known. Reported
        # as part of an estimate, never as a measurement.
        params = record.parameter_count or 0
        if params > 0:
            estimate += int(context * 2 * (params ** 0.5) / 1024)
    return estimate


def model_manager(root: Path | str | None = None, **kwargs: Any) -> ModelManager:
    """The manager at ``root`` (the default data folder when omitted)."""
    return ModelManager(root, **kwargs)


__all__ = [
    "STATE_DAMAGED",
    "STATE_MISSING",
    "STATE_NOT_DOWNLOADED",
    "STATE_ORDER",
    "STATE_READY",
    "STATE_UNKNOWN",
    "CompatibilityReport",
    "ModelEntry",
    "ModelManager",
    "ModelManagerStatus",
    "ModelNotFoundError",
    "OperationResult",
    "RegistryValidationError",
    "model_manager",
]
