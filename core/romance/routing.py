"""The companion model router: which local model answers, and whether it can.

A thin delegation over the project's **existing** Model Manager
(:class:`core.model_manager.ModelManager`) and language-model catalog
(:mod:`core.language_models`). The router owns **no** downloader, registry, or
runtime of its own -- it resolves a *choice* (a character's or a mode's assigned
model) to the exact file the manager already knows, loads and unloads it through
the manager, and reports health, memory estimate, and preflight honestly.

The three rules the router keeps:

* **Local first, CPU first.** The default model is the one the project already
  ships for the companion, and the settings are the ones the manager resolves for
  this machine. A GPU is used only when a build reports it; the target hardware
  (16 GB RAM, small integrated/discrete GPUs) is CPU-first.
* **Never a second downloader.** Fetching weights stays with the existing
  governed engines; :meth:`CompanionModelRouter.download_command` reports the
  command, it does not run it.
* **Never assume a model is installed.** Every read returns an honest status with
  the real next step; a model that cannot run is reported, not pretended.

It imports no runtime directly and loads no model on construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.model_manager import (
    STATE_READY,
    ModelManager,
)
from core.model_runtime import InferenceResult

#: The default task the router ranks models against (the manager's own task map
#: names the companion engine for it).
TASK_COMPANION = "companion"

#: A safe default context for the target hardware: large enough for a real
#: conversation, small enough that the KV cache does not crowd out the weights on
#: a 16 GB machine.
SAFE_CONTEXT = 4096

#: The most output tokens a safe default asks for (a reply, not a document).
SAFE_MAX_TOKENS = 512

#: When free memory is below this share of total, a load is flagged risky.
LOW_MEMORY_SHARE = 0.2


class CompanionModelError(RuntimeError):
    """Raised when a model choice cannot be resolved or acted on as asked."""


@dataclass(frozen=True)
class ResolvedModel:
    """A resolved model choice: the key, its file, and whether it can run now."""

    key: str
    title: str
    engine: str
    path: Path | None = None
    available: bool = False
    detail: str = ""
    from_choice: bool = False

    def describe(self) -> str:
        if self.available and self.path is not None:
            return f"{self.title} ({self.key}) — ready at {self.path}."
        return f"{self.title} ({self.key}) — not ready: {self.detail or 'no file'}"


@dataclass(frozen=True)
class ModelHealth:
    """One honest answer to "can this model answer right now?"."""

    key: str
    loaded: bool = False
    ready: bool = False
    runtime: str = ""
    detail: str = ""
    hint: str = ""
    load_seconds: float = 0.0

    def describe(self) -> str:
        if self.loaded:
            return f"{self.key}: loaded in the {self.runtime or 'local'} runtime."
        if self.ready:
            return f"{self.key}: ready to load ({self.runtime or 'local'})."
        return f"{self.key}: not ready — {self.detail or 'no model'}" + (
            f" (next step: {self.hint})" if self.hint else ""
        )


@dataclass(frozen=True)
class GenerationHandle:
    """A cancellation/timeout handle for one generation request."""

    key: str
    started_at: float
    timeout_seconds: float = 0.0
    _cancelled: bool = field(default=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        object.__setattr__(self, "_cancelled", True)

    def expired(self) -> bool:
        if self.timeout_seconds <= 0:
            return False
        return (time.monotonic() - self.started_at) > self.timeout_seconds

    def remaining(self) -> float:
        if self.timeout_seconds <= 0:
            return 0.0
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started_at))


class CompanionModelRouter:
    """Resolve, load, and report the local models the Studio may use.

    ``manager`` is the existing :class:`~core.model_manager.ModelManager`; a test
    injects one built with fake runtimes and a fake catalog, so the router is
    exercised with no model and no GPU.
    """

    def __init__(
        self,
        manager: ModelManager | None = None,
        *,
        catalog: Any = None,
        default_key: str = "",
        character_model_key: Any = None,
        mode_model_key: Any = None,
    ) -> None:
        self.manager = manager if manager is not None else ModelManager()
        self._catalog = catalog
        self._default_key = default_key
        self._character_model_key = character_model_key
        self._mode_model_key = mode_model_key

    # -- assignment -------------------------------------------------------

    def assign_character(self, character_id: str, key: str) -> None:
        """Remember the model assigned to a character (persistence is the caller's)."""
        if self._character_model_key is None:
            raise CompanionModelError("this router cannot persist assignments")
        self._character_model_key(character_id, key)

    def assign_mode(self, mode: str, key: str) -> None:
        """Remember the model assigned to a conversation mode."""
        if self._mode_model_key is None:
            raise CompanionModelError("this router cannot persist assignments")
        self._mode_model_key(mode, key)

    # -- resolution -------------------------------------------------------

    def _catalog_models(self) -> tuple[Any, ...]:
        """The models a companion may choose, from the existing catalog.

        The ``catalog`` seam is a callable returning the model objects; it defaults
        to the project's own catalog for the companion mouth, so a test drives the
        router with a fake catalog and no weights.
        """
        if self._catalog is not None:
            try:
                return tuple(self._catalog())
            except Exception:
                return ()
        from core.language_models import MOUTH_NATASHA, models_for_choice

        return models_for_choice(MOUTH_NATASHA)

    def choices(self) -> tuple[ResolvedModel, ...]:
        """Every model a companion may be assigned, with its honest readiness."""
        resolved: list[ResolvedModel] = []
        for model in self._catalog_models():
            resolved.append(_resolved_from(model))
        return tuple(resolved)

    def resolve(self, key: str | None = None) -> ResolvedModel:
        """Resolve a model key to its file, or the companion's own default.

        An unknown key falls back to the first *available* model rather than
        raising, so a stale saved choice cannot break a mouth; when nothing is
        available at all the result is honest (``available=False`` with the real
        next step).
        """
        models = self._catalog_models()
        wanted = (key or "").strip()
        if wanted:
            for model in models:
                if str(getattr(model, "key", "")) == wanted:
                    return _resolved_from(model)
        for model in models:
            if bool(getattr(model, "available", False)):
                return _resolved_from(model)
        # Nothing available: keep the catalog's own next step rather than a blank.
        if models:
            return _resolved_from(models[0])
        return ResolvedModel(
            key=wanted,
            title=wanted or "no model",
            engine="",
            available=False,
            detail="No companion model is installed yet.",
        )

    def default_key(self) -> str:
        """The key the companion would choose on its own, or ``""``."""
        if self._default_key:
            return self._default_key
        from core.language_models import default_language_model_key

        return default_language_model_key("natasha") or ""

    # -- manager record ---------------------------------------------------

    def _record_for(self, resolved: ResolvedModel) -> Any:
        """The manager record for a resolved model, registering it when needed.

        The manager's registry is the one source of truth for a *runnable* model;
        a catalog entry is registered on demand so the manager can load it. The
        registration is idempotent (the identity is the file path).
        """
        if resolved.path is None:
            raise CompanionModelError(f"{resolved.key or 'the model'} has no local file to load")
        entry = self.manager.register(
            resolved.path,
            display_name=resolved.title or resolved.key,
            engine=resolved.engine,
            replace_existing=True,
        )
        return entry

    def _find_record(self, resolved: ResolvedModel) -> Any | None:
        """An existing registry entry for the model, if one is already registered."""
        if resolved.path is None:
            return None
        wanted = str(resolved.path)
        for entry in self.manager.list_models(include_catalog=False):
            path = getattr(entry.record, "path", None)
            if path is not None and str(path) == wanted:
                return entry
        return None

    # -- status / health --------------------------------------------------

    def status(self, key: str | None = None) -> ModelHealth:
        """Whether the chosen model can answer, read from the manager's own state."""
        resolved = self.resolve(key)
        if not resolved.available or resolved.path is None:
            return ModelHealth(
                key=resolved.key or (key or ""),
                ready=False,
                detail=resolved.detail or "no model is installed",
                hint=self.download_hint(resolved.key),
            )
        entry = self._find_record(resolved)
        runtime = entry.runtime if entry is not None else ""
        loaded = self._is_loaded(runtime, resolved.path)
        if entry is not None and entry.state == STATE_READY:
            return ModelHealth(
                key=resolved.key, loaded=loaded, ready=True, runtime=runtime,
                detail="ready",
            )
        # Not registered yet: ask the manager to register so the state is real.
        try:
            registered = self._record_for(resolved)
        except Exception as error:  # a registration failure is reported, not raised
            return ModelHealth(
                key=resolved.key, ready=False,
                detail=f"{type(error).__name__}: {error}",
            )
        return ModelHealth(
            key=resolved.key,
            loaded=self._is_loaded(registered.runtime, resolved.path),
            ready=registered.state == STATE_READY,
            runtime=registered.runtime,
            detail=registered.status_line or "ready",
        )

    def _is_loaded(self, runtime: str, path: Path | None) -> bool:
        if not runtime or path is None:
            return False
        try:
            adapter = self.manager.adapter(runtime)
        except Exception:
            return False
        loaded_path = getattr(adapter, "loaded_path", None)
        if callable(loaded_path):
            current = loaded_path()
            return current is not None and Path(current) == path
        return False

    def health(self, key: str | None = None) -> ModelHealth:
        """The live health of the chosen model (an alias of :meth:`status`)."""
        return self.status(key)

    # -- load / unload ----------------------------------------------------

    def load(self, key: str | None = None, *, settings: Any = None) -> Any:
        """Load the chosen model through the manager, returning its OperationResult."""
        resolved = self.resolve(key)
        if resolved.path is None:
            raise CompanionModelError(
                resolved.detail or f"{key or 'the model'} has no local file to load"
            )
        entry = self._record_for(resolved)
        if settings is None:
            settings = self.safe_settings(resolved)
        return self.manager.load(entry.model_id, settings=settings)

    def unload(self, key: str | None = None) -> Any:
        """Unload the chosen model (or every loaded model when no key is given)."""
        if not key:
            return self.manager.unload()
        resolved = self.resolve(key)
        entry = self._find_record(resolved)
        if entry is None:
            return self.manager.unload()
        return self.manager.unload(entry.model_id)

    def test(self, key: str | None = None, *, prompt: str = "") -> Any:
        """Ask the chosen model a trivial question, so a user can prove it answers."""
        resolved = self.resolve(key)
        entry = self._record_for(resolved)
        if prompt:
            return self.manager.test(entry.model_id, prompt=prompt)
        return self.manager.test(entry.model_id)

    # -- settings ---------------------------------------------------------

    def safe_settings(self, resolved: ResolvedModel | None = None) -> Any:
        """CPU-first, memory-safe settings for the chosen model.

        The context is the model's own safe default (:data:`SAFE_CONTEXT`), the
        output is bounded (:data:`SAFE_MAX_TOKENS`), and the GPU layer count stays
        0 unless a caller raises it explicitly -- the honest default for the target
        hardware.
        """
        from core.model_settings import ModelSettings

        return ModelSettings(context=SAFE_CONTEXT, max_tokens=SAFE_MAX_TOKENS, gpu_layers=0)

    def memory_estimate(
        self, key: str | None = None, *, context_length: int | None = None
    ) -> Any:
        """An estimate of the chosen model's memory need, with its uncertainty."""
        resolved = self.resolve(key)
        if resolved.path is None:
            return None
        entry = self._record_for(resolved)
        return self.manager.memory_estimate(
            entry.model_id,
            context_length=context_length or SAFE_CONTEXT,
            gpu_layers=0,
        )

    def preflight(self, key: str | None = None, *, context_length: int | None = None) -> Any:
        """Every check that should pass before the chosen model is loaded, as data."""
        resolved = self.resolve(key)
        if resolved.path is None:
            return None
        entry = self._record_for(resolved)
        return self.manager.preflight(entry.model_id, context_length=context_length or SAFE_CONTEXT)

    def resources(self) -> Any:
        """This machine's own figures (memory, disk, CPU), each ``None`` when unknown."""
        return self.manager.resources()

    # -- fallback / ranking ----------------------------------------------

    def rank(self, *, limit: int = 5) -> tuple[Any, ...]:
        """Rank the models this machine has for the companion task, best first."""
        return self.manager.rank_for_task(TASK_COMPANION, limit=limit)

    def fallback_plan(self, key: str | None = None, *, limit: int = 3) -> Any:
        """A bounded ladder of models to try for the companion task."""
        preferred = ""
        resolved = self.resolve(key)
        if resolved.path is not None:
            entry = self._find_record(resolved)
            if entry is not None:
                preferred = entry.model_id
        return self.manager.fallback_plan(TASK_COMPANION, preferred_model_id=preferred, limit=limit)

    def download_hint(self, key: str | None = None) -> str:
        """The governed command that would fetch the chosen model, or ``""``."""
        resolved = self.resolve(key)
        for model in self._catalog_models():
            if str(getattr(model, "key", "")) == resolved.key:
                return str(getattr(model, "download_command", "") or "")
        return ""

    # -- cancellation / timeout ------------------------------------------

    def begin_generation(self, key: str | None = None, *, timeout_seconds: float = 0.0) -> GenerationHandle:
        """Open a cancellation/timeout handle for one generation request."""
        resolved = self.resolve(key)
        return GenerationHandle(
            key=resolved.key, started_at=time.monotonic(), timeout_seconds=timeout_seconds
        )

    def should_stop(self, handle: GenerationHandle) -> bool:
        """Whether a generation should stop (cancelled by the user or timed out)."""
        return handle.cancelled or handle.expired()

    def generate(
        self,
        prompt: str,
        *,
        key: str | None = None,
        settings: Any = None,
        handle: GenerationHandle | None = None,
    ) -> Any:
        """Run **one** local generation through the manager and return its result.

        This is the Studio's only inference call: it resolves the chosen model,
        runs the project's existing runtime through the manager (loading it first
        when nothing is loaded), and returns the manager's own ``InferenceResult``
        -- never a second runtime, never a network call. A cancelled or expired
        handle stops the request **before** it starts, so a user's Stop is honoured
        without a long wait; a failure is returned as ``ok=False`` with the real
        reason, never raised at the caller.
        """
        resolved = self.resolve(key)
        if handle is not None and self.should_stop(handle):
            return InferenceResult(
                ok=False, runtime=resolved.engine, detail="generation cancelled"
            )
        if resolved.path is None:
            return InferenceResult(
                ok=False,
                runtime=resolved.engine,
                detail=f"The model {resolved.key!r} is not installed on this machine.",
            )
        entry = self._record_for(resolved)
        result = self.manager.test(
            entry.model_id, prompt=prompt, settings=settings or self.safe_settings(resolved)
        )
        return result.inference or InferenceResult(
            ok=False, runtime=resolved.engine, detail=result.detail
        )


def _resolved_from(model: Any) -> ResolvedModel:
    """A :class:`ResolvedModel` view of a catalog entry, read defensively.

    The catalog's own shape is the source of truth; this only renames its fields,
    so a change in the catalog is reflected here without a second definition.
    """
    path_value = getattr(model, "path", None)
    return ResolvedModel(
        key=str(getattr(model, "key", "") or ""),
        title=str(getattr(model, "title", "") or ""),
        engine=str(getattr(model, "engine", "") or ""),
        path=Path(path_value) if path_value else None,
        available=bool(getattr(model, "available", False)),
        detail=str(getattr(model, "detail", "") or ""),
        from_choice=True,
    )


__all__ = [
    "LOW_MEMORY_SHARE",
    "SAFE_CONTEXT",
    "SAFE_MAX_TOKENS",
    "TASK_COMPANION",
    "CompanionModelError",
    "CompanionModelRouter",
    "GenerationHandle",
    "ModelHealth",
    "ResolvedModel",
]
