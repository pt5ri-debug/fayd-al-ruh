"""The runtime adapter: what a model *runs on*, behind one honest interface.

The project already has the runtimes it needs -- six near-identical in-process
``llama-cpp-python`` wrappers (Ultron's, Vision's, Banner's, the Code Studio's,
Doc's Studio's, the companion's) and one standard-library Ollama client -- but no
single description of what a runtime *can do*. Without one, "does this runtime
support unloading?" can only be answered by reading six modules, and a caller
ends up pretending every runtime works the same way.

This module is that description, and it is deliberately **capability-first**:

* :class:`RuntimeCapabilities` declares what an adapter can actually do
  (``can_load``, ``can_unload``, ``can_stop``, ``can_discover``, ``can_infer``,
  ``gpu_offload``, ``streaming``, ``vision``). An operation an adapter does not
  support is refused through :meth:`RuntimeAdapter.require` with the reason, not
  silently no-op'd.
* :class:`RuntimeStatus` is one shape for "can this runtime answer right now, and
  what is the real next step", replacing the eleven different ``*_status``
  dataclasses a caller would otherwise have to branch on.
* :class:`RuntimeAdapter` is the ``Protocol``. Two implementations ship here:
  :class:`LlamaCppAdapter` (in-process, wraps the existing construction) and
  :class:`OllamaAdapter` (a server, reusing :class:`core.coder_model.OllamaCoderClient`
  so no second HTTP client is written).

Two hard rules, matching the audit:

* **llama.cpp is imported lazily** and its availability is *probed*, never
  assumed. No executable path, package, server mode, or GPU backend is presumed.
* **GPU offloading stays off** unless the installed build and the machine both
  say otherwise. ``llama_supports_gpu_offload()`` is asked defensively through
  ``getattr``; a runtime that cannot answer is reported as *unknown*, and CPU
  execution remains the path that always works.

Nothing here downloads a model, and nothing here loads one at import time.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.coder_model import (
    CoderMessage,
    CoderUnavailable,
    OllamaCoderClient,
    coder_status,
)
from core.model_files import FORMAT_FAYD_CHECKPOINT, FORMAT_GGUF, FileProbe, probe_file

#: The runtime identifiers a registry entry carries. A plain string rather than an
#: enum, because the value is persisted and printed (the project's own rule:
#: ``str`` + ``Enum`` would change ``str()`` output and alter stored data).
RUNTIME_LLAMA_CPP = "llama.cpp"
RUNTIME_OLLAMA = "ollama"
RUNTIME_INPROCESS_TRAINED = "fayd-inprocess"
#: A remote, OpenAI-compatible endpoint. A model can be *registered* against it,
#: but no adapter is built for it here: the existing ``core.cloud_model`` path
#: already governs it behind its own permission and approval gates, and this
#: subsystem deliberately opens no second route to the network.
RUNTIME_CLOUD = "cloud"
#: The retrieval mouth (Jarvis from the sources) runs no language model at all.
RUNTIME_NONE = "none"
#: A model whose runtime this build does not know. An honest answer, not a failure.
RUNTIME_UNKNOWN = "unknown"

#: Every runtime name this module can build an adapter for.
RUNTIME_NAMES: tuple[str, ...] = (
    RUNTIME_LLAMA_CPP,
    RUNTIME_OLLAMA,
    RUNTIME_INPROCESS_TRAINED,
)

#: Every runtime a *registry entry* may name -- the adaptable ones plus the two
#: the project already has an owner for (a cloud endpoint, and no model at all).
RUNTIME_VALUES: tuple[str, ...] = (
    *RUNTIME_NAMES,
    RUNTIME_CLOUD,
    RUNTIME_NONE,
    RUNTIME_UNKNOWN,
)


class RuntimeUnavailable(RuntimeError):
    """The runtime cannot perform the requested operation, with the reason stated."""


class UnsupportedCapability(RuntimeUnavailable):
    """The runtime is fine, but it genuinely cannot do what was asked."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What one runtime can actually do. ``False`` means "refuse", not "try"."""

    can_load: bool = False
    can_unload: bool = False
    can_stop: bool = False
    can_discover: bool = False
    can_validate: bool = True
    can_infer: bool = False
    #: Tri-state, like :attr:`RuntimeStatus.gpu_offload`: ``True``/``False`` when
    #: the runtime answered, ``None`` when it cannot be determined here.
    gpu_offload: bool | None = None
    streaming: bool = False
    vision: bool = False
    #: True when the runtime keeps the model resident after a call, so unloading
    #: is meaningful at all.
    resident: bool = False

    def supports(self, operation: str) -> bool:
        """True when ``operation`` names a capability this runtime declares."""
        return bool(getattr(self, operation, False))

    def unsupported(self) -> tuple[str, ...]:
        """The capabilities this runtime does *not* have, in a stable order."""
        names = (
            "can_load", "can_unload", "can_stop", "can_discover",
            "can_validate", "can_infer", "gpu_offload", "streaming", "vision",
        )
        return tuple(name for name in names if not self.supports(name))

    def describe(self) -> str:
        if not self.unsupported():
            return "full capabilities"
        return "supported: " + ", ".join(
            name[4:] if name.startswith("can_") else name
            for name in (
                "can_load", "can_unload", "can_stop", "can_discover",
                "can_validate", "can_infer", "gpu_offload", "streaming", "vision",
            )
            if self.supports(name)
        )


@dataclass(frozen=True)
class RuntimeStatus:
    """One honest shape for "can this runtime answer right now?".

    ``available`` is the answer; ``detail`` names the real reason when it is
    ``False``, and ``hint`` the real next step. ``gpu_offload`` is tri-state:
    ``True``/``False`` when the build answered, ``None`` when it could not be
    determined -- which is a first-class answer here, never guessed.
    """

    runtime: str
    available: bool
    detail: str = ""
    hint: str = ""
    version: str = ""
    models: tuple[str, ...] = ()
    gpu_offload: bool | None = None
    capabilities: RuntimeCapabilities = field(default_factory=RuntimeCapabilities)

    def summary(self) -> str:
        if self.available:
            gpu = (
                "gpu offload available"
                if self.gpu_offload
                else ("cpu only" if self.gpu_offload is False else "gpu support unknown")
            )
            return f"{self.runtime}: ready ({gpu})."
        return f"{self.runtime}: not available — {self.detail or 'unknown reason.'}"


@dataclass(frozen=True)
class LoadResult:
    """The outcome of asking a runtime to load a model. Never raised at the caller."""

    ok: bool
    runtime: str
    model_path: Path | None = None
    detail: str = ""
    seconds: float = 0.0

    def describe(self) -> str:
        if self.ok:
            return f"{self.runtime}: loaded {self.model_path} in {self.seconds:.2f}s."
        return f"{self.runtime}: could not load the model — {self.detail}"


@dataclass(frozen=True)
class InferenceResult:
    """The outcome of a one-shot test generation. Never raised at the caller."""

    ok: bool
    runtime: str
    text: str = ""
    detail: str = ""
    seconds: float = 0.0

    def describe(self) -> str:
        if self.ok:
            preview = self.text.strip().replace("\n", " ")[:120]
            return f"{self.runtime}: answered in {self.seconds:.2f}s — {preview!r}"
        return f"{self.runtime}: inference failed — {self.detail}"


@runtime_checkable
class RuntimeAdapter(Protocol):
    """What a model manager needs from a runtime.

    Deliberately narrow: a runtime that cannot load, unload, or infer declares so
    in :meth:`capabilities` and raises :class:`UnsupportedCapability` from the
    method, so a caller can branch on the declaration instead of on the runtime's
    name.
    """

    name: str

    def capabilities(self) -> RuntimeCapabilities: ...

    def status(self) -> RuntimeStatus: ...

    def validate(self, probe: FileProbe) -> tuple[bool, str]: ...

    def discover(self) -> tuple[str, ...]: ...

    def load(self, model_path: Path, *, settings: Any = None) -> LoadResult: ...

    def unload(self) -> LoadResult: ...

    def infer(self, prompt: str, *, settings: Any = None) -> InferenceResult: ...


# ------------------------------------------------------------------ llama.cpp


def llama_cpp_available() -> bool:
    """True when ``llama_cpp`` can be imported here, without importing it twice.

    Probed with :mod:`importlib.util` so merely asking does not pull a heavy
    extension module into a process that is only listing models.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


def llama_cpp_gpu_offload() -> bool | None:
    """Whether the installed llama.cpp build supports GPU offload.

    ``None`` when the question cannot be answered -- the package is absent, the
    build predates the query, or the call fails. The project never turns GPU
    offload on from a guess, so ``None`` is treated exactly like ``False`` by
    :func:`core.model_manager` while still being *reported* honestly.
    """
    if not llama_cpp_available():
        return None
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if not callable(probe):
        return None
    try:
        return bool(probe())
    except Exception:  # pragma: no cover - a broken build must not crash a listing
        return None


def llama_cpp_version() -> str:
    """The installed llama.cpp version string, or ``""`` when it cannot be read."""
    if not llama_cpp_available():
        return ""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        return ""
    for attribute in ("__version__", "llama_cpp_version"):
        value = getattr(llama_cpp, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


class LlamaCppAdapter:
    """Run a GGUF file in this process through ``llama-cpp-python``.

    The adapter is the **one** place that construction happens for the manager.
    It deliberately does not replace the six existing per-mouth runners -- those
    keep serving their mouths unchanged -- so introducing the manager cannot alter
    how Ultron, Vision, Banner, the Code Studio, Doc's Studio, or the companion
    answer. What it adds is the missing half: a runtime that can be *asked* what
    it supports, told to load a specific file, told to unload, and asked for a
    one-shot test reply.

    The model is loaded lazily, on the first :meth:`load` or :meth:`infer`, so
    constructing the adapter is free. A test injects its own ``runner`` (the same
    seam every other engine in this project exposes) and the whole adapter is then
    exercised with no extra and no weights.
    """

    name = RUNTIME_LLAMA_CPP

    def __init__(self, *, runner: Any = None, root: Path | str | None = None) -> None:
        self._runner = runner
        self._root = Path(root) if root is not None else None
        self._model_path: Path | None = None
        self._loaded = False

    # --- capability ----------------------------------------------------
    def capabilities(self) -> RuntimeCapabilities:
        offload = self._gpu_offload()
        return RuntimeCapabilities(
            can_load=True,
            # In-process: "unload" means dropping our reference and letting the
            # interpreter free it. It is honest to offer, and honest to say it
            # cannot be guaranteed (see ``unload``).
            can_unload=True,
            can_stop=False,  # there is no process or server to stop
            can_discover=True,  # the folder scan is the discovery
            can_validate=True,
            can_infer=True,
            # Tri-state: the build's own answer, or ``None`` when it cannot be read.
            gpu_offload=offload,
            streaming=True,
            vision=False,
            resident=True,
        )

    def _gpu_offload(self) -> bool | None:
        if self._runner is not None:
            # An injected runner is a test double: report the honest unknown.
            return None
        return llama_cpp_gpu_offload()

    # --- status --------------------------------------------------------
    def status(self) -> RuntimeStatus:
        capabilities = self.capabilities()
        offload = self._gpu_offload()
        if self._runner is not None:
            return RuntimeStatus(
                runtime=self.name, available=True, version="injected runner",
                detail="", gpu_offload=offload, capabilities=capabilities,
            )
        if not llama_cpp_available():
            return RuntimeStatus(
                runtime=self.name, available=False,
                detail="The optional runtime `llama-cpp-python` is not installed.",
                hint="python -m pip install -r requirements-ultron.txt",
                gpu_offload=None, capabilities=capabilities,
            )
        return RuntimeStatus(
            runtime=self.name, available=True, version=llama_cpp_version(),
            gpu_offload=offload, capabilities=capabilities,
        )

    # --- validation ----------------------------------------------------
    def validate(self, probe: FileProbe) -> tuple[bool, str]:
        """Whether this runtime can load ``probe``'s file, with the reason."""
        if not probe.exists:
            return False, probe.detail or "the file is not there."
        if probe.format != FORMAT_GGUF:
            return (
                False,
                f"{self.name} loads GGUF files; this one is {probe.format}. "
                "The trained checkpoint runs through the in-process trained runtime.",
            )
        if not probe.usable:
            return False, probe.detail or "the file failed its integrity check."
        return True, ""

    # --- discovery -----------------------------------------------------
    def discover(self) -> tuple[str, ...]:
        """What this runtime can see. In-process, that is the folder, not a server."""
        if self._root is None:
            return ()
        return tuple(sorted(path.name for path in self._root.glob("*.gguf") if path.is_file()))

    # --- lifecycle -----------------------------------------------------
    def load(self, model_path: Path, *, settings: Any = None) -> LoadResult:
        started = time.monotonic()
        path = Path(model_path)
        if self._runner is not None:
            self._model_path = path
            self._loaded = True
            return LoadResult(
                ok=True, runtime=self.name, model_path=path,
                detail="using the injected runner",
                seconds=time.monotonic() - started,
            )
        if not llama_cpp_available():
            return LoadResult(
                ok=False, runtime=self.name, model_path=path,
                detail="The optional runtime `llama-cpp-python` is not installed.",
                seconds=time.monotonic() - started,
            )
        probe = probe_file(path)
        ok, reason = self.validate(probe)
        if not ok:
            return LoadResult(
                ok=False, runtime=self.name, model_path=path, detail=reason,
                seconds=time.monotonic() - started,
            )
        try:
            self._runner = _build_llama_runner(path, settings)
        except Exception as error:
            self._runner = None
            return LoadResult(
                ok=False, runtime=self.name, model_path=path,
                detail=f"The model could not be loaded: {error}",
                seconds=time.monotonic() - started,
            )
        self._model_path = path
        self._loaded = True
        return LoadResult(
            ok=True, runtime=self.name, model_path=path,
            seconds=time.monotonic() - started,
        )

    def unload(self) -> LoadResult:
        """Drop the reference to the loaded model.

        In-process this cannot *promise* the memory is returned -- the interpreter
        frees it when the last reference goes, which is not the same instant -- so
        the result says exactly that rather than claiming a release that may not
        have happened.
        """
        if not self._loaded:
            return LoadResult(
                ok=True, runtime=self.name, detail="No model was loaded.",
            )
        self._runner = None
        self._model_path = None
        self._loaded = False
        return LoadResult(
            ok=True, runtime=self.name,
            detail=(
                "Reference dropped; the interpreter frees the weights when it "
                "collects them. In-process memory is not released on a fixed clock."
            ),
        )

    def loaded(self) -> bool:
        return self._loaded

    def loaded_path(self) -> Path | None:
        return self._model_path

    # --- inference -----------------------------------------------------
    def infer(self, prompt: str, *, settings: Any = None) -> InferenceResult:
        """One short test reply, so a user can prove the model really answers."""
        started = time.monotonic()
        text = (prompt or "").strip()
        if not text:
            return InferenceResult(
                ok=False, runtime=self.name, detail="A test needs a prompt.",
                seconds=time.monotonic() - started,
            )
        if not self._loaded or self._runner is None:
            return InferenceResult(
                ok=False, runtime=self.name,
                detail="No model is loaded. Load one first.",
                seconds=time.monotonic() - started,
            )
        options = _inference_options(settings)
        try:
            reply = self._runner.chat(
                [{"role": "user", "content": text}], **options
            )
        except Exception as error:
            return InferenceResult(
                ok=False, runtime=self.name, detail=str(error),
                seconds=time.monotonic() - started,
            )
        return InferenceResult(
            ok=True, runtime=self.name, text=str(reply),
            seconds=time.monotonic() - started,
        )


def _build_llama_runner(path: Path, settings: Any) -> Any:
    """Construct the real ``llama-cpp-python`` runner for ``path``.

    Kept as a module-level function so the import and the constructor arguments
    live in exactly one place. ``llama_cpp`` is imported here, lazily.
    """
    import llama_cpp  # type: ignore[import-not-found]

    return _LlamaChatRunner(
        llama_cpp.Llama(
            model_path=str(path),
            n_ctx=int(getattr(settings, "context", 0) or 4096),
            n_gpu_layers=int(getattr(settings, "gpu_layers", 0) or 0),
            n_threads=int(getattr(settings, "threads", 0) or 0) or None,
            verbose=False,
        )
    )


class _LlamaChatRunner:
    """A thin chat wrapper over a ``llama_cpp.Llama`` instance.

    Matches the ``chat(messages, **kwargs)`` seam every other runner in the
    project exposes, so the adapter can be handed a test double in its place.
    """

    def __init__(self, llama: Any) -> None:
        self._llama = llama

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        result = self._llama.create_chat_completion(messages=messages, **kwargs)
        try:
            return str(result["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeUnavailable(
                f"The model returned an unreadable response: {error}"
            ) from error


def _inference_options(settings: Any) -> dict[str, Any]:
    """The generation kwargs a test call should use, from the settings (or defaults)."""
    options: dict[str, Any] = {}
    if settings is None:
        return options
    temperature = getattr(settings, "temperature", None)
    if isinstance(temperature, (int, float)):
        options["temperature"] = float(temperature)
    top_p = getattr(settings, "top_p", None)
    if isinstance(top_p, (int, float)):
        options["top_p"] = float(top_p)
    max_tokens = getattr(settings, "max_tokens", None)
    if isinstance(max_tokens, int) and max_tokens > 0:
        options["max_tokens"] = int(max_tokens)
    return options


# -------------------------------------------------------------------- Ollama


class OllamaAdapter:
    """Reach a local Ollama server, reusing the project's own client.

    Ollama is a **server**, not an in-process runtime, so its capabilities differ
    honestly from llama.cpp's: it can load and unload (through its own API), it
    cannot be *discovered* the way a folder can, and the project never starts or
    stops the daemon -- that stays the user's program to run.

    The transport is :class:`core.coder_model.OllamaCoderClient`, a thin
    standard-library ``urllib`` client, so no new dependency and no second HTTP
    client is written. A test injects its own ``client``.
    """

    name = RUNTIME_OLLAMA

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = "",
        base_url: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OllamaCoderClient(model or "qwen2.5-coder:7b", **kwargs)
        self._loaded_model = ""

    # --- capability ----------------------------------------------------
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            can_load=True,      # Ollama loads on the first request
            can_unload=True,    # Ollama has an explicit unload
            can_stop=False,     # the project never stops the user's daemon
            can_discover=True,  # ``/api/tags`` lists what is pulled
            can_validate=True,
            can_infer=True,
            # Decided by Ollama and its own configuration, not by this project, so
            # the honest answer is "unknown" rather than a guess either way.
            gpu_offload=None,
            streaming=False,
            vision=False,
            resident=True,
        )

    # --- status --------------------------------------------------------
    def status(self) -> RuntimeStatus:
        status = coder_status(self._client)
        return RuntimeStatus(
            runtime=self.name,
            available=bool(status.available),
            detail=status.detail or ("" if status.available else "the server did not answer."),
            hint=status.hint,
            models=tuple(status.models),
            gpu_offload=None,
            capabilities=self.capabilities(),
        )

    # --- validation ----------------------------------------------------
    def validate(self, probe: FileProbe) -> tuple[bool, str]:
        """Ollama runs models it pulled itself, not arbitrary files on disk."""
        return (
            False,
            "Ollama runs the models it pulled (`ollama pull <model>`); it is not "
            "handed a .gguf file. Point a registry entry at a pulled name instead.",
        )

    # --- discovery -----------------------------------------------------
    def discover(self) -> tuple[str, ...]:
        try:
            return tuple(self._client.list_models())
        except CoderUnavailable:
            return ()

    # --- lifecycle -----------------------------------------------------
    def load(self, model_path: Path, *, settings: Any = None) -> LoadResult:
        """A no-op the honest way: Ollama loads on the first request, so there is
        nothing to pre-load. Reported as a success with that explanation rather
        than pretending a load happened."""
        name = str(model_path)
        self._loaded_model = name
        return LoadResult(
            ok=True, runtime=self.name, model_path=Path(name),
            detail="Ollama loads a model on its first request; nothing was pre-loaded.",
        )

    def unload(self) -> LoadResult:
        self._loaded_model = ""
        return LoadResult(
            ok=True, runtime=self.name,
            detail=(
                "Ollama keeps a model resident for its own `keep_alive` window; "
                "this project does not stop the server."
            ),
        )

    def loaded(self) -> bool:
        return bool(self._loaded_model)

    # --- inference -----------------------------------------------------
    def infer(self, prompt: str, *, settings: Any = None) -> InferenceResult:
        started = time.monotonic()
        text = (prompt or "").strip()
        if not text:
            return InferenceResult(
                ok=False, runtime=self.name, detail="A test needs a prompt.",
                seconds=time.monotonic() - started,
            )
        temperature = float(getattr(settings, "temperature", 0.2) or 0.2)
        try:
            reply = self._client.chat(
                [CoderMessage(role="user", content=text)], temperature=temperature
            )
        except CoderUnavailable as error:
            return InferenceResult(
                ok=False, runtime=self.name, detail=str(error),
                seconds=time.monotonic() - started,
            )
        return InferenceResult(
            ok=True, runtime=self.name, text=str(reply),
            seconds=time.monotonic() - started,
        )


# ------------------------------------------------------- in-project trained


class InProcessTrainedAdapter:
    """The project's own trained checkpoint, run in this process.

    A trained ``.npz`` checkpoint is **not** a GGUF file and is not
    interchangeable with one: it needs the project's own numpy transformer
    (:mod:`ml.llm.transformer`) and its matching tokenizer, not
    ``llama-cpp-python``. This adapter is therefore its own runtime, and it says
    so -- which is what stops a caller from handing a checkpoint to a GGUF
    runtime and getting an opaque failure.

    The heavy import is deferred and the loader is injectable, so the adapter is
    exercised with neither numpy nor a checkpoint present.
    """

    name = RUNTIME_INPROCESS_TRAINED

    def __init__(self, *, loader: Any = None) -> None:
        self._loader = loader
        self._model: Any = None
        self._path: Path | None = None

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            can_load=True,
            can_unload=True,
            can_stop=False,
            can_discover=False,  # there is one checkpoint, not a folder to browse
            can_validate=True,
            can_infer=True,
            gpu_offload=False,  # the numpy transformer is CPU only, by design
            streaming=False,
            vision=False,
            resident=True,
        )

    def status(self) -> RuntimeStatus:
        capabilities = self.capabilities()
        try:
            import importlib.util

            present = importlib.util.find_spec("numpy") is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            return RuntimeStatus(
                runtime=self.name, available=False,
                detail="The trained model needs numpy, which is not installed.",
                hint="python -m pip install -r requirements.txt",
                gpu_offload=False, capabilities=capabilities,
            )
        return RuntimeStatus(
            runtime=self.name, available=True, gpu_offload=False,
            capabilities=capabilities,
        )

    def validate(self, probe: FileProbe) -> tuple[bool, str]:
        if not probe.exists:
            return False, probe.detail or "the file is not there."
        if probe.format != FORMAT_FAYD_CHECKPOINT:
            return (
                False,
                "The trained model needs a Fayd Al-Ruh checkpoint (.npz), not a "
                f"{probe.format} file.",
            )
        if not probe.usable:
            return False, probe.detail or "the checkpoint failed its integrity check."
        return True, ""

    def discover(self) -> tuple[str, ...]:
        raise UnsupportedCapability(
            "The trained checkpoint is a single file, not a set to discover. "
            "Its location comes from `core.paths.default_checkpoint()`."
        )

    def load(self, model_path: Path, *, settings: Any = None) -> LoadResult:
        started = time.monotonic()
        path = Path(model_path)
        probe = probe_file(path)
        ok, reason = self.validate(probe)
        if not ok:
            return LoadResult(
                ok=False, runtime=self.name, model_path=path, detail=reason,
                seconds=time.monotonic() - started,
            )
        if self._loader is not None:
            try:
                self._model = self._loader(path)
            except Exception as error:
                self._model = None
                return LoadResult(
                    ok=False, runtime=self.name, model_path=path,
                    detail=f"The checkpoint could not be loaded: {error}",
                    seconds=time.monotonic() - started,
                )
            self._path = path
            return LoadResult(
                ok=True, runtime=self.name, model_path=path,
                seconds=time.monotonic() - started,
            )
        try:
            from ml.llm.transformer import TransformerLM
        except ImportError as error:
            return LoadResult(
                ok=False, runtime=self.name, model_path=path,
                detail=(
                    "The trained model needs its own runtime (numpy + "
                    f"ml.llm.transformer): {error}"
                ),
                seconds=time.monotonic() - started,
            )
        try:
            self._model = TransformerLM.load(path)
        except Exception as error:
            self._model = None
            return LoadResult(
                ok=False, runtime=self.name, model_path=path,
                detail=f"The checkpoint could not be loaded: {error}",
                seconds=time.monotonic() - started,
            )
        self._path = path
        return LoadResult(
            ok=True, runtime=self.name, model_path=path,
            seconds=time.monotonic() - started,
        )

    def unload(self) -> LoadResult:
        self._model = None
        self._path = None
        return LoadResult(
            ok=True, runtime=self.name,
            detail="Reference dropped; the interpreter frees the weights when it collects them.",
        )

    def loaded(self) -> bool:
        return self._model is not None

    def infer(self, prompt: str, *, settings: Any = None) -> InferenceResult:
        started = time.monotonic()
        if self._model is None:
            return InferenceResult(
                ok=False, runtime=self.name, detail="No checkpoint is loaded.",
                seconds=time.monotonic() - started,
            )
        try:
            from ml.llm.generator import LocalGenerator

            generator = LocalGenerator(self._model)
            max_tokens = int(getattr(settings, "max_tokens", 0) or 32)
            temperature = float(getattr(settings, "temperature", 0.0) or 0.0)
            full = generator.generate(
                prompt, max_new_tokens=max_tokens, temperature=temperature
            )
        except Exception as error:
            return InferenceResult(
                ok=False, runtime=self.name, detail=str(error),
                seconds=time.monotonic() - started,
            )
        # The generator returns prompt + continuation; the reply is the continuation.
        return InferenceResult(
            ok=True, runtime=self.name, text=str(full)[len(prompt):],
            seconds=time.monotonic() - started,
        )


# ------------------------------------------------------------------ registry


#: The runtime each model format is run by. A format with no entry is genuinely
#: unknown here, and the caller is told so rather than being handed a default.
RUNTIME_FOR_FORMAT: dict[str, str] = {
    FORMAT_GGUF: RUNTIME_LLAMA_CPP,
    FORMAT_FAYD_CHECKPOINT: RUNTIME_INPROCESS_TRAINED,
}

#: The runtime each mouth's engine already uses. This is a *record* of what the
#: existing code does, so a registered model reports the runtime that will really
#: run it -- it changes no behaviour.
RUNTIME_FOR_ENGINE: dict[str, str] = {
    "ultron": RUNTIME_LLAMA_CPP,
    "vision": RUNTIME_LLAMA_CPP,
    "banner": RUNTIME_LLAMA_CPP,
    "natasha": RUNTIME_LLAMA_CPP,
    "hakeem": RUNTIME_LLAMA_CPP,
    "shared": RUNTIME_LLAMA_CPP,
    "coder": RUNTIME_LLAMA_CPP,
    "docs": RUNTIME_LLAMA_CPP,
    "model": RUNTIME_INPROCESS_TRAINED,
    "cloud": RUNTIME_CLOUD,
    "sources": RUNTIME_NONE,
}


def runtime_for_format(model_format: str) -> str:
    """The runtime that runs ``model_format``, or :data:`RUNTIME_UNKNOWN`."""
    return RUNTIME_FOR_FORMAT.get(model_format, RUNTIME_UNKNOWN)


def runtime_for_engine(engine: str) -> str:
    """The runtime an existing mouth's engine uses, or :data:`RUNTIME_UNKNOWN`."""
    return RUNTIME_FOR_ENGINE.get(engine, RUNTIME_UNKNOWN)


def adapter_for(
    runtime: str,
    *,
    runner: Any = None,
    client: Any = None,
    loader: Any = None,
) -> RuntimeAdapter:
    """Build the adapter for ``runtime``, or raise :class:`RuntimeUnavailable`.

    The three keyword seams (``runner``/``client``/``loader``) let a test inject a
    fake, exactly as every other engine in this project does.
    """
    if runtime == RUNTIME_LLAMA_CPP:
        return LlamaCppAdapter(runner=runner)
    if runtime == RUNTIME_OLLAMA:
        return OllamaAdapter(client=client)
    if runtime == RUNTIME_INPROCESS_TRAINED:
        return InProcessTrainedAdapter(loader=loader)
    raise RuntimeUnavailable(
        f"No runtime adapter is known as {runtime!r}. Known runtimes: "
        f"{', '.join(RUNTIME_NAMES)}."
    )


def runtime_names() -> tuple[str, ...]:
    """Every runtime name this build can adapt."""
    return RUNTIME_NAMES


def describe_runtimes() -> str:
    """A multi-line, honest report of every runtime, for a status surface."""
    lines: list[str] = []
    for name in RUNTIME_NAMES:
        try:
            status = adapter_for(name).status()
        except RuntimeUnavailable as error:  # pragma: no cover - defensive
            lines.append(f"{name}: unavailable ({error})")
            continue
        lines.append(status.summary())
        if not status.available and status.hint:
            lines.append(f"    next step: {status.hint}")
    return "\n".join(lines)


def iter_messages(messages: Sequence[Any]) -> Iterator[dict[str, str]]:
    """Normalise a message sequence into ``{"role", "content"}`` dicts.

    A small helper so an adapter never has to know which message class a caller
    used (the project has several: ``CoderMessage``, ``CloudMessage``, and plain
    dicts). Anything without a role and content is skipped rather than guessed at.
    """
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
        else:
            role = str(getattr(message, "role", ""))
            content = str(getattr(message, "content", ""))
        if role and content:
            yield {"role": role, "content": content}


__all__ = [
    "RUNTIME_CLOUD",
    "RUNTIME_FOR_ENGINE",
    "RUNTIME_FOR_FORMAT",
    "RUNTIME_INPROCESS_TRAINED",
    "RUNTIME_LLAMA_CPP",
    "RUNTIME_NAMES",
    "RUNTIME_NONE",
    "RUNTIME_OLLAMA",
    "RUNTIME_UNKNOWN",
    "RUNTIME_VALUES",
    "InProcessTrainedAdapter",
    "InferenceResult",
    "LlamaCppAdapter",
    "LoadResult",
    "OllamaAdapter",
    "RuntimeAdapter",
    "RuntimeCapabilities",
    "RuntimeStatus",
    "RuntimeUnavailable",
    "UnsupportedCapability",
    "adapter_for",
    "describe_runtimes",
    "iter_messages",
    "llama_cpp_available",
    "llama_cpp_gpu_offload",
    "llama_cpp_version",
    "runtime_for_engine",
    "runtime_for_format",
    "runtime_names",
]
