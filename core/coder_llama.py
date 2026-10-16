"""The Code Studio's *in-process* coding model: no server, no background daemon.

The Studio originally talked to a local **Ollama** server. That works, but it
means a separate program has to be installed and left running, holding a few
gigabytes of RAM even while the Studio is idle. This module is the alternative
the project already uses for its general mouth (:mod:`core.ultron`): load a
**GGUF** file straight into this process with ``llama-cpp-python`` and generate a
reply on demand. When nothing is generating, nothing is held.

The two runtimes are kept side by side rather than replacing one another:

* :class:`LlamaCppCoderClient` satisfies the same
  :class:`core.coder_model.CoderClient` protocol as
  :class:`core.coder_model.OllamaCoderClient`, so the agent, the service, and the
  window cannot tell them apart.
* :func:`coder_backend_status` reports *which* runtime can answer and, when none
  can, the single next step for each -- install the extra, pull the model, or
  download the weights.

Everything heavy is lazy. ``llama_cpp`` is imported only inside
:meth:`LlamaCppCoderClient._load`, so importing this module (and the whole CLI)
needs neither the extra nor a model file. Nothing here executes model output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from core.coder_model import (
    DEFAULT_CODER_MODEL,
    CoderClient,
    CoderMessage,
    CoderUnavailable,
    OllamaCoderClient,
    coder_status,
)
from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_CODER, pip_install_command
from core.paths import coder_models_dir
from core.permissions import Capability, PermissionPolicy

#: The registry file written after a download, naming the model to load.
REGISTRY_FILENAME = "model.json"

#: The tool name the download gate is keyed on.
DOWNLOAD_TOOL = "coder-model-download"

#: Context window for an in-process coding model. Code prompts are long (a file
#: plus the protocol), so this is larger than Ultron's conversational default.
DEFAULT_CONTEXT = 8192

#: A safe ceiling on the context, because RAM grows with it.
MAX_CONTEXT = 32768

#: A safe ceiling on one reply, so a mistyped run cannot generate forever.
MAX_TOKENS_CEILING = 4096

#: How many threads to use. ``0`` lets llama.cpp pick a sensible count.
DEFAULT_THREADS = 0

#: Offload to the GPU only when the user asks; the default is CPU-only so the
#: Studio works on a plain laptop.
DEFAULT_GPU_LAYERS = 0


class CoderModelUnavailable(CoderUnavailable):
    """The in-process coding model cannot run (no extra, no weights)."""


@dataclass(frozen=True)
class CoderModelSpec:
    """A known-good coding model that ``studio download`` can fetch.

    A spec bundles the Hugging Face repo, the exact GGUF filename, and a short
    description, so "a local coding model" is a named choice rather than a URL
    the user has to assemble. Sizes are the Q4 quantisation, the balance of
    quality and RAM that fits a laptop.
    """

    key: str
    title: str
    repo: str
    filename: str
    params: str
    approx_size_gb: float
    summary: str


#: The download catalog, keyed by name. ``qwen2.5-coder-7b`` is the default: it
#: is the same model family the Ollama path used, so behaviour is comparable.
CATALOG: dict[str, CoderModelSpec] = {
    "qwen2.5-coder-7b": CoderModelSpec(
        key="qwen2.5-coder-7b",
        title="Qwen2.5 Coder 7B Instruct (default)",
        repo="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        params="7B",
        approx_size_gb=4.7,
        summary="Strong at code, follows a strict JSON protocol; needs ~6 GB RAM.",
    ),
    "qwen2.5-coder-3b": CoderModelSpec(
        key="qwen2.5-coder-3b",
        title="Qwen2.5 Coder 3B Instruct (light)",
        repo="bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
        params="3B",
        approx_size_gb=2.0,
        summary="Lighter and faster; good when RAM or time is tight.",
    ),
    "qwen2.5-coder-1.5b": CoderModelSpec(
        key="qwen2.5-coder-1.5b",
        title="Qwen2.5 Coder 1.5B Instruct (smallest)",
        repo="bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        params="1.5B",
        approx_size_gb=1.1,
        summary="Runs on almost any machine; weaker on large multi-file changes.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen2.5-coder-7b"


def resolve_spec(name: str | None) -> CoderModelSpec:
    """The spec for ``name``, or the default when ``name`` is empty/unknown."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise CoderModelUnavailable(
            f"Unknown coding model {name!r}. Available: {', '.join(sorted(CATALOG))}."
        )
    return CATALOG[key]


def spec_names() -> tuple[str, ...]:
    """Every catalog key, sorted, for a menu."""
    return tuple(sorted(CATALOG))


def spec_summaries() -> tuple[tuple[str, str, float, str], ...]:
    """``(key, title, size in GB, summary)`` per catalog entry, for a menu."""
    return tuple(
        (spec.key, spec.title, spec.approx_size_gb, spec.summary)
        for spec in (CATALOG[key] for key in spec_names())
    )


# ------------------------------------------------------------------- discovery


def _coder_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else coder_models_dir()


def _registry_file(root: Path | str | None = None) -> Path:
    return _coder_dir(root) / REGISTRY_FILENAME


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The coding model file to load, or ``None`` when none is present.

    Prefers the file named in the registry (``model.json``, written by
    ``download``), then any single ``.gguf`` in the folder. When several exist
    with no registry, the largest is chosen -- the biggest file is almost always
    the highest-quality model the user meant to use.
    """
    directory = _coder_dir(root)
    registry = _registry_file(root)
    if registry.is_file():
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        name = data.get("file") if isinstance(data, dict) else None
        if name:
            candidate = directory / str(name)
            if candidate.is_file():
                return candidate
    candidates = sorted(
        (path for path in directory.glob("*.gguf") if path.is_file()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def model_is_cached(root: Path | str | None = None) -> bool:
    """True when at least one coding model file is present on disk."""
    return find_model_file(root) is not None


def extra_available() -> bool:
    """True when ``llama_cpp`` can be imported, without importing it here."""
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _require_extra() -> Any:
    """Import ``llama_cpp`` or raise a clear :class:`CoderModelUnavailable`."""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError as error:
        raise CoderModelUnavailable(
            "The in-process coding model needs the optional coder extra. Install it "
            f"with: {pip_install_command(FEATURE_CODER)}"
        ) from error
    return llama_cpp


# ---------------------------------------------------------------------- client


class CoderRunner(Protocol):
    """The seam a test injects so no real model is needed."""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class LlamaCppRunner:
    """The default runner: a thin ``llama-cpp-python`` chat wrapper.

    ``llama_cpp`` is imported here, lazily, so the module loads without the extra.
    The GGUF's own chat template is used by ``create_chat_completion``, so the
    model's expected prompt format is honoured without the project hard-coding one.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        llama_cpp = _require_extra()
        self._llama = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=context,
            n_gpu_layers=gpu_layers,
            n_threads=threads or None,
            verbose=False,
        )

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        result = self._llama.create_chat_completion(messages=messages, **kwargs)
        return _content_of(result)


def _content_of(result: Any) -> str:
    """Pull the assistant text out of llama.cpp's reply, defensively."""
    try:
        choices = result["choices"]
        message = choices[0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as error:
        raise CoderModelUnavailable(
            "The in-process coding model returned an unexpected reply shape."
        ) from error
    text = str(content or "").strip()
    if not text:
        raise CoderModelUnavailable("The in-process coding model returned an empty reply.")
    return text


class LlamaCppCoderClient:
    """A :class:`core.coder_model.CoderClient` backed by a local GGUF file.

    Satisfies the same protocol as :class:`OllamaCoderClient`, so the agent and
    the window cannot tell the two runtimes apart. The model is loaded once, on
    the first generation, and reused -- so a Studio session pays the load cost
    once and nothing sits in RAM before the first request.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: str = DEFAULT_CODER_MODEL,
        root: Path | str | None = None,
        runner: CoderRunner | None = None,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        #: Marks this client for :func:`core.coder_model.coder_status`, so the
        #: status line never tells an in-process user to start a server.
        self.backend = "in-process"
        self.root = root
        self._path = Path(model_path) if model_path is not None else None
        self._runner = runner
        self.context = context
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.max_tokens = max_tokens

    # --- model file -----------------------------------------------------
    def model_path(self) -> Path:
        """The GGUF file this client will load, resolved on demand."""
        if self._path is not None:
            if not self._path.is_file():
                raise CoderModelUnavailable(f"The coding model file is missing: {self._path}")
            return self._path
        found = find_model_file(self.root)
        if found is None:
            directory = _coder_dir(self.root)
            raise CoderModelUnavailable(
                f"No coding model file in {directory}. Download one with "
                "`python -m cli studio download --approve`, or drop a `.gguf` file there."
            )
        return found

    def is_installed(self) -> bool:
        """True when a model file is present and the runtime can load it."""
        if self._runner is not None:
            return True
        if not extra_available():
            return False
        try:
            return self.model_path().is_file()
        except CoderModelUnavailable:
            return False

    def missing_detail(self) -> str:
        """Why the in-process model cannot answer, in one actionable line."""
        if not extra_available():
            return (
                "The in-process coding model needs the optional coder extra. Install it "
                f"with: {pip_install_command(FEATURE_CODER)}"
            )
        return (
            f"No coding model file in {_coder_dir(self.root)}. Download one with "
            "`python -m cli studio download --approve`, or drop a `.gguf` file there."
        )

    def _load(self) -> CoderRunner:
        """Load the runner on first use, so construction never loads a model."""
        if self._runner is None:
            self._runner = LlamaCppRunner(
                self.model_path(),
                context=self.context,
                gpu_layers=self.gpu_layers,
                threads=self.threads,
            )
        return self._runner

    # --- CoderClient ----------------------------------------------------
    def list_models(self) -> tuple[str, ...]:
        """The model files present, so ``status`` can report honestly.

        The in-process runtime has no server to ask, so this reports what is on
        disk. An absent file yields an empty tuple rather than raising, so
        :func:`core.coder_model.coder_status` reads it as "not installed".
        """
        found = find_model_file(self.root)
        return (found.name,) if found is not None else ()

    def chat(self, messages: Sequence[CoderMessage], *, temperature: float = 0.2) -> str:
        """One reply from the in-process model."""
        if not messages:
            raise ValueError("At least one message is required.")
        runner = self._load()
        try:
            return runner.chat(
                [message.as_dict() for message in messages],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        except CoderUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - surfaced as an honest message
            raise CoderModelUnavailable(f"The in-process coding model failed: {error}") from error


# ---------------------------------------------------------------------- status


@dataclass
class BackendStatus:
    """Which coding runtime can answer, and the next step for each when none can."""

    backend: str
    available: bool
    detail: str = ""
    hint: str = ""

    def summary(self) -> str:
        if self.available:
            return f"Coding model ready ({self.backend})."
        return self.detail or "No coding model is ready."


def coder_backend_status(
    *,
    root: Path | str | None = None,
    client: CoderClient | None = None,
    probe_ollama: bool = True,
) -> BackendStatus:
    """Report which runtime can answer, preferring the in-process model.

    The in-process model is preferred because it needs no background daemon. When
    it cannot run, an existing Ollama setup is still honoured, so a user who
    already pulled a model there is not told to download a second copy.
    """
    if client is not None:
        status = coder_status(client)
        return BackendStatus("injected", status.available, detail=status.summary())

    model_file = find_model_file(root)
    if model_file is not None and extra_available():
        return BackendStatus(
            "in-process",
            True,
            detail=f"In-process coding model ready: {model_file.name}.",
        )

    directory = _coder_dir(root)
    if model_file is not None and not extra_available():
        return BackendStatus(
            "in-process",
            False,
            detail=(
                f"Found {model_file.name} but the coder extra is not installed."
            ),
            hint=pip_install_command(FEATURE_CODER),
        )
    if extra_available():
        return BackendStatus(
            "in-process",
            False,
            detail=f"No coding model file in {directory}.",
            hint="python -m cli studio download --approve",
        )

    if probe_ollama:
        status = coder_status(OllamaCoderClient())
        if status.available:
            return BackendStatus("ollama", True, detail=status.summary())
        ollama_detail = status.summary()
    else:
        ollama_detail = ""

    detail = (
        f"No in-process coding model: {directory} is empty and the coder extra "
        f"is not installed. Install it with `{pip_install_command(FEATURE_CODER)}` "
        "and download a model with `python -m cli studio download --approve`."
    )
    if ollama_detail:
        detail += f" Ollama also cannot answer: {ollama_detail}"
    return BackendStatus("none", False, detail=detail, hint=pip_install_command(FEATURE_CODER))


# -------------------------------------------------------------------- download


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Any = None,
) -> Path:
    """Download a coding model's weights into the coder folder.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Returns the model file path.
    """
    spec = resolve_spec(spec_key)
    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({DOWNLOAD_TOOL}),
        )
        approvals = approvals or _approve(DOWNLOAD_TOOL)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, DOWNLOAD_TOOL)
    if not decision.allowed:
        raise CoderModelUnavailable(f"Downloading {spec.key!r} is denied by policy ({decision.reason}).")
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise CoderModelUnavailable(
            f"Downloading {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _coder_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target

    try:
        import huggingface_hub
    except ImportError as error:
        raise CoderModelUnavailable(
            f"Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_CODER)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…")
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:  # noqa: BLE001 - surfaced as an honest message
        raise CoderModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
    _write_registry(directory, Path(path).name, spec.key)
    if progress is not None:
        progress(f"Saved {spec.title} to {path}.")
    return Path(path)


def _write_registry(directory: Path, filename: str, spec_key: str) -> None:
    """Record which model file to load, so the newest download wins."""
    try:
        (directory / REGISTRY_FILENAME).write_text(
            json.dumps({"file": filename, "spec": spec_key}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _approve(tool: str) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


__all__ = [
    "CATALOG",
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_TOOL",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "REGISTRY_FILENAME",
    "BackendStatus",
    "CoderModelSpec",
    "CoderModelUnavailable",
    "CoderRunner",
    "LlamaCppCoderClient",
    "LlamaCppRunner",
    "coder_backend_status",
    "download_model",
    "extra_available",
    "find_model_file",
    "model_is_cached",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
