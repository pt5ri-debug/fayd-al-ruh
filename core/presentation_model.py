"""The Presentation Studio's own authoring model: a local Qwen3 GGUF.

The Studio turns a discussion into an outline and an outline into slide bullets.
Those two jobs are *structured prose generation*, not retrieval: the general
assistant answers with cited extracts, which is exactly right for a grounded
answer but wrong for a slide -- the user asked for a title and three to five
bullets, and a retrieved paragraph is neither. So the Studio drafts with a model
of its own, chosen for the job: **Qwen3 8B Instruct**, strong at instruction
following and at producing the numbered/bulleted shape the Studio's deterministic
parsers already understand.

The runtime is the one the project already uses for its general mouth
(:mod:`core.ultron`) and the Code Studio (:mod:`core.coder_llama`): a **GGUF file
loaded straight into this process** with ``llama-cpp-python``. No server, no
background daemon holding RAM while the Studio is idle, and nothing sent off the
machine. Everything heavy is imported lazily, so the base install needs neither
the extra nor a model file, and this module imports with neither present.

Three things stay honest, as everywhere else in the project:

* :class:`PresentationModelClient` is the seam the Studio talks to, satisfied by
  both the in-process client and the Ollama client, so a test -- or a user with a
  different runtime -- can inject a replacement.
* :func:`presentation_model_status` reports *exactly* what is installed: whether
  the runtime can load a model, whether the weights are on disk, and the real
  next step when they are not.
* Nothing here executes model output. The model returns text; the Studio parses
  it with the same deterministic parsers it uses for any other answer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.coder_model import (
    CoderClient,
    CoderMessage,
    CoderUnavailable,
    OllamaCoderClient,
    coder_status,
)
from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_PRESENTATION, pip_install_command
from core.paths import presentation_models_dir
from core.permissions import Capability, PermissionPolicy

#: The registry file written after a download, naming the model to load.
REGISTRY_FILENAME = "model.json"

#: The tool name the download gate is keyed on.
DOWNLOAD_TOOL = "presentation-model-download"

#: The model the Studio drafts with by default, as an Ollama-style tag. The
#: in-process client carries the same name so the status line reads identically
#: whichever runtime is answering.
DEFAULT_PRESENTATION_MODEL = "qwen3:8b"

#: Context window for the authoring model. An outline or slide request carries the
#: deck title plus one heading, so a generous window costs little RAM.
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


class PresentationModelUnavailable(CoderUnavailable):
    """The Presentation Studio's authoring model cannot run (no extra, no weights)."""


@dataclass(frozen=True)
class PresentationModelSpec:
    """A known-good authoring model that ``presentation model-download`` can fetch.

    A spec bundles the Hugging Face repo, the exact GGUF filename, and a short
    description, so "a model for the Studio" is a named choice rather than a URL
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


#: The download catalog, keyed by name. ``qwen3-8b`` is the default the user
#: asked for; ``qwen3-4b`` is the lighter option for a machine with less RAM.
#: Both are the official Qwen GGUF repos (Apache-2.0).
CATALOG: dict[str, PresentationModelSpec] = {
    "qwen3-8b": PresentationModelSpec(
        key="qwen3-8b",
        title="Qwen3 8B Instruct (default)",
        repo="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        params="8B",
        approx_size_gb=5.0,
        summary="Strong instruction following; the default balance of quality and size.",
    ),
    "qwen3-4b": PresentationModelSpec(
        key="qwen3-4b",
        title="Qwen3 4B Instruct (light)",
        repo="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        params="4B",
        approx_size_gb=2.6,
        summary="Lighter and faster; good when RAM or time is tight.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen3-8b"


def resolve_spec(name: str | None) -> PresentationModelSpec:
    """The spec for ``name``, or the default when it is empty. Raises on a bad name."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise PresentationModelUnavailable(
            f"Unknown presentation model {name!r}. Available: {', '.join(sorted(CATALOG))}."
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


def _presentation_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else presentation_models_dir()


def _registry_file(root: Path | str | None = None) -> Path:
    return _presentation_dir(root) / REGISTRY_FILENAME


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The authoring model file to load, or ``None`` when none is present.

    Prefers the file named in the registry (``model.json``, written by
    ``download``), then any single ``.gguf`` in the folder. When several exist
    with no registry, the largest is chosen -- the biggest file is almost always
    the highest-quality model the user meant to use.
    """
    directory = _presentation_dir(root)
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
    """True when at least one authoring model file is present on disk."""
    return find_model_file(root) is not None


def extra_available() -> bool:
    """True when ``llama_cpp`` can be imported, without importing it here."""
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _require_extra() -> Any:
    """Import ``llama_cpp`` or raise a clear :class:`PresentationModelUnavailable`."""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError as error:
        raise PresentationModelUnavailable(
            "The in-process presentation model needs the optional presentation "
            f"extra. Install it with: {pip_install_command(FEATURE_PRESENTATION)}"
        ) from error
    return llama_cpp


# ---------------------------------------------------------------------- client


class PresentationRunner(Protocol):
    """The seam a test injects so no real model is needed."""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class LlamaCppPresentationRunner:
    """The default runner: a thin ``llama-cpp-python`` chat wrapper.

    ``llama_cpp`` is imported here, lazily, so the module loads without the extra.
    The GGUF's own chat template is used by ``create_chat_completion``, so the
    model's expected prompt format -- Qwen3's included -- is honoured without the
    project hard-coding one.
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
        raise PresentationModelUnavailable(
            "The in-process presentation model returned an unexpected reply shape."
        ) from error
    text = str(content or "").strip()
    if not text:
        raise PresentationModelUnavailable(
            "The in-process presentation model returned an empty reply."
        )
    return text


class LlamaCppPresentationClient:
    """A :class:`core.coder_model.CoderClient` backed by a local GGUF file.

    Satisfies the same protocol as :class:`core.coder_model.OllamaCoderClient`, so
    the Studio cannot tell the two runtimes apart. The model is loaded once, on
    the first generation, and reused -- so a Studio session pays the load cost
    once and nothing sits in RAM before the first request.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: str = DEFAULT_PRESENTATION_MODEL,
        root: Path | str | None = None,
        runner: PresentationRunner | None = None,
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
                raise PresentationModelUnavailable(
                    f"The presentation model file is missing: {self._path}"
                )
            return self._path
        found = find_model_file(self.root)
        if found is None:
            directory = _presentation_dir(self.root)
            raise PresentationModelUnavailable(
                f"No presentation model file in {directory}. Download one with "
                "`python -m cli presentation model-download --approve`, or drop a "
                "`.gguf` file there."
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
        except PresentationModelUnavailable:
            return False

    def missing_detail(self) -> str:
        """Why the in-process model cannot answer, in one actionable line."""
        if not extra_available():
            return (
                "The in-process presentation model needs the optional presentation "
                f"extra. Install it with: {pip_install_command(FEATURE_PRESENTATION)}"
            )
        return (
            f"No presentation model file in {_presentation_dir(self.root)}. Download "
            "one with `python -m cli presentation model-download --approve`, or drop "
            "a `.gguf` file there."
        )

    def _load(self) -> PresentationRunner:
        """Load the runner on first use, so construction never loads a model."""
        if self._runner is None:
            self._runner = LlamaCppPresentationRunner(
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
        except Exception as error:
            raise PresentationModelUnavailable(
                f"The in-process presentation model failed: {error}"
            ) from error


# ---------------------------------------------------------------------- status


@dataclass
class PresentationBackendStatus:
    """Which authoring runtime can answer, and the next step for each when none can."""

    backend: str
    available: bool
    detail: str = ""
    hint: str = ""

    def summary(self) -> str:
        if self.available:
            return f"Presentation model ready ({self.backend})."
        return self.detail or "No presentation model is ready."


@dataclass
class PresentationModelStatus:
    """An honest report of whether the Studio's authoring model can draft right now.

    Deliberately *not* the Code Studio's :class:`core.coder_model.CoderStatus`: its
    ``summary`` names a *coding* model, and reusing it would tell a presentation
    user about a coder. The shape is the same, so the two windows render a status
    line the same way, and the wording is this feature's own.
    """

    available: bool
    model: str
    backend: str
    installed: bool
    detail: str = ""
    hint: str = ""

    def summary(self) -> str:
        """A one-line, plain-language status for the window and the CLI."""
        if self.available:
            where = "in-process, no server" if self.backend == "in-process" else "Ollama"
            return f"Presentation model ready: {self.model} ({where})."
        return self.detail or "No presentation model is ready."


def presentation_backend_status(
    *,
    root: Path | str | None = None,
    client: CoderClient | None = None,
    probe_ollama: bool = True,
) -> PresentationBackendStatus:
    """Report which runtime can answer, preferring the in-process model.

    The in-process model is preferred because it needs no background daemon. When
    it cannot run, an existing Ollama setup is still honoured, so a user who
    already pulled ``qwen3:8b`` there is not told to download a second copy.
    """
    if client is not None:
        status = coder_status(client)
        return PresentationBackendStatus(
            "injected",
            status.available,
            detail=f"Presentation model ready ({client.model})." if status.available else status.detail,
        )

    model_file = find_model_file(root)
    if model_file is not None and extra_available():
        return PresentationBackendStatus(
            "in-process",
            True,
            detail=f"In-process presentation model ready: {model_file.name}.",
        )

    directory = _presentation_dir(root)
    if model_file is not None and not extra_available():
        return PresentationBackendStatus(
            "in-process",
            False,
            detail=f"Found {model_file.name} but the presentation extra is not installed.",
            hint=pip_install_command(FEATURE_PRESENTATION),
        )
    if extra_available():
        return PresentationBackendStatus(
            "in-process",
            False,
            detail=f"No presentation model file in {directory}.",
            hint="python -m cli presentation model-download --approve",
        )

    ollama_detail = ""
    if probe_ollama:
        status = coder_status(OllamaCoderClient(model=DEFAULT_PRESENTATION_MODEL))
        if status.available:
            return PresentationBackendStatus(
                "ollama", True, detail=f"Presentation model ready: {status.model} (Ollama)."
            )
        ollama_detail = (
            f" Ollama cannot answer either: `{status.model}` is not available there."
            if status.server_reachable
            else " Ollama is not running, so it cannot answer either."
        )

    detail = (
        f"No in-process presentation model: {directory} is empty and the presentation "
        f"extra is not installed. Install it with "
        f"`{pip_install_command(FEATURE_PRESENTATION)}` and download a model with "
        "`python -m cli presentation model-download --approve`."
    ) + ollama_detail
    return PresentationBackendStatus(
        "none", False, detail=detail, hint=pip_install_command(FEATURE_PRESENTATION)
    )


def presentation_model_status(
    *,
    root: Path | str | None = None,
    client: CoderClient | None = None,
    probe_ollama: bool = True,
) -> PresentationModelStatus:
    """An honest status of the Studio's authoring model, for the window and the CLI."""
    backend = presentation_backend_status(root=root, client=client, probe_ollama=probe_ollama)
    return PresentationModelStatus(
        available=backend.available,
        model=DEFAULT_PRESENTATION_MODEL,
        backend=backend.backend,
        installed=backend.available,
        detail=backend.detail,
        hint=backend.hint,
    )


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
    """Download an authoring model's weights into the presentation folder.

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
        raise PresentationModelUnavailable(
            f"Downloading {spec.key!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise PresentationModelUnavailable(
            f"Downloading {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _presentation_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target

    try:
        import huggingface_hub
    except ImportError as error:
        raise PresentationModelUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_PRESENTATION)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…")
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:
        raise PresentationModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
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
    "DEFAULT_PRESENTATION_MODEL",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_TOOL",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "REGISTRY_FILENAME",
    "LlamaCppPresentationClient",
    "LlamaCppPresentationRunner",
    "PresentationBackendStatus",
    "PresentationModelSpec",
    "PresentationModelStatus",
    "PresentationModelUnavailable",
    "PresentationRunner",
    "download_model",
    "extra_available",
    "find_model_file",
    "model_is_cached",
    "presentation_backend_status",
    "presentation_model_status",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
