"""Natasha's model layer: the local model she speaks through, GGUF or Ollama.

Every mouth in Fayd Al-Ruh names its model in one place: :mod:`core.ultron`,
:mod:`core.vision_model`, :mod:`core.banner`, :mod:`core.cloud_model`, and
:mod:`core.presentation_model` each own a catalog, a client seam, an honest status,
and a governed download. **Natasha** is the project's *companion*
(:mod:`companion.companion_agent`), and this module is her counterpart: it owns the
model she talks through, the choices a user can fetch, the honest report of what is
installed, and the one governed way to fetch a model.

She runs the model **in this process** by default -- a **GGUF file** loaded with
``llama-cpp-python``, the same runtime :mod:`core.ultron`, :mod:`core.vision_model`,
:mod:`core.banner`, and the two Studios use. No ``ollama serve`` daemon has to sit
in the background holding RAM while she is idle, and nothing is sent off the
machine. The file lives under the writable data folder
(:func:`core.paths.natasha_models_dir`, ``data/natasha``), git-ignored, so a
multi-gigabyte model never dirties the working tree.

An existing **Ollama** server is still honoured as a fallback: the project already
has a standard-library Ollama transport (:class:`core.coder_model.OllamaCoderClient`),
so it is *reused* rather than re-implemented, exactly as the other mouths reuse a
shared transport. The two runtimes sit side by side and satisfy the same
:class:`NatashaModelClient` protocol, so the companion cannot tell them apart.

Three things stay honest, as everywhere else in the project:

* :class:`NatashaModelClient` is the seam the companion talks to; the default
  in-process :class:`LlamaCppNatashaClient` and the fallback
  :class:`OllamaNatashaClient` both satisfy it, and a test injects its own, so the
  whole companion is exercised with no server and no weights.
* :func:`natasha_model_status` reports *exactly* what is installed: which runtime
  can answer, whether the weights are on disk (or the model is pulled), and the
  real next step when nothing can. It never pretends a model is present.
* :func:`download_model` (the GGUF weights) and :func:`pull_model` (an Ollama tag)
  both leave the machine, so both run only through the project's permission and
  human-approval gates, denied by default -- fetching several gigabytes is an
  external action, like every other model download.

Nothing here executes model output: the model returns text, and the companion's own
program logic (memory, tone, sessions) decides what to keep.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.coder_model import CoderMessage, CoderUnavailable, OllamaCoderClient
from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_NATASHA, pip_install_command
from core.paths import natasha_models_dir
from core.permissions import Capability, PermissionPolicy

from companion.personality import NAME

#: The model Natasha speaks through by default, named as an Ollama-style tag so a
#: status line reads the same whichever runtime answers. A small local Qwen3 model
#: that runs on a laptop CPU with a few gigabytes of RAM, and the one the project
#: already targets for local chat.
DEFAULT_MODEL = "qwen3:4b"

#: The local Ollama server address, used by the fallback runtime. The same default
#: the rest of the project uses (:data:`core.coder_model.DEFAULT_OLLAMA_URL`).
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: The command that pulls the default model into Ollama, shown verbatim when it is
#: missing there.
PULL_HINT = f"ollama pull {DEFAULT_MODEL}"

#: The command that fetches the default model's GGUF weights, shown when no model
#: file is present yet.
DOWNLOAD_HINT = "python -m cli natasha download --approve"

#: The external-tool names the permission/approval gates see. One per runtime, so a
#: policy can allow a GGUF download without also allowing an Ollama pull.
DOWNLOAD_TOOL = "natasha_model_download"
PULL_TOOL = "natasha_model_pull"

#: The registry file written after a download, naming the model file to load.
REGISTRY_FILENAME = "model.json"

#: The two runtimes, plus the "pick for me" value.
BACKEND_GGUF = "gguf"
BACKEND_OLLAMA = "ollama"
BACKEND_AUTO = "auto"

#: Accepted spellings of each runtime, so a user can write what reads naturally.
_BACKEND_ALIASES: dict[str, str] = {
    "": BACKEND_AUTO,
    "auto": BACKEND_AUTO,
    "gguf": BACKEND_GGUF,
    "in-process": BACKEND_GGUF,
    "llama-cpp": BACKEND_GGUF,
    "llamacpp": BACKEND_GGUF,
    "ollama": BACKEND_OLLAMA,
}

#: The canonical runtime names, for a menu and for validation.
BACKEND_NAMES: tuple[str, ...] = (BACKEND_AUTO, BACKEND_GGUF, BACKEND_OLLAMA)

#: Context window for the in-process model. A companion prompt carries the system
#: block, the remembered facts, and a bounded window of turns, so this is generous.
DEFAULT_CONTEXT = 8192

#: A safe ceiling on the context, because RAM grows with it.
MAX_CONTEXT = 32768

#: A safe ceiling on one reply, so a mistyped run cannot generate forever.
MAX_TOKENS_CEILING = 4096

#: How many threads to use. ``0`` lets llama.cpp pick a sensible count.
DEFAULT_THREADS = 0

#: Offload to the GPU only when the user asks; the default is CPU-only so Natasha
#: works on a plain laptop.
DEFAULT_GPU_LAYERS = 0


class CompanionModelUnavailable(CoderUnavailable):
    """Natasha's model cannot run (no runtime, no weights, no server, or no pull)."""


@dataclass(frozen=True)
class CompanionModelSpec:
    """A known-good model that ``natasha download`` (GGUF) or ``pull_model`` can fetch.

    A spec bundles both forms of the same model -- the Ollama tag *and* the Hugging
    Face repo with its exact GGUF filename -- so "the model Natasha speaks through"
    is a named choice, and switching runtime does not change which model she is.
    Sizes are the Q4 quantisation, the balance of quality and RAM that fits a laptop.
    """

    key: str
    title: str
    tag: str
    repo: str
    filename: str
    params: str
    approx_size_gb: float
    summary: str


#: The download catalog, keyed by name. ``qwen3-4b`` is the default: it reasons
#: before it answers, so it holds a warm conversation well, and it fits a modest
#: machine. ``qwen3-8b`` is stronger but heavier; ``qwen2.5-7b`` is a
#: non-reasoning model, so it answers faster with no hidden reasoning trace. The
#: GGUF repos are the ones the project already ships for Ultron, Vision, and the
#: Presentation Studio (Apache-2.0), so a user who fetched one downloads nothing new.
CATALOG: dict[str, CompanionModelSpec] = {
    "qwen3-4b": CompanionModelSpec(
        key="qwen3-4b",
        title="Qwen3 4B (default)",
        tag="qwen3:4b",
        repo="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        params="4B",
        approx_size_gb=2.6,
        summary="Warm and capable on a modest machine; reasons before it answers.",
    ),
    "qwen3-8b": CompanionModelSpec(
        key="qwen3-8b",
        title="Qwen3 8B",
        tag="qwen3:8b",
        repo="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        params="8B",
        approx_size_gb=5.0,
        summary="Stronger and more coherent; needs more RAM and time.",
    ),
    "qwen2.5-7b": CompanionModelSpec(
        key="qwen2.5-7b",
        title="Qwen2.5 7B (no hidden reasoning)",
        tag="qwen2.5:7b",
        repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        params="7B",
        approx_size_gb=4.7,
        summary="No reasoning trace, so replies come back faster and shorter.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen3-4b"


def resolve_spec(name: str | None) -> CompanionModelSpec:
    """The spec for ``name``, or the default when it is empty. Raises on a bad name."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise CompanionModelUnavailable(
            f"Unknown companion model {name!r}. Available: {', '.join(sorted(CATALOG))}."
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


def model_present(model: str, models: Sequence[str]) -> bool:
    """True when ``model`` is pulled, accepting Ollama's implicit ``:latest`` tag.

    Only an exact match (or the same name with the implicit ``:latest`` tag)
    counts: reporting a *different* size of the same family as present would be a
    lie the user would only discover when the run failed.
    """
    wanted = model.strip()
    return wanted in models or (":" not in wanted and f"{wanted}:latest" in models)


# ---------------------------------------------------------------------- extras


def extra_available() -> bool:
    """True when ``llama_cpp`` can be imported, without importing it here."""
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _require_extra() -> Any:
    """Import ``llama_cpp`` or raise a clear :class:`CompanionModelUnavailable`."""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError as error:
        raise CompanionModelUnavailable(
            "Natasha's in-process model needs the optional natasha extra. Install it "
            f"with: {pip_install_command(FEATURE_NATASHA)}"
        ) from error
    return llama_cpp


# ----------------------------------------------------------------- model files


def _natasha_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else natasha_models_dir()


def _registry_file(root: Path | str | None = None) -> Path:
    return _natasha_dir(root) / REGISTRY_FILENAME


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The model file Natasha should load, or ``None`` when none is present.

    Prefers the file named in the registry (``model.json``, written by
    ``download``), then any single ``.gguf`` in the folder. When several exist with
    no registry, the largest is chosen -- the biggest file is almost always the
    highest-quality model the user meant to use.
    """
    directory = _natasha_dir(root)
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
    """True when at least one model file is present on disk."""
    return find_model_file(root) is not None


# ---------------------------------------------------------------------- client


@runtime_checkable
class NatashaModelClient(Protocol):
    """What the companion needs from a model: a chat call and a model list.

    The same shape as :class:`core.coder_model.CoderClient`, satisfied by both the
    in-process :class:`LlamaCppNatashaClient` and the fallback
    :class:`OllamaNatashaClient`, and by a test's own client.
    """

    model: str

    def chat(self, messages: Sequence[CoderMessage], *, temperature: float = 0.2) -> str: ...

    def list_models(self) -> tuple[str, ...]: ...


class OllamaNatashaClient(OllamaCoderClient):
    """The fallback client: the project's Ollama transport, named for Natasha.

    It *is* :class:`core.coder_model.OllamaCoderClient` -- the same
    standard-library HTTP, the same ``/api/tags`` and ``/api/chat`` calls, the same
    optional ``max_tokens``/``think`` settings -- so no second Ollama client is
    written. The subclass only marks the seam for the companion and keeps the
    honest status line specific to her.
    """

    #: Marks the runtime for a status line, matching the other mouths.
    backend = "ollama"


class NatashaRunner(Protocol):
    """The seam that runs the in-process model: one chat call.

    The default runner is a ``llama-cpp-python`` wrapper; a test supplies its own
    so the engine is exercised without the extra or a model file.
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class LlamaCppNatashaRunner:
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


#: A reasoning model may wrap its answer in a thinking block. Matched
#: case-insensitively and non-greedily, so several blocks are all removed.
_THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)

#: An unterminated opening tag (the model ran out of tokens mid-thought) leaves
#: everything after it as reasoning, so the tail is dropped too.
_OPEN_THINKING = re.compile(r"<think(?:ing)?>.*\Z", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's `` thinking`` blocks, leaving the answer.

    Qwen3 and other reasoning models emit their chain of thought before the
    answer. In Ollama that trace is usually kept out of the reply by the ``think``
    setting, but not on every build; in-process there is no such switch at all. So
    the companion applies this to every reply, whichever runtime produced it -- the
    trace is not something Natasha said, and she stores and shows replies verbatim.
    An unterminated block (the model ran out of tokens mid-thought) is dropped to
    the end too.

    A reply that was *only* reasoning leaves the empty string, deliberately: the
    caller then reports an empty reply honestly, rather than showing the user the
    model's raw inner monologue as though Natasha had said it.
    """
    return _OPEN_THINKING.sub("", _THINKING.sub("", text)).strip()


def _content_of(result: Any) -> str:
    """Pull the assistant text out of llama.cpp's reply, defensively."""
    try:
        choices = result["choices"]
        message = choices[0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as error:
        raise CompanionModelUnavailable(
            "The in-process companion model returned an unexpected reply shape."
        ) from error
    text = str(content or "").strip()
    if not text:
        raise CompanionModelUnavailable(
            "The in-process companion model returned an empty reply."
        )
    return text


class LlamaCppNatashaClient:
    """A :class:`NatashaModelClient` backed by a local GGUF file.

    Satisfies the same protocol as :class:`OllamaNatashaClient`, so the companion
    cannot tell the two runtimes apart. The model is loaded once, on the first
    generation, and reused -- so a conversation pays the load cost once and nothing
    sits in RAM before the first reply.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: str = DEFAULT_MODEL,
        root: Path | str | None = None,
        runner: NatashaRunner | None = None,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        #: Marks this client for :func:`natasha_model_status`, so the status line
        #: never tells an in-process user to start a server.
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
                raise CompanionModelUnavailable(
                    f"The companion model file is missing: {self._path}"
                )
            return self._path
        found = find_model_file(self.root)
        if found is None:
            directory = _natasha_dir(self.root)
            raise CompanionModelUnavailable(
                f"No companion model file in {directory}. Download one with "
                f"`{DOWNLOAD_HINT}`, or drop a `.gguf` file there."
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
        except CompanionModelUnavailable:
            return False

    def missing_detail(self) -> str:
        """Why the in-process model cannot answer, in one actionable line."""
        if not extra_available():
            return (
                "Natasha's in-process model needs the optional natasha extra. "
                f"Install it with: {pip_install_command(FEATURE_NATASHA)}"
            )
        return (
            f"No companion model file in {_natasha_dir(self.root)}. Download one "
            f"with `{DOWNLOAD_HINT}`, or drop a `.gguf` file there."
        )

    def _load(self) -> NatashaRunner:
        """Load the runner on first use, so construction never loads a model."""
        if self._runner is None:
            self._runner = LlamaCppNatashaRunner(
                self.model_path(),
                context=self.context,
                gpu_layers=self.gpu_layers,
                threads=self.threads,
            )
        return self._runner

    # --- NatashaModelClient ---------------------------------------------
    def list_models(self) -> tuple[str, ...]:
        """The model files present, so ``status`` can report honestly.

        The in-process runtime has no server to ask, so this reports what is on
        disk. An absent file yields an empty tuple rather than raising, so
        :func:`natasha_model_status` reads it as "not installed".
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
            raise CompanionModelUnavailable(
                f"The in-process companion model failed: {error}"
            ) from error


# ---------------------------------------------------------------------- status


@dataclass
class NatashaModelStatus:
    """An honest report of whether Natasha's model can answer right now.

    The shape matches the other mouths' status objects, and the wording is
    Natasha's own: a missing runtime, a missing *model file*, a missing server, and
    a missing *Ollama tag* are reported separately, because the next step differs.
    """

    available: bool
    server_reachable: bool
    model: str
    installed: bool
    backend: str = BACKEND_OLLAMA
    models: tuple[str, ...] = ()
    detail: str = ""
    hint: str = PULL_HINT

    def summary(self) -> str:
        """A one-line, plain-language status for the CLI and the window."""
        if self.available:
            where = "in-process" if self.backend == "in-process" else "Ollama"
            return f"{NAME} is ready ({self.model}, {where})."
        if self.backend == "in-process":
            return self.detail or f"{NAME} has no local model file yet."
        if not self.server_reachable:
            return (
                f"Natasha cannot reach Ollama at the configured address. "
                f"Start it with `ollama serve`. ({self.detail})"
            )
        return (
            f"Ollama is running but `{self.model}` is not installed. "
            f"Install it with `ollama pull {self.model}`."
        )


def resolve_backend(
    name: str | None = None,
    *,
    root: Path | str | None = None,
    model_file: str | Path | None = None,
) -> str:
    """Which runtime Natasha will use: ``"gguf"`` or ``"ollama"``.

    ``"auto"`` (the default) prefers the in-process GGUF model when the extra is
    installed and a model is available -- either an explicitly named ``model_file``
    or one found under ``root`` -- because it needs no background daemon; otherwise
    it falls back to Ollama, so a user who already has a server running is never
    blocked. An explicit ``"gguf"``/``"ollama"`` is honoured as asked; an unknown
    name is refused rather than silently ignored.
    """
    key = (name or BACKEND_AUTO).strip().lower()
    if key not in _BACKEND_ALIASES:
        raise CompanionModelUnavailable(
            f"Unknown backend {name!r}. Choose 'auto', 'gguf', or 'ollama'."
        )
    chosen = _BACKEND_ALIASES[key]
    if chosen != BACKEND_AUTO:
        return chosen
    has_model = model_file is not None or find_model_file(root) is not None
    if extra_available() and has_model:
        return BACKEND_GGUF
    return BACKEND_OLLAMA


def build_client(
    *,
    backend: str | None = None,
    model: str = DEFAULT_MODEL,
    model_file: str | Path | None = None,
    root: Path | str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 600.0,
    max_tokens: int = 4096,
    think: bool = True,
    context: int = DEFAULT_CONTEXT,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
    threads: int = DEFAULT_THREADS,
) -> NatashaModelClient:
    """Build the client for the resolved runtime.

    The one place the choice is made, so the CLI, the chat mouth, and the window
    cannot disagree about which runtime Natasha speaks through. Explicit arguments
    rather than a config object keep this module free of a cycle with
    :mod:`companion.config`, which reads its defaults from here.
    """
    if resolve_backend(backend, root=root, model_file=model_file) == BACKEND_GGUF:
        return LlamaCppNatashaClient(
            model_file,
            model=model,
            root=root,
            context=context,
            gpu_layers=gpu_layers,
            threads=threads,
            max_tokens=max_tokens,
        )
    return OllamaNatashaClient(
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
        think=think,
    )


def natasha_model_status(
    *,
    client: NatashaModelClient | None = None,
    backend: str | None = None,
    root: Path | str | None = None,
    model: str = DEFAULT_MODEL,
) -> NatashaModelStatus:
    """Report whether Natasha's model can answer, without pretending.

    A missing runtime, a missing model file, a missing server, and a missing Ollama
    tag are reported separately, because the next step differs. An injected client
    (a test's) is honoured unchanged, and its own ``backend`` attribute decides how
    the report is worded.
    """
    if client is None:
        client = build_client(backend=backend, model=model, root=root)
    active = client
    kind = str(getattr(active, "backend", BACKEND_OLLAMA))
    try:
        models = active.list_models()
    except CoderUnavailable as error:
        return NatashaModelStatus(
            available=False,
            server_reachable=False,
            model=active.model,
            installed=False,
            backend=kind,
            detail=str(error),
        )
    installed = model_present(active.model, models)
    is_installed = getattr(active, "is_installed", None)
    if callable(is_installed):
        installed = bool(is_installed())
    if installed:
        detail = ""
    elif kind == "in-process":
        missing = getattr(active, "missing_detail", None)
        detail = missing() if callable(missing) else "No companion model file is present."
    else:
        detail = f"`{active.model}` is not pulled."
    hint = DOWNLOAD_HINT if kind == "in-process" else f"ollama pull {active.model}"
    return NatashaModelStatus(
        available=installed,
        server_reachable=True,
        model=active.model,
        installed=installed,
        backend=kind,
        models=models,
        detail=detail,
        hint=hint,
    )


# -------------------------------------------------------------------- download


#: Runs one command and returns ``(returncode, combined_output)``.
CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def _default_runner(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _gate(
    tool: str,
    *,
    approved: bool,
    permissions: PermissionPolicy | None,
    approvals: HumanApprovalGate | None,
) -> None:
    """Raise unless the permission policy *and* a human approval both allow ``tool``."""
    if approved:
        permissions = permissions or PermissionPolicy(
            allowed=frozenset(
                {
                    Capability.SYMBOLIC_SIMULATION,
                    Capability.SANDBOX_TASK,
                    Capability.EXTERNAL_TOOL,
                }
            ),
            allowed_external_tools=frozenset({tool}),
        )
        approvals = approvals or _approve(tool)
    policy = permissions or PermissionPolicy()
    gate = approvals or HumanApprovalGate()
    decision = policy.decide(Capability.EXTERNAL_TOOL, tool)
    if not decision.allowed:
        raise CompanionModelUnavailable(f"Denied by policy ({decision.reason}).")
    approval = gate.decide(tool)
    if not approval.approved:
        raise CompanionModelUnavailable(f"Needs human approval ({approval.reason}).")


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download a companion model's **GGUF weights** into the natasha folder.

    The default runtime is the in-process one, so this is the default fetch. The
    download leaves the machine, so it is treated as an external tool: denied by
    default, and it runs only when an approving :class:`PermissionPolicy` *and* a
    live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the CLI's
    convenience that records a human approval). Returns the model file path.
    """
    spec = resolve_spec(spec_key)
    _gate(DOWNLOAD_TOOL, approved=approved, permissions=permissions, approvals=approvals)

    directory = _natasha_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target

    try:
        import huggingface_hub
    except ImportError as error:
        raise CompanionModelUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_NATASHA)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…")
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:
        raise CompanionModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
    _write_registry(directory, Path(path).name, spec.key)
    if progress is not None:
        progress(f"Saved {spec.title} to {path}.")
    return Path(path)


def pull_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    runner: CommandRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Pull a companion model into the local **Ollama** server with ``ollama pull``.

    The fallback runtime. Fetching the weights leaves the machine, so it is
    governed exactly like :func:`download_model`: denied by default, and it runs
    only when an approving :class:`PermissionPolicy` *and* a live
    :class:`HumanApprovalGate` approval exist (or ``approved=True``). ``runner`` is
    the seam a test injects, so the gate is exercised with no real pull. Returns the
    Ollama tag that is now present.
    """
    spec = resolve_spec(spec_key)
    _gate(PULL_TOOL, approved=approved, permissions=permissions, approvals=approvals)

    command = ("ollama", "pull", spec.tag)
    run = runner or _default_runner
    if progress is not None:
        progress(f"Pulling {spec.title} ({spec.tag}, ~{spec.approx_size_gb:.1f} GB) with Ollama…")
    try:
        returncode, output = run(command)
    except OSError as error:
        raise CompanionModelUnavailable(
            "Could not run `ollama pull`: "
            f"{error}. Install Ollama and start it with `ollama serve`."
        ) from error
    if returncode != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else "ollama pull failed."
        raise CompanionModelUnavailable(f"Could not pull {spec.tag!r}: {tail}")
    if progress is not None:
        progress(f"Pulled {spec.tag}.")
    return spec.tag


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
    "BACKEND_AUTO",
    "BACKEND_GGUF",
    "BACKEND_NAMES",
    "BACKEND_OLLAMA",
    "CATALOG",
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_HINT",
    "DOWNLOAD_TOOL",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "NAME",
    "PULL_HINT",
    "PULL_TOOL",
    "REGISTRY_FILENAME",
    "CommandRunner",
    "CompanionModelSpec",
    "CompanionModelUnavailable",
    "LlamaCppNatashaClient",
    "LlamaCppNatashaRunner",
    "NatashaModelClient",
    "NatashaModelStatus",
    "NatashaRunner",
    "OllamaNatashaClient",
    "build_client",
    "download_model",
    "extra_available",
    "find_model_file",
    "model_is_cached",
    "model_present",
    "natasha_model_status",
    "pull_model",
    "resolve_backend",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
    "strip_thinking",
]
