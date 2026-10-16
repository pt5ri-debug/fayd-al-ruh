"""Vision -- the project's *petroleum* local language model.

The project already has three mouths. The grounded assistant answers only from
the petroleum corpus it was seeded with; :mod:`ai.mind` speaks the small
in-project checkpoint; and :mod:`core.ultron` runs a *general* advanced model
that talks about every field of life. **Vision** is a fourth, deliberately
different mouth: a real, advanced model that runs **locally** and specialises in
**petroleum engineering** -- reservoir, drilling, production, petrophysics, PVT,
facilities -- built on the OGAI Reasoner weights (a DeepSeek-architecture
engineering system for oil and gas operations).

It is wired in **directly below Ultron** in the numbered menu, the CLI
(``python -m cli vision``), and the GUI sidebar.

Design, matching the project's hard rules:

* **Local only.** The model is a GGUF file that runs on this machine through
  ``llama-cpp-python``; nothing is sent to a remote server. The model lives under
  the writable data folder (:func:`core.paths.vision_models_dir`,
  ``data/vision`` by default), git-ignored, never in the repository -- so a
  multi-gigabyte model never dirties the working tree or bloats a program
  update. Drop any ``.gguf`` there and Vision uses it; ``python -m cli vision
  download`` fetches a known-good one instead.
* **Optional extra.** ``llama_cpp`` is imported lazily, only when a reply is
  actually generated, so a base install keeps working and importing this module
  never pulls a heavy dependency. When it (or the model file) is missing,
  :class:`VisionUnavailable` carries a clear, actionable message. It shares
  Ultron's ``[ultron]`` runtime extra: the same ``llama-cpp-python`` +
  ``huggingface_hub`` pair runs both mouths.
* **Human approval for the download.** Downloading multi-gigabyte weights is an
  external action that leaves the machine, so it runs only through the project's
  permission and human-approval gates (:mod:`core.permissions`,
  :mod:`core.human_approval`), denied by default, exactly like a web search or
  the image model. A model already on disk needs no download and no approval.
* **Honesty.** Vision's answers are its own; they are not retrieved from the
  project's corpus and carry no citations, so every answer is labelled as coming
  from a petroleum local model rather than from verified sources.

The heavy work is delegated to an injectable :class:`VisionRunner`; the default
runner is a thin ``llama-cpp-python`` wrapper, so a test can supply its own
runner and exercise the whole engine without the extra or a model file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_ULTRON, pip_install_command
from core.paths import vision_models_dir
from core.permissions import Capability, PermissionPolicy

#: The assistant's name, used in every label and status line.
NAME = "Vision"

#: The external-tool name the permission/approval gates see for a download.
DOWNLOAD_TOOL = "vision_model_download"

#: The file that records which model file Vision should load (written on download).
REGISTRY_FILENAME = "model.json"

#: The standing instruction given to the model. Vision is a petroleum-engineering
#: specialist, so it is asked to answer as one -- with the units, the formula, and
#: the field practice -- and to admit uncertainty rather than invent a number.
SYSTEM_PROMPT = (
    "You are Vision, a petroleum-engineering specialist AI. You answer questions "
    "about reservoir engineering, drilling, production, petrophysics, PVT, "
    "well testing, and oil and gas facilities with technical depth. Give concrete, "
    "quantitative answers, name the formula and the units you use, and follow "
    "recognised industry practice (SPE, API, ISO). When a question is outside "
    "petroleum engineering, say so plainly instead of guessing. Be accurate and "
    "honest about uncertainty, and answer in the user's language."
)

#: The honesty line shown above every Vision answer. It must stay truthful: the
#: answer is the local model's own prose, not a cited extract from the corpus.
HEADER = "[vision] answered by the local petroleum model; not grounded in cited sources."

#: A short explanation for the menu and the help text.
BLURB = "Petroleum-engineering local model: reservoir, drilling, production, and PVT, with no subject filter."

#: Defaults chosen to be usable on a CPU-only machine with a modest model.
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_GPU_LAYERS = 0
DEFAULT_THREADS = 0  # 0 lets llama.cpp pick a sensible count.

#: A safe ceiling for a single reply, so a mistyped request cannot run forever.
MAX_TOKENS_CEILING = 4096
#: A safe ceiling for the context window (RAM grows with it).
MAX_CONTEXT = 32768


class VisionUnavailable(RuntimeError):
    """Raised when Vision cannot run (no extra, no model file, no weights)."""


@dataclass(frozen=True)
class VisionModelSpec:
    """A known-good petroleum model that ``vision download`` can fetch.

    A spec bundles the Hugging Face repo, the exact GGUF filename, and a short
    description, so "a petroleum-engineering model" is a named choice rather than
    a URL the user has to assemble.
    """

    key: str
    title: str
    repo: str
    filename: str
    params: str
    approx_size_gb: float
    summary: str
    #: When true the Hugging Face repo is gated: the user must accept its licence
    #: and be authenticated before the weights can be fetched. Kept honest so the
    #: CLI can say exactly what the next step is instead of failing with a 401.
    gated: bool = False


#: The download catalog, keyed by name. ``ogai-reasoner`` is the default: the
#: OGAI Reasoner engineering model this mouth is built on (petroleum-specific).
CATALOG: dict[str, VisionModelSpec] = {
    "ogai-reasoner": VisionModelSpec(
        key="ogai-reasoner",
        title="OGAI Reasoner (petroleum, default)",
        repo="GainEnergy/ogai-reasoner",
        filename="ogai-reasoner.gguf",
        params="7B",
        approx_size_gb=4.7,
        summary="Oil-and-gas engineering specialist (DeepSeek architecture); the default petroleum model.",
        gated=True,
    ),
    "ogai-8x7b": VisionModelSpec(
        key="ogai-8x7b",
        title="OGAI 8x7b (petroleum, larger)",
        repo="GainEnergy/OGAI-8x7b-Q4_K_M-GGUF",
        filename="ogai-8x7b-q4_k_m.gguf",
        params="8x7B MoE",
        approx_size_gb=28.4,
        summary="The larger OGAI mixture-of-experts model; needs a lot of RAM.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "ogai-reasoner"


def resolve_spec(name: str | None) -> VisionModelSpec:
    """The :class:`VisionModelSpec` for ``name`` (or the default). Raises on a bad name."""
    key = (name or DEFAULT_SPEC_KEY).strip().lower()
    if key in CATALOG:
        return CATALOG[key]
    raise ValueError(
        f"Unknown Vision model {name!r}; choose one of {', '.join(CATALOG)}."
    )


def spec_names() -> tuple[str, ...]:
    """The catalog keys, in declaration order."""
    return tuple(CATALOG)


def spec_summaries() -> tuple[tuple[str, str, float, str], ...]:
    """``(key, title, approx_size_gb, summary)`` for every catalog entry."""
    return tuple(
        (spec.key, spec.title, spec.approx_size_gb, spec.summary) for spec in CATALOG.values()
    )


class VisionRunner(Protocol):
    """The seam that runs the model: a chat call and an optional streamed one.

    The default runner is a ``llama-cpp-python`` wrapper; a test supplies its own
    so the engine is exercised without the extra or a model file.
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


# --------------------------------------------------------------------- files


def _vision_dir(root: Path | str | None) -> Path:
    return Path(root) if root is not None else vision_models_dir()


def _registry_file(root: Path | str | None) -> Path:
    return _vision_dir(root) / REGISTRY_FILENAME


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The model file Vision should load, or ``None`` when none is present.

    Prefers the file named in the registry (``model.json``, written by
    ``download``), then any single ``.gguf`` in the folder. When several ``.gguf``
    files exist with no registry, the largest is chosen -- the biggest file is
    almost always the highest-quality model the user meant to use.
    """
    directory = _vision_dir(root)
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
    """True when at least one Vision model file is present on disk."""
    return find_model_file(root) is not None


# ------------------------------------------------------------------- extras


def _require_extra() -> Any:
    """Import ``llama_cpp`` or raise a clear :class:`VisionUnavailable`."""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise VisionUnavailable(
            f"Vision needs the optional runtime `llama-cpp-python`. {pip_install_command(FEATURE_ULTRON)}"
        ) from error
    return llama_cpp


def _require_hub() -> Any:
    """Import ``huggingface_hub`` or raise a clear :class:`VisionUnavailable`."""
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise VisionUnavailable(
            "Downloading a Vision model needs `huggingface_hub`. "
            f"{pip_install_command(FEATURE_ULTRON)}"
        ) from error
    return huggingface_hub


# ------------------------------------------------------------------- engine


class VisionEngine:
    """Runs the local petroleum model and returns its replies.

    The engine is a thin, honest wrapper: it builds the chat messages (the
    standing system instruction plus the conversation so far), passes them to the
    runner, and returns the model's text. It never invents a capability -- with no
    runner, no extra, or no model file it raises :class:`VisionUnavailable`.
    """

    name = NAME

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        root: Path | str | None = None,
        runner: VisionRunner | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        context: int = DEFAULT_CONTEXT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._model_path = Path(model_path) if model_path is not None else None
        self.system_prompt = system_prompt
        if not 1 <= context <= MAX_CONTEXT:
            raise ValueError(f"context must be in [1, {MAX_CONTEXT}].")
        if not 1 <= max_tokens <= MAX_TOKENS_CEILING:
            raise ValueError(f"max_tokens must be in [1, {MAX_TOKENS_CEILING}].")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2].")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        self.context = context
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_layers = gpu_layers
        self.threads = threads
        self._runner = runner

    @property
    def model_path(self) -> Path | None:
        """The model file that will be loaded, or ``None`` when none is present."""
        if self._model_path is not None:
            return self._model_path
        return find_model_file(self._root)

    def is_available(self) -> bool:
        """True when a reply could be produced now (an injected runner always can)."""
        if self._runner is not None:
            return True
        path = self.model_path
        if path is None or not path.is_file():
            return False
        try:
            _require_extra()
        except VisionUnavailable:
            return False
        return True

    def _active_runner(self) -> VisionRunner:
        if self._runner is not None:
            return self._runner
        path = self.model_path
        if path is None or not path.is_file():
            raise VisionUnavailable(
                "No Vision model found. Put a `.gguf` file in "
                f"{_vision_dir(self._root)} or run `python -m cli vision download --approve`."
            )
        self._runner = LlamaCppRunner(
            path,
            context=self.context,
            gpu_layers=self.gpu_layers,
            threads=self.threads,
        )
        return self._runner

    def build_messages(
        self, history: Sequence[tuple[str, str]], prompt: str
    ) -> list[dict[str, str]]:
        """Assemble the chat messages: the system prompt, ``history``, then ``prompt``.

        ``history`` is a sequence of ``(role, text)`` pairs with ``role`` one of
        ``"user"``/``"assistant"``; the newest turn is ``prompt``.
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for role, text in history:
            if role not in {"user", "assistant"} or not text.strip():
                continue
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": prompt})
        return messages

    def reply(
        self,
        prompt: str,
        *,
        history: Sequence[tuple[str, str]] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> str:
        """One non-streamed reply to ``prompt`` in the context of ``history``."""
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Vision needs a question to answer.")
        messages = self.build_messages(history, text)
        runner = self._active_runner()
        result = runner.chat(
            messages,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return result.strip()

    def stream_reply(
        self,
        prompt: str,
        *,
        history: Sequence[tuple[str, str]] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> Iterator[str]:
        """Yield a reply piece by piece when the runner can stream, else in one piece."""
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Vision needs a question to answer.")
        messages = self.build_messages(history, text)
        runner = self._active_runner()
        options = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        streamer = getattr(runner, "stream", None)
        if callable(streamer):
            yield from streamer(messages, **options)
            return
        yield runner.chat(messages, **options)

    def describe(self) -> str:
        """A one-line, honest description of the model in use."""
        path = self.model_path
        if path is None:
            return f"{NAME}: no local model file yet."
        return f"{NAME}: {path.name} ({path.stat().st_size / 1e9:.1f} GB) at {path.parent}."


class LlamaCppRunner:
    """The default runner: a thin ``llama-cpp-python`` chat wrapper.

    ``llama_cpp`` is imported here, lazily, so the module (and the whole CLI) loads
    without the extra. The GGUF's own chat template is used by
    ``create_chat_completion``, so the model's expected prompt format is honoured
    without the project hard-coding one.
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

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        for chunk in self._llama.create_chat_completion(messages=messages, stream=True, **kwargs):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece


def _content_of(result: Any) -> str:
    """The message text of a non-streamed chat completion, defensively."""
    try:
        return str(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:  # pragma: no cover - malformed runner
        raise VisionUnavailable(f"The model returned an unreadable response: {error}") from error


# ----------------------------------------------------------------- download


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download a petroleum model's weights into the Vision folder.

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
        raise VisionUnavailable(f"Downloading {spec.key!r} is denied by policy ({decision.reason}).")
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise VisionUnavailable(f"Downloading {spec.key!r} needs human approval ({approval.reason}).")

    directory = _vision_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target
    hub = _require_hub()
    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(
            f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…"
        )
    try:
        path = hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:  # noqa: BLE001 - surfaced as an honest message
        hint = (
            " The repo is gated: accept its licence on Hugging Face and sign in "
            "(`huggingface-cli login`) first."
            if spec.gated
            else ""
        )
        raise VisionUnavailable(f"Could not download {spec.key!r}: {error}.{hint}") from error
    _write_registry(directory, Path(path).name, spec.key)
    if progress is not None:
        progress(f"Saved {spec.title} to {path}.")
    return Path(path)


def _write_registry(directory: Path, filename: str, spec_key: str) -> None:
    """Record which model file Vision should load, so the newest download wins."""
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
    "BLURB",
    "CATALOG",
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DOWNLOAD_TOOL",
    "HEADER",
    "LlamaCppRunner",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "NAME",
    "REGISTRY_FILENAME",
    "SYSTEM_PROMPT",
    "VisionEngine",
    "VisionModelSpec",
    "VisionRunner",
    "VisionUnavailable",
    "download_model",
    "find_model_file",
    "model_is_cached",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
