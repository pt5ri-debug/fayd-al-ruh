"""Doc's Studio's own vision-language model: a local Qwen3-VL GGUF, in-process.

Doc's Studio reads *documents* -- text files, Word, PDFs, and images -- and lets
the user ask about what is inside them. Two of those jobs cannot be done by the
project's text models:

* a **scanned** PDF (or a photographed page) has no text layer at all, so there
  is nothing for a text model to read; and
* a question about a *table*, a *figure*, or a *layout* is a question about the
  page's appearance, not about its words.

So the Studio uses a model chosen for the job: **Qwen3-VL-4B-Instruct**, a
vision-language model that reads the page *image* as well as the extracted text.
It is the same runtime the project already uses for its other local models
(:mod:`core.ultron`, :mod:`core.vision_model`, :mod:`core.presentation_model`): a
**GGUF file loaded straight into this process** with ``llama-cpp-python``. No
server, no background daemon holding RAM while the Studio is idle, and nothing
sent off the machine.

A vision model needs **two** files, and this module keeps them together:

* the language model (``Qwen3VL-4B-Instruct-Q4_K_M.gguf``), and
* the **multimodal projector** (``mmproj-Qwen3VL-4B-Instruct-F16.gguf``), the
  small adapter that turns image patches into the language model's embedding
  space.

``llama-cpp-python`` drives the pair through its ``MTMDChatHandler`` (the
``mtmd`` library), so a message may carry an ``image_url`` alongside its text --
exactly the shape :mod:`core.docs_studio` builds. A model file without its
projector is a *partial* install, and the status says so rather than failing
later with an obscure error.

Three things stay honest, as everywhere else in the project:

* :class:`DocsRunner` is the seam the engine talks to, so a test -- or a user
  with a different runtime -- can inject a replacement and exercise the whole
  Studio with no extra and no weights.
* :func:`docs_model_status` reports *exactly* what is installed: whether the
  runtime can load a model, whether **both** files are on disk, and the real next
  step when they are not.
* Nothing here executes model output. The model returns text; the Studio shows it
  and labels it as the local model's own reading, never as a cited extract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_DOCS, pip_install_command
from core.paths import docs_models_dir
from core.permissions import Capability, PermissionPolicy

#: The name used in every label and status line.
NAME = "Doc's Studio model"

#: The registry file written after a download, naming the model to load.
REGISTRY_FILENAME = "model.json"

#: The tool name the download gate is keyed on.
DOWNLOAD_TOOL = "docs-model-download"

#: The model the Studio reads documents with by default, as a short tag. Kept in
#: the ``qwen3-vl-*`` family so the status line reads consistently.
DEFAULT_DOCS_MODEL = "qwen3-vl-4b"

#: Context window for a vision request. A document page plus its extracted text
#: and the question is a large prompt, and image tokens are numerous, so this is
#: generous; it is also the single biggest lever on RAM.
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

#: The prefix a multimodal projector file carries, so it can be told apart from
#: the language model it belongs to.
MMPROJ_PREFIX = "mmproj"

#: The honesty header shown above every model reading. It must stay truthful: the
#: answer is the local model's own reading, not a cited extract from the corpus.
HEADER = "[docs] read by the local vision model; not grounded in cited sources."

#: A short explanation for the menu and the help text.
BLURB = (
    "Read any document -- text, Word, PDF, scanned PDF, or an image -- with a "
    "local vision model, and ask questions about it."
)


class DocsModelUnavailable(RuntimeError):
    """The Studio's vision model cannot run (no extra, no weights, no projector)."""


@dataclass(frozen=True)
class DocsModelSpec:
    """A known-good vision-language model ``docs download`` can fetch.

    A spec bundles the Hugging Face repo, the exact GGUF filenames for **both**
    halves of a vision model, and a short description, so "a model for the
    Studio" is a named choice rather than a URL the user has to assemble. Sizes
    are the Q4 quantisation of the language model plus the F16 projector.
    """

    key: str
    title: str
    repo: str
    filename: str
    mmproj_filename: str
    params: str
    approx_size_gb: float
    summary: str


#: The download catalog, keyed by name. ``qwen3-vl-4b`` is the default the user
#: asked for; ``qwen3-vl-2b`` is the lighter option for a machine with less RAM,
#: and ``qwen3-vl-8b`` the heavier one for a machine that can afford it. All are
#: the official Qwen GGUF repos (Apache-2.0).
CATALOG: dict[str, DocsModelSpec] = {
    "qwen3-vl-2b": DocsModelSpec(
        key="qwen3-vl-2b",
        title="Qwen3-VL 2B Instruct (light)",
        repo="Qwen/Qwen3-VL-2B-Instruct-GGUF",
        filename="Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen3VL-2B-Instruct-F16.gguf",
        params="2B",
        approx_size_gb=1.6,
        summary="Lightest and fastest; good when RAM or time is tight.",
    ),
    "qwen3-vl-4b": DocsModelSpec(
        key="qwen3-vl-4b",
        title="Qwen3-VL 4B Instruct (default)",
        repo="Qwen/Qwen3-VL-4B-Instruct-GGUF",
        filename="Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen3VL-4B-Instruct-F16.gguf",
        params="4B",
        approx_size_gb=3.4,
        summary="The default balance of reading quality and size.",
    ),
    "qwen3-vl-8b": DocsModelSpec(
        key="qwen3-vl-8b",
        title="Qwen3-VL 8B Instruct (strong)",
        repo="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        filename="Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        mmproj_filename="mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        params="8B",
        approx_size_gb=5.9,
        summary="The strongest reading; needs more RAM and time.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen3-vl-4b"


def resolve_spec(name: str | None) -> DocsModelSpec:
    """The spec for ``name``, or the default when it is empty. Raises on a bad name."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise DocsModelUnavailable(
            f"Unknown docs model {name!r}. Available: {', '.join(sorted(CATALOG))}."
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


def _docs_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else docs_models_dir()


def _registry_file(root: Path | str | None = None) -> Path:
    return _docs_dir(root) / REGISTRY_FILENAME


def _registry(root: Path | str | None = None) -> dict[str, Any]:
    """The registry's contents, or an empty mapping when it is absent or corrupt."""
    path = _registry_file(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _largest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(
        (path for path in directory.glob(pattern) if path.is_file()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_model_file(root: Path | str | None = None) -> Path | None:
    """The language-model file to load, or ``None`` when none is present.

    Prefers the file named in the registry (``model.json``, written by
    ``download``), then the largest non-projector ``.gguf`` in the folder. The
    projector is excluded by name so the two halves are never confused -- a
    projector is a small adapter, so picking "the largest .gguf" would otherwise
    hand it to the language-model loader.
    """
    directory = _docs_dir(root)
    named = _registry(root).get("file")
    if named:
        candidate = directory / str(named)
        if candidate.is_file():
            return candidate
    candidates = sorted(
        (
            path
            for path in directory.glob("*.gguf")
            if path.is_file() and not path.name.startswith(MMPROJ_PREFIX)
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_mmproj_file(root: Path | str | None = None) -> Path | None:
    """The multimodal projector to load, or ``None`` when none is present.

    Prefers the file named in the registry, then the largest ``mmproj-*.gguf`` in
    the folder. A projector is required: without it the model can read text but
    cannot see a page image, which is the whole point of this Studio.
    """
    directory = _docs_dir(root)
    named = _registry(root).get("mmproj")
    if named:
        candidate = directory / str(named)
        if candidate.is_file():
            return candidate
    return _largest(directory, f"{MMPROJ_PREFIX}-*.gguf")


def model_is_cached(root: Path | str | None = None) -> bool:
    """True when **both** halves of the vision model are present on disk."""
    return find_model_file(root) is not None and find_mmproj_file(root) is not None


def extra_available() -> bool:
    """True when ``llama_cpp`` can be imported, without importing it here."""
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _require_extra() -> Any:
    """Import ``llama_cpp`` or raise a clear :class:`DocsModelUnavailable`."""
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocsModelUnavailable(
            "The in-process vision model needs the optional docs extra. Install it "
            f"with: {pip_install_command(FEATURE_DOCS)}"
        ) from error
    return llama_cpp


def _require_mtmd() -> Any:
    """The ``MTMDChatHandler`` class, or raise an actionable message.

    Vision support arrives with the ``mtmd`` library, so an older
    ``llama-cpp-python`` that predates it must say so plainly rather than failing
    with an attribute error deep inside a reply.
    """
    from llama_cpp import llama_chat_format  # type: ignore[import-not-found]

    handler = getattr(llama_chat_format, "MTMDChatHandler", None)
    if handler is None:
        raise DocsModelUnavailable(
            "This llama-cpp-python build has no vision support (no MTMDChatHandler). "
            "Upgrade it with: python -m pip install --upgrade llama-cpp-python"
        )
    return handler


# ---------------------------------------------------------------------- client


class DocsRunner(Protocol):
    """The seam a test injects so no real model is needed."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


class LlamaCppDocsRunner:
    """The default runner: a thin ``llama-cpp-python`` multimodal chat wrapper.

    ``llama_cpp`` is imported here, lazily, so the module loads without the extra.
    The language model and its projector are loaded together and the GGUF's own
    chat template is used by ``create_chat_completion``, so the model's expected
    prompt format -- Qwen3-VL's included -- is honoured without the project
    hard-coding one.
    """

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        *,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        llama_cpp = _require_extra()
        handler_class = _require_mtmd()
        self._handler = handler_class(clip_model_path=str(mmproj_path))
        self._llama = llama_cpp.Llama(
            model_path=str(model_path),
            chat_handler=self._handler,
            n_ctx=context,
            n_gpu_layers=gpu_layers,
            n_threads=threads or None,
            verbose=False,
        )

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        result = self._llama.create_chat_completion(messages=messages, **kwargs)
        return _content_of(result)


def _content_of(result: Any) -> str:
    """Pull the assistant text out of llama.cpp's reply, defensively."""
    try:
        choices = result["choices"]
        message = choices[0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as error:
        raise DocsModelUnavailable(
            "The in-process vision model returned an unexpected reply shape."
        ) from error
    text = str(content or "").strip()
    if not text:
        raise DocsModelUnavailable(
            "The in-process vision model returned an empty reply."
        )
    return text


class LlamaCppDocsClient:
    """A small client over a local Qwen3-VL GGUF pair.

    The model is loaded once, on the first request, and reused -- so a Studio
    session pays the load cost once and nothing sits in RAM before the first
    question.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        mmproj_path: str | Path | None = None,
        *,
        model: str = DEFAULT_DOCS_MODEL,
        root: Path | str | None = None,
        runner: DocsRunner | None = None,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        threads: int = DEFAULT_THREADS,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        #: Marks this client for the status helpers, so the status line never
        #: tells an in-process user to start a server.
        self.backend = "in-process"
        self.root = root
        self._path = Path(model_path) if model_path is not None else None
        self._mmproj = Path(mmproj_path) if mmproj_path is not None else None
        self._runner = runner
        self.context = context
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.max_tokens = max_tokens

    # --- model files ----------------------------------------------------
    def model_path(self) -> Path:
        """The language-model file this client will load, resolved on demand."""
        if self._path is not None:
            if not self._path.is_file():
                raise DocsModelUnavailable(f"The docs model file is missing: {self._path}")
            return self._path
        found = find_model_file(self.root)
        if found is None:
            raise DocsModelUnavailable(self.missing_detail())
        return found

    def mmproj_path(self) -> Path:
        """The multimodal projector this client will load, resolved on demand."""
        if self._mmproj is not None:
            if not self._mmproj.is_file():
                raise DocsModelUnavailable(
                    f"The docs vision projector is missing: {self._mmproj}"
                )
            return self._mmproj
        found = find_mmproj_file(self.root)
        if found is None:
            directory = _docs_dir(self.root)
            if find_model_file(self.root) is not None:
                raise DocsModelUnavailable(
                    f"Found a docs model file in {directory} but no `mmproj-*.gguf` "
                    "vision projector beside it, so it cannot see a page image. "
                    "Download the projector with `python -m cli docs download "
                    "--approve`."
                )
            raise DocsModelUnavailable(self.missing_detail())
        return found

    def is_installed(self) -> bool:
        """True when both files are present and the runtime can load them."""
        if self._runner is not None:
            return True
        if not extra_available():
            return False
        try:
            return self.model_path().is_file() and self.mmproj_path().is_file()
        except DocsModelUnavailable:
            return False

    def missing_detail(self) -> str:
        """Why the in-process model cannot answer, in one actionable line."""
        directory = _docs_dir(self.root)
        if not extra_available():
            return (
                "The in-process vision model needs the optional docs extra. Install "
                f"it with: {pip_install_command(FEATURE_DOCS)}"
            )
        if find_model_file(self.root) is None:
            return (
                f"No docs model file in {directory}. Download one with "
                "`python -m cli docs download --approve`, or drop a `.gguf` file "
                "there."
            )
        if find_mmproj_file(self.root) is None:
            return (
                f"Found a docs model file in {directory} but no `mmproj-*.gguf` vision "
                "projector beside it, so it cannot see a page image. Download the "
                "projector with `python -m cli docs download --approve`."
            )
        return (
            f"The docs model files in {directory} are not usable by this "
            f"llama-cpp-python build. Install the docs extra with: "
            f"{pip_install_command(FEATURE_DOCS)}"
        )

    def _load(self) -> DocsRunner:
        """Load the runner on first use, so construction never loads a model."""
        if self._runner is None:
            self._runner = LlamaCppDocsRunner(
                self.model_path(),
                self.mmproj_path(),
                context=self.context,
                gpu_layers=self.gpu_layers,
                threads=self.threads,
            )
        return self._runner

    # --- one request ----------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        """One reply from the in-process model."""
        if not messages:
            raise ValueError("At least one message is required.")
        runner = self._load()
        try:
            return runner.chat(
                messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        except DocsModelUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - surfaced as an honest message
            raise DocsModelUnavailable(f"The in-process vision model failed: {error}") from error


# ---------------------------------------------------------------------- status


@dataclass
class DocsModelStatus:
    """An honest report of whether the Studio's vision model can read right now."""

    available: bool
    model: str
    backend: str
    installed: bool
    detail: str = ""
    hint: str = ""

    def summary(self) -> str:
        """A one-line, plain-language status for the window and the CLI."""
        if self.available:
            return f"Docs model ready: {self.model} (in-process, no server)."
        return self.detail or "No docs model is ready."


def docs_model_status(
    *,
    root: Path | str | None = None,
    client: LlamaCppDocsClient | None = None,
) -> DocsModelStatus:
    """An honest status of the Studio's vision model, for the window and the CLI.

    The in-process runtime is the only one this Studio supports: a vision model
    needs its projector file, which is a local concern, so there is no server
    fallback to probe here.
    """
    if client is not None:
        return DocsModelStatus(
            available=client.is_installed(),
            model=client.model,
            backend=client.backend,
            installed=client.is_installed(),
            detail="" if client.is_installed() else client.missing_detail(),
        )

    model_file = find_model_file(root)
    mmproj_file = find_mmproj_file(root)
    if model_file is not None and mmproj_file is not None and extra_available():
        return DocsModelStatus(
            available=True,
            model=DEFAULT_DOCS_MODEL,
            backend="in-process",
            installed=True,
            detail=f"In-process docs model ready: {model_file.name} + {mmproj_file.name}.",
        )
    if model_file is not None and mmproj_file is None:
        return DocsModelStatus(
            available=False,
            model=DEFAULT_DOCS_MODEL,
            backend="in-process",
            installed=False,
            detail=(
                f"Found {model_file.name} but no `mmproj-*.gguf` vision projector "
                f"beside it in {_docs_dir(root)}, so it cannot see a page image."
            ),
            hint="python -m cli docs download --approve",
        )
    if model_file is not None and not extra_available():
        return DocsModelStatus(
            available=False,
            model=DEFAULT_DOCS_MODEL,
            backend="in-process",
            installed=False,
            detail=f"Found {model_file.name} but the docs extra is not installed.",
            hint=pip_install_command(FEATURE_DOCS),
        )
    if extra_available():
        return DocsModelStatus(
            available=False,
            model=DEFAULT_DOCS_MODEL,
            backend="in-process",
            installed=False,
            detail=f"No docs model file in {_docs_dir(root)}.",
            hint="python -m cli docs download --approve",
        )
    return DocsModelStatus(
        available=False,
        model=DEFAULT_DOCS_MODEL,
        backend="none",
        installed=False,
        detail=(
            f"No in-process docs model: {_docs_dir(root)} is empty and the docs extra "
            f"is not installed. Install it with `{pip_install_command(FEATURE_DOCS)}` "
            "and download a model with `python -m cli docs download --approve`."
        ),
        hint=pip_install_command(FEATURE_DOCS),
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
) -> tuple[Path, Path]:
    """Download a vision model's weights **and** its projector into the docs folder.

    The download leaves the machine, so it is treated as an external tool: denied
    by default, and it runs only when an approving :class:`PermissionPolicy` *and*
    a live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the
    CLI's convenience that records a human approval). Both halves are fetched,
    because a vision model without its projector cannot read a page image.
    Returns ``(model_path, mmproj_path)``.
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
        raise DocsModelUnavailable(
            f"Downloading {spec.key!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise DocsModelUnavailable(
            f"Downloading {spec.key!r} needs human approval ({approval.reason})."
        )

    directory = _docs_dir(root)
    target = directory / spec.filename
    projector = directory / spec.mmproj_filename
    if target.is_file() and projector.is_file():
        return target, projector

    try:
        import huggingface_hub
    except ImportError as error:
        raise DocsModelUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_DOCS)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(
            f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…"
        )
    try:
        model_path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
        if progress is not None:
            progress("Downloading the vision projector (mmproj)…")
        mmproj_path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.mmproj_filename, local_dir=str(directory)
        )
    except Exception as error:  # noqa: BLE001 - surfaced as an honest message
        raise DocsModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
    _write_registry(
        directory, Path(model_path).name, Path(mmproj_path).name, spec.key
    )
    if progress is not None:
        progress(f"Saved {spec.title} to {directory}.")
    return Path(model_path), Path(mmproj_path)


def _write_registry(
    directory: Path, filename: str, mmproj_filename: str, spec_key: str
) -> None:
    """Record which model files to load, so the newest download wins."""
    try:
        (directory / REGISTRY_FILENAME).write_text(
            json.dumps(
                {"file": filename, "mmproj": mmproj_filename, "spec": spec_key},
                ensure_ascii=False,
                indent=2,
            ),
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
    "DEFAULT_DOCS_MODEL",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_TOOL",
    "HEADER",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "MMPROJ_PREFIX",
    "NAME",
    "REGISTRY_FILENAME",
    "DocsModelSpec",
    "DocsModelStatus",
    "DocsModelUnavailable",
    "DocsRunner",
    "LlamaCppDocsClient",
    "LlamaCppDocsRunner",
    "docs_model_status",
    "download_model",
    "extra_available",
    "find_mmproj_file",
    "find_model_file",
    "model_is_cached",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
