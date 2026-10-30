"""Hakeem's model layer: his **own** local model, in his own folder.

Hakeem is the project's scholar of religion and history, and he is a character in
his own right -- so he has his **own** model folder (``data/hakeem``) and his own
default model (``qwen3.5-4b``), separate from the companion Natasha's, exactly as
his SQLite database is a separate file. A user who wants the two characters on two
different models can have it; a user who wants them on the same one downloads it
once and shares it.

Three things are Hakeem's own, and they live here:

* the **catalog** of models he can fetch (``qwen3.5-4b`` by default, plus the
  smaller and larger Qwen3.5 sizes);
* his **folder** (``data/hakeem``), so his weights never mix with Natasha's;
* :func:`hakeem_model_status`, worded so a status line never says "Natasha".

Everything that is genuinely shared is **reused**, not re-implemented: the
in-process GGUF client, the Ollama fallback client, the reasoning-trace switches,
and the client seam all come from :mod:`companion.model`, because they are the
project's one runtime, not a character's. And the project's **public**
shared-model folder (:mod:`core.shared_models`, ``data/shared_models``) is a
fallback: a model downloaded there answers for Hakeem too, so a shared model is
one download for every character.

Honest by construction, like every other mouth: :func:`hakeem_model_status`
reports *exactly* what is installed, and :func:`download_model` /
:func:`pull_model` leave the machine, so both run only through the project's
permission and human-approval gates, denied by default.

Nothing here executes model output: the model returns text, and Hakeem's own
program logic (memory, sessions, personality) decides what to keep.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from companion.model import (
    BACKEND_AUTO,
    BACKEND_GGUF,
    BACKEND_NAMES,
    BACKEND_OLLAMA,
    DEFAULT_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_REPLY_CONTEXT,
    DEFAULT_THREADS,
    MAX_CONTEXT,
    MAX_TOKENS_CEILING,
    NO_THINK_TOKEN,
    REGISTRY_FILENAME,
    CompanionModelSpec,
    CompanionModelUnavailable,
    LlamaCppNatashaClient,
    LlamaCppNatashaRunner,
    NatashaModelClient,
    OllamaNatashaClient,
    apply_no_think,
    extra_available,
    find_model_in,
    model_present,
    strip_thinking,
    thinking_kwargs,
)
from companion.personality import NAME as _COMPANION_NAME
from core.coder_model import CoderUnavailable
from core.human_approval import HumanApprovalGate
from core.optional_deps import FEATURE_HAKEEM, pip_install_command
from core.paths import hakeem_models_dir, shared_models_dir
from core.permissions import Capability, PermissionPolicy

from hakeem.personality import ARABIC_NAME, NAME

#: The name a status line and a message use.
LABEL = NAME

#: Hakeem's default model: a general Qwen3.5 model, capable enough for a careful
#: scholarly answer and small enough to run on a laptop CPU.
DEFAULT_MODEL = "qwen3.5:4b"

#: The command that fetches Hakeem's weights, shown when none is present.
DOWNLOAD_HINT = "python -m cli hakeem download --approve"

#: The command that pulls Hakeem's model into Ollama, shown for the fallback.
PULL_HINT = f"ollama pull {DEFAULT_MODEL}"

#: The external-tool names the permission/approval gates see. One per runtime, so a
#: policy can allow a GGUF download without also allowing an Ollama pull.
DOWNLOAD_TOOL = "hakeem_model_download"
PULL_TOOL = "hakeem_model_pull"


class HakeemModelUnavailable(CompanionModelUnavailable):
    """Raised when Hakeem's model cannot run (no runtime, no weights, no server)."""


#: The client seam, named for Hakeem. It is the companion's protocol unchanged --
#: the same in-process GGUF client and the same Ollama client satisfy both.
HakeemModelClient = NatashaModelClient

#: The runner seam for the in-process model (the companion's own), re-exported so
#: a test can inject one without importing the companion module directly.
HakeemRunner = LlamaCppNatashaRunner


#: Hakeem's catalog: the Qwen3.5 family, ``qwen3.5-4b`` first as his default. It
#: reuses the companion's spec dataclass (same fields, same meaning) rather than
#: defining a second shape, so the two catalogs can never drift apart in structure.
CATALOG: dict[str, CompanionModelSpec] = {
    "qwen3.5-4b": CompanionModelSpec(
        key="qwen3.5-4b",
        title="Qwen3.5 4B (default)",
        tag="qwen3.5:4b",
        repo="unsloth/Qwen3.5-4B-GGUF",
        filename="Qwen3.5-4B-Q4_K_M.gguf",
        params="4B",
        approx_size_gb=2.5,
        summary="Hakeem's own model: capable and careful, and quick on a modest CPU.",
    ),
    "qwen3.5-2b": CompanionModelSpec(
        key="qwen3.5-2b",
        title="Qwen3.5 2B (lighter)",
        tag="qwen3.5:2b",
        repo="unsloth/Qwen3.5-2B-GGUF",
        filename="Qwen3.5-2B-Q4_K_M.gguf",
        params="2B",
        approx_size_gb=1.2,
        summary="Lighter and faster; good when RAM or time is tight.",
    ),
    "qwen3.5-0.8b": CompanionModelSpec(
        key="qwen3.5-0.8b",
        title="Qwen3.5 0.8B (lightest)",
        tag="qwen3.5:0.8b",
        repo="unsloth/Qwen3.5-0.8B-GGUF",
        filename="Qwen3.5-0.8B-Q4_K_M.gguf",
        params="0.8B",
        approx_size_gb=0.5,
        summary="The lightest choice: a reply in a second or two on a CPU.",
    ),
    "qwen3.5-9b": CompanionModelSpec(
        key="qwen3.5-9b",
        title="Qwen3.5 9B",
        tag="qwen3.5:9b",
        repo="unsloth/Qwen3.5-9B-GGUF",
        filename="Qwen3.5-9B-Q4_K_M.gguf",
        params="9B",
        approx_size_gb=5.5,
        summary="Stronger and more coherent; needs more RAM and time.",
    ),
}

#: The default download target.
DEFAULT_SPEC_KEY = "qwen3.5-4b"


def resolve_spec(name: str | None) -> CompanionModelSpec:
    """The spec for ``name``, or the default when it is empty. Raises on a bad name."""
    key = (name or "").strip().lower()
    if not key:
        return CATALOG[DEFAULT_SPEC_KEY]
    if key not in CATALOG:
        raise HakeemModelUnavailable(
            f"Unknown Hakeem model {name!r}. Available: {', '.join(sorted(CATALOG))}."
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


# ----------------------------------------------------------------- model files


def _hakeem_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else hakeem_models_dir()


def _shared_dir(root: Path | str | None = None) -> Path | None:
    """The public shared folder Hakeem falls back to (``None`` with explicit root)."""
    return None if root is not None else shared_models_dir()


def find_model_file(root: Path | str | None = None) -> Path | None:
    """Hakeem's model file, or ``None`` when none is present.

    His **own** folder (``data/hakeem``) is checked first; when it holds no model,
    the project's public shared folder (``data/shared_models``) is used, so a model
    downloaded once for every character answers for him too. A caller that passed an
    explicit ``root`` gets that folder alone.
    """
    return find_model_in(_hakeem_dir(root), _shared_dir(root))


def model_is_cached(root: Path | str | None = None) -> bool:
    """True when at least one model file is present on disk."""
    return find_model_file(root) is not None


def _registry_file(root: Path | str | None = None) -> Path:
    return _hakeem_dir(root) / REGISTRY_FILENAME


# ---------------------------------------------------------------------- status


@dataclass
class HakeemModelStatus:
    """An honest report of whether Hakeem's model can answer right now.

    The shape matches the companion's status object; the wording is Hakeem's own,
    so a status line never calls him Natasha. A missing runtime, a missing *model
    file*, a missing server, and a missing *Ollama tag* are reported separately,
    because the next step differs.
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
                "Hakeem cannot reach Ollama at the configured address. "
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
    """Which runtime Hakeem will use: ``"gguf"`` or ``"ollama"``.

    ``"auto"`` (the default) prefers the in-process GGUF model when the extra is
    installed and a model is available in his own folder or the shared one;
    otherwise it falls back to Ollama. An explicit ``"gguf"``/``"ollama"`` is
    honoured as asked; an unknown name is refused rather than silently ignored.
    """
    key = (name or BACKEND_AUTO).strip().lower()
    aliases = {
        "": BACKEND_AUTO,
        "auto": BACKEND_AUTO,
        "gguf": BACKEND_GGUF,
        "in-process": BACKEND_GGUF,
        "llama-cpp": BACKEND_GGUF,
        "llamacpp": BACKEND_GGUF,
        "ollama": BACKEND_OLLAMA,
    }
    if key not in aliases:
        raise HakeemModelUnavailable(
            f"Unknown backend {name!r}. Choose 'auto', 'gguf', or 'ollama'."
        )
    chosen = aliases[key]
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    think: bool = False,
    context: int = DEFAULT_CONTEXT,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
    threads: int = DEFAULT_THREADS,
    keep_alive: str | None = DEFAULT_KEEP_ALIVE,
) -> HakeemModelClient:
    """Build the client for the resolved runtime, named for Hakeem.

    The one place the choice is made, so the CLI, the chat mouth, and the window
    cannot disagree about which runtime Hakeem speaks through. The client is the
    companion's own, labelled for Hakeem, so a missing-file message names him.
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
            think=think,
            label=LABEL,
            download_hint=DOWNLOAD_HINT,
            model_dir=hakeem_models_dir() if root is None else None,
            feature=FEATURE_HAKEEM,
        )
    return OllamaNatashaClient(
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
        think=think,
        num_ctx=context,
        keep_alive=keep_alive,
    )


def hakeem_model_status(
    *,
    client: HakeemModelClient | None = None,
    backend: str | None = None,
    root: Path | str | None = None,
    model: str = DEFAULT_MODEL,
) -> HakeemModelStatus:
    """Report whether Hakeem's model can answer, without pretending.

    A missing runtime, a missing model file, a missing server, and a missing Ollama
    tag are reported separately, because the next step differs. ``root`` defaults to
    Hakeem's own model folder (with the shared folder as a fallback).
    """
    if client is None:
        client = build_client(backend=backend, model=model, root=root)
    active = client
    kind = str(getattr(active, "backend", BACKEND_OLLAMA))
    try:
        models = active.list_models()
    except (CoderUnavailable, CompanionModelUnavailable) as error:
        return HakeemModelStatus(
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
        detail = missing() if callable(missing) else "No Hakeem model file is present."
    else:
        detail = f"`{active.model}` is not pulled."
    hint = DOWNLOAD_HINT if kind == "in-process" else f"ollama pull {active.model}"
    return HakeemModelStatus(
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
        raise HakeemModelUnavailable(f"Denied by policy ({decision.reason}).")
    approval = gate.decide(tool)
    if not approval.approved:
        raise HakeemModelUnavailable(f"Needs human approval ({approval.reason}).")


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    root: Path | str | None = None,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download a Hakeem model's **GGUF weights** into his own folder.

    The download leaves the machine, so it is treated as an external tool: denied by
    default, and it runs only when an approving :class:`PermissionPolicy` *and* a
    live :class:`HumanApprovalGate` approval exist (or ``approved=True``, the CLI's
    convenience that records a human approval). Returns the model file path.
    """
    spec = resolve_spec(spec_key)
    _gate(DOWNLOAD_TOOL, approved=approved, permissions=permissions, approvals=approvals)

    directory = _hakeem_dir(root)
    target = directory / spec.filename
    if target.is_file():
        return target

    try:
        import huggingface_hub
    except ImportError as error:
        raise HakeemModelUnavailable(
            "Downloading a model needs `huggingface_hub`. Install it with: "
            f"{pip_install_command(FEATURE_HAKEEM)}"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(f"Downloading {spec.title} (~{spec.approx_size_gb:.1f} GB) to {directory}…")
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=spec.repo, filename=spec.filename, local_dir=str(directory)
        )
    except Exception as error:
        raise HakeemModelUnavailable(f"Could not download {spec.key!r}: {error}") from error
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
    """Pull a Hakeem model into the local **Ollama** server with ``ollama pull``.

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
        raise HakeemModelUnavailable(
            "Could not run `ollama pull`: "
            f"{error}. Install Ollama and start it with `ollama serve`."
        ) from error
    if returncode != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else "ollama pull failed."
        raise HakeemModelUnavailable(f"Could not pull {spec.tag!r}: {tail}")
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


def companion_name() -> str:
    """The name of the companion layer Hakeem reuses (for honest messages)."""
    return _COMPANION_NAME


__all__ = [
    "ARABIC_NAME",
    "BACKEND_AUTO",
    "BACKEND_GGUF",
    "BACKEND_NAMES",
    "BACKEND_OLLAMA",
    "CATALOG",
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_KEEP_ALIVE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_REPLY_CONTEXT",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_HINT",
    "DOWNLOAD_TOOL",
    "LABEL",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "NAME",
    "NO_THINK_TOKEN",
    "PULL_HINT",
    "PULL_TOOL",
    "CompanionModelSpec",
    "HakeemModelClient",
    "HakeemModelStatus",
    "HakeemModelUnavailable",
    "HakeemRunner",
    "LlamaCppNatashaClient",
    "OllamaNatashaClient",
    "apply_no_think",
    "build_client",
    "companion_name",
    "download_model",
    "extra_available",
    "find_model_file",
    "hakeem_model_status",
    "model_is_cached",
    "model_present",
    "pull_model",
    "resolve_backend",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
    "strip_thinking",
    "thinking_kwargs",
]
