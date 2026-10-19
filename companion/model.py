"""Natasha's model layer: the local Ollama model she speaks through.

Every mouth in Fayd Al-Ruh names its model in one place: :mod:`core.ultron`,
:mod:`core.vision_model`, :mod:`core.banner`, :mod:`core.cloud_model`, and
:mod:`core.presentation_model` each own a catalog, a client seam, an honest status,
and a governed download. **Natasha** is the project's *companion*
(:mod:`companion.companion_agent`), and this module is her counterpart: it owns the
model she talks through, the choices a user can fetch, the honest report of what is
installed, and the one governed way to fetch a model.

Natasha speaks through a **local Ollama server** -- the same runtime the project's
Code Studio and Presentation Studio can use -- not a GGUF loaded in-process. Ollama
is a plain HTTP server, so the transport is the project's own
:class:`core.coder_model.OllamaCoderClient` (standard library only, no new
dependency), *reused* here rather than re-implemented, exactly as the other mouths
reuse a shared transport. The model itself is an Ollama tag (``qwen3:4b`` by
default), pulled from the Ollama registry into the server's own store -- there is
no ``data/`` folder to manage, so this module has no ``find_model_file``.

Three things stay honest, as everywhere else in the project:

* :class:`NatashaModelClient` is the seam the companion talks to; the default
  :class:`OllamaNatashaClient` is the project's Ollama client, and a test injects
  its own, so the whole companion is exercised with no server and no weights.
* :func:`natasha_model_status` reports *exactly* what is installed: whether the
  server answers, whether the model is pulled, and the real next step when it is
  not. It never pretends a model is present.
* :func:`download_model` runs ``ollama pull`` only through the project's
  permission and human-approval gates, denied by default -- fetching several
  gigabytes is an external action, like every other model download.

Nothing here executes model output: the model returns text, and the companion's own
program logic (memory, tone, sessions) decides what to keep.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from companion.personality import NAME
from core.coder_model import CoderMessage, CoderUnavailable, OllamaCoderClient
from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionPolicy

#: The Ollama tag Natasha speaks through by default. A small local Qwen3 model
#: that runs on a laptop CPU with a few gigabytes of RAM, and the one the project
#: already targets for local chat.
DEFAULT_MODEL = "qwen3:4b"

#: The local Ollama server address. The same default the rest of the project uses
#: (:data:`core.coder_model.DEFAULT_OLLAMA_URL`).
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: The command that installs the default model, shown verbatim when it is missing.
PULL_HINT = f"ollama pull {DEFAULT_MODEL}"

#: The external-tool name the permission/approval gates see for a model pull.
DOWNLOAD_TOOL = "natasha_model_download"


class CompanionModelUnavailable(CoderUnavailable):
    """Natasha's model cannot run (the server is down or the model is not pulled)."""


@dataclass(frozen=True)
class CompanionModelSpec:
    """A known-good Ollama model that ``natasha download`` can pull.

    A spec bundles the Ollama tag and a short description, so "the model Natasha
    speaks through" is a named choice rather than a tag the user has to assemble.
    Sizes are the default Ollama quantisation, the balance of quality and RAM that
    fits a laptop.
    """

    key: str
    title: str
    tag: str
    params: str
    approx_size_gb: float
    summary: str


#: The download catalog, keyed by name. ``qwen3-4b`` is the default: it reasons
#: before it answers, so it holds a warm conversation well, and it fits a modest
#: machine. ``qwen3-8b`` is stronger but heavier; ``qwen2.5-7b`` is a
#: non-reasoning model, so it answers faster with no hidden reasoning trace.
CATALOG: dict[str, CompanionModelSpec] = {
    "qwen3-4b": CompanionModelSpec(
        key="qwen3-4b",
        title="Qwen3 4B (default)",
        tag="qwen3:4b",
        params="4B",
        approx_size_gb=2.6,
        summary="Warm and capable on a modest machine; reasons before it answers.",
    ),
    "qwen3-8b": CompanionModelSpec(
        key="qwen3-8b",
        title="Qwen3 8B",
        tag="qwen3:8b",
        params="8B",
        approx_size_gb=5.2,
        summary="Stronger and more coherent; needs more RAM and time.",
    ),
    "qwen2.5-7b": CompanionModelSpec(
        key="qwen2.5-7b",
        title="Qwen2.5 7B (no hidden reasoning)",
        tag="qwen2.5:7b",
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


# ---------------------------------------------------------------------- client


def model_present(model: str, models: Sequence[str]) -> bool:
    """True when ``model`` is pulled, accepting Ollama's implicit ``:latest`` tag.

    Only an exact match (or the same name with the implicit ``:latest`` tag)
    counts: reporting a *different* size of the same family as present would be a
    lie the user would only discover when the run failed.
    """
    wanted = model.strip()
    return wanted in models or (":" not in wanted and f"{wanted}:latest" in models)


@runtime_checkable
class NatashaModelClient(Protocol):
    """What the companion needs from a model: a chat call and a model list.

    The same shape as :class:`core.coder_model.CoderClient`, satisfied by the
    default :class:`OllamaNatashaClient` and by a test's own client.
    """

    model: str

    def chat(self, messages: Sequence[CoderMessage], *, temperature: float = 0.2) -> str: ...

    def list_models(self) -> tuple[str, ...]: ...


class OllamaNatashaClient(OllamaCoderClient):
    """The default client: the project's Ollama transport, named for Natasha.

    It *is* :class:`core.coder_model.OllamaCoderClient` -- the same
    standard-library HTTP, the same ``/api/tags`` and ``/api/chat`` calls, the same
    optional ``max_tokens``/``think`` settings -- so no second Ollama client is
    written. The subclass only marks the seam for the companion and keeps the
    honest status line specific to her.
    """

    #: Marks the runtime for a status line, matching the other mouths.
    backend = "ollama"


# ---------------------------------------------------------------------- status


@dataclass
class NatashaModelStatus:
    """An honest report of whether Natasha's model can answer right now.

    The shape matches the other mouths' status objects, and the wording is
    Natasha's own: a missing server and a missing *model* are reported separately,
    because the next step differs (start Ollama, or pull the model).
    """

    available: bool
    server_reachable: bool
    model: str
    installed: bool
    models: tuple[str, ...] = ()
    detail: str = ""
    hint: str = PULL_HINT

    def summary(self) -> str:
        """A one-line, plain-language status for the CLI and the window."""
        if self.available:
            return f"{NAME} is ready ({self.model})."
        if not self.server_reachable:
            return (
                f"Natasha cannot reach Ollama at the configured address. "
                f"Start it with `ollama serve`. ({self.detail})"
            )
        return (
            f"Ollama is running but `{self.model}` is not installed. "
            f"Install it with `ollama pull {self.model}`."
        )


def natasha_model_status(*, client: NatashaModelClient | None = None) -> NatashaModelStatus:
    """Report whether Natasha's model can answer, without pretending.

    A missing server and a missing model are reported separately, because the next
    step differs: start Ollama, or pull the model. An injected client (a test's)
    is honoured unchanged.
    """
    active: NatashaModelClient = client or OllamaNatashaClient(model=DEFAULT_MODEL)
    model = active.model
    try:
        models = active.list_models()
    except CoderUnavailable as error:
        return NatashaModelStatus(
            available=False,
            server_reachable=False,
            model=model,
            installed=False,
            detail=str(error),
        )
    installed = model_present(model, models)
    detail = "" if installed else f"`{model}` is not pulled."
    return NatashaModelStatus(
        available=installed,
        server_reachable=True,
        model=model,
        installed=installed,
        models=models,
        detail=detail,
        hint=f"ollama pull {model}",
    )


# -------------------------------------------------------------------- download


#: Runs one command and returns ``(returncode, combined_output)``.
CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def _default_runner(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def download_model(
    spec_key: str = DEFAULT_SPEC_KEY,
    *,
    approved: bool = False,
    permissions: PermissionPolicy | None = None,
    approvals: HumanApprovalGate | None = None,
    runner: CommandRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Pull a companion model into the local Ollama server with ``ollama pull``.

    Fetching the weights leaves the machine, so it is treated as an external tool:
    denied by default, and it runs only when an approving :class:`PermissionPolicy`
    *and* a live :class:`HumanApprovalGate` approval exist (or ``approved=True``,
    the CLI's convenience that records a human approval). ``runner`` is the seam a
    test injects, so the gate is exercised with no real pull. Returns the Ollama
    tag that is now present.
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
        raise CompanionModelUnavailable(
            f"Pulling {spec.key!r} is denied by policy ({decision.reason})."
        )
    approval = gate.decide(DOWNLOAD_TOOL)
    if not approval.approved:
        raise CompanionModelUnavailable(
            f"Pulling {spec.key!r} needs human approval ({approval.reason})."
        )

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


def _approve(tool: str) -> HumanApprovalGate:
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="cli-user")
    request.approve("cli-reviewer")
    return gate


__all__ = [
    "CATALOG",
    "CommandRunner",
    "CompanionModelSpec",
    "CompanionModelUnavailable",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SPEC_KEY",
    "DOWNLOAD_TOOL",
    "NAME",
    "NatashaModelClient",
    "NatashaModelStatus",
    "OllamaNatashaClient",
    "PULL_HINT",
    "download_model",
    "model_present",
    "natasha_model_status",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
