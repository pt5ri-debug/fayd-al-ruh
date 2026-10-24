"""Hakeem's model layer: the **same** local model Natasha, the companion, uses.

The task was explicit: Hakeem stands on the model the project's companion already
speaks through. So this module is deliberately a *thin* layer, not a second
implementation. The catalog, the runtime choice (an in-process GGUF file loaded
with ``llama-cpp-python``, or an existing Ollama server as a fallback), the model
folder, the governed download, and the client seam all live in
:mod:`companion.model`; Hakeem reuses every one of them.

One catalog and one model folder mean Hakeem and Natasha can never disagree about
what is installed -- and a user who already fetched the companion's weights
downloads **nothing** for Hakeem. He is a different *character* (see
:mod:`hakeem.personality`) speaking through the same *model*.

Two things are Hakeem's own, so a status line never says "Natasha":

* :class:`HakeemModelClient` is the client seam the agent talks to -- the same
  shape as the companion's, so the same in-process and Ollama clients satisfy it;
* :func:`hakeem_model_status` reports *exactly* what is installed, with Hakeem's
  own wording, and never pretends a model is present.

Nothing here executes model output: the model returns text, and Hakeem's own
program logic (memory, sessions, personality) decides what to keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from companion.model import (
    BACKEND_AUTO,
    BACKEND_GGUF,
    BACKEND_NAMES,
    BACKEND_OLLAMA,
    CATALOG,
    DEFAULT_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SPEC_KEY,
    DEFAULT_THREADS,
    MAX_CONTEXT,
    MAX_TOKENS_CEILING,
    CompanionModelSpec,
    CompanionModelUnavailable,
    LlamaCppNatashaClient,
    LlamaCppNatashaRunner,
    NatashaModelClient,
    NatashaModelStatus,
    OllamaNatashaClient,
    build_client,
    download_model,
    extra_available,
    find_model_file,
    model_is_cached,
    model_present,
    pull_model,
    resolve_backend,
    resolve_spec,
    spec_names,
    spec_summaries,
    strip_thinking,
)
from companion.personality import NAME as _COMPANION_NAME

from hakeem.personality import ARABIC_NAME, NAME

#: The command that fetches the shared model's weights, shown when none is present.
DOWNLOAD_HINT = "python -m cli hakeem download --approve"

#: The command that pulls the shared model into Ollama, shown for the fallback.
PULL_HINT = f"ollama pull {DEFAULT_MODEL}"


class HakeemModelUnavailable(CompanionModelUnavailable):
    """Raised when Hakeem's (shared) model cannot run.

    A subclass of the companion's own error, so a caller can catch either the
    Hakeem-specific type or the general one. It shares the companion's model, so
    it shares the companion's failure modes: no runtime, no weights, no server,
    or no pull.
    """


#: The client seam, named for Hakeem. It is the companion's protocol unchanged --
#: the same in-process GGUF client and the same Ollama client satisfy both.
HakeemModelClient = NatashaModelClient

#: The runner seam for the in-process model (the companion's own), re-exported so
#: a test can inject one without importing the companion module directly.
HakeemRunner = LlamaCppNatashaRunner


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


def hakeem_model_status(
    *,
    client: HakeemModelClient | None = None,
    backend: str | None = None,
    root: Path | str | None = None,
    model: str = DEFAULT_MODEL,
) -> HakeemModelStatus:
    """Report whether Hakeem's (shared) model can answer, without pretending.

    The check is the companion's own (:func:`companion.model.natasha_model_status`),
    because it is the same model and the same runtime; only the report is reworded
    so it names Hakeem. ``root`` defaults to the companion's model folder, so the
    two characters share one set of weights.
    """
    companion_status: NatashaModelStatus = _status(client=client, backend=backend, root=root, model=model)
    return HakeemModelStatus(
        available=companion_status.available,
        server_reachable=companion_status.server_reachable,
        model=companion_status.model,
        installed=companion_status.installed,
        backend=companion_status.backend,
        models=companion_status.models,
        detail=companion_status.detail,
        hint=DOWNLOAD_HINT if companion_status.backend == "in-process" else PULL_HINT,
    )


def _status(
    *,
    client: HakeemModelClient | None,
    backend: str | None,
    root: Path | str | None,
    model: str,
) -> NatashaModelStatus:
    """The companion's status check, imported lazily so the two stay in step."""
    from companion.model import natasha_model_status

    return natasha_model_status(client=client, backend=backend, root=root, model=model)


def companion_name() -> str:
    """The name of the companion whose model Hakeem shares (for honest messages)."""
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
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_THREADS",
    "DOWNLOAD_HINT",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "NAME",
    "PULL_HINT",
    "CompanionModelSpec",
    "HakeemModelClient",
    "HakeemModelStatus",
    "HakeemModelUnavailable",
    "HakeemRunner",
    "LlamaCppNatashaClient",
    "OllamaNatashaClient",
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
]
