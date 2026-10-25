"""The coding model the Code Studio writes code with.

The Studio's agent needs a model that is good at *code* specifically. This module
owns that choice: a local **Ollama** server serving ``qwen2.5-coder:7b`` by
default. Ollama is a plain HTTP server, so this client is written against the
standard library only (``urllib``): no new dependency is added to the base
install, and the whole module imports without a running server.

Three things are deliberately separated, matching the project's honesty rules:

* :class:`CoderClient` is the thin HTTP transport (``/api/tags`` to list models,
  ``/api/chat`` to hold a conversation). It is a ``Protocol`` so a test -- or a
  user with a different runtime -- can inject a replacement.
* :class:`OllamaCoderClient` is the default implementation.
* :func:`coder_status` reports *exactly* what is installed: whether the server
  answers, whether the model is pulled, and what to run when it is not. It never
  pretends a model is present.

Nothing here executes model output. The model returns text; the caller decides
what, if anything, to do with it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

#: The coding model the Studio uses by default. Small enough to run on a laptop
#: CPU with a few gigabytes of RAM, and tuned for code rather than prose.
DEFAULT_CODER_MODEL = "qwen2.5-coder:7b"

#: The local Ollama server address. Overridable so a user can point at another
#: port or machine without editing code.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: The command that installs the model, shown verbatim when it is missing.
PULL_HINT = f"ollama pull {DEFAULT_CODER_MODEL}"


class CoderUnavailable(RuntimeError):
    """Raised when the coding model cannot be reached or is not installed."""


@dataclass(frozen=True)
class CoderMessage:
    """One turn sent to the coding model."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty.")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class CoderClient(Protocol):
    """What the Studio needs from a coding model: a chat call and a model list."""

    model: str

    def chat(self, messages: Sequence[CoderMessage], *, temperature: float = 0.2) -> str: ...

    def list_models(self) -> tuple[str, ...]: ...


@dataclass
class CoderStatus:
    """An honest report of whether the coding model can answer right now."""

    available: bool
    server_reachable: bool
    model: str
    installed: bool
    models: tuple[str, ...] = ()
    detail: str = ""
    hint: str = PULL_HINT
    #: Which runtime this describes: ``"ollama"`` (a server) or ``"in-process"``
    #: (a GGUF loaded in this process). Only the wording of :meth:`summary`
    #: depends on it, so the two runtimes are reported accurately.
    backend: str = "ollama"

    def summary(self) -> str:
        """A one-line, plain-language status for the window and the CLI."""
        if self.available:
            if self.backend == "in-process":
                return f"Coding model ready: {self.model} (in-process, no server)."
            return f"Coding model ready: {self.model} (Ollama)."
        if self.backend == "in-process":
            return self.detail or "The in-process coding model is not ready."
        if not self.server_reachable:
            return (
                f"Ollama is not running, so the coding model cannot answer. "
                f"Start it (`ollama serve`), then `{self.hint}`."
            )
        return f"Ollama is running but `{self.model}` is not pulled. Run `{self.hint}`."


class OllamaCoderClient:
    """A standard-library Ollama client for a code model.

    The client is intentionally thin: it posts a chat request and returns the
    assistant's text. It never retries a failed generation silently and never
    invents a reply -- a transport problem raises :class:`CoderUnavailable` with
    the server's own message so the user sees the real cause.
    """

    def __init__(
        self,
        model: str = DEFAULT_CODER_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        think: bool | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A coding model name is required.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        #: Cap on generated tokens (Ollama's ``num_predict``); ``None`` leaves the
        #: server's own default.
        self.max_tokens = max_tokens
        #: Toggle for a reasoning model's hidden reasoning trace (Ollama's
        #: ``think``); ``None`` leaves the server's own default.
        self.think = think
        #: Context window to ask Ollama for (``num_ctx``). Left unset, Ollama picks
        #: its own default and *reloads* the model whenever a longer prompt needs a
        #: bigger window -- which is the first-call delay a chat sees. Sending a
        #: value keeps the window stable across turns.
        self.num_ctx = num_ctx
        #: How long Ollama keeps the model resident (``keep_alive``), e.g. ``"30m"``.
        #: A longer value stops the model being unloaded between turns, so the next
        #: reply does not pay the load cost again.
        self.keep_alive = keep_alive

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request)

    def _get(self, path: str) -> dict[str, Any]:
        return self._send(urllib.request.Request(f"{self.base_url}{path}", method="GET"))

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace") if error.fp else str(error)
            raise CoderUnavailable(
                f"Ollama returned HTTP {error.code} for {request.full_url}: {detail[:300]}"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise CoderUnavailable(
                f"Could not reach Ollama at {self.base_url}: {error}"
            ) from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise CoderUnavailable(f"Ollama returned a non-JSON reply: {body[:200]}") from error
        if not isinstance(parsed, dict):
            raise CoderUnavailable("Ollama returned an unexpected reply shape.")
        return parsed

    def list_models(self) -> tuple[str, ...]:
        """Every model the local server has pulled, sorted."""
        payload = self._get("/api/tags")
        models = payload.get("models") or []
        names = sorted(
            str(entry.get("name", ""))
            for entry in models
            if isinstance(entry, dict) and entry.get("name")
        )
        return tuple(names)

    def chat(self, messages: Sequence[CoderMessage], *, temperature: float = 0.2) -> str:
        """One non-streamed reply from the coding model."""
        if not messages:
            raise ValueError("At least one message is required.")
        options: dict[str, Any] = {"temperature": temperature}
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        reply = self._post("/api/chat", payload)
        message = reply.get("message") or {}
        content = str(message.get("content", "")).strip()
        if not content:
            raise CoderUnavailable("The coding model returned an empty reply.")
        return content


def _model_present(model: str, models: Sequence[str]) -> bool:
    """True when ``model`` is pulled, accepting Ollama's implicit ``:latest`` tag.

    Only an exact match (or the same name with the implicit ``:latest`` tag)
    counts: reporting a *different* size of the same family as present would be a
    lie the user would only discover when the run failed.
    """
    wanted = model.strip()
    if wanted in models:
        return True
    return ":" not in wanted and f"{wanted}:latest" in models


def coder_status(client: CoderClient | None = None) -> CoderStatus:
    """Report whether the coding model can answer, without pretending.

    A missing server and a missing model are reported separately, because the
    next step differs: start Ollama, or pull the model. A client that runs the
    model *in this process* declares ``backend = "in-process"`` and supplies its
    own ``missing_detail()``, so the wording never tells an in-process user to
    start a server that does not exist.
    """
    client = client or OllamaCoderClient()
    backend = str(getattr(client, "backend", "ollama"))
    try:
        models = client.list_models()
    except CoderUnavailable as error:
        return CoderStatus(
            available=False,
            server_reachable=False,
            model=client.model,
            installed=False,
            detail=str(error),
            backend=backend,
        )
    installed = _model_present(client.model, models)
    is_installed = getattr(client, "is_installed", None)
    if callable(is_installed):
        installed = bool(is_installed())
    if installed:
        detail = ""
    elif backend == "in-process":
        missing = getattr(client, "missing_detail", None)
        detail = missing() if callable(missing) else "No coding model file is present."
    else:
        detail = f"`{client.model}` is not pulled."
    return CoderStatus(
        available=installed,
        server_reachable=True,
        model=client.model,
        installed=installed,
        models=models,
        detail=detail,
        backend=backend,
    )


__all__ = [
    "DEFAULT_CODER_MODEL",
    "DEFAULT_OLLAMA_URL",
    "PULL_HINT",
    "CoderClient",
    "CoderMessage",
    "CoderStatus",
    "CoderUnavailable",
    "OllamaCoderClient",
    "coder_status",
]
