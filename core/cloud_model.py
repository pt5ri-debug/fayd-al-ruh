"""Talk to a remote, cloud-hosted language model -- off by default, on request.

Every other mouth in this project runs **on this machine**: the grounded
assistant reads its own corpus, ``--mind`` speaks the in-project checkpoint, and
Ultron / Vision / Banner / the Studio's authoring model load a GGUF file into
this process. This module adds the one mouth that deliberately does *not*: a
**remote** model reached over the network, so a user who has an account with a
cloud provider (or a free tier, or their own OpenAI-compatible server) can talk
to it from the same chat.

Design, matching the project's hard rules:

* **Denied by default.** A cloud call sends the user's words off the machine, so
  it is treated exactly like a web search: it runs only when an approving
  :class:`~core.permissions.PermissionPolicy` grants the external capability
  *and* a live :class:`~core.human_approval.HumanApprovalGate` approval exists.
  Selecting the cloud mouth in the chat is the user's explicit approval; a bare
  :class:`CloudEngine` reaches nothing.
* **The key is never the project's.** The API key is read from an environment
  variable (or passed in explicitly) and is never written to the repository, a
  config file, or a log. :func:`redact_secret` scrubs it from every message this
  module produces, so an error string can be shown, logged, or pasted safely.
* **No new dependency.** The client is plain ``urllib`` + ``json`` from the
  standard library, so a base install pulls nothing and the whole module imports
  without a network or an account. (That is also why there is no
  ``requirements-cloud.txt``: there is no package to install. The "next step"
  when a provider is unusable is setting an environment variable, which
  :func:`cloud_status` states.)
* **Honest reporting.** :func:`cloud_status` distinguishes "no key set" from "the
  endpoint refused us" and names the real next step; it makes **no** network call
  unless ``probe=True``, so a status check never surprises the user with a
  request. A cloud answer is labelled as the remote model's own prose and is
  never presented as grounded in cited sources.

The protocol is the **OpenAI-compatible** chat-completions shape
(``POST {base_url}/chat/completions``), which OpenAI, Anthropic, Gemini, Groq,
Mistral, DeepSeek, OpenRouter, Together, and local servers (Ollama, LM Studio,
vLLM, llama.cpp's own server) all expose. One client therefore reaches all of
them; a provider entry is just a base URL, a default model, and the environment
variable its key lives in.

The transport is the seam a test injects: :class:`OpenAICompatibleClient` takes a
``transport`` callable, so the whole engine is exercised with no network and no
account. Nothing here executes model output; the model returns text and the
caller decides what, if anything, to do with it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.audit_log import AuditLog
from core.human_approval import HumanApprovalGate
from core.permissions import Capability, PermissionDecision, PermissionPolicy

#: The name used in labels and status lines.
NAME = "Cloud"

#: The standing instruction given to a cloud model. It is asked to be accurate
#: and to admit uncertainty; the project adds no subject filter of its own.
SYSTEM_PROMPT = (
    "You are answering inside the Fayd Al-Ruh project. Answer directly and "
    "helpfully, in the user's language. Be accurate and concrete; when you are "
    "unsure, say so plainly."
)

#: The external-tool name the permission/approval gates see for one cloud reply.
CHAT_TOOL = "cloud_chat"

#: The honesty line shown above every cloud answer. It must stay truthful: the
#: answer is the remote model's own prose, sent over the network, and it is *not*
#: grounded in the project's cited sources.
HEADER = "[cloud] answered by a remote model over the network; not grounded in cited sources."

#: A short explanation for the menu and the help text.
BLURB = "Remote cloud model (ChatGPT, Claude, Gemini, and others), opt-in and off by default."

#: The environment variable a *custom* endpoint's key is read from.
DEFAULT_API_KEY_ENV = "FAYD_CLOUD_API_KEY"

#: A safe ceiling on one reply, so a mistyped request cannot run away.
MAX_TOKENS_CEILING = 8192

#: A safe ceiling on the sampling temperature (both OpenAI and Anthropic accept
#: up to 2.0).
MAX_TEMPERATURE = 2.0

#: How long to wait for a reply. A cloud model can be slow, but an unanswered
#: request must not hang the chat forever.
DEFAULT_TIMEOUT = 120.0


class CloudUnavailable(RuntimeError):
    """Raised when a cloud model cannot be reached, is not configured, or is denied."""


@dataclass(frozen=True)
class CloudProvider:
    """One remote (or local-but-remote-shaped) OpenAI-compatible endpoint.

    A provider bundles the base URL, the model to ask for by default, and the
    environment variable its API key lives in, so "talk to ChatGPT" is a named
    choice rather than a URL the user has to assemble. ``api_key_env`` is empty
    for an endpoint that needs no key (a server on this machine).
    """

    key: str
    title: str
    base_url: str
    default_model: str
    api_key_env: str
    summary: str
    #: A few other models the provider is known to serve, for the status report.
    models: tuple[str, ...] = ()
    #: True when the endpoint is expected to run on this machine, so the wording
    #: never claims data left the machine for a local server.
    local: bool = False
    #: True when the provider offers a free tier (a real, no-cost quota), so the
    #: window and the CLI can point a user at a provider that needs no payment.
    free: bool = False
    #: Where a user creates a free (or paid) API key for this provider. Empty when
    #: the endpoint needs no key (a local server), so a hint never sends a user to
    #: a signup page they do not need.
    signup_url: str = ""

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_env)


#: The provider catalog, keyed by name. Every entry speaks the OpenAI-compatible
#: chat-completions protocol, so one client reaches them all.
CATALOG: dict[str, CloudProvider] = {
    "openai": CloudProvider(
        key="openai",
        title="OpenAI (ChatGPT)",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        summary="OpenAI's own API (the ChatGPT models). Needs a paid API key.",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1", "o4-mini"),
        signup_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": CloudProvider(
        key="anthropic",
        title="Anthropic (Claude)",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-5-sonnet-latest",
        api_key_env="ANTHROPIC_API_KEY",
        summary="Anthropic's Claude models through their OpenAI-compatible endpoint.",
        models=("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"),
        signup_url="https://console.anthropic.com/settings/keys",
    ),
    "gemini": CloudProvider(
        key="gemini",
        title="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        summary="Google's Gemini models; there is a free tier for light use.",
        models=(
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
        ),
        free=True,
        signup_url="https://aistudio.google.com/apikey",
    ),
    "groq": CloudProvider(
        key="groq",
        title="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        summary="Very fast open-model hosting with a free tier.",
        models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        free=True,
        signup_url="https://console.groq.com/keys",
    ),
    "openrouter": CloudProvider(
        key="openrouter",
        title="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.1-8b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        summary="One key for many providers; several models are free to use.",
        models=(
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
        ),
        free=True,
        signup_url="https://openrouter.ai/keys",
    ),
    "mistral": CloudProvider(
        key="mistral",
        title="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
        summary="Mistral's own models; there is a free tier for light use.",
        models=("mistral-small-latest", "mistral-large-latest", "open-mistral-nemo"),
        free=True,
        signup_url="https://console.mistral.ai/api-keys",
    ),
    "deepseek": CloudProvider(
        key="deepseek",
        title="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        summary="DeepSeek's chat and reasoning models.",
        models=("deepseek-chat", "deepseek-reasoner"),
        signup_url="https://platform.deepseek.com/api_keys",
    ),
    "ollama": CloudProvider(
        key="ollama",
        title="Ollama (this machine)",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        api_key_env="",
        summary="An Ollama server on this machine, through its OpenAI-compatible endpoint.",
        models=("qwen2.5:7b", "llama3.2", "qwen3:8b"),
        local=True,
    ),
    "lmstudio": CloudProvider(
        key="lmstudio",
        title="LM Studio / vLLM (this machine)",
        base_url="http://localhost:1234/v1",
        default_model="local-model",
        api_key_env="",
        summary="A local OpenAI-compatible server (LM Studio, vLLM, llama.cpp) on this machine.",
        local=True,
    ),
}

#: The provider used when the user names none.
DEFAULT_PROVIDER_KEY = "openai"


def resolve_provider(name: str | CloudProvider | None) -> CloudProvider:
    """The :class:`CloudProvider` for ``name`` (or the default). Raises on a bad name.

    Accepts a :class:`CloudProvider` back unchanged, so a caller that already has
    one (or ``None``) can pass it straight through.
    """
    if isinstance(name, CloudProvider):
        return name
    key = (name or DEFAULT_PROVIDER_KEY).strip().lower()
    if key in CATALOG:
        return CATALOG[key]
    raise ValueError(f"Unknown cloud provider {name!r}; choose one of {', '.join(CATALOG)}.")


def provider_names() -> tuple[str, ...]:
    """The catalog keys, in declaration order."""
    return tuple(CATALOG)


def provider_summaries() -> tuple[tuple[str, str, str, str, bool, bool], ...]:
    """``(key, title, default_model, summary, local, free)`` for every catalog entry."""
    return tuple(
        (spec.key, spec.title, spec.default_model, spec.summary, spec.local, spec.free)
        for spec in CATALOG.values()
    )


def free_provider_names() -> tuple[str, ...]:
    """The providers that offer a no-cost tier, in catalog order.

    Used to point a user who does not want to pay at a provider that works
    without a card (Gemini, Groq, OpenRouter, Mistral), or at a local server.
    """
    return tuple(key for key, spec in CATALOG.items() if spec.free)


def free_provider_summaries() -> tuple[tuple[str, str, str, str, str], ...]:
    """``(key, title, api_key_env, signup_url, summary)`` for the free-tier providers.

    One row per provider that needs **no card**: the key variable to set, where to
    create the key, and what the provider is for. This is the single source the
    CLI's ``providers`` list and the window's "free providers" hint both read, so
    the two can never disagree about which services are free or where to sign up.
    """
    return tuple(
        (spec.key, spec.title, spec.api_key_env, spec.signup_url, spec.summary)
        for spec in CATALOG.values()
        if spec.free
    )


def provider_models(name: str | CloudProvider | None) -> tuple[str, ...]:
    """The models the catalog knows a provider serves, default first.

    The provider's ``default_model`` is put first (so a menu opens on the model
    the project ships with) followed by the rest of its known lineup, with no
    repeats. A provider with no listed lineup still yields its default, so the
    picker is never empty. This is the *catalog's* knowledge, not a live query:
    ``cloud models`` asks the endpoint for the real, current list.
    """
    spec = resolve_provider(name)
    ordered = [spec.default_model, *spec.models]
    seen: list[str] = []
    for model in ordered:
        name_ = model.strip()
        if name_ and name_ not in seen:
            seen.append(name_)
    return tuple(seen)


# ------------------------------------------------------- the saved choice


#: Where the user's *chosen* provider is remembered, so "use Gemini" or "use
#: Groq" survives a restart and the next chat needs no flag. Persisted through
#: the project's own :class:`storage.json_store.JsonStore`, like every other
#: preference -- never a key, only the provider's name.
PROVIDER_NAMESPACE = "cloud"
PROVIDER_KEY = "provider"

#: Where the user's *chosen model* per provider is remembered, so a provider whose
#: default was retired (a provider renames and shuts models down) can be pointed at
#: a working one once -- ``cloud use gemini --model gemini-3.8-flash`` -- instead of
#: repeating ``--model`` on every call. Only a model *name*, never a key.
MODEL_KEY = "models"


def load_provider_key(root: str | Path) -> str:
    """The provider the user last chose, or the default; never raises.

    A stored value that is no longer in the catalog (a removed provider) falls
    back to the default rather than failing, so a stale file can never make the
    cloud mouth unusable.
    """
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root).load(PROVIDER_NAMESPACE, PROVIDER_KEY)
    except Exception:  # a broken store must not stop the cloud mouth loading
        return DEFAULT_PROVIDER_KEY
    name = saved.get("provider") if isinstance(saved, dict) else None
    key = str(name).strip().lower() if name else ""
    return key if key in CATALOG else DEFAULT_PROVIDER_KEY


def save_provider_key(root: str | Path, key: str) -> str:
    """Remember ``key`` as the chosen provider. Returns the key that was stored.

    Raises :class:`ValueError` for an unknown provider, so a typo is reported
    rather than silently stored. A store error is swallowed: choosing a provider
    must not fail just because the data folder is read-only.
    """
    resolved = resolve_provider(key).key
    from storage.json_store import JsonStore

    try:
        JsonStore(root).save(PROVIDER_NAMESPACE, PROVIDER_KEY, {"provider": resolved})
    except Exception:
        pass
    return resolved


def selected_provider(root: str | Path, explicit: str | None = None) -> CloudProvider:
    """The provider to use: an explicit choice, else the saved one, else the default."""
    if explicit:
        return resolve_provider(explicit)
    return resolve_provider(load_provider_key(root))


def load_model_key(root: str | Path, provider_key: str) -> str | None:
    """The model the user last chose for ``provider_key``, or ``None``.

    Lets a provider whose built-in default was retired be pointed at a working
    model once, instead of repeating ``--model`` on every call. Never raises: a
    broken store simply means "no saved model", so the provider default applies.
    """
    try:
        resolved = resolve_provider(provider_key).key
    except ValueError:
        return None
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root).load(PROVIDER_NAMESPACE, MODEL_KEY)
    except Exception:
        return None
    if not isinstance(saved, dict):
        return None
    name = saved.get(resolved)
    return str(name).strip() or None if name else None


def save_model_key(root: str | Path, provider_key: str, model: str) -> str:
    """Remember ``model`` as the model to use for ``provider_key``.

    Raises :class:`ValueError` for an unknown provider or an empty model name, so
    a typo is reported rather than stored. A store error is swallowed, so choosing
    a model cannot fail just because the data folder is read-only.
    """
    resolved = resolve_provider(provider_key).key
    name = (model or "").strip()
    if not name:
        raise ValueError("A model name is required.")
    from storage.json_store import JsonStore

    try:
        store = JsonStore(root)
        current = store.load(PROVIDER_NAMESPACE, MODEL_KEY)
        mapping = dict(current) if isinstance(current, dict) else {}
        mapping[resolved] = name
        store.save(PROVIDER_NAMESPACE, MODEL_KEY, mapping)
    except Exception:
        pass
    return name


def selected_model(
    root: str | Path, provider: CloudProvider | str | None, explicit: str | None = None
) -> str:
    """The model to ask for: an explicit name, else the saved one, else the default.

    The order matters: a user who typed ``--model`` this once gets it, a user who
    saved a model for this provider gets that, and otherwise the provider's own
    default applies.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    spec = resolve_provider(provider)
    return load_model_key(root, spec.key) or spec.default_model


def redact_secret(text: str, secret: str | None) -> str:
    """Remove ``secret`` from ``text`` so an error can be shown or logged safely.

    This module never puts the key into a message itself, but a transport error
    (or a redirect URL) could echo it back. Replacing it here keeps every string
    this module produces safe to display, log, or paste.
    """
    if not secret:
        return text
    return text.replace(secret, "***")


@dataclass(frozen=True)
class CloudMessage:
    """One chat turn sent to the remote model."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty.")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class CloudClient(Protocol):
    """What the engine needs from a cloud transport: a chat call and a model list.

    The shape matches :class:`core.coder_model.CoderClient`, so a cloud client can
    also serve the Code Studio's agent unchanged.
    """

    model: str

    def chat(
        self,
        messages: Sequence[CloudMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    def list_models(self) -> tuple[str, ...]: ...


#: The seam a test injects: send one HTTP request, return ``(status, body)``.
#: The default implementation is :func:`_urllib_transport`.
Transport = Callable[[str, dict[str, str], bytes | None, float], "tuple[int, str]"]


def _urllib_transport(
    url: str, headers: dict[str, str], data: bytes | None, timeout: float
) -> tuple[int, str]:
    """The default transport: one non-streaming HTTP request, via ``urllib``."""
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace") if error.fp else str(error)
        return int(error.code), body
    except (urllib.error.URLError, OSError) as error:
        raise CloudUnavailable(f"Could not reach {url}: {error}") from error


class OpenAICompatibleClient:
    """A standard-library client for any OpenAI-compatible chat endpoint.

    The client is deliberately thin: it posts one chat request and returns the
    assistant's text. It never retries a failed generation silently and never
    invents a reply -- a transport problem raises :class:`CloudUnavailable` with
    the endpoint's own message (key redacted) so the user sees the real cause.

    The API key is read from the provider's environment variable unless one is
    passed explicitly; it is never stored anywhere on disk.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: CloudProvider | str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Transport | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        spec = provider if isinstance(provider, CloudProvider) else resolve_provider(provider)
        self.provider = spec
        self.model = (model or spec.default_model).strip()
        if not self.model:
            raise ValueError("A cloud model name is required.")
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.api_key_env = api_key_env if api_key_env is not None else spec.api_key_env
        self._api_key = api_key
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self._extra_headers = dict(extra_headers or {})

    # ------------------------------------------------------------- the key

    @property
    def api_key(self) -> str | None:
        """The key to send, from the explicit value or the environment variable.

        Read at call time, so a user who exports the variable after the process
        started is still picked up. Never written anywhere.
        """
        if self._api_key is not None:
            return self._api_key.strip() or None
        if not self.api_key_env:
            return None
        return (os.environ.get(self.api_key_env) or "").strip() or None

    @property
    def needs_key(self) -> bool:
        return self.provider.needs_key

    def key_available(self) -> bool:
        """True when the client has everything it needs to *try* a request."""
        return bool(self.api_key) or not self.needs_key

    def missing_key_hint(self) -> str:
        """The one next step when no key is set, naming the variable exactly."""
        if not self.needs_key:
            return f"No key is needed for {self.provider.title}."
        return (
            f"Set the {self.provider.api_key_env} environment variable to your "
            f"{self.provider.title} API key, then try again."
        )

    # --------------------------------------------------------- the request

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        key = self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        status, body = self._transport(
            f"{self.base_url}{path}", self._headers(), data, self.timeout
        )
        return self._parse(status, body)

    def _get(self, path: str) -> dict[str, Any]:
        status, body = self._transport(f"{self.base_url}{path}", self._headers(), None, self.timeout)
        return self._parse(status, body)

    def _parse(self, status: int, body: str) -> dict[str, Any]:
        safe = redact_secret(body, self.api_key)
        if status >= 400:
            raise CloudUnavailable(self._error_message(status, safe))
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise CloudUnavailable(
                f"{self.provider.title} returned a non-JSON reply: {safe[:200]}"
            ) from error
        if not isinstance(parsed, dict):
            raise CloudUnavailable(f"{self.provider.title} returned an unexpected reply shape.")
        return parsed

    def _error_message(self, status: int, safe: str) -> str:
        """An honest error that names the *next step*, not just the HTTP code.

        A retired or misspelled model is the most common failure (providers
        rename and shut models down), and a bare ``HTTP 404`` leaves the user
        stuck. So a 404/400 that mentions the model points at the two fixes:
        list what the endpoint really serves, and pick a different one.
        """
        head = f"{self.provider.title} returned HTTP {status}: {safe[:300]}"
        low = safe.lower()
        model_gone = status in (400, 404) and (
            "model" in low and ("not found" in low or "no longer available" in low
                                or "not available" in low or "does not exist" in low
                                or "unsupported" in low or "shut down" in low
                                or "deprecated" in low)
        )
        if not model_gone:
            return head
        return (
            f"{head}\n"
            f"The model {self.model!r} is not available at {self.provider.title}. "
            f"See what it serves with: python -m cli cloud models --provider "
            f"{self.provider.key}\n"
            f"Then ask with another one, e.g.: python -m cli cloud --provider "
            f"{self.provider.key} --model <name> \"your question\" "
            f"(or set it as the default for this provider)."
        )

    def _payload(
        self,
        messages: Sequence[CloudMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "stream": False,
        }
        if temperature is not None:
            if not 0.0 <= temperature <= MAX_TEMPERATURE:
                raise ValueError(f"temperature must be in [0, {MAX_TEMPERATURE}].")
            payload["temperature"] = temperature
        if max_tokens is not None:
            if not 1 <= max_tokens <= MAX_TOKENS_CEILING:
                raise ValueError(f"max_tokens must be in [1, {MAX_TOKENS_CEILING}].")
            payload["max_tokens"] = max_tokens
        return payload

    def chat(
        self,
        messages: Sequence[CloudMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One non-streamed reply from the remote model."""
        if not messages:
            raise ValueError("At least one message is required.")
        if not self.key_available():
            raise CloudUnavailable(self.missing_key_hint())
        reply = self._post(
            "/chat/completions", self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        )
        content = _content_of(reply, self.provider.title)
        if not content:
            raise CloudUnavailable(f"{self.provider.title} returned an empty reply.")
        return content

    def list_models(self) -> tuple[str, ...]:
        """Every model the endpoint advertises, sorted.

        Some OpenAI-compatible endpoints (notably Anthropic's compatibility
        layer) do not implement ``GET /models``; that is reported as
        :class:`CloudUnavailable` rather than guessed at, so a caller can fall
        back to the catalog's own list.
        """
        payload = self._get("/models")
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise CloudUnavailable(f"{self.provider.title} did not return a model list.")
        names = sorted(
            str(entry.get("id", ""))
            for entry in entries
            if isinstance(entry, dict) and entry.get("id")
        )
        return tuple(names)


def _content_of(reply: dict[str, Any], provider_title: str) -> str:
    """The assistant text of a chat completion, defensively."""
    try:
        choices = reply["choices"]
        message = choices[0]["message"]
        return str(message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise CloudUnavailable(
            f"{provider_title} returned an unreadable response: {error}"
        ) from error


# ------------------------------------------------------------------ status


@dataclass
class CloudStatus:
    """An honest report of whether a cloud model can answer right now.

    ``probe`` is what makes the difference between "we have a key, so we can
    try" and "the endpoint answered us": without a probe no request is made, so a
    status check never surprises the user with a network call.
    """

    provider: str
    model: str
    available: bool
    key_required: bool
    key_present: bool
    local: bool = False
    reachable: bool | None = None
    models: tuple[str, ...] = ()
    detail: str = ""
    hint: str = ""

    def summary(self) -> str:
        """A one-line, plain-language status for the window and the CLI."""
        where = "on this machine" if self.local else "a remote service"
        if self.available:
            if self.reachable is False:
                return f"{self.provider}: key is set, but the endpoint did not answer ({self.model})."
            return f"{self.provider}: ready to answer with {self.model} ({where})."
        if self.detail:
            return f"{self.provider}: {self.detail}"
        return f"{self.provider}: not ready."


def cloud_status(
    client: CloudClient | None = None,
    *,
    provider: CloudProvider | str | None = None,
    model: str | None = None,
    probe: bool = False,
) -> CloudStatus:
    """Report whether a cloud model can answer, without pretending.

    A missing *key* and an unreachable *endpoint* are reported separately, because
    the next step differs. No network request is made unless ``probe=True``.
    """
    client = client or OpenAICompatibleClient(model=model, provider=provider)
    spec = getattr(client, "provider", None) or resolve_provider(provider)
    key_required = bool(getattr(spec, "needs_key", False))
    key_present = True
    if hasattr(client, "key_available"):
        key_present = bool(client.key_available())
    elif key_required:
        env = getattr(spec, "api_key_env", "")
        key_present = bool((os.environ.get(env) or "").strip())

    status = CloudStatus(
        provider=spec.title,
        model=client.model,
        available=key_present,
        key_required=key_required,
        key_present=key_present,
        local=bool(getattr(spec, "local", False)),
    )
    if not key_present:
        status.detail = f"no API key is set ({spec.api_key_env} is empty)."
        status.hint = (
            f"Set {spec.api_key_env} to your {spec.title} API key, then try again."
        )
        return status
    if not probe:
        return status
    try:
        status.models = client.list_models()
        status.reachable = True
    except CloudUnavailable as error:
        status.reachable = False
        status.available = False
        status.detail = str(error)
        status.hint = (
            f"Check the endpoint ({spec.base_url}) and your key. "
            "Some providers do not implement GET /models; that is not fatal, "
            "a chat request may still work."
        )
    return status


# ------------------------------------------------------------------ engine


def granted_permissions(tool: str = CHAT_TOOL) -> PermissionPolicy:
    """The permission policy that lets one cloud chat request run."""
    return PermissionPolicy(
        allowed=frozenset(
            {
                Capability.SYMBOLIC_SIMULATION,
                Capability.SANDBOX_TASK,
                Capability.EXTERNAL_TOOL,
            }
        ),
        allowed_external_tools=frozenset({tool}),
    )


def approved_gate(tool: str = CHAT_TOOL) -> HumanApprovalGate:
    """A live approval for one cloud chat request, signed by the local user."""
    gate = HumanApprovalGate()
    request = gate.request(tool, requested_by="local-user")
    request.approve("local-user-reviewer")
    return gate


class CloudEngine:
    """Runs a remote model behind the project's permission, approval, and audit gates.

    The engine is a thin, honest wrapper: it builds the chat messages (the
    standing system instruction plus the conversation so far), asks the client,
    and returns the model's text. It never invents a capability -- with no
    approval, no key, or no reachable endpoint it raises
    :class:`CloudUnavailable` with the real next step.
    """

    name = NAME

    def __init__(
        self,
        client: CloudClient | None = None,
        *,
        provider: CloudProvider | str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        permissions: PermissionPolicy | None = None,
        approvals: HumanApprovalGate | None = None,
        audit: AuditLog | None = None,
        approved: bool = False,
    ) -> None:
        self.client: CloudClient = client or OpenAICompatibleClient(
            model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout=timeout,
        )
        self.provider = getattr(self.client, "provider", None) or resolve_provider(provider)
        if approved:
            permissions = permissions or granted_permissions()
            approvals = approvals or approved_gate()
        self.permissions = permissions or PermissionPolicy()
        self.approvals = approvals or HumanApprovalGate()
        self.audit = audit or AuditLog()
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        if temperature is not None and not 0.0 <= temperature <= MAX_TEMPERATURE:
            raise ValueError(f"temperature must be in [0, {MAX_TEMPERATURE}].")
        if max_tokens is not None and not 1 <= max_tokens <= MAX_TOKENS_CEILING:
            raise ValueError(f"max_tokens must be in [1, {MAX_TOKENS_CEILING}].")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def api_key(self) -> str | None:
        getter = getattr(self.client, "api_key", None)
        return getter if isinstance(getter, str) or getter is None else None

    def is_available(self) -> bool:
        """True when a reply could be attempted now (approval *and* key present)."""
        if not self.permissions.decide(Capability.EXTERNAL_TOOL, CHAT_TOOL).allowed:
            return False
        if not self.approvals.decide(CHAT_TOOL).approved:
            return False
        checker = getattr(self.client, "key_available", None)
        if callable(checker):
            return bool(checker())
        return True

    def _authorise(self) -> None:
        """Record the decision and refuse with the real reason when not allowed."""
        decision = self.permissions.decide(Capability.EXTERNAL_TOOL, CHAT_TOOL)
        approval = self.approvals.decide(CHAT_TOOL)
        allowed = decision.allowed and approval.approved
        reason = decision.reason if not decision.allowed else approval.reason
        self.audit.record_decision(
            PermissionDecision(decision.capability, decision.target, allowed, reason),
            actor="cloud-mouth",
            metadata={"provider": self.provider.key, "model": self.model},
        )
        if not decision.allowed:
            raise CloudUnavailable(
                f"Talking to {self.provider.title} is denied by policy ({decision.reason}). "
                "Select the cloud mouth explicitly to approve it."
            )
        if not approval.approved:
            raise CloudUnavailable(
                f"Talking to {self.provider.title} needs human approval ({approval.reason})."
            )
        checker = getattr(self.client, "key_available", None)
        if callable(checker) and not checker():
            missing = getattr(self.client, "missing_key_hint", None)
            raise CloudUnavailable(
                missing() if callable(missing) else "No API key is set for this provider."
            )

    def build_messages(
        self, history: Sequence[tuple[str, str]], prompt: str
    ) -> list[CloudMessage]:
        """Assemble the chat messages: the system prompt, ``history``, then ``prompt``."""
        messages: list[CloudMessage] = [CloudMessage("system", self.system_prompt)]
        for role, text in history:
            if role not in {"user", "assistant"} or not text.strip():
                continue
            messages.append(CloudMessage(role, text))
        messages.append(CloudMessage("user", prompt))
        return messages

    def reply(
        self,
        prompt: str,
        *,
        history: Sequence[tuple[str, str]] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One reply to ``prompt`` in the context of ``history``."""
        text = (prompt or "").strip()
        if not text:
            raise ValueError("The cloud model needs a question to answer.")
        self._authorise()
        messages = self.build_messages(history, text)
        result = self.client.chat(
            messages,
            temperature=self.temperature if temperature is None else temperature,
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
    ) -> Iterator[str]:
        """Yield a reply piece by piece when the client can stream, else in one piece."""
        text = (prompt or "").strip()
        if not text:
            raise ValueError("The cloud model needs a question to answer.")
        self._authorise()
        messages = self.build_messages(history, text)
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        streamer = getattr(self.client, "stream_chat", None)
        if callable(streamer):
            yield from streamer(messages, temperature=temperature, max_tokens=max_tokens)
            return
        yield self.client.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def describe(self) -> str:
        """A one-line, honest description of the endpoint in use."""
        where = "on this machine" if self.provider.local else "a remote service"
        return (
            f"{NAME}: {self.provider.title} / {self.model} ({where}, {self.provider.base_url})."
        )


__all__ = [
    "BLURB",
    "CATALOG",
    "CHAT_TOOL",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_PROVIDER_KEY",
    "DEFAULT_TIMEOUT",
    "HEADER",
    "MAX_TEMPERATURE",
    "MAX_TOKENS_CEILING",
    "MODEL_KEY",
    "NAME",
    "PROVIDER_KEY",
    "PROVIDER_NAMESPACE",
    "SYSTEM_PROMPT",
    "CloudClient",
    "CloudEngine",
    "CloudMessage",
    "CloudProvider",
    "CloudStatus",
    "CloudUnavailable",
    "OpenAICompatibleClient",
    "approved_gate",
    "cloud_status",
    "free_provider_names",
    "free_provider_summaries",
    "granted_permissions",
    "load_model_key",
    "load_provider_key",
    "provider_models",
    "provider_names",
    "provider_summaries",
    "redact_secret",
    "resolve_provider",
    "save_model_key",
    "save_provider_key",
    "selected_model",
    "selected_provider",
]
