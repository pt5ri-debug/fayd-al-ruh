"""Banner -- the project's *presentation* local language model.

The project already has four mouths. The grounded assistant answers only from the
petroleum corpus it was seeded with; :mod:`ai.mind` speaks the small in-project
checkpoint; :mod:`core.ultron` runs a *general* advanced model; and
:mod:`core.vision_model` runs a *petroleum-engineering* specialist. **Banner** is
a fifth, deliberately different mouth: a real, advanced model that runs
**locally** and is dedicated to the **Presentation Studio** -- turning a
discussion into an outline, and an outline into slide content.

It is the same model the Studio already drafts with (**Qwen3 8B Instruct**,
``qwen3:8b``), promoted here to a first-class mouth so it can be chosen, spoken
to, downloaded, and inspected exactly like Ultron and Vision. This module is
deliberately a *thin* layer, not a second implementation: the catalog, the model
folder, the download gate, and the in-process GGUF runner all live in
:mod:`core.presentation_model`, and Banner reuses them. One catalog and one model
file mean the Studio and Banner can never disagree about what is installed, and a
user who fetched the weights for the Studio downloads nothing extra for Banner.

Design, matching the project's hard rules:

* **Local only.** The model is a GGUF file that runs on this machine through
  ``llama-cpp-python``; nothing is sent to a remote server. It lives under the
  writable data folder (:func:`core.paths.presentation_models_dir`,
  ``data/presentation`` by default), git-ignored, never in the repository -- so a
  multi-gigabyte model never dirties the working tree or bloats a program update.
* **Optional extra.** ``llama_cpp`` is imported lazily, only when a reply is
  actually generated, so a base install keeps working and importing this module
  never pulls a heavy dependency. When it (or the model file) is missing,
  :class:`BannerUnavailable` carries a clear, actionable message. It shares the
  ``[presentation]`` runtime extra: the same ``llama-cpp-python`` +
  ``huggingface_hub`` pair runs the Studio's authoring model.
* **Human approval for the download.** Downloading multi-gigabyte weights is an
  external action that leaves the machine, so it runs only through the project's
  permission and human-approval gates, denied by default -- the Studio's own gate
  is reused unchanged. A model already on disk needs no download and no approval.
* **Honesty.** Banner's answers are its own; they are not retrieved from the
  project's corpus and carry no citations, so every answer is labelled as coming
  from a presentation local model rather than from verified sources.

The heavy work is delegated to an injectable :class:`BannerRunner` (the same shape
as the Studio's :class:`core.presentation_model.PresentationRunner`); a test can
supply its own runner and exercise the whole engine without the extra or a model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from core.coder_model import CoderUnavailable
from core.presentation_model import (
    CATALOG,
    DEFAULT_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_PRESENTATION_MODEL,
    DEFAULT_SPEC_KEY,
    DEFAULT_THREADS,
    MAX_CONTEXT,
    MAX_TOKENS_CEILING,
    LlamaCppPresentationRunner,
    PresentationModelSpec,
    PresentationModelUnavailable,
    download_model,
    extra_available,
    find_model_file,
    model_is_cached,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from core.paths import presentation_models_dir

#: The assistant's name, used in every label and status line.
NAME = "Banner"

#: The model tag Banner carries, so a status line reads the same as the Studio's.
MODEL_TAG = DEFAULT_PRESENTATION_MODEL

#: The standing instruction given to the model. Banner is the Presentation
#: Studio's author, so it is asked to answer as a scientific and technical
#: presentation author: the exact structure the Studio's parsers read, and no
#: commentary that would end up as a slide bullet.
SYSTEM_PROMPT = (
    "You are Banner, the Presentation Studio's scientific and technical "
    "presentation author. You turn a discussion into an outline and an outline "
    "into slide content. Answer with exactly the structure the request asks for: "
    "no preamble, no commentary, no markdown code fences. For an outline, write "
    "each section as a heading on its own line and indent its subsections beneath "
    "it. For slide content, write a short slide title on the first line, then 3 to "
    "5 concise bullets, one per line. Be accurate and concrete, name the formula "
    "and the units where they apply, and follow recognised industry practice "
    "(SPE, API, ISO). Answer in the user's language."
)

#: The honesty line shown above every Banner answer. It must stay truthful: the
#: answer is the local model's own prose, not a cited extract from the corpus.
HEADER = "[banner] answered by the local presentation model; not grounded in cited sources."

#: A short explanation for the menu and the help text.
BLURB = "Presentation local model: drafts the Studio's outlines and slide content from a discussion."

#: Defaults chosen to be usable on a CPU-only machine with a modest model.
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95


class BannerUnavailable(PresentationModelUnavailable):
    """Raised when Banner cannot run (no extra, no model file, no weights)."""


class BannerRunner(Protocol):
    """The seam that runs the model: a chat call over a list of chat messages.

    This is the same shape as :class:`core.presentation_model.PresentationRunner`,
    so the Studio's runner (and a test's) satisfies it unchanged.
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


def _banner_dir(root: Path | str | None) -> Path:
    return Path(root) if root is not None else presentation_models_dir()


class BannerEngine:
    """Runs the local presentation model and returns its replies.

    The engine is a thin, honest wrapper: it builds the chat messages (the
    standing system instruction plus the conversation so far), passes them to the
    runner, and returns the model's text. It never invents a capability -- with no
    runner, no extra, or no model file it raises :class:`BannerUnavailable`.
    """

    name = NAME

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        root: Path | str | None = None,
        runner: BannerRunner | None = None,
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
        return extra_available()

    def _active_runner(self) -> BannerRunner:
        if self._runner is not None:
            return self._runner
        path = self.model_path
        if path is None or not path.is_file():
            raise BannerUnavailable(
                "No Banner model found. Put a `.gguf` file in "
                f"{_banner_dir(self._root)} or run `python -m cli banner download --approve`."
            )
        self._runner = LlamaCppPresentationRunner(
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
        """One non-streamed reply to ``prompt`` in the context of ``history``.

        An empty reply is returned as an empty string rather than raised, so the
        mouth above can substitute its own honest "produced no answer" line.
        """
        text = (prompt or "").strip()
        if not text:
            raise ValueError("Banner needs a question to answer.")
        messages = self.build_messages(history, text)
        runner = self._active_runner()
        try:
            result = runner.chat(
                messages,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p if top_p is None else top_p,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
        except (BannerUnavailable, CoderUnavailable):
            raise
        except Exception as error:  # noqa: BLE001 - surfaced as an honest message
            raise BannerUnavailable(f"Banner could not answer: {error}") from error
        return str(result or "").strip()

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
            raise ValueError("Banner needs a question to answer.")
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


__all__ = [
    "BLURB",
    "CATALOG",
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SPEC_KEY",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_THREADS",
    "HEADER",
    "MAX_CONTEXT",
    "MAX_TOKENS_CEILING",
    "MODEL_TAG",
    "NAME",
    "SYSTEM_PROMPT",
    "BannerEngine",
    "BannerRunner",
    "BannerUnavailable",
    "PresentationModelSpec",
    "download_model",
    "extra_available",
    "find_model_file",
    "model_is_cached",
    "resolve_spec",
    "spec_names",
    "spec_summaries",
]
