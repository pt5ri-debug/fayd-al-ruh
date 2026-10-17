"""Draft presentation material with the Studio's own local model.

The Presentation Studio asks one question at a time ("propose an outline for this
deck", "write slide bullets about this heading") and parses the reply with the
same deterministic parsers it uses for any other answer. This module adapts the
Studio's authoring model (:mod:`core.presentation_model`) to that ``ask`` shape:
a question in, plain text out.

It exists as its own layer for two reasons:

* the Studio's ``ask`` callback is ``Callable[[str], str]``, and the model client
  speaks in messages -- the adapter is where a system prompt is added, so the
  model is told *once* to answer with the requested structure and no commentary;
* a reasoning model may wrap its answer in `` thinking…`` blocks, which would end
  up as slide bullets. :func:`strip_thinking` removes them, so a thinking model
  still produces a clean outline.

Nothing here executes model output; the Studio's parsers decide what the text
means.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.coder_model import CoderClient, CoderMessage, OllamaCoderClient
from core.presentation_model import (
    DEFAULT_PRESENTATION_MODEL,
    LlamaCppPresentationClient,
    PresentationModelUnavailable,
    presentation_backend_status,
)

#: The instruction sent ahead of every authoring request. It asks for exactly the
#: shape the Studio's parsers read, so no model-specific post-processing is needed.
SYSTEM_PROMPT = (
    "You are a scientific and technical presentation author. Answer with exactly "
    "the structure the request asks for: no preamble, no commentary, no markdown "
    "code fences. For an outline, write each section as a heading on its own line "
    "and indent its subsections beneath it. For slide content, write a short slide "
    "title on the first line, then 3 to 5 concise bullets, one per line."
)

#: A reasoning model may wrap its answer in a thinking block. Matched
#: case-insensitively and non-greedily, so several blocks are all removed.
_THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)

#: An unterminated opening tag (the model ran out of tokens mid-thought) leaves
#: everything after it as reasoning, so the tail is dropped too.
_OPEN_THINKING = re.compile(r"<think(?:ing)?>.*\Z", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's `` thinking`` blocks, leaving the answer.

    Qwen3 and other reasoning models emit their chain of thought before the
    answer. That reasoning is not slide content, so it is dropped; when nothing
    is left, the original text is returned so a caller still has something to
    show rather than a silently empty answer.
    """
    cleaned = _OPEN_THINKING.sub("", _THINKING.sub("", text)).strip()
    return cleaned or text.strip()


class PresentationAuthor:
    """Adapts an authoring model to the Studio's ``ask`` callback.

    Call it with a question and it returns the model's answer as text -- the same
    contract the Studio's grounded assistant satisfies, so the service cannot tell
    which one it is talking to.
    """

    def __init__(
        self,
        client: CoderClient,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.3,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature

    def __call__(self, question: str) -> str:
        """Ask the authoring model one question and return its answer."""
        messages = [
            CoderMessage("system", self.system_prompt),
            CoderMessage("user", question),
        ]
        answer = self.client.chat(messages, temperature=self.temperature)
        return strip_thinking(answer)

    def describe(self) -> str:
        """One honest line naming the model and runtime that will answer."""
        backend = str(getattr(self.client, "backend", "ollama"))
        where = "in-process" if backend == "in-process" else "Ollama"
        return f"{self.client.model} ({where})"


def load_presentation_author(
    *,
    model_path: str | Path | None = None,
    root: Path | str | None = None,
    runner: Any = None,
    temperature: float = 0.3,
) -> PresentationAuthor:
    """Build the Studio's authoring model, preferring the in-process GGUF.

    The in-process model needs no background daemon, so it is used when the extra
    and the weights are both present. A reachable Ollama serving the same model is
    honoured otherwise, so a user who already pulled it there is not asked to
    download a second copy. Raises :class:`PresentationModelUnavailable` when
    neither runtime can answer, so the caller reports the real next step.
    """
    in_process = LlamaCppPresentationClient(model_path=model_path, root=root, runner=runner)
    if in_process.is_installed():
        return PresentationAuthor(in_process, temperature=temperature)
    if runner is None:
        ollama = OllamaCoderClient(model=DEFAULT_PRESENTATION_MODEL)
        status = presentation_backend_status(root=root)
        if status.backend == "ollama" and status.available:
            return PresentationAuthor(ollama, temperature=temperature)
    raise PresentationModelUnavailable(in_process.missing_detail())


__all__ = [
    "SYSTEM_PROMPT",
    "PresentationAuthor",
    "load_presentation_author",
    "strip_thinking",
]
