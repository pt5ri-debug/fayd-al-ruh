"""Answer from the trained mind -- the in-project language model -- not the sources.

The ``chat`` command normally answers the way Jarvis does: it retrieves the
passages that support an answer and cites them. This module adds a second,
deliberately different mouth: it asks the **trained model** directly, with no
retrieval and no citations, so the quality of its own prose is visible.

That is a measurement, not a smarter answer: the model here is small, trained
inside the repository, and its output is *not* checked against sources. Because a
byte-level continuation model leaves ragged edges -- a leading fragment, a
sentence that stops mid-word, an unfinished final clause, a repeated sentence, a
run of repeated punctuation -- the continuation is passed through
:func:`core.composition.compose_model_answer` first, so the mouth reads as
connected sentences while still speaking the model's own words (it never
paraphrases or adds a fact). The answer is labelled as coming from the model so
it is never mistaken for a grounded, cited one, and ``MindResult.raw_text`` keeps
the untouched continuation for inspection.

Everything in this module is self-contained. It does not touch the assistant,
the Jarvis loop, the controls, or the session builder, so enabling it cannot
change how Jarvis or the default chat behaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.assistant import AssistantAnswer
from core.composition import compose_model_answer
from core.conversation import Conversation
from core.paths import default_checkpoint

#: Wording shown above every model answer. It must stay honest: the answer is the
#: model's own prose, not a cited extract, so it never looks like grounded output.
#: The prose is tidied by :func:`core.composition.compose_model_answer`, but it
#: is still the model's own words -- no fact is added -- so it stays ungrounded.
MIND_HEADER = "[mind] answered from the trained model only; not grounded in cited sources."

#: One line explaining the model switches, appended by ``chat`` to the help text.
MIND_HELP = (
    "Model mode: type 'from the model' to answer from the trained model (no sources), "
    "'from the sources' to answer from retrieved passages with citations again."
)

#: Shown in place of a body when the model produced nothing wordbearing (an empty
#: continuation, or a stopping run such as ")))))" that tidies away). Without it
#: the answer would be the header alone -- a message with no text, which the GUI
#: renders as an empty bubble carrying only its Copy button.
MIND_EMPTY = "The model produced no connected words for this question."

#: The controls ``chat`` recognises to switch mouths (kept here, not in the shared
#: Jarvis controls, so the Jarvis command's wording is untouched).
SWITCH_TO_MODEL = (
    "from the model", "use the model", "answer from the model", "model on",
    "من الموديل", "الموديل", "استخدم الموديل",
)
SWITCH_TO_SOURCES = (
    "from the sources", "from the knowledge", "use the sources", "model off",
    "answer from the sources", "من المصادر", "من المعرفة", "المصادر",
)


def load_mind_model(
    checkpoint: str | Path | None = None,
    *,
    temperature: float = 0.0,
    seed: int = 0,
    max_new_tokens: int = 96,
):
    """Load the trained language model as the answering engine.

    Deferred import keeps this module importable without numpy (the whole project
    tolerates a missing ``numpy`` until a model is actually needed). A missing
    checkpoint raises ``FileNotFoundError`` with the trainer's hint, so ``chat``
    can tell the user to train one instead of failing obscurely.
    """
    from ai.local_model import LocalLanguageModel

    return LocalLanguageModel(
        checkpoint,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )


def render_prompt(question: str) -> str:
    """The prompt the question is turned into.

    A plain completion prompt: the model is trained on short declarative prose,
    so a question/answer frame would only put it out of distribution. Keeping the
    frame minimal asks the model to continue from the question as written.
    """
    return f"Question: {question.strip()}\nAnswer:"


def mind_status(checkpoint: str | Path | None = None) -> dict[str, Any]:
    """A small, honest description of the checkpoint that will answer.

    Includes the release manifest's ``trained_at`` when it is readable, so a
    reader can tell which training run the answers came from.
    """
    path = Path(checkpoint) if checkpoint is not None else default_checkpoint()
    status: dict[str, Any] = {"checkpoint": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        status["size_bytes"] = stat.st_size
        status["modified"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    manifest = _manifest_info()
    if manifest:
        status.update(manifest)
    return status


def status_line(status: dict[str, Any]) -> str:
    """One readable line summarising :func:`mind_status` for the terminal."""
    if not status.get("exists"):
        return f"Model mode: no checkpoint at {status['checkpoint']}."
    parts = [f"Model mode: {Path(status['checkpoint']).name}"]
    if status.get("trained_at"):
        parts.append(f"trained {status['trained_at']}")
    if status.get("size_bytes"):
        parts.append(f"{status['size_bytes'] // 1024} KB")
    return " | ".join(parts) + "."


def _manifest_info() -> dict[str, Any]:
    """``trained_at``/``version`` from the release manifest, when present."""
    import json

    from core.paths import artifacts_dir, bundled_artifacts_dir

    trained = artifacts_dir() / "release_manifest.json"
    path = trained if trained.exists() else bundled_artifacts_dir() / "release_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    info: dict[str, Any] = {}
    if isinstance(data, dict):
        if data.get("trained_at"):
            info["trained_at"] = data["trained_at"]
        if data.get("version"):
            info["version"] = data["version"]
    return info


@dataclass
class MindResult:
    """The raw result of one model-only answer."""

    question: str
    text: str
    header: str = MIND_HEADER
    raw_text: str = ""


class MindAssistant:
    """Answers from the trained model only, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    can call it unchanged when ``chat`` swaps it in. It never retrieves, never
    searches the web, and returns empty citations on purpose: an answer built
    this way has no source behind it, and saying so is the point.
    """

    def __init__(
        self,
        model,
        *,
        max_new_tokens: int = 96,
        temperature: float | None = None,
        seed: int | None = None,
        header: str = MIND_HEADER,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        # Generation settings the model was loaded with. Passing them explicitly
        # to the generator matters: the generator's own defaults (0.8 / unseeded)
        # would otherwise override the checkpoint's greedy, reproducible settings
        # for the mind mouth -- the same defect that made ``--mind-temperature 0``
        # still sample.
        self.temperature = getattr(model, "temperature", 0.0) if temperature is None else temperature
        self.seed = getattr(model, "seed", None) if seed is None else seed
        self.header = header
        self.memory = None

    def answer(self, question: str) -> MindResult:
        """Generate the model's continuation for one question.

        The continuation is tidied by :func:`core.composition.compose_model_answer`
        before it is returned, so the mouth shows readable sentences rather than a
        fragment that stops mid-word. That helper never rewrites the model's
        words; it only keeps the whole sentences the model finished and centres
        them on the question. When the model produced nothing wordbearing the
        tidied text is empty, and ``ask`` substitutes :data:`MIND_EMPTY` so the
        answer is never a header with no body. ``MindResult.raw_text`` keeps the
        untouched output for inspection.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        prompt = render_prompt(question)
        raw = self._generate(prompt)
        return MindResult(
            question=question,
            text=compose_model_answer(raw, question=question),
            header=self.header,
            raw_text=raw,
        )

    def ask(
        self,
        question: str,
        *,
        conversation: Conversation | None = None,
        limit: int = 4,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> AssistantAnswer:
        """Match ``GroundedAssistant.ask``: record the turn and answer from the model."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        conversation.add_user(question)
        result = self.answer(question)
        body = result.text.strip() or MIND_EMPTY
        text = f"{result.header}\n\n{body}"
        conversation.add_assistant(text, ())
        return AssistantAnswer(
            text=text,
            citations=(),
            passages=(),
            confidence=0.0,
            used_model=True,
            conversation_id=conversation.id,
        )

    def _generate(self, prompt: str) -> str:
        """The continuation only -- never the prompt echoed back."""
        full = self.model.generator.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        return full[len(prompt):].strip()


__all__ = [
    "MIND_EMPTY",
    "MIND_HEADER",
    "MIND_HELP",
    "SWITCH_TO_MODEL",
    "SWITCH_TO_SOURCES",
    "MindAssistant",
    "MindResult",
    "load_mind_model",
    "mind_status",
    "render_prompt",
    "status_line",
]
