"""Answer from the trained mind -- the in-project language model -- not the sources.

The ``chat`` command normally answers the way Jarvis does: it retrieves the
passages that support an answer and cites them. This module adds a second,
deliberately different mouth: it asks the **trained model** directly, with no
retrieval and no citations, so the quality of its own prose is visible.

That is a measurement, not a smarter answer: the model here is small, trained
inside the repository, and its output is *not* checked against sources. The
continuation is composed into clean prose by :mod:`core.composition` -- the same
deterministic verbaliser the grounded answers use -- but only the model's own
sentences survive, so the answer is still an honest sample of what the model
learned. Every answer is labelled as coming from the model so it is never
mistaken for a grounded, cited one.

Everything in this module is self-contained. It does not touch the assistant,
the Jarvis loop, the controls, or the session builder, so enabling it cannot
change how Jarvis or the default chat behaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.assistant import AssistantAnswer
from core.composition import is_coherent, model_answer, topic_phrase
from core.conversation import Conversation
from core.paths import default_checkpoint

#: Wording shown above every model answer. It must stay honest: the answer is the
#: model's own prose, not a cited extract, so it never looks like grounded output.
MIND_HEADER = "[mind] answered from the trained model only; not grounded in cited sources."

#: One line explaining the model switches, appended by ``chat`` to the help text.
MIND_HELP = (
    "Model mode: type 'from the model' to answer from the trained model (no sources), "
    "'from the sources' to answer from retrieved passages with citations again."
)

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

    The model is trained on short declarative sentences, so the prompt must look
    like the prose it was trained on: a bare ``Question:``/``Answer:`` frame sits
    out of distribution and the model continues it with noise. Anchoring the
    prompt on the subject -- ``"Permeability is "`` -- puts the model back in
    distribution, and it completes the sentence instead of drifting. The subject
    is read from the question with the composer's :func:`topic_phrase`, so the
    same phrase names the topic in the answer body.
    """
    topic = topic_phrase(question)
    if topic and topic != "this question":
        return f"{topic[:1].upper()}{topic[1:]} is "
    return f"{question.strip().rstrip('?.!')} is "


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
        status["modified"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
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


#: Shown in place of prose when the continuation holds no complete sentence. It
#: keeps the answer honest -- the model did not say anything usable -- instead of
#: printing a broken fragment as though it were an answer.
MIND_NO_ANSWER = "(the model produced no complete sentence for this question)"


@dataclass
class MindResult:
    """The composed result of one model-only answer."""

    question: str
    text: str
    header: str = MIND_HEADER


class MindAssistant:
    """Answers from the trained model only, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    can call it unchanged when ``chat`` swaps it in. It never retrieves, never
    searches the web, and returns empty citations on purpose: an answer built
    this way has no source behind it, and saying so is the point.
    """

    def __init__(self, model, *, max_new_tokens: int = 96, header: str = MIND_HEADER) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.header = header
        self.memory = None

    def answer(self, question: str) -> MindResult:
        """Generate the model's answer for one question, as composed prose.

        The prompt anchors the sentence on the subject (``"Permeability is "``),
        so the model's continuation is the predicate of a real sentence; prompt
        and continuation are then composed together and reduced to whole
        sentences. The result is the model's own words, restated as prose.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        prompt = render_prompt(question)
        text = model_answer(question, prompt + self._generate(prompt))
        if not is_coherent(text):
            text = MIND_NO_ANSWER
        return MindResult(question=question, text=text, header=self.header)

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
        text = f"{result.header}\n\n{result.text.strip()}"
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
        full = self.model.generator.generate(prompt, max_new_tokens=self.max_new_tokens)
        return full[len(prompt):].strip()


__all__ = [
    "MIND_HEADER",
    "MIND_HELP",
    "MIND_NO_ANSWER",
    "MindAssistant",
    "MindResult",
    "SWITCH_TO_MODEL",
    "SWITCH_TO_SOURCES",
    "load_mind_model",
    "mind_status",
    "render_prompt",
    "status_line",
]
