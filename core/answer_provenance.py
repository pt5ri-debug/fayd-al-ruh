"""What answered a question, and what the answer rests on.

Every window in the project shows an answer; until now none of them said *which*
model produced it or *which* sources it used. The per-mouth windows knew their own
model (``gui/model_chat``), the main window knew its own mouth (``gui/app``), and
each studio knew its own engine -- so a reader had to remember which window they
were in to know what they were reading. This module is the one place that turns
those facts into a readable, bilingual line, so every surface can show the same
thing without growing its own copy that drifts.

Two pieces:

* :func:`describe_answer_model` -- the readable name of the model that answered.
  It reads the answering assistant's own ``describe()`` (every mouth already has
  one, so nothing is duplicated), and falls back to the shared language-model
  catalog for a mouth that cannot describe itself. It never invents a name: an
  unknown mouth with no description yields an honest "no model named".
* :class:`AnswerProvenance` -- the frozen record of one answer's provenance: the
  model, the mouth it answered through, the cited sources, and whether the answer
  used the internet or was learned. :meth:`AnswerProvenance.describe` renders it
  as the two labelled lines a window paints under the answer.

**Honest by construction**: a mouth that retrieves rather than generates says so
(``sources``), a model-backed mouth with no citations says its answer is its own
words with no cited sources (never an empty "Sources:" heading), and a question
whose language is Arabic gets Arabic labels. Nothing here imports Tk, loads a
model, or reaches the network -- it only *names* what already happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.voices import detect_language

#: The label under which the answering model is named, per language.
MODEL_LABEL = {"en": "Answer model", "ar": "موديل الإجابة"}
#: The label under which the cited sources are listed, per language.
SOURCES_LABEL = {"en": "Sources", "ar": "المصادر"}
#: What a model-backed answer with no citations honestly rests on.
NO_SOURCES = {
    "en": "the model's own words — no cited sources",
    "ar": "كلام الموديل نفسه — بلا مصادر مقتبسة",
}
#: What a retrieval answer that fell back to the extractive path rests on.
RETRIEVAL_SOURCES = {
    "en": "the project's own knowledge corpus (retrieval, no model)",
    "ar": "قاعدة معرفة المشروع نفسها (استرجاع بلا موديل)",
}
#: Shown when no model could be named at all, so the line is never blank.
UNKNOWN_MODEL = {"en": "not named by this surface", "ar": "غير مُسمّى في هذه النافذة"}


def _language(text: str, configured: str = "") -> str:
    """``"ar"`` or ``"en"`` -- the configured choice, else the text's own script."""
    if configured in ("ar", "en"):
        return configured
    return detect_language(text)


def _describe_via_catalog(mouth: str) -> str:
    """The shared catalog's honest one-line description of a mouth's model."""
    try:
        from core import language_models as catalog

        key = catalog.default_language_model_key(mouth)
        model = catalog.find_language_model(mouth, key) if key else None
        if model is not None:
            return model.describe()
        return catalog.ENGINE_LABELS.get(mouth, "")
    except Exception:
        return ""


def describe_answer_model(
    assistant: Any = None,
    *,
    mouth: str = "",
    model_key: str | None = None,
) -> str:
    """A readable name for the model that answered, or an honest placeholder.

    The answering assistant's own ``describe()`` is preferred, because each mouth
    already states exactly what it runs (``Ultron: qwen2.5-7b.gguf (4.0 GB) at
    data/ultron.``, ``Cloud: OpenAI / gpt-4o-mini (a remote service, ...)``). A
    ``CharacterMouth`` forwards an unknown attribute to the wrapped mouth, so the
    call reaches the real engine through the wrapper too. When the assistant
    cannot describe itself the shared language-model catalog is asked for the
    mouth's chosen or default model, and only when *both* are silent does the
    honest placeholder come back -- a name is never invented.
    """
    describe = getattr(assistant, "describe", None)
    if callable(describe):
        try:
            text = str(describe()).strip()
        except Exception:
            text = ""
        if text:
            return text
    if mouth:
        text = _describe_via_catalog(mouth)
        if text:
            return text
    return ""


@dataclass(frozen=True)
class AnswerProvenance:
    """What answered one question, and what the answer rests on.

    ``model_label`` is the readable model description (empty when none could be
    named), ``mouth`` the answering mouth's key, and ``sources`` the citations the
    answer carried. ``used_web``/``learned`` record whether the internet was
    consulted and whether the answer was learned into the store, and ``language``
    selects the labels. Nothing here is computed at render time, so a caller that
    stores the record keeps exactly what was true when the answer was made.
    """

    model_label: str = ""
    mouth: str = ""
    model_key: str = ""
    sources: tuple[str, ...] = ()
    used_web: bool = False
    learned: bool = False
    language: str = "en"
    #: True when the grounded assistant answered from retrieval without a model.
    retrieval_only: bool = False
    extra: tuple[str, ...] = field(default_factory=tuple)
    """Any further provenance lines a surface wants to add (e.g. "pages read: 3"),
    rendered verbatim after the model and sources lines so a studio can state what
    it used without inventing a second format."""

    # ------------------------------------------------------------- rendering

    def model_line(self) -> str:
        """The ``Answer model: …`` line, in the answer's language."""
        label = MODEL_LABEL[self.language]
        name = self.model_label.strip() or UNKNOWN_MODEL[self.language]
        return f"{label}: {name}"

    def sources_line(self) -> str:
        """The ``Sources: …`` line, naming the citations or what the answer is."""
        label = SOURCES_LABEL[self.language]
        if self.sources:
            joiner = "، " if self.language == "ar" else " · "
            return f"{label}: {joiner.join(self.sources)}"
        if self.retrieval_only:
            return f"{label}: {RETRIEVAL_SOURCES[self.language]}"
        return f"{label}: {NO_SOURCES[self.language]}"

    def lines(self, *, include_sources: bool = True) -> tuple[str, ...]:
        """Every provenance line, in the order a window paints them.

        ``include_sources=False`` drops the sources line, for a caller that already
        rendered the citations itself (the turn renderer draws its own ``Sources:``
        block), so the sources are never listed twice.
        """
        if include_sources:
            return (self.model_line(), self.sources_line(), *self.extra)
        return (self.model_line(), *self.extra)

    def describe(self, *, include_sources: bool = True) -> str:
        """The whole provenance as one readable block."""
        return "\n".join(self.lines(include_sources=include_sources))


def answer_provenance(
    assistant: Any = None,
    *,
    mouth: str = "",
    model_key: str | None = None,
    citations: tuple[str, ...] = (),
    used_web: bool = False,
    learned: bool = False,
    retrieval_only: bool = False,
    question: str = "",
    language: str = "",
    extra: tuple[str, ...] = (),
) -> AnswerProvenance:
    """Assemble one answer's provenance from what already exists.

    The model is named through :func:`describe_answer_model` (the assistant's own
    ``describe()`` first), the sources are the answer's own citations, and the
    language is the configured choice when given, else the script of ``question``
    (an Arabic question gets Arabic labels). Nothing is fetched, loaded, or
    invented here.
    """
    resolved = _language(question, language)
    label = describe_answer_model(assistant, mouth=mouth, model_key=model_key)
    return AnswerProvenance(
        model_label=label,
        mouth=mouth,
        model_key=model_key or "",
        sources=tuple(citations),
        used_web=bool(used_web),
        learned=bool(learned),
        language=resolved,
        retrieval_only=bool(retrieval_only),
        extra=tuple(extra),
    )


__all__ = [
    "MODEL_LABEL",
    "NO_SOURCES",
    "RETRIEVAL_SOURCES",
    "SOURCES_LABEL",
    "UNKNOWN_MODEL",
    "AnswerProvenance",
    "answer_provenance",
    "describe_answer_model",
]
