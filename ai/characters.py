"""Give every answering character its own personality, memory, and learning.

The project offers seven characters inside the chat -- **Jarvis** (the grounded,
cited assistant), **Ultron** (the general local model), **Vision** (the petroleum
local model), **Banner** (the presentation model), **Natasha** (the companion),
**Hakeem** (the scholar of religion and history), and **Cloud** (the remote
model) -- and its own **Fayd Al-Ruh model**. :mod:`core.character_learning` owns
the *data*: each character's traits, the facts it was told, and the exchanges it
took part in, all kept in small files under the git-ignored ``data/characters``
folder.

This module is the *wiring*. :class:`CharacterMouth` wraps any mouth's assistant
and does two things, once, for every mouth:

* it **prepends the character's standing instruction** to the model the mouth
  speaks through (the engine's ``system_prompt``, or the grounded assistant's
  ``system_instruction``), so the character's traits shape every reply -- and the
  facts the character has learned are re-rendered into that instruction before
  each answer, so its memory grows with the conversation;
* it **records each exchange** into the character's own learning file -- the
  question, the answer, and where the answer came from (the web, a learned
  statement, cited sources) -- so the character's memory is durable and the same
  files train the project's own model (see
  :func:`core.character_learning.character_learning_sentences`).

The wrapper is deliberately transparent: an unknown attribute is forwarded to the
mouth it wraps, so every existing caller (the Jarvis loop, the switcher, the
service's ``_retriever``) keeps working unchanged. It never widens a character's
authority: the persona is *guidance for how to speak*, and the mouth's own
honesty rules (the groundedness gate, the cloud approval gate, a character's own
boundaries) still run underneath it.

**The personality is configuration-driven.** The standing instruction is rendered
from the character's saved :class:`~core.character_config.CharacterConfig` (loaded
through :mod:`core.character_registry`) rather than from the hard-coded persona
alone, so an edit in the Characters window really changes how the character speaks
-- and the persona's own identity, traits, and rules remain the *defaults*, so a
character that was never edited behaves exactly as before. The character's learning
records are written through :mod:`core.learning_pipeline` into its own isolated
:mod:`core.character_learning_store`, so the same exchange that is remembered in the
character's facts/exchanges file is also a learning record with provenance.

Nothing here imports Tk, and no model is loaded by importing this module, so
building a character mouth is cheap and testable with no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.best_effort import best_effort
from core.character_learning import (
    CHARACTER_KEYS,
    PROJECT_MODEL_KEY,
    CharacterLearning,
    persona_for,
)

#: The mouths that are *characters* with a learning record. The project's own
#: model is included, so its own exchanges are remembered too, but the corpus
#: bridge that trains it reads the seven characters (see
#: :data:`core.character_learning.CHARACTER_KEYS`).
CHARACTER_MOUTHS: tuple[str, ...] = (*CHARACTER_KEYS, PROJECT_MODEL_KEY)


def is_character(mouth: str) -> bool:
    """True when ``mouth`` is one of the characters (or the project's own model)."""
    return mouth in CHARACTER_MOUTHS


class CharacterMouth:
    """A mouth's assistant with a character's personality, memory, and learning.

    ``ask`` behaves exactly like the wrapped assistant's, then records the
    exchange; every other attribute is forwarded, so the wrapper is invisible to
    callers. The character's standing instruction is re-rendered before each
    answer, so a fact learned earlier in the conversation is in force for the next
    one.
    """

    def __init__(
        self,
        key: str,
        assistant: Any,
        *,
        root: Any = None,
        in_memory: bool = False,
        registry: Any = None,
        learning_store: Any = None,
    ) -> None:
        self.key = key
        self.assistant = assistant
        self.learning = CharacterLearning(key, root=root, in_memory=in_memory)
        self._persona = persona_for(key)
        # The editable configuration drives the standing instruction. It is loaded
        # through the one registry (so an edit in the Characters window is in force
        # on the next answer) and falls back to the persona-derived default, so a
        # character that was never edited behaves exactly as before.
        self._registry = registry
        self._config: Any = None
        self._load_config()
        # The learning pipeline writes a provenance-carrying record for each
        # exchange into the character's *own* isolated store. The store is built
        # from the same ``root`` (a test's temp folder), so it can never be pointed
        # at another character's directory.
        self._pipeline: Any = None
        self._learning_store = learning_store
        self._build_pipeline()
        # The base instruction the mouth was built with, kept so the character's
        # own words are always *added in front of* it rather than replacing the
        # mouth's honesty rules.
        engine = getattr(assistant, "engine", None)
        self._base_prompt = getattr(engine, "system_prompt", None)
        self._base_instruction = getattr(assistant, "system_instruction", None)
        self._apply_instruction()

    # ---------------------------------------------------------- configuration

    def _load_config(self) -> None:
        """Load the character's effective configuration (never raising)."""
        try:
            from core.character_registry import default_registry

            registry = self._registry if self._registry is not None else default_registry()
            self._config = registry.load(self.key)
        except (ValueError, OSError):
            self._config = None

    def _build_pipeline(self) -> None:
        """Build the learning pipeline for this character's own store."""
        try:
            from core.character_learning_store import CharacterLearningStore
            from core.learning_pipeline import LearningPipeline

            store = self._learning_store
            if store is None:
                store = CharacterLearningStore(self.key, root=self._store_root())
            self._pipeline = LearningPipeline(self.key, store=store, config=self._config)
        except (ValueError, OSError):
            self._pipeline = None

    def _store_root(self) -> Any:
        """The data root the learning store should use (a test's, or the default)."""
        # ``CharacterLearning`` keeps its own root; reuse the same base so a test
        # that passes a temp ``root`` keeps *both* stores inside it. The record
        # exposes its root (``None`` for an in-memory record, which the store reads
        # as "the default data directory").
        root = getattr(self.learning, "root", None)
        if root is None:
            return None
        # ``CharacterLearning`` stores under ``<root>/characters/learning``; the
        # isolated learning store belongs under the data root itself.
        root = Path(root)
        if root.name == "learning" and root.parent.name == "characters":
            return root.parent.parent
        return root

    @property
    def config(self) -> Any:
        """The character's current configuration (or ``None`` if unavailable)."""
        return self._config

    def reload_config(self) -> Any:
        """Re-read the configuration, so an APPLY takes effect on the next answer."""
        self._load_config()
        if self._pipeline is not None:
            self._pipeline.set_config(self._config)
        return self._config

    # ------------------------------------------------------------ personality

    def _apply_instruction(self) -> None:
        """Render the character's instruction (with its facts) into the mouth."""
        # Re-read the record from disk first: a fact the character learned in
        # another instance of the program (or through the CLI) must be in force
        # for this answer, not only the facts this wrapper happened to load.
        self.learning.reload()
        instruction = self.system_prompt()
        engine = getattr(self.assistant, "engine", None)
        if engine is not None and isinstance(self._base_prompt, str):
            engine.system_prompt = f"{instruction}\n\n{self._base_prompt}"
        if isinstance(self._base_instruction, str):
            self.assistant.system_instruction = f"{instruction}\n\n{self._base_instruction}"

    def system_prompt(self) -> str:
        """The character's current standing instruction, for inspection.

        Rendered from the editable configuration when one is available, and from
        the character's persona otherwise, so the two always agree by default.
        """
        facts = self.learning.facts()
        config = self._config
        if config is not None:
            return str(config.system_prompt(learned_facts=facts))
        return self._persona.system_prompt(learned_facts=facts)

    # ---------------------------------------------------------------- asking

    def ask(self, question: str, **kwargs: Any):
        """Answer as the wrapped mouth, then remember the exchange."""
        self._apply_instruction()
        answer = self.assistant.ask(question, **kwargs)
        text = str(getattr(answer, "text", "") or "")
        used_web = bool(getattr(answer, "used_web", False))
        learned = bool(getattr(answer, "learned", False))
        citations = tuple(getattr(answer, "citations", ()) or ())
        self.learning.remember(
            question,
            text,
            used_web=used_web,
            learned=learned,
            citations=citations,
        )
        self._record_learning(question, text, used_web=used_web, citations=citations)
        return answer

    def _record_learning(
        self,
        question: str,
        text: str,
        *,
        used_web: bool,
        citations: tuple[str, ...],
    ) -> None:
        """Write a provenance-carrying learning record for this exchange.

        The record's *source* is honest: an answer the mouth got from the web is
        ``internet`` (with the citations as its evidence), otherwise it is the
        mouth's own language model. A record is never born verified, and a failure
        here is best-effort -- a learning record must never break a chat.
        """
        pipeline = self._pipeline
        if pipeline is None or not text.strip():
            return
        from core.learning_records import SOURCE_LANGUAGE_MODEL

        with best_effort(f"record {self.key}'s exchange as a learning record"):
            if used_web:
                pipeline.learn_from_internet(
                    question=question,
                    answer=text,
                    url=citations[0] if citations else "",
                    evidence=citations,
                )
            else:
                pipeline.learn(
                    source_type=SOURCE_LANGUAGE_MODEL,
                    question=question,
                    answer=text,
                    source_reference=self._model_reference(),
                )

    def _model_reference(self) -> str:
        """A readable name for the model that answered, for the record's provenance."""
        config = self._config
        if config is not None:
            model = config.model
            if model.key:
                return f"language model {model.key}"
            if model.model:
                return f"language model {model.model}"
        return "the mouth's own language model"

    # ------------------------------------------------------- transparent proxy

    #: The attributes that belong to the wrapper itself; every other read *and
    #: write* is forwarded to the wrapped mouth, so a caller that sets
    #: ``assistant.web_enabled`` (the Jarvis ``web on`` control) reaches the real
    #: assistant rather than a shadow copy on the wrapper.
    _INTERNAL = frozenset(
        {
            "key",
            "assistant",
            "learning",
            "_persona",
            "_base_prompt",
            "_base_instruction",
            "_registry",
            "_config",
            "_pipeline",
            "_learning_store",
        }
    )

    def __getattr__(self, name: str) -> Any:
        # ``assistant`` is set in __init__; anything else is the wrapped mouth's.
        return getattr(object.__getattribute__(self, "assistant"), name)

    @property  # type: ignore[misc]
    def __class__(self):  # type: ignore[override]
        """Report the wrapped assistant's own class.

        The wrapper is transparent: ``isinstance(character, GroundedAssistant)``
        and ``character.__class__`` answer for the real mouth, so a caller (or a
        test) that checks the mouth's type keeps working unchanged.
        """
        return object.__getattribute__(self, "assistant").__class__

    def __setattr__(self, name: str, value: Any) -> None:
        if name in CharacterMouth._INTERNAL or "assistant" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.assistant, name, value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CharacterMouth({self.key!r}, {self.assistant!r})"


def wrap_character(
    mouth: str,
    assistant: Any,
    *,
    root: Any = None,
    in_memory: bool = False,
    registry: Any = None,
    learning_store: Any = None,
) -> Any:
    """Wrap ``assistant`` as ``mouth``'s character, or return it unchanged.

    A mouth that is not a character is returned as-is, so this is safe to call on
    every mouth.
    """
    if not is_character(mouth):
        return assistant
    return CharacterMouth(
        mouth,
        assistant,
        root=root,
        in_memory=in_memory,
        registry=registry,
        learning_store=learning_store,
    )


__all__ = [
    "CHARACTER_MOUTHS",
    "CharacterMouth",
    "is_character",
    "wrap_character",
]
