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
and does three things, once, for every mouth:

* it **prepends the character's standing instruction** to the model the mouth
  speaks through (the engine's ``system_prompt``, the grounded assistant's
  ``system_instruction``, or -- for a mouth that exposes neither, such as the
  companion or the scholar -- a ``system_prompt`` the wrapper itself supplies and
  the mouth's own prompt builder reads), so the character's traits shape every
  reply;
* it **recalls the character's own memory into that instruction**: the durable
  facts it was told *and* a bounded window of the exchanges it took part in, so a
  character remembers the conversation rather than only recording it -- and the
  recall is re-rendered as the memory grows;
* it **records each exchange** into the character's own learning file -- the
  question, the answer (with the mouth's honesty header stripped, so the stored
  text is the model's own words), and where the answer came from (the web, a
  learned statement, cited sources) -- so the character's memory is durable and the
  same files train the project's own model (see
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

#: How many of a character's own past exchanges are recalled into its standing
#: instruction. A character's memory is meant to *reach its prompt*; the window is
#: small on purpose, so recalling memory never floods a small model's context.
RECALL_LIMIT = 3

#: Re-render the standing instruction (and therefore the recalled memory) every
#: this-many turns. Re-rendering on *every* turn would append the previous answer
#: to the instruction it produced, which a small model can learn to echo; a small
#: period keeps the recall current without that trap.
RECALL_REFRESH_TURNS = 5


def _honesty_headers() -> frozenset[str]:
    """Every mouth's honesty header, read from the one place each is defined.

    Collecting them (rather than trying one mouth's stripper) means an answer is
    stripped of whichever header it actually carries, whatever mouth wrote it.
    """
    labels: set[str] = set()
    for module_name, attribute in (
        ("ai.ultron_mouth", "ULTRON_HEADER"),
        ("ai.vision_mouth", "VISION_HEADER"),
        ("ai.banner_mouth", "BANNER_HEADER"),
        ("ai.cloud_mouth", "CLOUD_HEADER"),
        ("ai.natasha_mouth", "HEADER"),
        ("ai.hakeem_mouth", "HEADER"),
        ("ai.mind", "MIND_HEADER"),
    ):
        try:
            import importlib

            module = importlib.import_module(module_name)
            value = getattr(module, attribute)
        except (ImportError, AttributeError):
            continue
        if isinstance(value, str) and value:
            labels.add(value)
    return frozenset(labels)


def _strip_honesty_header(text: str) -> str:
    """Drop a mouth's honesty header from an answer before it is remembered.

    Each model-backed mouth writes a header above its answer (``[ultron] answered
    by the local general model; not grounded in cited sources.``). The header is
    honest on screen, but it is *not* something the character said, so storing it
    would put a bracketed label into the character's memory and into the training
    text the project's own model learns from. A mouth without one is left as-is.
    """
    if not text:
        return text
    labels = _honesty_headers()
    lines = text.splitlines()
    while lines and lines[0].strip() in labels:
        lines.pop(0)
    return "\n".join(lines).strip()


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
        # Whether the wrapper had to *supply* the mouth's ``system_prompt`` because
        # the mouth exposed none (the companion and the scholar read their
        # instruction through their own personality, so the wrapper writes the
        # character block onto the assistant and their prompt builder reads it).
        # Remembered so a re-render overwrites the wrapper's own block rather than
        # appending to it a second time.
        self._supplied_prompt = (
            engine is None
            and not isinstance(self._base_instruction, str)
        )
        # The turn counter that paces the recall re-render, and the last question
        # the wrapper asked the mouth (so the just-answered exchange is never fed
        # back to the model that produced it).
        self._turns = 0
        self._last_question = ""
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

    def _apply_instruction(self, *, exclude_questions: tuple[str, ...] = ()) -> None:
        """Render the character's instruction (with its facts and memory) into the mouth."""
        # Re-read the record from disk first: a fact the character learned in
        # another instance of the program (or through the CLI) must be in force
        # for this answer, not only the facts this wrapper happened to load.
        self.learning.reload()
        instruction = self.system_prompt(exclude_questions=exclude_questions)
        engine = getattr(self.assistant, "engine", None)
        if engine is not None and isinstance(self._base_prompt, str):
            engine.system_prompt = f"{instruction}\n\n{self._base_prompt}"
        elif self._supplied_prompt:
            # The mouth exposed no engine prompt (the companion, the scholar), so
            # the wrapper supplies one and the mouth's own prompt builder reads it.
            self.assistant.system_prompt = instruction
        if isinstance(self._base_instruction, str):
            self.assistant.system_instruction = f"{instruction}\n\n{self._base_instruction}"

    def system_prompt(self, *, exclude_questions: tuple[str, ...] = ()) -> str:
        """The character's current standing instruction, for inspection.

        Rendered from the editable configuration when one is available, and from
        the character's persona otherwise, so the two always agree by default. The
        character's durable facts and a bounded window of its own past exchanges
        are rendered in, so its memory is in force for the next answer.
        """
        facts = self.learning.facts()
        recalled = tuple(
            exchange.recall_line()
            for exchange in self.learning.recall(
                limit=RECALL_LIMIT, exclude_questions=exclude_questions
            )
        )
        config = self._config
        if config is not None:
            return str(config.system_prompt(learned_facts=facts, recalled_memory=recalled))
        return self._persona.system_prompt(learned_facts=facts, recalled_memory=recalled)

    # ---------------------------------------------------------------- asking

    def ask(self, question: str, **kwargs: Any):
        """Answer as the wrapped mouth, then remember the exchange.

        The standing instruction is re-rendered before the answer so the character's
        memory is in force -- but the re-render (which now carries the recalled
        memory) is paced: doing it on *every* turn would append the previous answer
        to the very instruction that produced it, which a small model can learn to
        echo. Facts and an explicit reload still take effect at once; the recalled
        window is refreshed every few turns, and always on the first answer.
        """
        refresh = self._turns % RECALL_REFRESH_TURNS == 0
        if refresh:
            self._apply_instruction(exclude_questions=(self._last_question, question))
        else:
            self.learning.reload()
        self._turns += 1
        self._last_question = question
        answer = self.assistant.ask(question, **kwargs)
        raw = str(getattr(answer, "text", "") or "")
        # The mouth's honesty header is honest on screen but is not the character's
        # own words, so it is stripped before the exchange is remembered -- keeping
        # it out of the character's memory and out of the training text built from it.
        text = _strip_honesty_header(raw) or raw
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
        # The character's own configuration governs whether it learns at all: a
        # character whose Learning section is switched off keeps a memory (the
        # exchange is still remembered above) but writes no learning record, so the
        # switch in the Characters window really turns learning off.
        config = self._config
        if config is not None and not config.learning.enabled:
            return
        if used_web and config is not None and not config.learning.learn_from_internet:
            # A character not allowed to learn from the internet still records what
            # it said, but as its own model's answer -- never as web knowledge.
            used_web = False
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
            "_supplied_prompt",
            "_turns",
            "_last_question",
        }
    )

    # --------------------------------------------------------------- memory API

    def remember_fact(self, text: str) -> bool:
        """Teach this character one durable fact; ``True`` when it was new.

        This is the one write path a caller needs to make a character's memory
        grow from outside a chat -- the CLI (``characters remember``) and the
        Characters window both call it, and the fact is in force on the next answer
        because the standing instruction is re-rendered from the record.
        """
        stored = self.learning.add_fact(text)
        if stored:
            self._apply_instruction()
        return stored

    def facts(self) -> tuple[str, ...]:
        """The durable facts this character has been told."""
        return self.learning.facts()

    def forget_facts(self) -> None:
        """Forget everything this character learned (its facts and exchanges)."""
        self.learning.clear()
        self._apply_instruction()

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
