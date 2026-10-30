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

Nothing here imports Tk, and no model is loaded by importing this module, so
building a character mouth is cheap and testable with no display.
"""

from __future__ import annotations

from typing import Any

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
    ) -> None:
        self.key = key
        self.assistant = assistant
        self.learning = CharacterLearning(key, root=root, in_memory=in_memory)
        self._persona = persona_for(key)
        # The base instruction the mouth was built with, kept so the character's
        # own words are always *added in front of* it rather than replacing the
        # mouth's honesty rules.
        engine = getattr(assistant, "engine", None)
        self._base_prompt = getattr(engine, "system_prompt", None)
        self._base_instruction = getattr(assistant, "system_instruction", None)
        self._apply_instruction()

    # ------------------------------------------------------------ personality

    def _apply_instruction(self) -> None:
        """Render the character's instruction (with its facts) into the mouth."""
        # Re-read the record from disk first: a fact the character learned in
        # another instance of the program (or through the CLI) must be in force
        # for this answer, not only the facts this wrapper happened to load.
        self.learning.reload()
        instruction = self._persona.system_prompt(learned_facts=self.learning.facts())
        engine = getattr(self.assistant, "engine", None)
        if engine is not None and isinstance(self._base_prompt, str):
            engine.system_prompt = f"{instruction}\n\n{self._base_prompt}"
        if isinstance(self._base_instruction, str):
            self.assistant.system_instruction = f"{instruction}\n\n{self._base_instruction}"

    def system_prompt(self) -> str:
        """The character's current standing instruction, for inspection."""
        return self._persona.system_prompt(learned_facts=self.learning.facts())

    # ---------------------------------------------------------------- asking

    def ask(self, question: str, **kwargs: Any):
        """Answer as the wrapped mouth, then remember the exchange."""
        self._apply_instruction()
        answer = self.assistant.ask(question, **kwargs)
        self.learning.remember(
            question,
            str(getattr(answer, "text", "") or ""),
            used_web=bool(getattr(answer, "used_web", False)),
            learned=bool(getattr(answer, "learned", False)),
            citations=tuple(getattr(answer, "citations", ()) or ()),
        )
        return answer

    # ------------------------------------------------------- transparent proxy

    #: The attributes that belong to the wrapper itself; every other read *and
    #: write* is forwarded to the wrapped mouth, so a caller that sets
    #: ``assistant.web_enabled`` (the Jarvis ``web on`` control) reaches the real
    #: assistant rather than a shadow copy on the wrapper.
    _INTERNAL = frozenset(
        {"key", "assistant", "learning", "_persona", "_base_prompt", "_base_instruction"}
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
) -> Any:
    """Wrap ``assistant`` as ``mouth``'s character, or return it unchanged.

    A mouth that is not a character is returned as-is, so this is safe to call on
    every mouth.
    """
    if not is_character(mouth):
        return assistant
    return CharacterMouth(mouth, assistant, root=root, in_memory=in_memory)


__all__ = [
    "CHARACTER_MOUTHS",
    "CharacterMouth",
    "is_character",
    "wrap_character",
]
