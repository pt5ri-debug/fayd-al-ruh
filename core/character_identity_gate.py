"""The held-out gate for character identity reaching an answer.

The task's third metric: a character's personality must actually reach the
**text of its answer**, not merely sit in a configuration file. This module is
the catalogue -- a held-out set of hand-written judgements about what the
character wiring must do -- scored as one number and wired into ``evaluation``,
so a regression in the identity is a **gate failure**, not a passing suite.

**What is measured, and how it is honest.** A character's answer is produced by a
language model, which CI does not run. So the gate uses the one faithful seam the
project already has (see :mod:`ai.characters`): a mouth that answers *from the
instruction it was given* -- a small stand-in that returns its own system prompt,
exactly as a model conditioned on that prompt would open its reply. Every
judgement builds the **real** :class:`~ai.characters.CharacterMouth` over that
mouth and reads what the wrapper actually wrote onto the engine and what the
answer actually carried, so a wrapper that stopped applying a character's
identity would fail here even though the persona object still existed.

Several judgements are **negative controls**: a character's identity must not
contain *another* character's name (the identity does not bleed across mouths),
an unknown key must fall back to the project's own model rather than a wrong
character, and a non-character mouth must be left unwrapped.

The catalogue also checks the **unification** this phase adds: the mouth key and
the owner name resolve to one character, a fact written through one vocabulary is
visible through the other, and a rename in one store is the name the other store
reports.

The module is dependency-free, Tk-free, and loads no model, so it runs on a fresh
clone with no display and no weights.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.character_identity import (
    IDENTITY_OWNERS,
    all_identities,
    identity_for,
    same_character,
)
from core.character_learning import (
    CHARACTER_KEYS,
    PROJECT_MODEL_KEY,
    CharacterLearning,
    persona_for,
)

#: The owner keys, in the project's order, that every identity judgement walks.
_ALL_OWNERS: tuple[str, ...] = (*CHARACTER_KEYS, PROJECT_MODEL_KEY)


@dataclass(frozen=True)
class IdentityJudgement:
    """One held-out claim about the character identity, and how to check it."""

    name: str
    kind: str
    #: The character a per-character judgement exercises (unused by the global kinds).
    character: str = ""
    #: A short note, for a failure report.
    note: str = ""


#: The held-out catalogue. The ``reaches-*`` kinds are the task's own claim -- a
#: character's identity reaches the answer text; the ``unify-*`` kinds are the
#: one-identity-across-windows property; the rest are the negative controls.
IDENTITY_JUDGEMENTS: tuple[IdentityJudgement, ...] = (
    # -- every character has its own, distinct identity -----------------------
    IdentityJudgement("the seven characters and the project model all resolve", "resolve-all"),
    IdentityJudgement("every identity is distinct", "distinct"),
    IdentityJudgement(
        "a mouth key and its owner name are the same character",
        "same-character",
        note="sources and jarvis must be one identity, not two",
    ),
    # -- the identity reaches the instruction the model reads -----------------
    IdentityJudgement("ultron's identity reaches the mouth's instruction", "instruction", character="ultron"),
    IdentityJudgement("vision's identity reaches the mouth's instruction", "instruction", character="vision"),
    IdentityJudgement("banner's identity reaches the mouth's instruction", "instruction", character="banner"),
    IdentityJudgement("natasha's identity reaches the mouth's instruction", "instruction", character="natasha"),
    IdentityJudgement("hakeem's identity reaches the mouth's instruction", "instruction", character="hakeem"),
    IdentityJudgement("cloud's identity reaches the mouth's instruction", "instruction", character="cloud"),
    IdentityJudgement("jarvis's identity reaches the mouth's instruction", "instruction", character="sources"),
    # -- and it reaches the *answer text* -------------------------------------
    IdentityJudgement("ultron's identity reaches its answer text", "answer", character="ultron"),
    IdentityJudgement("vision's identity reaches its answer text", "answer", character="vision"),
    IdentityJudgement("banner's identity reaches its answer text", "answer", character="banner"),
    IdentityJudgement("natasha's identity reaches its answer text", "answer", character="natasha"),
    IdentityJudgement("hakeem's identity reaches its answer text", "answer", character="hakeem"),
    IdentityJudgement("cloud's identity reaches its answer text", "answer", character="cloud"),
    IdentityJudgement("jarvis's identity reaches its answer text", "answer", character="sources"),
    # -- the character's memory reaches the instruction ------------------------
    IdentityJudgement(
        "a fact a character is told reaches its next instruction",
        "memory-reaches",
        character="vision",
    ),
    # -- negative controls ----------------------------------------------------
    IdentityJudgement(
        "a character's instruction names no other character",
        "no-bleed",
        note="negative control: identity must not bleed across mouths",
    ),
    IdentityJudgement(
        "an unknown key falls back to the project's own model",
        "unknown-falls-back",
        note="negative control: an unknown character is never a wrong character",
    ),
    IdentityJudgement(
        "a non-character mouth is left unwrapped",
        "non-character-untouched",
        note="negative control: wrapping must be safe on every mouth",
    ),
    # -- one identity across the two windows (the unification) ----------------
    IdentityJudgement(
        "a fact written under the owner name is read under the mouth key",
        "unify-memory",
        character="sources",
        note="the two windows must share one memory file",
    ),
    IdentityJudgement(
        "a rename in one store is the name the other reports",
        "unify-name",
        character="ultron",
        note="the two windows must share one display name",
    ),
    IdentityJudgement(
        "two live writers to one character's memory do not clobber each other",
        "unify-concurrent",
        character="hakeem",
        note="one file, two windows: neither write may drop the other's fact",
    ),
)


class _EchoEngine:
    """A minimal engine: it carries a system prompt, nothing else."""

    def __init__(self) -> None:
        self.system_prompt: str = ""


class _EchoAnswer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.citations: tuple[str, ...] = ()
        self.used_web = False
        self.learned = False


class _EchoMouth:
    """A faithful stand-in: it answers *from the instruction it was given*.

    A real model conditioned on its system prompt opens its reply from that
    prompt; returning the prompt is the one deterministic stand-in for that, so a
    judgement can check that a character's identity reached the *answer text* --
    not only the prompt object -- with no model and no weights. It is documented
    as a seam, not a mock of the character wiring: the wiring under test (the
    wrapper applying the identity to the engine) is entirely real.
    """

    def __init__(self) -> None:
        self.engine = _EchoEngine()

    def ask(self, question: str, **kwargs: Any) -> _EchoAnswer:
        return _EchoAnswer(self.engine.system_prompt)


@contextmanager
def _temp_root() -> Iterator[Path]:
    """A throwaway data root, with ``FAYD_DATA_DIR`` pointed at it for the block.

    Every store the identity touches -- the learning record, the mouth profile,
    and the character configuration -- resolves its folder from the data
    directory, so pointing that at a temp folder keeps the whole judgement inside
    it. The previous value is restored even if the body raises, so one judgement
    can never leak a data root into the next.
    """
    import os

    previous = os.environ.get("FAYD_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="fayd-identity-") as folder:
        root = Path(folder)
        os.environ["FAYD_DATA_DIR"] = str(root)
        try:
            yield root
        finally:
            if previous is None:
                os.environ.pop("FAYD_DATA_DIR", None)
            else:
                os.environ["FAYD_DATA_DIR"] = previous


def _instruction_for(character: str, root: Path) -> str:
    """The instruction the wrapper actually wrote onto the mouth's engine."""
    from ai.characters import wrap_character

    mouth = _EchoMouth()
    character_mouth = wrap_character(character, mouth, root=root)
    written = str(getattr(mouth.engine, "system_prompt", "") or "")
    if written:
        return written
    reader = getattr(character_mouth, "system_prompt", None)
    return str(reader()) if callable(reader) else ""


def _judgement_holds(judgement: IdentityJudgement) -> bool:
    """Run one judgement against the real wiring; ``True`` when it holds."""
    kind = judgement.kind

    if kind == "resolve-all":
        return all(identity_for(owner).name for owner in IDENTITY_OWNERS)

    if kind == "distinct":
        signatures = {identity.signature() for identity in all_identities()}
        return len(signatures) == len(IDENTITY_OWNERS)

    if kind == "same-character":
        return same_character("sources", "jarvis") and same_character("model", "fayd_al_ruh")

    if kind == "instruction":
        with _temp_root() as root:
            instruction = _instruction_for(judgement.character, root)
        persona = persona_for(judgement.character)
        return (
            persona.name in instruction
            and persona.role in instruction
            and any(trait in instruction for trait in persona.traits)
        )

    if kind == "answer":
        with _temp_root() as root:
            from ai.characters import wrap_character

            mouth = _EchoMouth()
            character_mouth = wrap_character(judgement.character, mouth, root=root)
            answer = character_mouth.ask("tell me about yourself")
        persona = persona_for(judgement.character)
        text = str(getattr(answer, "text", "") or "")
        return persona.name in text and persona.role in text

    if kind == "memory-reaches":
        with _temp_root() as root:
            from ai.characters import wrap_character

            CharacterLearning(judgement.character, root=root).add_fact("the field is in the north")
            mouth = _EchoMouth()
            character_mouth = wrap_character(judgement.character, mouth, root=root)
            character_mouth.ask("where is the field?")
            instruction = str(mouth.engine.system_prompt or "")
        return "the field is in the north" in instruction

    if kind == "no-bleed":
        with _temp_root() as root:
            other_names = {
                identity.name
                for identity in all_identities()
                if identity.owner != "ultron"
            }
            instruction = _instruction_for("ultron", root)
        return all(name not in instruction for name in other_names)

    if kind == "unknown-falls-back":
        # An unknown key is not a character, so the wrapper leaves the mouth alone
        # (the negative control below) -- but the *identity resolver* must never
        # answer with a wrong character: it falls back to the project's own model.
        return identity_for("not-a-real-character").name == persona_for(PROJECT_MODEL_KEY).name

    if kind == "non-character-untouched":
        from ai.characters import wrap_character

        mouth = _EchoMouth()
        return wrap_character("not-a-character", mouth) is mouth

    if kind == "unify-memory":
        with _temp_root() as root:
            # Write under the owner name (the Characters surface's vocabulary)...
            CharacterLearning("jarvis", root=root).add_fact("the user works offshore")
            # ...and read it back under the mouth key (the chat's vocabulary).
            seen = CharacterLearning("sources", root=root).facts()
        return "the user works offshore" in seen

    if kind == "unify-name":
        with _temp_root() as root:
            from core import mouth_profile as profiles
            from core.character_registry import CharacterRegistry

            profiles.set_name("ultron", "Ultra")
            profile_name = profiles.load_profile("ultron").name
            config_name = CharacterRegistry(data_root=root).load("ultron").general.name
            # The identity the surfaces resolve must agree with both stores.
            identity_name = identity_for("ultron", registry=CharacterRegistry(data_root=root)).name
        return profile_name == "Ultra" and config_name == "Ultra" and identity_name == "Ultra"

    if kind == "unify-concurrent":
        with _temp_root() as root:
            first = CharacterLearning(judgement.character, root=root)
            second = CharacterLearning(judgement.character, root=root)
            first.add_fact("fact from the first window")
            second.add_fact("fact from the second window")
            first.reload()
            merged = set(first.facts())
        return {"fact from the first window", "fact from the second window"} <= merged

    return False


def character_identity_accuracy(
    judgements: tuple[IdentityJudgement, ...] = IDENTITY_JUDGEMENTS,
) -> float:
    """The share of held-out identity judgements the wiring satisfies.

    A judgement counts only when the real wrapper does exactly what it must -- the
    character's identity reaches the instruction *and* the answer text, no other
    character's name leaks in, an unknown key falls back to the project's own
    model, and the two windows share one memory and one name. So a regression in
    the identity wiring fails the relevant judgement. Returns ``0.0`` for an empty
    catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "IDENTITY_JUDGEMENTS",
    "IdentityJudgement",
    "character_identity_accuracy",
]
