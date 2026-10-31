"""One identity per character, shared by every surface that shows it.

The project talks to the same seven characters from several places: the main
chat window (the header's model selector and the composer's model row), each
character's **own** dedicated window, and the **Characters** window and CLI. Until
this module existed those surfaces each kept their own idea of a character:

* the memory was keyed on the *mouth* key in one place (``sources``) and on the
  *owner* name in another (``jarvis``), so a fact a character learned in one
  surface was invisible in the other -- two files for one character;
* the display name lived in two stores (the mouth profile and the character
  configuration), so renaming a character in one window left the other showing
  the old name.

This module is the **one** place that answers "who is this character": its owner
name, its chat mouth, its display name, its role, and whether it is private. Every
surface resolves a character through :func:`identity_for`, so the two windows can
never disagree, and :func:`identity_signature` / :func:`same_character` make the
equality checkable rather than assumed.

It is dependency-free and Tk-free, and it loads no model, so importing it is free
and a test can exercise it with no display.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.character_learning import CHARACTER_KEYS, PROJECT_MODEL_KEY, persona_for
from core.learning_isolation import (
    CHARACTER_OWNERS,
    MAIN_MODEL_OWNER,
    is_private,
    mouth_key,
    owner_for,
)

#: Every owner whose identity this module resolves: the seven characters, then the
#: project's own model.
IDENTITY_OWNERS: tuple[str, ...] = (*CHARACTER_OWNERS, MAIN_MODEL_OWNER)


@dataclass(frozen=True)
class CharacterIdentity:
    """Who a character is, whichever surface is asking.

    ``owner`` is the learning/directory name (``jarvis``), ``mouth`` the chat key
    (``sources``), and ``name`` the display name -- read from the character's own
    configuration when one is available, else the persona's. ``role`` and
    ``private`` round it out. Two surfaces that resolve the same character get
    equal values here, which is what "one identity" means in practice.
    """

    owner: str
    mouth: str
    name: str
    role: str
    private: bool

    def signature(self) -> tuple[str, str, str]:
        """The stable identity triple two surfaces must agree on."""
        return (self.owner, self.mouth, self.name)

    def describe(self) -> str:
        private = ", private records" if self.private else ""
        return f"{self.name} ({self.owner} / {self.mouth}){private}"


def identity_for(key: str, *, registry: object | None = None) -> CharacterIdentity:
    """Resolve ``key`` (either vocabulary) to its one identity.

    The name is read from the character's effective configuration -- the file the
    Characters window edits -- so a rename there is the name every window shows;
    with no configuration available the persona's name is used, so an unknown or
    unconfigured character still resolves rather than raising.
    """
    owner = owner_for(key)
    mouth = mouth_key(owner)
    persona = persona_for(mouth)
    name = _configured_name(owner, registry) or persona.name
    return CharacterIdentity(
        owner=owner,
        mouth=mouth,
        name=name,
        role=persona.role,
        private=is_private(owner),
    )


def _configured_name(owner: str, registry: object | None) -> str:
    """The character's configured name, or ``""`` when none can be read."""
    try:
        if registry is None:
            from core.character_registry import default_registry

            registry = default_registry()
        return str(registry.load(owner).general.name).strip()  # type: ignore[attr-defined]
    except (ValueError, OSError, AttributeError):
        return ""


def all_identities(*, registry: object | None = None) -> tuple[CharacterIdentity, ...]:
    """Every character's identity, in the project's order (characters then model)."""
    return tuple(identity_for(owner, registry=registry) for owner in IDENTITY_OWNERS)


def character_mouths() -> tuple[str, ...]:
    """The chat mouth keys of the seven characters (not the project's own model)."""
    return tuple(CHARACTER_KEYS)


def project_model_identity(*, registry: object | None = None) -> CharacterIdentity:
    """The identity of the project's own model (the one that learns from them)."""
    return identity_for(PROJECT_MODEL_KEY, registry=registry)


def identity_signature(key: str, *, registry: object | None = None) -> tuple[str, str, str]:
    """The identity triple for ``key`` -- the value two surfaces must agree on."""
    return identity_for(key, registry=registry).signature()


def same_character(
    key_a: str,
    key_b: str,
    *,
    registry: object | None = None,
) -> bool:
    """True when two keys name the **same** character.

    ``same_character("sources", "jarvis")`` is ``True``: the mouth key and the
    owner name are two spellings of one identity, which is exactly the fact the
    two windows must agree on.
    """
    return identity_for(key_a, registry=registry).owner == identity_for(
        key_b, registry=registry
    ).owner


__all__ = [
    "IDENTITY_OWNERS",
    "CharacterIdentity",
    "all_identities",
    "character_mouths",
    "identity_for",
    "identity_signature",
    "project_model_identity",
    "same_character",
]
