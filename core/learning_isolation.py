"""The isolation rules between the seven characters and the main model.

The task's security section is explicit, and it must hold in the **backend**, not
merely be hidden in the user interface:

* a character can write only to its **own** learning directory;
* a character cannot read or modify another character's records;
* the main Fayd Al-Ruh model may read permitted knowledge from **six** characters
  -- Jarvis, Ultron, Vision, Banner, Hakeem, and Cloud;
* the main model must **never** read Natasha's records;
* the main model writes everything it acquires into its **own** store.

This module is the one place those rules are written down. It is pure policy --
no file is opened here -- so it is dependency-free, testable on its own, and
cannot be bypassed by a caller that skips a check: a store resolves its directory
from :func:`store_dir`, and the sharing service asks :func:`decide` before it
touches a path at all.

**Two vocabularies, one identity.** The chat calls the answering mouths by the keys
it has always used (``sources`` is Jarvis, ``model`` is the project's own model),
while the task names the characters and their directories directly (``jarvis``,
``fayd_al_ruh``). :data:`OWNERS` are the *directory* names -- so the on-disk layout
is exactly ``data/learning/fayd_al_ruh/characters/jarvis/`` -- and
:func:`owner_for` accepts either vocabulary, so no caller has to remember which one
it holds.

``PRIVATE_OWNERS`` is deliberately a **deny-list of one** rather than an allow-list:
the exception is Natasha and only Natasha, exactly as the task states, and any
character added later is shareable by default (the safe direction: a new
character's knowledge reaches the main model, and marking one private is a
deliberate act).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.character_learning import PROJECT_MODEL_KEY
from core.paths import character_learning_store_dir, main_model_learning_dir

#: The directory/owner name of the project's own model -- the one that learns
#: *from* the characters. The task names its directory ``fayd_al_ruh``.
MAIN_MODEL_OWNER = "fayd_al_ruh"

#: The mouth key the chat uses for the project's own model (``model``), kept so
#: both vocabularies resolve to the same owner.
MAIN_MODEL_MOUTH = PROJECT_MODEL_KEY

#: The character *directory* names, in the order the project shows the characters.
#: These are the names the task uses for the learning folders.
CHARACTER_OWNERS: tuple[str, ...] = (
    "jarvis",
    "ultron",
    "vision",
    "banner",
    "natasha",
    "hakeem",
    "cloud",
)

#: The chat's mouth key -> the owner (directory) name. ``sources`` is Jarvis; the
#: rest already match. This is the one mapping between the two vocabularies.
MOUTH_TO_OWNER: dict[str, str] = {
    "sources": "jarvis",
    "ultron": "ultron",
    "vision": "vision",
    "banner": "banner",
    "natasha": "natasha",
    "hakeem": "hakeem",
    "cloud": "cloud",
    PROJECT_MODEL_KEY: MAIN_MODEL_OWNER,
}

#: The reverse map, so a caller that holds an owner name can name its mouth.
OWNER_TO_MOUTH: dict[str, str] = {owner: mouth for mouth, owner in MOUTH_TO_OWNER.items()}

#: Every learning owner: the seven characters, then the project's own model.
OWNERS: tuple[str, ...] = (*CHARACTER_OWNERS, MAIN_MODEL_OWNER)

#: Owners whose records the main model must never read. **Natasha and only
#: Natasha**, exactly as the task states; the reason is recorded here so the rule
#: is explained where it is enforced.
PRIVATE_OWNERS: frozenset[str] = frozenset({"natasha"})

#: The characters the main model may learn from: the seven minus the private ones.
SHAREABLE_OWNERS: tuple[str, ...] = tuple(
    key for key in CHARACTER_OWNERS if key not in PRIVATE_OWNERS
)

#: Why Natasha is excluded, in one sentence, so a report can state it rather than
#: assert it.
PRIVATE_REASON = (
    "Natasha's records are private by design: the companion's conversations are "
    "personal, so the main model never retrieves, indexes, imports, or learns "
    "from them."
)

#: The same reason, keyed by owner, so a per-character report can state it without
#: repeating the string.
PRIVATE_REASONS: dict[str, str] = {"natasha": PRIVATE_REASON}


class LearningAccessError(PermissionError):
    """Raised when a read or write crosses a character's learning boundary.

    It subclasses :class:`PermissionError` so a caller that already handles a
    refused permission catches it too, while a caller that wants to be specific
    can catch this exact type.
    """


@dataclass(frozen=True)
class AccessDecision:
    """Whether ``reader`` may read ``owner``'s records, and why."""

    reader: str
    owner: str
    allowed: bool
    reason: str

    def describe(self) -> str:
        verdict = "allowed" if self.allowed else "denied"
        return f"{self.reader} -> {self.owner}: {verdict} ({self.reason})"


def owner_for(key: str) -> str:
    """The owner (directory) name for a mouth key **or** an owner name.

    Accepts either vocabulary -- ``"sources"`` and ``"jarvis"`` both give
    ``"jarvis"``; ``"model"`` and ``"fayd_al_ruh"`` both give ``"fayd_al_ruh"`` --
    so a caller never has to remember which one it holds. An unknown key is
    returned unchanged, and :func:`is_owner` is what rejects it.
    """
    cleaned = (key or "").strip()
    if cleaned in MOUTH_TO_OWNER:
        return MOUTH_TO_OWNER[cleaned]
    return cleaned


def is_owner(key: str) -> bool:
    """True when ``key`` (either vocabulary) names a known learning owner."""
    return owner_for(key) in OWNERS


def is_main_model(key: str) -> bool:
    """True when ``key`` is the project's own model (the cross-character reader)."""
    return owner_for(key) == MAIN_MODEL_OWNER


def is_private(key: str) -> bool:
    """True when ``key``'s records are never shared (Natasha)."""
    return owner_for(key) in PRIVATE_OWNERS


def is_shareable(key: str) -> bool:
    """True when the main model may read ``key``'s records."""
    return owner_for(key) in SHAREABLE_OWNERS


def mouth_key(owner: str) -> str:
    """The chat's mouth key for an owner name (``jarvis`` -> ``sources``)."""
    resolved = owner_for(owner)
    return OWNER_TO_MOUTH.get(resolved, resolved)


def store_dir(owner: str) -> Path:
    """The learning directory for ``owner`` -- the one it may read and write.

    A character's directory is its own isolated folder under
    ``data/learning/fayd_al_ruh/characters/<key>``; the main model's is
    ``data/learning/fayd_al_ruh/main``. Resolving the path in **one** place is
    what makes isolation structural: a store never takes a path from a caller.
    """
    resolved = owner_for(owner)
    if resolved not in OWNERS:
        raise LearningAccessError(f"{owner!r} is not a known learning owner.")
    if resolved == MAIN_MODEL_OWNER:
        return main_model_learning_dir() / "main"
    return character_learning_store_dir(resolved)


def store_dir_for(owner: str, data_root: Path | str | None = None) -> Path:
    """Like :func:`store_dir`, but under an explicit data root.

    A test (or a caller with its own data folder) passes ``data_root`` and gets the
    *same* isolated layout beneath it -- ``<data_root>/learning/fayd_al_ruh/main``
    or ``<data_root>/learning/fayd_al_ruh/characters/<key>``. The isolation
    property is preserved exactly: the owner still determines the path, so even a
    custom root cannot place one character's store where another would read it.
    """
    resolved = owner_for(owner)
    if resolved not in OWNERS:
        raise LearningAccessError(f"{owner!r} is not a known learning owner.")
    if data_root is None:
        return store_dir(resolved)
    base = Path(data_root) / "learning" / MAIN_MODEL_OWNER
    if resolved == MAIN_MODEL_OWNER:
        return base / "main"
    return base / "characters" / resolved


def decide(reader: str, owner: str) -> AccessDecision:
    """Whether ``reader`` may read ``owner``'s records.

    The rules, in order:

    * an unknown reader or owner is denied (nothing to read, or nothing to read
      it as);
    * a reader reading **its own** store is always allowed -- that is not sharing;
    * the main model may read a **shareable** character, and is denied a private
      one (Natasha);
    * any other cross-character read is denied: Jarvis cannot read Ultron, Ultron
      cannot read Vision, and so on.

    This function is pure: it opens no file. The caller acts on the decision and
    records it through :mod:`core.learning_audit`.
    """
    reader_owner = owner_for(reader)
    target = owner_for(owner)
    if reader_owner not in OWNERS:
        return AccessDecision(reader_owner, target, False, f"{reader!r} is not a learning owner.")
    if target not in OWNERS:
        return AccessDecision(reader_owner, target, False, f"{owner!r} is not a learning owner.")
    if reader_owner == target:
        return AccessDecision(
            reader_owner, target, True, "a character may always read its own records."
        )
    if reader_owner == MAIN_MODEL_OWNER:
        if target in PRIVATE_OWNERS:
            return AccessDecision(reader_owner, target, False, PRIVATE_REASON)
        return AccessDecision(
            reader_owner, target, True, "the main model may learn from a shareable character."
        )
    return AccessDecision(
        reader_owner,
        target,
        False,
        "characters cannot read another character's private records.",
    )


def assert_read(reader: str, owner: str) -> None:
    """Raise :class:`LearningAccessError` when the read is not permitted.

    The store and the sharing service both call this *before* they build a path,
    so a refused read never even resolves a directory.
    """
    decision = decide(reader, owner)
    if not decision.allowed:
        raise LearningAccessError(decision.reason)


def assert_owner(owner: str) -> None:
    """Raise :class:`LearningAccessError` when ``owner`` is not a known owner."""
    if not is_owner(owner):
        raise LearningAccessError(f"{owner!r} is not a known learning owner.")


__all__ = [
    "CHARACTER_OWNERS",
    "MAIN_MODEL_MOUTH",
    "MAIN_MODEL_OWNER",
    "MOUTH_TO_OWNER",
    "OWNERS",
    "OWNER_TO_MOUTH",
    "PRIVATE_OWNERS",
    "PRIVATE_REASON",
    "PRIVATE_REASONS",
    "SHAREABLE_OWNERS",
    "AccessDecision",
    "LearningAccessError",
    "assert_owner",
    "assert_read",
    "decide",
    "is_main_model",
    "is_owner",
    "is_private",
    "is_shareable",
    "mouth_key",
    "owner_for",
    "store_dir",
    "store_dir_for",
]
