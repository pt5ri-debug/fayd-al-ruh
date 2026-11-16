"""The character avatar: upload a picture, or generate one, per character.

A Studio character -- a companion or the user's own persona -- may carry a
picture (``CompanionCharacter.avatar_path``). This module is the one engine that
puts a real file there, and it deliberately **reuses the project's own image
machinery** rather than writing a second renderer or downloader:

* **Upload.** :meth:`CharacterAvatarManager.set_from_file` copies a picture the
  user chose into the character's own folder, re-encoding nothing -- the file is
  copied byte for byte, so a PNG stays a PNG and a JPEG stays a JPEG. A file that
  is not an image by its extension is refused rather than stored.
* **Generate.** :meth:`CharacterAvatarManager.generate` renders a picture from a
  prompt through :mod:`core.imagegen`'s procedural renderer (dependency-free,
  deterministic, always available) or its optional learned backend, and writes it
  under the same folder. A prompt the caller does not supply is built from the
  character's own profile, so "generate an avatar" needs no typing.
* **Clear.** :meth:`clear` removes the file and the record's reference, so the
  character falls back to its initial-letter badge.

**Honest by construction.** Every write goes through the character's own folder
under ``data/romance/characters/<character_id>/`` (git-ignored), resolved by the
same policy the character's other own files use, so one character can never reach
another's picture. A missing file, a non-image extension, a character that does
not exist, or a generation that produced nothing is an :class:`AvatarError` with
a readable reason -- never a silently empty avatar. Nothing here loads a model on
import and nothing reaches the network.

The module is dependency-free, imports no Tk, and is the one engine the Studio
window and the CLI both drive.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.imagegen import (
    DEFAULT_WIDTH,
    GeneratedImage,
    ImageRequest,
    load_image_generator,
)

#: The image extensions an upload may carry. A file with any other suffix is
#: refused rather than copied, so a stray document never becomes an "avatar".
AVATAR_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".ppm",
)

#: The two backends the avatar generator offers: the always-available procedural
#: renderer, and the optional learned diffusion backend.
BACKEND_PROCEDURAL = "procedural"
BACKEND_DIFFUSION = "diffusion"
AVATAR_BACKENDS: tuple[str, ...] = (BACKEND_PROCEDURAL, BACKEND_DIFFUSION)

#: The square size a generated avatar is rendered at (the project's own default).
AVATAR_SIZE = DEFAULT_WIDTH


class AvatarError(ValueError):
    """Raised when an avatar operation cannot be carried out as asked."""


@dataclass(frozen=True)
class AvatarResult:
    """What one avatar operation produced: the file, the character, and a note."""

    character_id: str
    path: Path
    source: str
    description: str

    @property
    def ok(self) -> bool:
        return bool(self.path)

    def describe(self) -> str:
        return self.description or str(self.path)


def is_image_name(name: str) -> bool:
    """Whether ``name`` carries an image extension an upload accepts."""
    return Path(name).suffix.lower() in AVATAR_SUFFIXES


def avatar_prompt(name: str, *, appearance: str = "", biography: str = "", gender: str = "") -> str:
    """A default generation prompt built from the character's own profile.

    Deliberately **not** a portrait instruction: the project's procedural renderer
    is abstract, and a learned backend is optional, so the prompt describes the
    character (its name, its own appearance note, its gender) and lets whichever
    renderer is available interpret it. A character with nothing recorded still
    yields a usable prompt from its name alone.
    """
    parts = [f"avatar for {name.strip() or 'a companion'}"]
    if gender.strip():
        parts.append(gender.strip())
    if appearance.strip():
        parts.append(appearance.strip())
    elif biography.strip():
        parts.append(biography.strip()[:160])
    parts.append("portrait, soft light")
    return ", ".join(parts)


class CharacterAvatarManager:
    """Upload, generate, and clear one character's picture.

    ``root`` is the Studio's data root -- the same one the storage service and
    :mod:`core.romance.character_files` take -- so an avatar lands beside the
    character's other own files, and one character's folder is never reachable
    from another's.
    """

    def __init__(
        self,
        storage: Any,
        *,
        root: str | Path | None = None,
        generator: Any = None,
    ) -> None:
        self.storage = storage
        self.domain = storage.domain
        self._root = Path(root) if root is not None else Path(storage.root)
        #: The image generator. ``None`` builds the procedural one lazily; a test
        #: injects its own, so the engine is exercised with no backend at all.
        self._generator = generator

    # -- paths ------------------------------------------------------------

    def folder(self, character_id: str) -> Path:
        """The character's own folder (``<root>/characters/<id>``)."""
        return self._characters_root() / _safe_name(character_id)

    def _characters_root(self) -> Path:
        """Where the Studio keeps each character's own files.

        Prefers the project's own resolver when the storage service exposes one,
        so the avatar lives beside the character's knowledge and learning files.
        """
        resolver = getattr(self.storage, "characters_dir", None)
        if callable(resolver):
            try:
                return Path(resolver())
            except Exception:
                pass
        return self._root / "characters"

    def current_path(self, character_id: str) -> Path | None:
        """The picture the character currently has, or ``None``."""
        character = self._character(character_id)
        raw = str(getattr(character, "avatar_path", "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.folder(character_id) / path
        return path if path.is_file() else None

    # -- upload -----------------------------------------------------------

    def set_from_file(self, character_id: str, source: str | Path) -> AvatarResult:
        """Copy a picture the user chose into the character's own folder.

        The file is copied byte for byte (never re-encoded), so a JPEG stays a
        JPEG. A source that is missing, is not a file, or does not carry an image
        extension is refused with a readable reason.
        """
        self._character(character_id)
        src = Path(source)
        if not src.is_file():
            raise AvatarError(f"no file at {source!r}")
        if not is_image_name(src.name):
            raise AvatarError(
                f"{src.name!r} is not an image; choose one of "
                f"{', '.join(AVATAR_SUFFIXES)}"
            )
        folder = self.folder(character_id)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"avatar{src.suffix.lower()}"
        if src.resolve() != target.resolve():
            shutil.copyfile(src, target)
        self._store(character_id, target)
        return AvatarResult(
            character_id=character_id,
            path=target,
            source="upload",
            description=f"Avatar set from {src.name}.",
        )

    # -- generate ---------------------------------------------------------

    def generate(
        self,
        character_id: str,
        *,
        prompt: str = "",
        backend: str = BACKEND_PROCEDURAL,
        seed: int | None = None,
        width: int = AVATAR_SIZE,
        height: int = AVATAR_SIZE,
    ) -> AvatarResult:
        """Render a picture for the character and store it as its avatar.

        The prompt defaults to one built from the character's own profile, so the
        caller may simply ask for an avatar. An unknown backend, or a backend
        whose optional runtime is missing, is reported as an :class:`AvatarError`
        naming the real next step rather than returning an empty picture.
        """
        character = self._character(character_id)
        if backend not in AVATAR_BACKENDS:
            raise AvatarError(
                f"unknown backend {backend!r}; choose one of {', '.join(AVATAR_BACKENDS)}"
            )
        text = prompt.strip() or avatar_prompt(
            str(getattr(character, "name", "") or ""),
            appearance=str(getattr(character, "appearance", "") or ""),
            biography=str(getattr(character, "biography", "") or ""),
            gender=str(getattr(character, "gender", "") or ""),
        )
        image = self._render(text, backend=backend, seed=seed, width=width, height=height)
        folder = self.folder(character_id)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "avatar.png"
        image.save(target)
        self._store(character_id, target)
        note = getattr(image, "description", "") or ""
        return AvatarResult(
            character_id=character_id,
            path=target,
            source=f"generated:{backend}",
            description=f"Avatar generated ({backend}). {note}".strip(),
        )

    def _render(
        self, prompt: str, *, backend: str, seed: int | None, width: int, height: int
    ) -> GeneratedImage:
        request = ImageRequest(prompt=prompt, width=width, height=height, seed=seed)
        if backend == BACKEND_PROCEDURAL and self._generator is None:
            generator = load_image_generator(BACKEND_PROCEDURAL)
        elif backend == BACKEND_PROCEDURAL:
            generator = self._generator
        else:
            generator = self._generator or self._load_diffusion()
        return generator.generate(request)

    def _load_diffusion(self) -> Any:
        try:
            return load_image_generator(BACKEND_DIFFUSION)
        except Exception as error:  # optional extra or weights missing
            raise AvatarError(
                "the learned image backend is not installed; install the image "
                f"extra or generate with '{BACKEND_PROCEDURAL}' ({error})"
            ) from error

    # -- clear ------------------------------------------------------------

    def clear(self, character_id: str) -> AvatarResult:
        """Remove the character's picture and its reference.

        The file is deleted when it is inside the character's own folder; a file
        the record points at *elsewhere* is left exactly where it is (the Studio
        only ever removes what it wrote).
        """
        self._character(character_id)
        path = self.current_path(character_id)
        if path is not None and self.folder(character_id) in path.parents:
            try:
                path.unlink()
            except OSError:
                pass
        self.domain.update("companion_characters", character_id, avatar_path="")
        return AvatarResult(
            character_id=character_id,
            path=Path(),
            source="cleared",
            description="Avatar cleared.",
        )

    # -- internals --------------------------------------------------------

    def _character(self, character_id: str) -> Any:
        character = self.domain.get("companion_characters", character_id)
        if character is None:
            raise AvatarError(f"no character with id {character_id!r}")
        return character

    def _store(self, character_id: str, path: Path) -> None:
        """Record the avatar on the character, as a path **relative** to its folder.

        Relative is deliberate: the folder moves with the data root, so a saved
        record stays valid when the Studio is pointed at another data folder.
        """
        try:
            stored = str(path.relative_to(self.folder(character_id)))
        except ValueError:
            stored = str(path)
        self.domain.update("companion_characters", character_id, avatar_path=stored)


def _safe_name(character_id: str) -> str:
    """A filesystem-safe folder name for a character id (never a path escape)."""
    cleaned = "".join(ch for ch in character_id if ch.isalnum() or ch in "-_")
    return cleaned or "character"


__all__ = [
    "AVATAR_BACKENDS",
    "AVATAR_SIZE",
    "AVATAR_SUFFIXES",
    "BACKEND_DIFFUSION",
    "BACKEND_PROCEDURAL",
    "AvatarError",
    "AvatarResult",
    "CharacterAvatarManager",
    "avatar_prompt",
    "is_image_name",
]
