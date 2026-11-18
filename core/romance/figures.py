"""The four ready-made bodies the Body & shape studio shows, and imported models.

The Body & shape window used to open on a single generic figure that a user had
reshaped many times, so there was no clear starting point. It now opens on one of
**four** complete human bodies -- a **woman**, a **girl**, a **man**, and a
**boy** -- chosen from the character being edited: a character recorded as a woman
shows the woman's body, a boy the boy's, and so on. The four are the project's own
parametric figures (see :mod:`core.scene3d_figures`), each fused into one
connected skin on real human proportions, so every one of them already reads as a
person rather than a pile of parts.

A user who has a **ready-made model** -- a DAZ 3D export, a MakeHuman OBJ, or any
OBJ/STL/PLY mesh -- can *import* it into one of the four slots (see
:meth:`FigureLibrary.import_model`). The imported model then **is** that slot's
figure: the studio shows it, and the existing region tools resize, move, rotate,
and extend it. An imported file is stored under the git-ignored Studio data
(``data/romance/figures/<slot>``) and never copied into the repository, and
nothing here opens a network connection -- fetching a DAZ model is the user's own
download, and the program only reads the file it is handed.

Honest by construction: a slot with nothing imported falls back to that slot's own
built-in figure, so the studio never shows an empty window; and importing refuses a
file this build cannot read with the real reason, rather than storing a broken one.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.romance.body import preview_scene, render_avatar, render_preview
from core.scene3d_figures import FIGURE_PRESETS, FigureParameters

#: The four figure slots, in the order the picker shows them.
SLOT_WOMAN = "woman"
SLOT_GIRL = "girl"
SLOT_MAN = "man"
SLOT_BOY = "boy"
FIGURE_SLOTS: tuple[str, ...] = (SLOT_WOMAN, SLOT_GIRL, SLOT_MAN, SLOT_BOY)

#: A slot with no character to choose from opens on the woman's figure -- the
#: project's documented neutral default -- so the studio is never empty.
DEFAULT_SLOT = SLOT_WOMAN

#: Bilingual slot titles.
SLOT_TITLES: dict[str, tuple[str, str]] = {
    SLOT_WOMAN: ("Woman", "امرأة"),
    SLOT_GIRL: ("Girl", "بنت"),
    SLOT_MAN: ("Man", "رجل"),
    SLOT_BOY: ("Boy", "ولد"),
}

#: The age at which a recorded character is shown as an **adult** figure. A
#: character recorded as younger gets the younger figure of its gender, so a minor
#: is never drawn with an adult body; an unrecorded age is shown as an adult, the
#: way the studio behaved before, because the program cannot invent an age.
ADULT_AGE = 18

#: The name of the small record kept in each slot's folder.
RECORD_NAME = "figure.json"


def slot_title(slot: str, language: str = "en") -> str:
    """The readable title of a slot, in English or Arabic (the key if unknown)."""
    pair = SLOT_TITLES.get(slot)
    if pair is None:
        return slot
    return pair[1] if language == "ar" else pair[0]


def slot_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(slot, title)`` for every figure, in the picker's own order."""
    return tuple((slot, slot_title(slot, language)) for slot in FIGURE_SLOTS)


def slot_parameters(slot: str) -> FigureParameters:
    """The built-in body parameters for a slot (an unknown slot is the default)."""
    key = slot if slot in FIGURE_SLOTS else DEFAULT_SLOT
    return FIGURE_PRESETS[key]


def _gender_kind(gender: str) -> str:
    """``"female"``, ``"male"``, or ``""`` from a character's own recorded gender.

    The check is deliberately broad -- an English word, an Arabic word, or the
    usual abbreviations -- but it never *guesses*: a blank or non-binary gender
    returns ``""`` so the caller can fall back to the documented default.
    """
    text = (gender or "").strip().casefold()
    if not text:
        return ""
    female = ("woman", "female", "feminine", "girl", "lady", "mother", "sister",
              "أنثى", "امرأة", "بنت", "فتاة", "سيدة", "أم", "أخت")
    male = ("man", "male", "masculine", "boy", "gentleman", "father", "brother",
            "ذكر", "رجل", "ولد", "فتى", "سيد", "أب", "أخ")
    if any(word in text for word in female):
        return "female"
    if any(word in text for word in male):
        return "male"
    return ""


def _age_of(character: Any) -> int | None:
    """A character's own recorded age, or ``None`` when it was never recorded."""
    if character is None:
        return None
    value = getattr(character, "fictional_age", None)
    if value is None:
        inner = getattr(character, "character", None)
        value = getattr(inner, "fictional_age", None) if inner is not None else None
    if value is None:
        return None
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    return age if age > 0 else None


def _gender_of(character: Any) -> str:
    """A character's own recorded gender string (``""`` when unrecorded)."""
    if character is None:
        return ""
    value = getattr(character, "gender", None)
    if value is None:
        inner = getattr(character, "character", None)
        value = getattr(inner, "gender", "") if inner is not None else ""
    return str(value or "")


def slot_for_character(character: Any) -> str:
    """The figure slot a character opens on, from its own recorded gender and age.

    A woman shows the woman's body and a man the man's; a character recorded as
    under :data:`ADULT_AGE` shows the younger figure of its gender, so a minor is
    never drawn with an adult body. An unrecorded or non-binary gender falls back
    to the documented default (the woman's figure) rather than a made-up choice.
    """
    female = _gender_kind(_gender_of(character)) == "female"
    male = _gender_kind(_gender_of(character)) == "male"
    age = _age_of(character)
    young = age is not None and age < ADULT_AGE
    if male:
        return SLOT_BOY if young else SLOT_MAN
    # A woman, an unknown gender, or a non-binary one: the woman's figure is the
    # project's documented neutral default.
    return SLOT_GIRL if (young and female) else SLOT_WOMAN


def slot_parameters_for_character(character: Any) -> tuple[str, FigureParameters]:
    """The slot a character opens on **and** that slot's body parameters, named.

    The name is kept from the character so the studio's own label is right, and the
    slot's measurements are the built-in figure's, so opening a woman's body really
    shows a woman's shape.
    """
    slot = slot_for_character(character)
    base = slot_parameters(slot)
    name = str(getattr(character, "name", "") or "") or base.name
    return slot, base.with_changes(name=name)


@dataclass(frozen=True)
class ImportedFigure:
    """One imported model recorded in a slot."""

    slot: str
    filename: str
    source: str = ""
    note: str = ""

    def path(self, root: Path) -> Path:
        return Path(root) / self.slot / self.filename

    def to_primitive(self) -> dict[str, Any]:
        return {"slot": self.slot, "filename": self.filename,
                "source": self.source, "note": self.note}


class FigureLibrary:
    """The four figure slots and the models imported into them.

    ``root`` is the folder the slots live in; with none, the Studio's own
    git-ignored ``data/romance/figures`` is used. A slot with nothing imported
    falls back to its built-in figure, so every read has an answer and the studio
    never shows an empty window.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root
        from core.paths import romance_figures_dir

        return romance_figures_dir()

    # -- reads ------------------------------------------------------------

    def imported(self, slot: str) -> ImportedFigure | None:
        """The model imported into ``slot``, or ``None`` when it holds the built-in.

        A record whose file has gone is reported as **nothing imported** rather than
        as a broken slot, so a slot a user deleted the file from falls back to its
        built-in figure instead of failing to draw.
        """
        if slot not in FIGURE_SLOTS:
            return None
        record = self._root_path(slot) / RECORD_NAME
        if not record.is_file():
            return None
        try:
            raw = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        filename = str(raw.get("filename") or "")
        if not filename:
            return None
        entry = ImportedFigure(
            slot=slot,
            filename=filename,
            source=str(raw.get("source") or ""),
            note=str(raw.get("note") or ""),
        )
        if not entry.path(self.root).is_file():
            return None
        return entry

    def has_imported(self, slot: str) -> bool:
        return self.imported(slot) is not None

    def summary(self, slot: str) -> str:
        """A one-line, honest description of what a slot is showing."""
        entry = self.imported(slot)
        title = slot_title(slot)
        if entry is None:
            return f"{title}: the built-in figure."
        origin = f" (imported from {entry.source})" if entry.source else " (imported)"
        return f"{title}: {entry.filename}{origin}."

    def slots(self) -> tuple[tuple[str, str, bool], ...]:
        """``(slot, title, imported?)`` for every figure, in the picker's order."""
        return tuple(
            (slot, slot_title(slot), self.has_imported(slot)) for slot in FIGURE_SLOTS
        )

    # -- the mesh a slot shows -------------------------------------------

    def mesh(self, slot: str, *, height: float | None = None) -> Any:
        """The mesh a slot shows: the imported model when there is one, else built.

        The imported model is normalised into the studio's own space (metres, feet
        on the ground, centred) at ``height`` (the slot's own height by default), so
        the region boxes and the camera framing mean the same thing they do for the
        built-in figure.
        """
        from core.romance.body import figure_mesh

        params = slot_parameters(slot)
        entry = self.imported(slot)
        if entry is None:
            return figure_mesh(params)
        from core.body_editor import load_body_mesh, normalize_mesh

        loaded = load_body_mesh(entry.path(self.root))
        return normalize_mesh(loaded, height=height or params.height)

    # -- imports ----------------------------------------------------------

    def import_model(
        self,
        slot: str,
        source: str | Path,
        *,
        origin: str = "",
        note: str = "",
        move: bool = False,
    ) -> ImportedFigure:
        """Import a ready-made model into ``slot`` and record where it came from.

        The file is **read** first (so a format this build cannot parse is refused
        with the real reason and nothing is stored), then copied into the slot's own
        folder under the git-ignored Studio data -- never into the repository -- and
        the previous model in the slot, if any, is replaced. ``move=True`` moves the
        file instead of copying it, for a user who wants the original left in place
        nowhere. ``origin`` records where it came from (a URL, a product name, a
        local path) and is shown to the user.
        """
        if slot not in FIGURE_SLOTS:
            raise ValueError(
                f"Unknown figure slot {slot!r}; choose from {', '.join(FIGURE_SLOTS)}."
            )
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"No such model file: {path}")
        # Read it first: a file this build cannot parse must not be stored, so the
        # slot keeps whatever it had and the user gets the real reason.
        from core.body_editor import load_body_mesh

        load_body_mesh(path)
        folder = self._root_path(slot)
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / path.name
        if destination.resolve() != path.resolve():
            if move:
                shutil.move(str(path), str(destination))
            else:
                shutil.copy2(path, destination)
        entry = ImportedFigure(
            slot=slot,
            filename=destination.name,
            source=origin or str(path),
            note=note,
        )
        self._write_record(slot, entry)
        return entry

    def clear(self, slot: str) -> bool:
        """Forget the model imported into ``slot`` (returns whether one was removed)."""
        if slot not in FIGURE_SLOTS:
            return False
        folder = self._root_path(slot)
        entry = self.imported(slot)
        record = folder / RECORD_NAME
        removed = False
        if entry is not None:
            try:
                entry.path(self.root).unlink()
                removed = True
            except OSError:
                removed = False
        if record.is_file():
            try:
                record.unlink()
                removed = True
            except OSError:
                removed = False
        return removed

    # -- rendering --------------------------------------------------------

    def render(
        self,
        slot: str,
        *,
        view: str = "full",
        width: int | None = None,
        height: int | None = None,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        zoom: float = 1.0,
        avatar: bool = False,
    ) -> bytes:
        """Render a slot's figure (imported or built-in) and return the PNG bytes.

        The window shows the result directly. An imported model is drawn without the
        built-in hair, because a ready-made model carries its own.
        """
        params = slot_parameters(slot)
        entry = self.imported(slot)
        mesh = None
        if entry is not None:
            from core.body_editor import load_body_mesh, normalize_mesh

            mesh = normalize_mesh(load_body_mesh(entry.path(self.root)), height=params.height)
        if avatar:
            return render_avatar(params, body_mesh=mesh)
        kwargs: dict[str, Any] = {"view": view, "body_mesh": mesh,
                                  "azimuth": azimuth, "elevation": elevation, "zoom": zoom}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        return render_preview(params, **kwargs)

    def scene(self, slot: str, **kwargs: Any) -> Any:
        """The :class:`core.scene3d.Scene` a slot renders (used by the tests)."""
        params = slot_parameters(slot)
        entry = self.imported(slot)
        mesh = None
        if entry is not None:
            from core.body_editor import load_body_mesh, normalize_mesh

            mesh = normalize_mesh(load_body_mesh(entry.path(self.root)), height=params.height)
        return preview_scene(params, body_mesh=mesh, **kwargs)

    # -- internals --------------------------------------------------------

    def _root_path(self, slot: str) -> Path:
        return self.root / slot

    def _write_record(self, slot: str, entry: ImportedFigure) -> None:
        folder = self._root_path(slot)
        folder.mkdir(parents=True, exist_ok=True)
        record = folder / RECORD_NAME
        payload = json.dumps(entry.to_primitive(), indent=2, ensure_ascii=False)
        record.write_text(payload, encoding="utf-8")


def figure_library(root: str | Path | None = None) -> FigureLibrary:
    """The Studio's figure library (a thin convenience constructor)."""
    return FigureLibrary(root)
