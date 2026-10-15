"""The image-editing service: load a picture, apply a written instruction, save it.

This is the Tk-free half of the window's "Edit an image from text…" screen, the
counterpart of :mod:`gui.image_gen`. It wraps :mod:`core.imageedit` with the small
conveniences a chat needs: it reads the instruction, applies the recognised
operations in order, writes the result under the data folder, and returns an
:class:`ImageEditReport` the window can render or the CLI can print.

Nothing here invents a capability. The engine is a deterministic interpreter over
a documented keyword vocabulary (see :mod:`core.imageedit`), not a learned
instruction-following model, and every report says so through its ``learned``
flag (always ``False`` here) and its description of the operations applied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.imageedit import (
    EditPlan,
    ImageEditError,
    Raster,
    apply_plan,
    decode_image,
    plan_edits,
)
from core.imagegen import default_image_dir

#: The extensions this build can read: PNG and binary PPM.
READABLE_SUFFIXES = (".png", ".ppm")

#: The instruction shown when the field is empty, so the window can start helpful.
EXAMPLE_INSTRUCTION = "rotate 90 and brighten, then crop to a square"


@dataclass(frozen=True)
class ImageEditReport:
    """The outcome of one edit, ready for a window or a terminal."""

    ok: bool
    message: str
    source: str = ""
    path: str = ""
    instruction: str = ""
    operations: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    #: True only when a *learned* model produced the change (never, here).
    learned: bool = False

    def summary(self) -> str:
        """A one-block, human-readable report."""
        if not self.ok:
            return f"Image edit failed: {self.message}"
        lines = [
            f"Edited image saved to {self.path}",
            f"  size      : {self.width}x{self.height}, {self.size_bytes} bytes",
            f"  applied   : {'; '.join(self.operations) or 'nothing'}",
        ]
        if self.unknown:
            lines.append(f"  not used  : {', '.join(self.unknown)}")
        lines.append(
            "  note      : deterministic keyword edits, not a learned model."
        )
        return "\n".join(lines)


@dataclass
class EditSession:
    """One image being edited: the current raster, its path, and its history.

    The window keeps a session so an instruction is applied to the *current* result
    (a chain of edits), while the original file is never touched. ``history``
    records the instructions applied, so the window can show and undo them.
    """

    source_path: str
    raster: Raster
    history: list[EditPlan] = field(default_factory=list)
    result_path: str = ""

    @property
    def applied(self) -> tuple[str, ...]:
        return tuple(op.description for plan in self.history for op in plan.ops)

    @property
    def unknown(self) -> tuple[str, ...]:
        words: list[str] = []
        for plan in self.history:
            for word in plan.unknown:
                if word not in words:
                    words.append(word)
        return tuple(words)


class ImageEditService:
    """Load an image, apply a written instruction, and save the result as a PNG."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._on_progress = on_progress

    @property
    def root(self) -> Path | None:
        """The explicit storage root, or ``None`` for the default data folder."""
        return self._root

    @property
    def image_dir(self) -> Path:
        """Where edited images are written when no explicit path is given."""
        return default_image_dir(self._root)

    def for_backend(self) -> "ImageEditService":
        """A service on the same storage root (kept for symmetry with image_gen)."""
        return ImageEditService(root=self._root, on_progress=self._on_progress)

    def load(self, source: str | Path) -> EditSession:
        """Decode ``source`` into a new editing session; raises on a bad file."""
        raster = decode_image(source)
        return EditSession(source_path=str(source), raster=raster)

    def plan(self, instruction: str) -> EditPlan:
        """Interpret an instruction without applying it (for a live preview)."""
        return plan_edits(instruction)

    def apply(
        self,
        session: EditSession,
        instruction: str,
        *,
        output: str | Path | None = None,
    ) -> ImageEditReport:
        """Apply ``instruction`` to the session's current image and save a PNG.

        The session's raster is advanced only when the instruction changed
        something, so an unrecognised instruction never mutates the working image.
        """
        plan = self.plan(instruction)
        if not plan.ops:
            return ImageEditReport(
                ok=False,
                message=(
                    "I could not find an edit in that instruction. Try words like "
                    "rotate, flip, resize, crop, brighten, contrast, grayscale, blur."
                ),
                source=session.source_path,
                instruction=instruction,
                unknown=plan.unknown,
            )
        self._progress(f"Applying {len(plan.ops)} operation(s)…")
        try:
            result = apply_plan(session.raster, plan)
        except (ImageEditError, ValueError) as error:
            return ImageEditReport(
                ok=False,
                message=str(error),
                source=session.source_path,
                instruction=instruction,
                unknown=plan.unknown,
            )
        target = Path(output) if output is not None else self._default_path(session, plan)
        try:
            result.save_png(target)
        except OSError as error:
            return ImageEditReport(
                ok=False,
                message=f"Could not save the edited image: {error}",
                source=session.source_path,
                instruction=instruction,
            )
        session.raster = result
        session.history.append(plan)
        session.result_path = str(target)
        return ImageEditReport(
            ok=True,
            message=f"Edited image saved to {target}.",
            source=session.source_path,
            path=str(target),
            instruction=instruction,
            operations=tuple(op.description for op in plan.ops),
            unknown=plan.unknown,
            width=result.width,
            height=result.height,
            size_bytes=len(result.to_png()),
        )

    def save_copy(self, session: EditSession, target: str | Path) -> ImageEditReport:
        """Write the session's current image wherever the user chooses."""
        try:
            path = session.raster.save_png(target)
        except OSError as error:
            return ImageEditReport(ok=False, message=f"Could not save a copy: {error}")
        return ImageEditReport(
            ok=True,
            message=f"Saved a copy to {path}.",
            path=str(path),
            width=session.raster.width,
            height=session.raster.height,
            size_bytes=len(session.raster.to_png()),
        )

    def render_working(self, session: EditSession) -> str:
        """Persist the session's current image and return its path.

        The window uses this to show a preview after an Undo (which changes the
        in-memory raster without producing a new saved file); a single working
        file per source is overwritten, so no temporary files accumulate.
        """
        path = self.working_path(session)
        session.raster.save_png(path)
        session.result_path = str(path)
        return str(path)

    def working_path(self, session: EditSession) -> Path:
        """A stable file that mirrors the session's current raster."""
        stem = Path(session.source_path).stem or "image"
        return self.image_dir / f"{stem}-working.png"

    def _progress(self, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(message)

    def _default_path(self, session: EditSession, plan: EditPlan) -> Path:
        stem = Path(session.source_path).stem or "image"
        step = len(session.history) + 1
        return self.image_dir / f"{stem}-edited-{step}.png"


__all__ = [
    "EXAMPLE_INSTRUCTION",
    "EditSession",
    "ImageEditReport",
    "ImageEditService",
    "READABLE_SUFFIXES",
]
