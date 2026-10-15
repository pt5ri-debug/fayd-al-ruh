"""The image-editing service: load a picture, apply a written instruction, save it.

This is the Tk-free half of the window's "Edit an image from text…" screen, the
counterpart of :mod:`gui.image_gen`. It wraps :mod:`core.imageedit` with the small
conveniences a chat needs: it reads the instruction, applies the recognised
operations in order, writes the result under the data folder, and returns an
:class:`ImageEditReport` the window can render or the CLI can print.

There are two backends, and the report always says which one ran:

* the **deterministic** editor (:mod:`core.imageedit`) -- a documented keyword
  vocabulary of pixel operations (rotate, brighten, crop…), fast and exact;
* the optional **learned** editor
  (:class:`core.imagegen_diffusion.InstructionImageEditor`, an InstructPix2Pix
  model) -- it follows a written instruction to make a *semantic* change the pixel
  operations cannot: make a person smile, change a pose, or alter the movement,
  re-synthesising the picture.

Nothing here invents a capability: the deterministic path says it is not a model,
and the learned path only runs when its optional extra and weights are present,
reporting honestly (through ``learned``) which produced the change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

#: An example of the kind of instruction only the learned backend can follow.
LEARNED_EXAMPLE_INSTRUCTION = "make the person smile, then change the pose to a wave"

#: The deterministic backend's name, and the learned one's.
BACKEND_DETERMINISTIC = "deterministic"
BACKEND_LEARNED = "learned"


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
    #: True only when a *learned* model produced the change.
    learned: bool = False
    #: The backend's honest description of what it did (empty for the default path).
    description: str = ""

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
        if self.learned:
            note = self.description or "learned model edit."
            lines.append(f"  note      : {note}")
        else:
            lines.append(
                "  note      : deterministic keyword edits, not a learned model."
            )
        return "\n".join(lines)


@dataclass
class EditStep:
    """One applied edit: a label for the history, and whether a model produced it."""

    label: str
    learned: bool = False


@dataclass
class EditSession:
    """One image being edited: the current raster, its path, and its history.

    The window keeps a session so an instruction is applied to the *current* result
    (a chain of edits), while the original file is never touched. ``history``
    records the edits applied, and ``undo_stack`` keeps the previous raster of each
    step so **Undo works for both backends** -- a learned edit cannot be replayed,
    so the previous state is kept rather than recomputed.
    """

    source_path: str
    raster: Raster
    history: list[EditStep] = field(default_factory=list)
    undo_stack: list[Raster] = field(default_factory=list)
    unknown_words: list[str] = field(default_factory=list)
    result_path: str = ""

    @property
    def applied(self) -> tuple[str, ...]:
        return tuple(step.label for step in self.history)

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(self.unknown_words)

    @property
    def learned_used(self) -> bool:
        """True when any applied edit came from the learned model."""
        return any(step.learned for step in self.history)


class ImageEditService:
    """Load an image, apply a written instruction, and save the result as a PNG."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        backend: str = BACKEND_DETERMINISTIC,
        preset: str | None = None,
        model: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        image_guidance: float | None = None,
        negative_prompt: str | None = None,
        on_progress: Callable[[str], None] | None = None,
        editor: object | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._backend_name = backend
        self._preset = preset
        self._model = model
        self._steps = steps
        self._guidance = guidance
        self._image_guidance = image_guidance
        self._negative_prompt = negative_prompt
        self._on_progress = on_progress
        if editor is not None:
            # An injected editor is always the learned one, so the service's
            # dispatch matches the editor a test (or a caller) supplied.
            self._backend_name = BACKEND_LEARNED
            self._editor: Any = editor
        else:
            self._editor = self._build_editor()

    # -- construction -----------------------------------------------------

    def _build_editor(self) -> object | None:
        """The learned editor, or ``None`` for the deterministic backend."""
        if self._backend_name != BACKEND_LEARNED:
            return None
        from core.imagegen_diffusion import InstructionImageEditor

        options: dict[str, Any] = {"root": self._root}
        if self._preset is not None:
            options["preset"] = self._preset
        if self._model is not None:
            options["model"] = self._model
        if self._steps is not None:
            options["steps"] = self._steps
        if self._guidance is not None:
            options["guidance"] = self._guidance
        if self._image_guidance is not None:
            options["image_guidance"] = self._image_guidance
        if self._negative_prompt is not None:
            options["negative_prompt"] = self._negative_prompt
        return InstructionImageEditor(**options)

    @property
    def backend(self) -> str:
        """``deterministic`` or ``learned``."""
        return self._backend_name

    @property
    def learned(self) -> bool:
        return self._backend_name == BACKEND_LEARNED

    @property
    def root(self) -> Path | None:
        """The explicit storage root, or ``None`` for the default data folder."""
        return self._root

    @property
    def image_dir(self) -> Path:
        """Where edited images are written when no explicit path is given."""
        return default_image_dir(self._root)

    def for_backend(
        self,
        backend: str,
        *,
        preset: str | None = None,
        model: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        image_guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> "ImageEditService":
        """A service on the same storage root that uses a different backend."""
        return ImageEditService(
            root=self._root,
            backend=backend,
            preset=preset,
            model=model,
            steps=steps,
            guidance=guidance,
            image_guidance=image_guidance,
            negative_prompt=negative_prompt,
            on_progress=self._on_progress,
        )

    def backend_status(self) -> tuple[tuple[str, bool, str], ...]:
        """Each backend and whether it can run now, for a menu or a status line."""
        from core.imagegen_diffusion import (
            DEFAULT_EDIT_PRESET,
            EDIT_MODEL,
            DiffusionUnavailable,
            edit_preset_names,
        )

        learned_note = (
            f"Instruction-following diffusion model ({EDIT_MODEL}): makes semantic "
            "changes (expression, pose, movement). Presets: "
            f"{', '.join(edit_preset_names())} (default {DEFAULT_EDIT_PRESET})."
        )
        status: list[tuple[str, bool, str]] = [
            (
                BACKEND_DETERMINISTIC,
                True,
                "Dependency-free keyword editor: exact pixel operations (not a model).",
            )
        ]
        try:
            available = self.for_backend(BACKEND_LEARNED).available()
        except DiffusionUnavailable:
            available = False
        status.append((BACKEND_LEARNED, available, learned_note))
        return tuple(status)

    def available(self) -> bool:
        """True when the selected backend can actually run."""
        if self._editor is None:
            return True
        checker = getattr(self._editor, "is_available", None)
        return True if not callable(checker) else bool(checker())

    def download_model(
        self,
        *,
        approved: bool = False,
        preset: str | None = None,
    ) -> ImageEditReport:
        """Download the learned editor's weights (external action; needs approval)."""
        from core.imagegen_diffusion import DiffusionUnavailable, download_edit_model

        try:
            path = download_edit_model(
                root=self._root, approved=approved, progress=self._progress
            )
        except (DiffusionUnavailable, ValueError, OSError) as error:
            return ImageEditReport(ok=False, message=str(error), learned=True)
        return ImageEditReport(
            ok=True,
            message=f"The editing model is cached at {path}.",
            path=str(path),
            learned=True,
        )

    # -- loading and planning ---------------------------------------------

    def load(self, source: str | Path) -> EditSession:
        """Decode ``source`` into a new editing session; raises on a bad file."""
        raster = decode_image(source)
        return EditSession(source_path=str(source), raster=raster)

    def plan(self, instruction: str) -> EditPlan:
        """Interpret an instruction without applying it (for a live preview)."""
        return plan_edits(instruction)

    # -- applying ---------------------------------------------------------

    def apply(
        self,
        session: EditSession,
        instruction: str,
        *,
        output: str | Path | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageEditReport:
        """Apply ``instruction`` to the session's current image and save a PNG.

        Dispatches to the selected backend. The session's raster is advanced only
        when the edit changed something, so an unrecognised instruction never
        mutates the working image.
        """
        if self.learned:
            return self._apply_learned(
                session, instruction, output=output, seed=seed, negative_prompt=negative_prompt
            )
        return self._apply_deterministic(session, instruction, output=output)

    def _apply_deterministic(
        self,
        session: EditSession,
        instruction: str,
        *,
        output: str | Path | None = None,
    ) -> ImageEditReport:
        plan = self.plan(instruction)
        if not plan.ops:
            return ImageEditReport(
                ok=False,
                message=(
                    "I could not find an edit in that instruction. Try words like "
                    "rotate, flip, resize, crop, brighten, contrast, grayscale, blur, "
                    "or switch to the learned editor for changes like “make him smile”."
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
        target = Path(output) if output is not None else self._default_path(session)
        try:
            result.save_png(target)
        except OSError as error:
            return ImageEditReport(
                ok=False,
                message=f"Could not save the edited image: {error}",
                source=session.source_path,
                instruction=instruction,
            )
        session.undo_stack.append(session.raster.copy())
        session.raster = result
        session.history.append(
            EditStep(label="; ".join(op.description for op in plan.ops), learned=False)
        )
        for word in plan.unknown:
            if word not in session.unknown_words:
                session.unknown_words.append(word)
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
            learned=False,
        )

    def _apply_learned(
        self,
        session: EditSession,
        instruction: str,
        *,
        output: str | Path | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageEditReport:
        text = (instruction or "").strip()
        if not text:
            return ImageEditReport(
                ok=False,
                message="Write an instruction first.",
                source=session.source_path,
                instruction=instruction,
                learned=True,
            )
        if self._editor is None:  # pragma: no cover - construction guards this
            return ImageEditReport(
                ok=False,
                message="The learned editor is not configured.",
                learned=True,
            )
        self._progress("Running the learned editing model (this can take a while)…")
        try:
            result = self._editor.edit(
                session.raster, text, seed=seed, negative_prompt=negative_prompt
            )
        except (ImageEditError, RuntimeError, ValueError) as error:
            return ImageEditReport(
                ok=False,
                message=str(error),
                source=session.source_path,
                instruction=instruction,
                learned=True,
            )
        target = Path(output) if output is not None else self._default_path(session)
        try:
            result.save_png(target)
        except OSError as error:
            return ImageEditReport(
                ok=False,
                message=f"Could not save the edited image: {error}",
                source=session.source_path,
                instruction=instruction,
                learned=True,
            )
        session.undo_stack.append(session.raster.copy())
        session.raster = result
        session.history.append(EditStep(label=f"learned: {text}", learned=True))
        session.result_path = str(target)
        describe = getattr(self._editor, "describe", None)
        description = describe(text) if callable(describe) else ""
        return ImageEditReport(
            ok=True,
            message=f"Edited image saved to {target}.",
            source=session.source_path,
            path=str(target),
            instruction=text,
            operations=(f"learned edit: {text}",),
            width=result.width,
            height=result.height,
            size_bytes=len(result.to_png()),
            learned=True,
            description=description,
        )

    # -- undo and saving --------------------------------------------------

    def undo(self, session: EditSession) -> bool:
        """Step back one edit (works for both backends); True when something changed."""
        if not session.undo_stack:
            return False
        session.raster = session.undo_stack.pop()
        if session.history:
            session.history.pop()
        return True

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
            learned=session.learned_used,
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

    def _default_path(self, session: EditSession) -> Path:
        stem = Path(session.source_path).stem or "image"
        step = len(session.history) + 1
        return self.image_dir / f"{stem}-edited-{step}.png"


__all__ = [
    "BACKEND_DETERMINISTIC",
    "BACKEND_LEARNED",
    "EXAMPLE_INSTRUCTION",
    "EditSession",
    "EditStep",
    "ImageEditReport",
    "ImageEditService",
    "LEARNED_EXAMPLE_INSTRUCTION",
    "READABLE_SUFFIXES",
]
