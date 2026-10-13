"""The text-to-image service: build a request, render it, and save the file.

This is the Tk-free half of the window's "Generate an image from text…" screen.
It wraps :mod:`core.imagegen` with the small conveniences a chat needs: it reads
an optional size out of the prompt ("a red spiral 768x512"), keeps the saved
files under the data folder, and returns a :class:`ImageReport` the window can
render or the CLI can print. The window only collects the prompt and shows the
result, so all of this is testable without a display.

Nothing here invents a capability. The default backend is the dependency-free
:class:`~core.imagegen.ProceduralImageGenerator`, and every report carries its
honest description and a ``learned`` flag that is ``False`` for it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.imagegen import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    GeneratedImage,
    ImageGenerator,
    ImageRequest,
    default_image_dir,
    image_backend_status,
    image_filename,
    load_image_generator,
)

#: ``1024x768`` / ``1024×768`` / ``1024 by 768`` at the end of a prompt.
_SIZE_RE = re.compile(r"[,\s]*\b(\d{2,4})\s*(?:x|×|by)\s*(\d{2,4})\s*$", re.IGNORECASE)


def parse_size(prompt: str) -> tuple[str, int, int]:
    """Split a trailing ``WxH`` off the prompt; return ``(prompt, width, height)``.

    A prompt with no size token keeps its text and returns the defaults. The size
    token is removed from the prompt so the words sent to the backend describe the
    picture, not its dimensions.
    """
    match = _SIZE_RE.search(prompt)
    if not match:
        return prompt.strip(), DEFAULT_WIDTH, DEFAULT_HEIGHT
    width, height = int(match.group(1)), int(match.group(2))
    cleaned = prompt[: match.start()].strip()
    return (cleaned or prompt.strip()), width, height


@dataclass(frozen=True)
class ImageReport:
    """The outcome of one generation, ready for a window or a terminal."""

    ok: bool
    prompt: str
    message: str
    path: str = ""
    backend: str = ""
    learned: bool = False
    seed: int = 0
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    description: str = ""

    def summary(self) -> str:
        """A one-block, human-readable report."""
        if not self.ok:
            return f"Image generation failed: {self.message}"
        kind = "learned model" if self.learned else "procedural renderer"
        return (
            f"Image saved to {self.path}\n"
            f"  backend : {self.backend} ({kind})\n"
            f"  size    : {self.width}x{self.height}, {self.size_bytes} bytes\n"
            f"  {self.description}"
        )


class ImageGenerationService:
    """Generate an image from a text prompt and save it as a PNG file."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        generator: ImageGenerator | None = None,
        backend: str = "procedural",
        model: str | None = None,
        preset: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._backend_name = backend
        self._on_progress = on_progress
        if generator is not None:
            self._generator = generator
        else:
            options: dict[str, object] = {}
            if backend == "diffusion":
                if model is not None:
                    options["model"] = model
                if preset is not None:
                    options["preset"] = preset
                if steps is not None:
                    options["steps"] = steps
                if guidance is not None:
                    options["guidance"] = guidance
                if negative_prompt is not None:
                    options["negative_prompt"] = negative_prompt
                options["root"] = self._root
            self._generator = load_image_generator(backend, **options)

    @property
    def backend(self) -> str:
        return self._generator.name

    @property
    def learned(self) -> bool:
        return bool(getattr(self._generator, "learned", False))

    @property
    def image_dir(self) -> Path:
        """Where generated images are written when no explicit path is given."""
        return default_image_dir(self._root)

    @property
    def root(self) -> Path | None:
        """The explicit storage root, or ``None`` for the default data folder."""
        return self._root

    def for_backend(
        self,
        backend: str,
        *,
        model: str | None = None,
        preset: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> ImageGenerationService:
        """A service on the same storage root that uses a different backend."""
        return ImageGenerationService(
            root=self._root,
            backend=backend,
            model=model,
            preset=preset,
            steps=steps,
            guidance=guidance,
            negative_prompt=negative_prompt,
            on_progress=self._on_progress,
        )

    def backend_status(self) -> tuple[tuple[str, bool, str], ...]:
        """Every backend and whether it can run now, for a menu or a status line."""
        return image_backend_status(self._root)

    def available(self) -> bool:
        """True when the selected backend can actually run (so a menu can fall back)."""
        checker = getattr(self._generator, "is_available", None)
        return True if not callable(checker) else bool(checker())

    def generate(
        self,
        prompt: str,
        *,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        output: str | Path | None = None,
    ) -> ImageReport:
        """Render ``prompt`` and save the PNG; never raises for a bad request."""
        text, parsed_width, parsed_height = parse_size(prompt or "")
        if not text:
            return ImageReport(ok=False, prompt="", message="Write a prompt first.")
        try:
            request = ImageRequest(
                prompt=text,
                width=width if width is not None else parsed_width,
                height=height if height is not None else parsed_height,
                seed=seed,
                negative_prompt=negative_prompt,
            )
            image = self._generator.generate(request)
            target = Path(output) if output is not None else self._default_path(image)
            image.save(target)
        except (ValueError, OSError, RuntimeError) as error:
            return ImageReport(
                ok=False,
                prompt=text,
                message=str(error),
                backend=self.backend,
                learned=self.learned,
            )
        return self._to_report(image, target)

    def download_model(
        self,
        *,
        approved: bool = False,
        model: str | None = None,
        preset: str | None = None,
    ) -> ImageReport:
        """Download the learned model's weights (external action; needs approval)."""
        from core.imagegen_diffusion import (
            DiffusionUnavailable,
            download_model,
            resolve_preset,
        )

        try:
            name = model or resolve_preset(preset).model
        except ValueError as error:
            return ImageReport(ok=False, prompt="", message=str(error))
        try:
            path = download_model(
                name, root=self._root, approved=approved, progress=self._progress
            )
        except (DiffusionUnavailable, ValueError, OSError) as error:
            return ImageReport(ok=False, prompt=name, message=str(error))
        return ImageReport(
            ok=True,
            prompt=name,
            message=f"Model {name!r} is cached at {path}.",
            path=str(path),
            backend="diffusion",
            learned=True,
        )

    def _progress(self, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(message)

    def _default_path(self, image: GeneratedImage) -> Path:
        return self.image_dir / image_filename(image.prompt, image.seed)

    def _to_report(self, image: GeneratedImage, target: Path) -> ImageReport:
        return ImageReport(
            ok=True,
            prompt=image.prompt,
            message=f"Image saved to {target}.",
            path=str(target),
            backend=image.backend,
            learned=image.learned,
            seed=image.seed,
            width=image.width,
            height=image.height,
            size_bytes=image.size_bytes,
            description=image.description,
        )


__all__ = [
    "ImageGenerationService",
    "ImageReport",
    "parse_size",
]
