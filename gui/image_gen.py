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
from dataclasses import dataclass
from pathlib import Path

from core.imagegen import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    GeneratedImage,
    ImageGenerator,
    ImageRequest,
    default_image_dir,
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
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._generator = generator if generator is not None else load_image_generator(backend)

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

    def generate(
        self,
        prompt: str,
        *,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
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
            )
            image = self._generator.generate(request)
            target = Path(output) if output is not None else self._default_path(image)
            image.save(target)
        except (ValueError, OSError) as error:
            return ImageReport(ok=False, prompt=text, message=str(error))
        return self._to_report(image, target)

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
