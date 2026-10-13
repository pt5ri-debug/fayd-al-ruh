"""Optional text-to-image: turn a written prompt into a real image file.

Image generation is *optional* and *additive*, exactly like the project's other
perception layer (:mod:`core.vision`). The base install stays dependency-light:
the default backend renders a PNG with only the standard library (``zlib`` and
``struct``), and a heavier backend can be dropped in behind the
:class:`ImageGenerator` protocol without the window, the CLI, or the service
knowing the difference.

Honesty is a hard rule here. The default :class:`ProceduralImageGenerator` does
**not** pretend to be a learned text-to-image model. It derives a deterministic
seed from the prompt and paints an abstract geometric composition from that seed;
the same prompt always yields the same picture, and the returned
:class:`GeneratedImage` says plainly that it is a geometric rendering, not a
depiction of the words. A real diffusion backend is a separate, optional,
explicitly-approved path -- never the default, and never silent.

The module is Tk-free and dependency-free at import time, so the generator and
its output are fully testable on a fresh clone.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

RGB = tuple[int, int, int]

#: The size used when a request does not name one.
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
#: A hard ceiling so a mistyped size can never allocate an unbounded buffer.
MAX_DIMENSION = 2048


@dataclass(frozen=True)
class ImageRequest:
    """One image request: the prompt, an optional size, an optional seed, and a
    negative prompt.

    ``seed`` overrides the prompt-derived seed, so the same picture can be
    reproduced or varied on demand; ``None`` means "derive it from the prompt".
    ``negative_prompt`` names what to steer away from (the learned backend uses a
    photorealistic default when it is ``None``); the dependency-free renderer
    ignores it.
    """

    prompt: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    seed: int | None = None
    negative_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("An image prompt cannot be empty.")
        for name, value in (("width", self.width), ("height", self.height)):
            if not 1 <= int(value) <= MAX_DIMENSION:
                raise ValueError(f"Image {name} must be in [1, {MAX_DIMENSION}].")


@dataclass(frozen=True)
class GeneratedImage:
    """A rendered image, its bytes, and an honest note about what produced it."""

    prompt: str
    width: int
    height: int
    png: bytes
    backend: str
    seed: int
    description: str
    #: True only when a *learned* text-to-image model drew the picture.
    learned: bool = False

    @property
    def size_bytes(self) -> int:
        return len(self.png)

    def save(self, path: str | Path) -> Path:
        """Write the PNG to ``path`` (creating parent directories) and return it."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.png)
        return target


def encode_png(width: int, height: int, rows: Iterable[bytes]) -> bytes:
    """Encode 8-bit RGB rows (3 bytes per pixel) as a PNG, standard library only.

    Each row must be exactly ``width * 3`` bytes. The result is a valid PNG with
    a single ``IDAT`` chunk; no colour profile or ancillary chunk is written.
    """
    raw = bytearray()
    expected = width * 3
    count = 0
    for row in rows:
        if len(row) != expected:
            raise ValueError(f"Each row must be {expected} bytes, got {len(row)}.")
        raw.append(0)  # filter type 0 (None) for this scanline
        raw.extend(row)
        count += 1
    if count != height:
        raise ValueError(f"Expected {height} rows, got {count}.")

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- palette


def _seed_for(prompt: str, seed: int | None) -> int:
    """A 32-bit seed that depends on the prompt *and* the optional explicit seed.

    The prompt always contributes, so two different prompts never collapse to the
    same picture; an explicit ``seed`` only varies or reproduces that picture.
    """
    digest = hashlib.sha256(prompt.strip().casefold().encode("utf-8")).digest()
    base = int.from_bytes(digest[:4], "big")
    if seed is None:
        return base
    mixed = hashlib.sha256(digest + (seed & 0xFFFFFFFF).to_bytes(4, "big")).digest()
    return int.from_bytes(mixed[:4], "big")


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> RGB:
    """Convert HSV (each in ``[0, 1]``) to an 8-bit RGB triple."""
    hue = hue % 1.0
    index = int(hue * 6) % 6
    fraction = hue * 6 - int(hue * 6)
    p = value * (1 - saturation)
    q = value * (1 - fraction * saturation)
    t = value * (1 - (1 - fraction) * saturation)
    table = {
        0: (value, t, p),
        1: (q, value, p),
        2: (p, value, t),
        3: (p, q, value),
        4: (t, p, value),
        5: (value, p, q),
    }[index]
    return tuple(int(round(channel * 255)) for channel in table)  # type: ignore[return-value]


class _Random:
    """A tiny deterministic PRNG (xorshift32), so a seed fully fixes the picture."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF or 0x1
        if self._state == 0:
            self._state = 0x1

    def next(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state

    def unit(self) -> float:
        return self.next() / 0xFFFFFFFF

    def between(self, low: float, high: float) -> float:
        return low + (high - low) * self.unit()

    def integer(self, low: int, high: int) -> int:
        return low + int(self.unit() * (high - low + 1))


# --------------------------------------------------------------------------- backends


@runtime_checkable
class ImageGenerator(Protocol):
    """Turns an :class:`ImageRequest` into a :class:`GeneratedImage`."""

    name: str
    learned: bool

    def generate(self, request: ImageRequest) -> GeneratedImage: ...


class ProceduralImageGenerator:
    """A dependency-free backend: a deterministic, abstract geometric composition.

    It is a *renderer*, not a model. The prompt selects a palette and a set of
    shapes through a seeded PRNG, so it is reproducible and instant, but it does
    not understand the words. This is stated in every :class:`GeneratedImage`.
    """

    name = "procedural"
    learned = False

    def __init__(self, *, shapes: int = 9) -> None:
        if shapes < 1:
            raise ValueError("A composition needs at least one shape.")
        self.shapes = shapes

    def generate(self, request: ImageRequest) -> GeneratedImage:
        width, height = request.width, request.height
        seed = _seed_for(request.prompt, request.seed)
        rng = _Random(seed)

        hue = rng.unit()
        top = _hsv_to_rgb(hue, 0.55, 0.35)
        bottom = _hsv_to_rgb(hue + 0.12, 0.65, 0.85)
        canvas = bytearray()
        for y in range(height):
            blend = y / max(1, height - 1)
            row = bytes(
                int(round(top[c] * (1 - blend) + bottom[c] * blend)) for c in range(3)
            ) * width
            canvas.extend(row)

        for _ in range(self.shapes):
            centre_x = rng.between(0, width)
            centre_y = rng.between(0, height)
            radius = rng.between(min(width, height) * 0.08, min(width, height) * 0.3)
            colour = _hsv_to_rgb(rng.unit(), rng.between(0.4, 0.9), rng.between(0.6, 1.0))
            _blend_circle(canvas, width, height, centre_x, centre_y, radius, colour, 0.5)

        rows = (bytes(canvas[offset : offset + width * 3]) for offset in range(0, len(canvas), width * 3))
        png = encode_png(width, height, rows)
        return GeneratedImage(
            prompt=request.prompt,
            width=width,
            height=height,
            png=png,
            backend=self.name,
            seed=seed,
            learned=False,
            description=(
                f"Abstract geometric image {width}x{height}, rendered deterministically "
                f"from the prompt (seed {seed}). This is a procedural composition, not a "
                "learned depiction of the words."
            ),
        )


def _blend_circle(
    canvas: bytearray,
    width: int,
    height: int,
    centre_x: float,
    centre_y: float,
    radius: float,
    colour: RGB,
    alpha: float,
) -> None:
    """Alpha-blend a filled circle into the RGB canvas, inside its bounding box."""
    left = max(0, int(centre_x - radius))
    right = min(width - 1, int(centre_x + radius))
    top = max(0, int(centre_y - radius))
    bottom = min(height - 1, int(centre_y + radius))
    radius_sq = radius * radius
    for y in range(top, bottom + 1):
        dy = y - centre_y
        for x in range(left, right + 1):
            dx = x - centre_x
            if dx * dx + dy * dy > radius_sq:
                continue
            offset = (y * width + x) * 3
            for channel in range(3):
                base = canvas[offset + channel]
                canvas[offset + channel] = int(round(base * (1 - alpha) + colour[channel] * alpha))


def load_image_generator(name: str = "procedural", **options: Any) -> ImageGenerator:
    """Return the named backend; raises ``ValueError`` for one this build lacks.

    ``procedural`` is always available. ``diffusion`` is the optional learned
    backend and is imported lazily, so this module never pulls torch on a base
    install; ``options`` are forwarded to it (``model``, ``steps``, ``guidance``,
    ``root``, ``runner``).
    """
    if name == "procedural":
        return ProceduralImageGenerator()
    if name == "diffusion":
        from core.imagegen_diffusion import DiffusionImageGenerator

        return DiffusionImageGenerator(**options)
    raise ValueError(
        f"Image generator {name!r} is not available in this build; "
        "use 'procedural' or 'diffusion'."
    )


def image_backend_names() -> tuple[str, ...]:
    """The backend names this build knows about, the learned one included."""
    return ("procedural", "diffusion")


def image_backend_status(root: Path | None = None) -> tuple[tuple[str, bool, str], ...]:
    """Report each backend as ``(name, available_now, note)``.

    ``procedural`` is always available. ``diffusion`` reports whether the optional
    extra is installed *and* its weights are cached; the note says what is missing
    so the CLI and the window can tell the user exactly what to do.
    """
    status: list[tuple[str, bool, str]] = [
        ("procedural", True, "Dependency-free deterministic renderer (not a model).")
    ]
    from core.optional_deps import FEATURE_IMAGE, pip_install_command

    try:
        from core.imagegen_diffusion import (
            DEFAULT_MODEL,
            DEFAULT_PRESET,
            DiffusionUnavailable,
            DiffusionImageGenerator,
            _require_extra,
            preset_names,
        )
    except ImportError:  # pragma: no cover - the module ships with the project
        status.append(("diffusion", False, "The learned backend module is missing."))
        return tuple(status)
    generator = DiffusionImageGenerator(root=root)
    presets = ", ".join(preset_names())
    try:
        _require_extra()
    except DiffusionUnavailable:
        status.append(
            (
                "diffusion",
                False,
                "Learned backend needs the optional extra: "
                + pip_install_command(FEATURE_IMAGE)
                + f"  Presets: {presets}.",
            )
        )
        return tuple(status)
    if generator.is_available():
        status.append(
            (
                "diffusion",
                True,
                f"Learned model {DEFAULT_MODEL!r} (preset {DEFAULT_PRESET!r}) is cached "
                f"and ready. Presets: {presets}.",
            )
        )
    else:
        status.append(
            (
                "diffusion",
                False,
                f"Extra installed; model {DEFAULT_MODEL!r} (preset {DEFAULT_PRESET!r}) "
                "is not cached yet (run: python -m cli image download --approve). "
                f"Presets: {presets}.",
            )
        )
    return tuple(status)


# --------------------------------------------------------------------------- storage


def default_image_dir(root: Path | None = None) -> Path:
    """Where generated images are kept when the caller does not choose a path."""
    if root is not None:
        return Path(root)
    from core.paths import data_dir

    return data_dir() / "images"


def image_filename(prompt: str, seed: int, *, extension: str = "png") -> str:
    """A stable, filesystem-safe file name for a prompt and seed."""
    digest = hashlib.sha256(f"{prompt.strip().casefold()}:{seed}".encode("utf-8")).hexdigest()
    return f"image-{digest[:12]}.{extension}"


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "GeneratedImage",
    "ImageGenerator",
    "ImageRequest",
    "MAX_DIMENSION",
    "ProceduralImageGenerator",
    "RGB",
    "default_image_dir",
    "encode_png",
    "image_backend_names",
    "image_backend_status",
    "image_filename",
    "load_image_generator",
]
