"""Optional multimodal perception: describe an image and reason over it.

Perception is *optional* and *additive*: the cognitive core and the assistant work
without it, and when an image is supplied it is turned into a short, honest text
description that joins the same grounded pipeline as any other evidence. Nothing
here pretends to recognise objects it cannot see; it reports measurable image
properties (size, aspect, brightness, colour balance, edge density) and attaches
the description as a citable document.

The image format is a plain, dependency-free PPM (``P6``) buffer, so the layer is
fully testable on a fresh clone. A real vision encoder can be dropped in behind
the :class:`VisionEncoder` protocol without touching the assistant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ImageBuffer:
    """A simple RGB image: width, height, and one ``(r, g, b)`` triple per pixel."""

    width: int
    height: int
    pixels: tuple[RGB, ...]

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Image dimensions must be positive.")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("Pixel count must equal width * height.")
        if any(not all(0 <= channel <= 255 for channel in pixel) for pixel in self.pixels):
            raise ValueError("RGB channels must be between 0 and 255.")

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 4)

    def to_ppm(self) -> bytes:
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        body = bytes(channel for pixel in self.pixels for channel in pixel)
        return header + body

    @classmethod
    def from_ppm(cls, payload: bytes) -> "ImageBuffer":
        # Parse the ASCII header: magic, width, height, maxval, then raw bytes.
        fields: list[bytes] = []
        index = 0
        while len(fields) < 4:
            while index < len(payload) and payload[index : index + 1].isspace():
                index += 1
            if index < len(payload) and payload[index : index + 1] == b"#":
                while index < len(payload) and payload[index : index + 1] != b"\n":
                    index += 1
                continue
            start = index
            while index < len(payload) and not payload[index : index + 1].isspace():
                index += 1
            fields.append(payload[start:index])
        index += 1  # single whitespace after maxval
        if fields[0] != b"P6":
            raise ValueError("Only binary PPM (P6) images are supported.")
        width, height, maxval = (int(field) for field in fields[1:4])
        if maxval != 255:
            raise ValueError("Only 8-bit PPM images are supported.")
        body = payload[index : index + width * height * 3]
        if len(body) != width * height * 3:
            raise ValueError("PPM payload is truncated.")
        pixels = tuple(
            (body[offset], body[offset + 1], body[offset + 2])
            for offset in range(0, len(body), 3)
        )
        return cls(width, height, pixels)

    def write_ppm(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_ppm())

    @classmethod
    def read_ppm(cls, path) -> "ImageBuffer":
        return cls.from_ppm(Path(path).read_bytes())

    @classmethod
    def solid(cls, width: int, height: int, colour: RGB) -> "ImageBuffer":
        return cls(width, height, tuple(colour for _ in range(width * height)))


@dataclass(frozen=True)
class ImageFeatures:
    """Measurable properties of an image, all in ``[0, 1]`` except size."""

    width: int
    height: int
    aspect_ratio: float
    brightness: float
    saturation: float
    edge_density: float

    @property
    def is_landscape(self) -> bool:
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


def extract_features(image: ImageBuffer) -> ImageFeatures:
    """Compute brightness, saturation, and edge density from raw pixels."""
    total = len(image.pixels)
    brightness_sum = 0.0
    saturation_sum = 0.0
    for red, green, blue in image.pixels:
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        brightness_sum += maximum
        saturation_sum += 0.0 if maximum == 0 else (maximum - minimum) / maximum

    edges = 0
    for row in range(image.height):
        for column in range(image.width - 1):
            left = image.pixels[row * image.width + column]
            right = image.pixels[row * image.width + column + 1]
            if abs(sum(left) - sum(right)) > 90:
                edges += 1
    horizontal_pairs = image.height * (image.width - 1)

    return ImageFeatures(
        width=image.width,
        height=image.height,
        aspect_ratio=image.aspect_ratio,
        brightness=round(brightness_sum / total / 255, 4),
        saturation=round(saturation_sum / total, 4),
        edge_density=round(edges / horizontal_pairs, 4) if horizontal_pairs else 0.0,
    )


@runtime_checkable
class VisionEncoder(Protocol):
    """Turns an image into a text description the assistant can cite."""

    name: str

    def describe(self, image: ImageBuffer) -> str: ...


class HeuristicVisionEncoder:
    """A dependency-free encoder that reports only what it can measure."""

    name = "heuristic"

    def describe(self, image: ImageBuffer) -> str:
        features = extract_features(image)
        orientation = (
            "landscape" if features.is_landscape else "portrait" if features.is_portrait else "square"
        )
        tone = (
            "dark"
            if features.brightness < 0.35
            else "bright"
            if features.brightness > 0.7
            else "medium-lit"
        )
        detail = (
            "high detail"
            if features.edge_density > 0.25
            else "low detail"
            if features.edge_density < 0.05
            else "medium detail"
        )
        return (
            f"{orientation.capitalize()} image {features.width}x{features.height} "
            f"(aspect {features.aspect_ratio}). "
            f"Lighting is {tone} (brightness {features.brightness:.2f}), "
            f"saturation {features.saturation:.2f}, "
            f"and {detail} (edge density {features.edge_density:.2f})."
        )


def load_vision_encoder(name: str = "heuristic") -> VisionEncoder:
    if name == "heuristic":
        return HeuristicVisionEncoder()
    raise ValueError(
        f"Vision encoder {name!r} is not available in this build; use 'heuristic'."
    )


@dataclass(frozen=True)
class ImageAnswer:
    """The answer to a question about an image, with the description as evidence."""

    question: str
    description: str
    answer: str
    used_model: bool
    citations: tuple[str, ...]


class MultimodalAssistant:
    """Answers questions about an image by grounding the description in the RAG store."""

    def __init__(self, assistant, *, encoder: VisionEncoder | None = None) -> None:
        self.assistant = assistant
        self.encoder = encoder or HeuristicVisionEncoder()

    def describe(self, image: ImageBuffer) -> str:
        return self.encoder.describe(image)

    def ask_about_image(self, question: str, image: ImageBuffer, *, limit: int = 4) -> ImageAnswer:
        description = self.describe(image)
        grounded_question = f"{question}\nImage description: {description}"
        result = self.assistant.ask(grounded_question, limit=limit)
        return ImageAnswer(
            question=question,
            description=description,
            answer=result.text,
            used_model=result.used_model,
            citations=result.citations,
        )


__all__ = [
    "HeuristicVisionEncoder",
    "ImageAnswer",
    "ImageBuffer",
    "ImageFeatures",
    "MultimodalAssistant",
    "RGB",
    "VisionEncoder",
    "extract_features",
    "load_vision_encoder",
]
