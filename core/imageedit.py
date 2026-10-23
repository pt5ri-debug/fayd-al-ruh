"""Edit an existing image from a written instruction: the dependency-free engine.

The project's other image layer (:mod:`core.imagegen`) turns text into a *new*
picture. This module is its counterpart: it takes an image the user already has
and applies the change they describe in words -- "rotate 90 and brighten",
"grayscale", "resize to 512x512", "crop square".

Honesty is a hard rule, exactly as in :mod:`core.imagegen`. This is **not** a
learned instruction-following image model. It is a deterministic interpreter: a
small, documented vocabulary of keywords is matched in the order they appear, and
each match becomes a plain pixel operation (rotate, flip, resize, crop,
brightness, contrast, saturation, grayscale, invert, sepia, warm/cool, blur,
sharpen, threshold). The same image and the same words always give the same
picture, and anything the interpreter does not understand is *reported*, never
silently dropped.

Decoding is standard-library only (a real PNG reader with all five scanline
filters, plus binary PPM), and encoding reuses :func:`core.imagegen.encode_png`.
No third-party package is imported, so the engine and its output are fully
testable on a fresh clone.
"""

from __future__ import annotations

import re
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.imagegen import MAX_DIMENSION, encode_png

RGB = tuple[int, int, int]

#: PNG colour type -> number of channels (samples per pixel).
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
#: PNG colour type -> a human name, for error messages.
_COLOUR_NAMES = {
    0: "grayscale",
    2: "truecolour",
    3: "palette",
    4: "grayscale+alpha",
    6: "truecolour+alpha",
}


class ImageEditError(ValueError):
    """Raised when an image cannot be decoded or an operation cannot be applied."""


# --------------------------------------------------------------------------- raster


@dataclass
class Raster:
    """A plain RGB bitmap: ``width``, ``height``, and three bytes per pixel.

    The bytes live in one ``bytearray`` (row-major, ``R`` then ``G`` then ``B``),
    which keeps the pixel operations fast and the memory small. Alpha, if the
    source had any, is already composited over white by :func:`decode_image`, so
    every pixel here is fully opaque.
    """

    width: int
    height: int
    data: bytearray

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ImageEditError("Image dimensions must be positive.")
        if len(self.data) != self.width * self.height * 3:
            raise ImageEditError("Pixel data must be width * height * 3 bytes.")

    @classmethod
    def solid(cls, width: int, height: int, colour: RGB) -> Raster:
        """A single-colour bitmap, handy for tests and padding."""
        return cls(width, height, bytearray(bytes(colour) * (width * height)))

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / self.height, 4)

    def copy(self) -> Raster:
        return Raster(self.width, self.height, bytearray(self.data))

    def pixel(self, x: int, y: int) -> RGB:
        offset = (y * self.width + x) * 3
        return (self.data[offset], self.data[offset + 1], self.data[offset + 2])

    def pixels(self) -> Iterator[RGB]:
        data = self.data
        for offset in range(0, len(data), 3):
            yield (data[offset], data[offset + 1], data[offset + 2])

    def to_image_buffer(self) -> Any:
        """Convert to :class:`core.vision.ImageBuffer` for the perception layer."""
        from core.vision import ImageBuffer

        return ImageBuffer(
            self.width,
            self.height,
            tuple(self.pixels()),
        )

    @classmethod
    def from_image_buffer(cls, image: Any) -> Raster:
        """Build a raster from a :class:`core.vision.ImageBuffer`."""
        data = bytearray()
        for red, green, blue in image.pixels:
            data.extend((red, green, blue))
        return cls(image.width, image.height, data)

    def to_png(self) -> bytes:
        """Encode as a PNG (reusing the project's standard-library encoder)."""
        stride = self.width * 3
        rows = (
            bytes(self.data[y * stride : (y + 1) * stride]) for y in range(self.height)
        )
        return encode_png(self.width, self.height, rows)

    def save_png(self, path: str | Path) -> Path:
        """Write the PNG to ``path`` (creating parents) and return the path."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_png())
        return target


# --------------------------------------------------------------------------- decoding


def decode_image(path: str | Path) -> Raster:
    """Decode a PNG or binary PPM file into a :class:`Raster`.

    Raises :class:`ImageEditError` with a readable message for anything the
    engine cannot read (an unknown format, a truncated file, an interlaced PNG,
    or a colour type it does not handle).
    """
    payload = Path(path).expanduser().read_bytes()
    return decode_image_bytes(payload)


def decode_image_bytes(payload: bytes) -> Raster:
    """Decode image bytes; see :func:`decode_image`."""
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return _decode_png(payload)
    if payload[:2] == b"P6":
        return _decode_ppm(payload)
    raise ImageEditError(
        "Unsupported image format. This build reads PNG and binary PPM (P6) files."
    )


def _decode_png(payload: bytes) -> Raster:
    width = height = bitdepth = colour_type = interlace = -1
    palette: tuple[RGB, ...] = ()
    transparency: bytes = b""
    idat = bytearray()
    index = 8  # skip the 8-byte signature
    while index + 8 <= len(payload):
        length = struct.unpack(">I", payload[index : index + 4])[0]
        tag = payload[index + 4 : index + 8]
        data = payload[index + 8 : index + 8 + length]
        if len(data) != length:
            raise ImageEditError("The PNG file is truncated.")
        if tag == b"IHDR":
            width, height, bitdepth, colour_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", data
            )
        elif tag == b"PLTE":
            palette = tuple(
                (data[i], data[i + 1], data[i + 2]) for i in range(0, len(data) - 2, 3)
            )
        elif tag == b"tRNS":
            transparency = data
        elif tag == b"IDAT":
            idat.extend(data)
        elif tag == b"IEND":
            break
        index += 12 + length

    if width < 1 or height < 1:
        raise ImageEditError("The PNG file has no IHDR header.")
    if colour_type not in _CHANNELS:
        raise ImageEditError(
            f"PNG colour type {colour_type} ({_COLOUR_NAMES.get(colour_type, '?')}) "
            "is not supported."
        )
    if bitdepth not in (1, 2, 4, 8, 16):
        raise ImageEditError(f"PNG bit depth {bitdepth} is not supported.")
    if interlace:
        raise ImageEditError("Interlaced (Adam7) PNGs are not supported.")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise ImageEditError(f"The PNG image data could not be decompressed: {error}") from error

    channels = _CHANNELS[colour_type]
    scanline_bytes = (width * channels * bitdepth + 7) // 8
    filter_bpp = max(1, (channels * bitdepth + 7) // 8)
    recon = _unfilter(raw, height, scanline_bytes, filter_bpp)
    return _png_pixels(
        recon, width, height, channels, bitdepth, colour_type, palette, transparency
    )


def _unfilter(raw: bytes, height: int, scanline_bytes: int, bpp: int) -> bytearray:
    """Reverse the per-scanline PNG filters into a contiguous sample buffer."""
    expected = height * (scanline_bytes + 1)
    if len(raw) < expected:
        raise ImageEditError("The PNG image data is shorter than its header claims.")
    out = bytearray(height * scanline_bytes)
    prev_start = 0
    for row in range(height):
        base = row * (scanline_bytes + 1)
        filter_type = raw[base]
        line = raw[base + 1 : base + 1 + scanline_bytes]
        out_start = row * scanline_bytes
        for i in range(scanline_bytes):
            value = line[i]
            left = out[out_start + i - bpp] if i >= bpp else 0
            up = out[prev_start + i] if row > 0 else 0
            up_left = out[prev_start + i - bpp] if (row > 0 and i >= bpp) else 0
            if filter_type == 0:
                result = value
            elif filter_type == 1:
                result = value + left
            elif filter_type == 2:
                result = value + up
            elif filter_type == 3:
                result = value + ((left + up) >> 1)
            elif filter_type == 4:
                result = value + _paeth(left, up, up_left)
            else:
                raise ImageEditError(f"Unknown PNG scanline filter {filter_type}.")
            out[out_start + i] = result & 0xFF
        prev_start = out_start
    return out


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_pixels(
    recon: bytearray,
    width: int,
    height: int,
    channels: int,
    bitdepth: int,
    colour_type: int,
    palette: tuple[RGB, ...],
    transparency: bytes,
) -> Raster:
    """Turn the unfiltered sample buffer into opaque RGB pixels."""
    maxval = (1 << bitdepth) - 1
    data = bytearray(width * height * 3)

    def sample(row: int, pixel: int, channel: int) -> int:
        if bitdepth == 8:
            return recon[(row * width + pixel) * channels + channel]
        if bitdepth == 16:
            offset = (row * width + pixel) * channels * 2 + channel * 2
            return recon[offset]  # take the high byte, dropping to 8-bit
        bit_index = (pixel * channels + channel) * bitdepth
        byte = recon[row * ((width * channels * bitdepth + 7) // 8) + (bit_index >> 3)]
        shift = 8 - bitdepth - (bit_index & 7)
        return (byte >> shift) & maxval

    for row in range(height):
        for pixel in range(width):
            out = (row * width + pixel) * 3
            if colour_type in (0, 4):
                grey = sample(row, pixel, 0)
                if bitdepth < 8:
                    grey = grey * 255 // maxval
                if colour_type == 4:
                    alpha = sample(row, pixel, 1)
                    grey = (grey * alpha + 255 * (255 - alpha)) // 255
                data[out] = data[out + 1] = data[out + 2] = grey
            elif colour_type in (2, 6):
                red = sample(row, pixel, 0)
                green = sample(row, pixel, 1)
                blue = sample(row, pixel, 2)
                if colour_type == 6:
                    alpha = sample(row, pixel, 3)
                    red = (red * alpha + 255 * (255 - alpha)) // 255
                    green = (green * alpha + 255 * (255 - alpha)) // 255
                    blue = (blue * alpha + 255 * (255 - alpha)) // 255
                data[out], data[out + 1], data[out + 2] = red, green, blue
            else:  # palette
                index = sample(row, pixel, 0)
                if index >= len(palette):
                    raise ImageEditError("The PNG palette index is out of range.")
                red, green, blue = palette[index]
                alpha = transparency[index] if index < len(transparency) else 255
                if alpha != 255:
                    red = (red * alpha + 255 * (255 - alpha)) // 255
                    green = (green * alpha + 255 * (255 - alpha)) // 255
                    blue = (blue * alpha + 255 * (255 - alpha)) // 255
                data[out], data[out + 1], data[out + 2] = red, green, blue
    return Raster(width, height, data)


def _decode_ppm(payload: bytes) -> Raster:
    fields: list[bytes] = []
    index = 0
    while len(fields) < 4:
        while index < len(payload) and payload[index : index + 1].isspace():
            index += 1
        if payload[index : index + 1] == b"#":
            while index < len(payload) and payload[index : index + 1] != b"\n":
                index += 1
            continue
        start = index
        while index < len(payload) and not payload[index : index + 1].isspace():
            index += 1
        fields.append(payload[start:index])
    index += 1
    if fields[0] != b"P6":
        raise ImageEditError("Only binary PPM (P6) images are supported.")
    width, height, maxval = (int(field) for field in fields[1:4])
    if maxval != 255:
        raise ImageEditError("Only 8-bit PPM images are supported.")
    body = payload[index : index + width * height * 3]
    if len(body) != width * height * 3:
        raise ImageEditError("The PPM payload is truncated.")
    return Raster(width, height, bytearray(body))


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class EditOp:
    """One resolved operation: a ``kind``, its parameters, and how to describe it."""

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class EditPlan:
    """The operations a written instruction asked for, and what was not understood.

    ``ops`` are in the order the words appear, so "resize then rotate" and
    "rotate then resize" do different things, as written. ``unknown`` lists the
    phrases the interpreter could not turn into an operation; the window shows
    them so nothing is silently ignored.
    """

    instruction: str
    ops: tuple[EditOp, ...]
    unknown: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.ops)

    def describe(self) -> str:
        if not self.ops:
            return "No operation was recognised."
        return "; ".join(op.description for op in self.ops)


# --- keyword patterns. Each entry: a regex, and a builder from the match to an op.

#: A percentage like ``20%`` or a bare number like ``1.5``; both are captured, so
#: a rule can tell "brighten 20%" from "brighten 20".
_AMOUNT = r"(?:(?P<pct>-?\d{1,3})\s*%|(?P<num>\d+(?:\.\d+)?))"

# Words that carry no operation of their own and must not be reported as unknown.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "to", "it", "this", "that", "image", "picture",
        "photo", "please", "then", "make", "turn", "of", "into", "by", "with",
        "some", "bit", "more", "less", "very", "slightly", "up", "down", "degree",
        "degrees", "colour", "color", "colours", "colors", "quite", "really",
        "just", "for", "me", "its", "also", "somewhat",
    }
)

#: A percentage like ``20%`` or a bare number like ``1.5``; both are captured, so
#: a rule can tell "brighten 20%" from "brighten 20".
_AMOUNT = r"(?:(?P<pct>-?\d{1,3})\s*%|(?P<num>\d+(?:\.\d+)?))"
#: A direction word, shared by the rotate rule.
_DIRECTION = (
    r"(?P<dir>clockwise|anticlockwise|anticlock-wise|counterclockwise|"
    r"counter-clockwise|anti-clockwise|left|right)"
)


def _clamp_scale(value: float) -> float:
    return max(0.05, min(20.0, value))


def _amount(match: re.Match[str], *, percent_scales: bool = False) -> float | None:
    """Read a rule's captured amount as a plain number, or ``None`` if absent."""
    groups = match.groupdict()
    if groups.get("pct") is not None:
        return float(groups["pct"]) / 100 if percent_scales else float(groups["pct"])
    if groups.get("num") is not None:
        return float(groups["num"])
    return None


def _brightness_op(match: re.Match[str]) -> EditOp:
    token = match.group(0).lower()
    reducing = any(
        word in token for word in ("reduce", "lower", "decrease", "dim", "dark")
    )
    amount = _amount(match, percent_scales=True)
    if amount is not None:
        factor = 1.0 - amount if reducing else 1.0 + amount
    elif reducing:
        factor = 0.75
    else:
        factor = 1.25
    factor = max(0.1, min(3.0, factor))
    return EditOp("brightness", {"factor": factor}, f"brightness ×{factor:g}")


def _contrast_op(match: re.Match[str]) -> EditOp:
    token = match.group(0).lower()
    reducing = any(
        word in token for word in ("less", "reduce", "decrease", "lower", "soft")
    )
    amount = _amount(match, percent_scales=True)
    if amount is not None:
        factor = 1.0 - amount if reducing else 1.0 + amount
        factor = factor if abs(factor) <= 4 else amount
    elif reducing:
        factor = 0.75
    else:
        factor = 1.35
    factor = _clamp_scale(factor)
    return EditOp("contrast", {"factor": factor}, f"contrast ×{factor:g}")


def _saturation_op(match: re.Match[str]) -> EditOp:
    token = match.group(0).lower()
    if "desaturate" in token:
        return EditOp("saturation", {"factor": 0.0}, "desaturate fully")
    reducing = any(
        word in token for word in ("less", "reduce", "decrease", "lower", "muted", "dull")
    )
    amount = _amount(match, percent_scales=True)
    if amount is not None:
        factor = 1.0 - amount if reducing else 1.0 + amount
        factor = factor if abs(factor) <= 4 else amount
    elif reducing:
        factor = 0.5
    else:
        factor = 1.4
    factor = _clamp_scale(factor)
    return EditOp("saturation", {"factor": factor}, f"saturation ×{factor:g}")


def _rotate_op(match: re.Match[str]) -> EditOp:
    groups = match.groupdict()
    direction = (groups.get("dir") or "").lower()
    degrees = groups.get("deg")
    if degrees is not None:
        value = int(degrees) % 360
    elif any(word in direction for word in ("left", "counter", "anti")):
        value = 270
    else:
        value = 90
    if value not in (0, 90, 180, 270):
        value = 90
    return EditOp("rotate", {"degrees": value}, f"rotate {value}°")


def _resize_op(match: re.Match[str]) -> EditOp:
    groups = match.groupdict()
    if groups.get("pct") is not None:
        factor = float(groups["pct"]) / 100
        return EditOp("resize", {"scale": factor}, f"scale ×{factor:g}")
    if groups.get("num") is not None:  # "scale 2" means twice as large
        factor = float(groups["num"])
        return EditOp("resize", {"scale": factor}, f"scale ×{factor:g}")
    width, height = int(groups["w"]), int(groups["h"])
    return EditOp("resize", {"width": width, "height": height}, f"resize to {width}×{height}")


def _crop_op(match: re.Match[str]) -> EditOp:
    width, height = int(match.group("w")), int(match.group("h"))
    return EditOp("crop", {"width": width, "height": height}, f"crop to {width}×{height}")


def _blur_op(match: re.Match[str]) -> EditOp:
    radius = match.group("radius")
    value = max(1, min(50, int(float(radius)))) if radius else 2
    return EditOp("blur", {"radius": value}, f"blur (radius {value})")


def _simple(kind: str, description: str) -> Any:
    def build(_match: re.Match[str]) -> EditOp:
        return EditOp(kind, {}, description)

    return build


_SIZE_GROUPS = r"(?P<w>\d{1,4})\s*(?:x|×|by)\s*(?P<h>\d{1,4})"

#: ``(regex, builder)`` pairs. ``finditer`` matches of *all* rules are pooled and
#: sorted by start position, then by length (longest first), so a longer match at
#: the same position (``brighten``) wins over a shorter prefix (``bright``) and the
#: resulting operation order follows the words.
_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (
        re.compile(
            r"(?:rotate|turn|spin)\s*(?:it\s*)?(?:by\s*)?"
            r"(?:(?P<deg>\d{1,3})\s*(?:°|deg(?:rees)?)?|"
            r"(?P<dir>clockwise|anticlockwise|counterclockwise|counter-clockwise|"
            r"anti-clockwise|left|right))"
        ),
        _rotate_op,
    ),
    (
        re.compile(r"(?:flip|mirror)\s*(?:it\s*)?(?:horizontally|horizontal|left-right|lr)"),
        _simple("flip_h", "flip horizontally"),
    ),
    (
        re.compile(r"(?:flip|mirror)\s*(?:it\s*)?(?:vertically|vertical|upside-down|upside|ud)"),
        _simple("flip_v", "flip vertically"),
    ),
    (
        re.compile(rf"(?:resize|scale|shrink|enlarge|zoom)\s*(?:it\s*)?(?:to\s*)?(?:{_SIZE_GROUPS}|{_AMOUNT})"),
        _resize_op,
    ),
    (
        re.compile(rf"crop\s*(?:it\s*)?(?:to\s*|the\s*center\s*)?(?:square\s*)?{_SIZE_GROUPS}"),
        _crop_op,
    ),
    (
        re.compile(r"crop\s*(?:it\s*)?(?:to\s*)?(?:a\s*)?square"),
        _simple("crop_square", "crop to a square"),
    ),
    (
        re.compile(
            rf"(?:(?P<verb>reduce|lower|decrease|dim|darken|increase|boost|raise|brighten|lighten)\s+)?"
            rf"(?:the\s+)?"
            rf"(?:brightness|brighten|brighter|bright|lightness|lighten|lighter|light|"
            rf"darkness|darken|darker|dark|exposure)"
            rf"\s*(?:by\s*)?{_AMOUNT}?"
        ),
        _brightness_op,
    ),
    (
        re.compile(
            rf"(?:(?P<verb>more|less|increase|reduce|decrease|boost|lower|add|remove)\s+)?"
            rf"contrast\s*(?:by\s*)?{_AMOUNT}?"
        ),
        _contrast_op,
    ),
    (
        re.compile(
            rf"(?:(?P<verb>more|less|increase|reduce|decrease|boost|lower|add|remove)\s+)?"
            rf"(?:saturat\w*|vivid\w*|desaturate)\s*(?:by\s*)?{_AMOUNT}?"
        ),
        _saturation_op,
    ),
    (
        re.compile(r"blur\s*(?:it\s*)?(?:by\s*)?(?P<radius>\d+)?\s*(?:px|pixels?)?|blur\s+the"),
        _blur_op,
    ),
    (
        re.compile(r"sharpen\s*(?:it\s*)?(?:a\s*bit)?|sharp(?:er)?\b"),
        _simple("sharpen", "sharpen"),
    ),
    (
        re.compile(r"grayscale|greyscale|black\s*(?:and|&)\s*white|b\s*&\s*w|monochrome"),
        _simple("grayscale", "grayscale"),
    ),
    (re.compile(r"invert|negat\w*|negative"), _simple("invert", "invert colours")),
    (re.compile(r"sepia"), _simple("sepia", "sepia tone")),
    (re.compile(r"\b(?:warm|warmer)\b"), _simple("warm", "warm tint")),
    (re.compile(r"\b(?:cool|cooler)\b"), _simple("cool", "cool tint")),
    (re.compile(r"threshold\s*(?:at\s*)?(?P<radius>\d{1,3})?|posterize"), _simple("threshold", "threshold to black and white")),
)

#: Arabic aliases, kept as a secondary vocabulary (project convention). Each is
#: ``(regex, builder)`` and behaves exactly like its English counterpart.
_ARABIC_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (re.compile(r"أبيض\s*و\s*أسود|ابيض\s*و\s*اسود|رمادي|تدرج\s*رمادي"), _simple("grayscale", "grayscale")),
    (re.compile(r"اعكس|عكس\s*الألوان|عكس\s*الالوان|سلبي"), _simple("invert", "invert colours")),
    (re.compile(r"دوّر|دور\s*(?:بـ?\s*)?(?:يمين|يمينا|ساعة)?"), _rotate_op),
    (re.compile(r"اقلب\s*(?:أفقيا|افقيا)?"), _simple("flip_h", "flip horizontally")),
    (re.compile(r"سيبيا"), _simple("sepia", "sepia tone")),
    (re.compile(r"ضبابي|ضبّب"), _simple("blur", "blur (radius 2)")),
    (re.compile(r"أفتح|افتح|تفتيح"), _brightness_op),
    (re.compile(r"أغمق|اغمق|تغميق"), _brightness_op),
)

_ALL_RULES: tuple[tuple[re.Pattern[str], Any], ...] = _RULES + _ARABIC_RULES


def plan_edits(instruction: str) -> EditPlan:
    """Interpret a written instruction as an ordered list of operations.

    Operations are collected in the order their words appear, so phrasing decides
    the order of application. Text that matches nothing is returned in ``unknown``
    (the meaningful words, stopwords removed) so the caller can say plainly what
    it did not understand.
    """
    text = (instruction or "").strip()
    if not text:
        return EditPlan(instruction="", ops=(), unknown=())

    lowered = text.casefold()
    matches: list[tuple[int, int, EditOp]] = []
    spans: list[tuple[int, int]] = []
    for pattern, build in _ALL_RULES:
        for match in pattern.finditer(lowered):
            matches.append((match.start(), match.end(), build(match)))
            spans.append((match.start(), match.end()))

    # Order by position; at the same position the longer match wins, and a match
    # contained in one already taken is dropped (so "brighten" does not also fire
    # the shorter "bright", and "desaturate" does not also fire "saturate").
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    ops: list[EditOp] = []
    taken: list[tuple[int, int]] = []
    for start, end, op in matches:
        if any(taken_start <= start and end <= taken_end for taken_start, taken_end in taken):
            continue
        ops.append(op)
        taken.append((start, end))
    unknown = _unknown_words(lowered, taken)
    return EditPlan(instruction=text, ops=tuple(ops), unknown=unknown)


def _unknown_words(lowered: str, spans: list[tuple[int, int]]) -> tuple[str, ...]:
    """The meaningful words the interpreter did not consume."""
    covered = bytearray(len(lowered))
    for start, end in spans:
        for position in range(max(0, start), min(len(lowered), end)):
            covered[position] = 1
    leftover = "".join(
        " " if covered[i] else lowered[i] for i in range(len(lowered))
    )
    words: list[str] = []
    for word in re.findall(r"[a-z]{3,}", leftover):
        if word in _STOPWORDS or word in words:
            continue
        words.append(word)
    return tuple(words)


# --------------------------------------------------------------------------- operations


def apply_plan(raster: Raster, plan: EditPlan) -> Raster:
    """Apply every operation in ``plan`` to ``raster`` in order."""
    result = raster
    for op in plan.ops:
        result = apply_op(result, op)
    return result


def apply_op(raster: Raster, op: EditOp) -> Raster:
    """Apply a single operation, dispatching on its kind."""
    handler = _OPERATIONS.get(op.kind)
    if handler is None:
        raise ImageEditError(f"Unknown edit operation {op.kind!r}.")
    return handler(raster, op.params)


def _op_rotate(raster: Raster, params: Mapping[str, Any]) -> Raster:
    degrees = int(params.get("degrees", 90)) % 360
    if degrees == 0:
        return raster.copy()
    if degrees == 180:
        return Raster(raster.width, raster.height, bytearray(_chunks_reversed(raster.data, 3)))
    if degrees not in (90, 270):
        raise ImageEditError("Rotation must be 90, 180, or 270 degrees.")
    width, height = raster.height, raster.width
    out = bytearray(width * height * 3)
    src = raster.data
    for y in range(raster.height):
        for x in range(raster.width):
            if degrees == 90:  # clockwise: (x, y) -> (height-1-y, x)
                nx, ny = raster.height - 1 - y, x
            else:  # 270: counter-clockwise: (x, y) -> (y, width-1-x)
                nx, ny = y, raster.width - 1 - x
            src_offset = (y * raster.width + x) * 3
            dst_offset = (ny * width + nx) * 3
            out[dst_offset : dst_offset + 3] = src[src_offset : src_offset + 3]
    return Raster(width, height, out)


def _chunks_reversed(data: bytearray, size: int) -> Iterable[int]:
    for offset in range(len(data) - size, -1, -size):
        yield from data[offset : offset + size]


def _op_flip_h(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    stride = raster.width * 3
    out = bytearray(len(raster.data))
    for y in range(raster.height):
        row = raster.data[y * stride : (y + 1) * stride]
        for x in range(raster.width):
            source = (raster.width - 1 - x) * 3
            out[y * stride + x * 3 : y * stride + x * 3 + 3] = row[source : source + 3]
    return Raster(raster.width, raster.height, out)


def _op_flip_v(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    stride = raster.width * 3
    out = bytearray(len(raster.data))
    for y in range(raster.height):
        source = (raster.height - 1 - y) * stride
        out[y * stride : (y + 1) * stride] = raster.data[source : source + stride]
    return Raster(raster.width, raster.height, out)


def _op_resize(raster: Raster, params: Mapping[str, Any]) -> Raster:
    if "scale" in params:
        factor = float(params["scale"])
        width = max(1, round(raster.width * factor))
        height = max(1, round(raster.height * factor))
    else:
        width = int(params.get("width") or raster.width)
        height = int(params.get("height") or raster.height)
    width = max(1, min(MAX_DIMENSION, width))
    height = max(1, min(MAX_DIMENSION, height))
    if width == raster.width and height == raster.height:
        return raster.copy()
    return _bilinear(raster, width, height)


def _bilinear(raster: Raster, width: int, height: int) -> Raster:
    """Resample with bilinear interpolation (deterministic, integer maths)."""
    src = raster.data
    out = bytearray(width * height * 3)
    x_ratio = (raster.width - 1) / max(1, width - 1) if width > 1 else 0.0
    y_ratio = (raster.height - 1) / max(1, height - 1) if height > 1 else 0.0
    for y in range(height):
        sy = y * y_ratio
        y0 = int(sy)
        y1 = min(y0 + 1, raster.height - 1)
        wy = sy - y0
        for x in range(width):
            sx = x * x_ratio
            x0 = int(sx)
            x1 = min(x0 + 1, raster.width - 1)
            wx = sx - x0
            out_offset = (y * width + x) * 3
            for channel in range(3):
                top = src[(y0 * raster.width + x0) * 3 + channel]
                top2 = src[(y0 * raster.width + x1) * 3 + channel]
                bottom = src[(y1 * raster.width + x0) * 3 + channel]
                bottom2 = src[(y1 * raster.width + x1) * 3 + channel]
                upper = top + (top2 - top) * wx
                lower = bottom + (bottom2 - bottom) * wx
                out[out_offset + channel] = int(upper + (lower - upper) * wy + 0.5) & 0xFF
    return Raster(width, height, out)


def _op_crop(raster: Raster, params: Mapping[str, Any]) -> Raster:
    width = int(params.get("width") or raster.width)
    height = int(params.get("height") or raster.height)
    width = max(1, min(width, raster.width))
    height = max(1, min(height, raster.height))
    left = (raster.width - width) // 2
    top = (raster.height - height) // 2
    out = bytearray(width * height * 3)
    for y in range(height):
        source = ((top + y) * raster.width + left) * 3
        out[y * width * 3 : (y + 1) * width * 3] = raster.data[source : source + width * 3]
    return Raster(width, height, out)


def _op_crop_square(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    side = min(raster.width, raster.height)
    return _op_crop(raster, {"width": side, "height": side})


def _luma(red: int, green: int, blue: int) -> int:
    return (red * 299 + green * 587 + blue * 114) // 1000


def _map_channels(raster: Raster, transform: Any) -> Raster:
    out = bytearray(len(raster.data))
    data = raster.data
    for offset in range(0, len(data), 3):
        red, green, blue = transform(data[offset], data[offset + 1], data[offset + 2])
        out[offset] = red & 0xFF
        out[offset + 1] = green & 0xFF
        out[offset + 2] = blue & 0xFF
    return Raster(raster.width, raster.height, out)


def _op_brightness(raster: Raster, params: Mapping[str, Any]) -> Raster:
    factor = float(params.get("factor", 1.25))
    return _map_channels(raster, lambda r, g, b: (int(r * factor), int(g * factor), int(b * factor)))


def _op_contrast(raster: Raster, params: Mapping[str, Any]) -> Raster:
    factor = float(params.get("factor", 1.35))

    def apply(red: int, green: int, blue: int) -> RGB:
        return (
            int((red - 128) * factor + 128),
            int((green - 128) * factor + 128),
            int((blue - 128) * factor + 128),
        )

    return _map_channels(raster, apply)


def _op_saturation(raster: Raster, params: Mapping[str, Any]) -> Raster:
    factor = float(params.get("factor", 1.4))

    def apply(red: int, green: int, blue: int) -> RGB:
        grey = _luma(red, green, blue)
        return (
            int(grey + (red - grey) * factor),
            int(grey + (green - grey) * factor),
            int(grey + (blue - grey) * factor),
        )

    return _map_channels(raster, apply)


def _op_grayscale(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    def apply(red: int, green: int, blue: int) -> RGB:
        grey = _luma(red, green, blue)
        return (grey, grey, grey)

    return _map_channels(raster, apply)


def _op_invert(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    return _map_channels(raster, lambda r, g, b: (255 - r, 255 - g, 255 - b))


def _op_sepia(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    def apply(red: int, green: int, blue: int) -> RGB:
        return (
            int(red * 0.393 + green * 0.769 + blue * 0.189),
            int(red * 0.349 + green * 0.686 + blue * 0.168),
            int(red * 0.272 + green * 0.534 + blue * 0.131),
        )

    return _map_channels(raster, apply)


def _op_warm(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    return _map_channels(raster, lambda r, g, b: (r + 22, g + 8, b - 22))


def _op_cool(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    return _map_channels(raster, lambda r, g, b: (r - 18, g + 4, b + 26))


def _op_threshold(raster: Raster, params: Mapping[str, Any]) -> Raster:
    level = int(params.get("level", 128))

    def apply(red: int, green: int, blue: int) -> RGB:
        value = 255 if _luma(red, green, blue) >= level else 0
        return (value, value, value)

    return _map_channels(raster, apply)


def _op_blur(raster: Raster, params: Mapping[str, Any]) -> Raster:
    radius = max(1, int(params.get("radius", 2)))
    return _box_blur(raster, radius)


def _box_blur(raster: Raster, radius: int) -> Raster:
    """A separable box blur (two O(n) passes with a running sum)."""
    width, height = raster.width, raster.height
    horizontal = bytearray(len(raster.data))
    data = raster.data
    for y in range(height):
        row = y * width * 3
        for channel in range(3):
            window = 0
            count = 0
            for x in range(-radius, width + radius):
                if x + radius < width and 0 <= x + radius:
                    window += data[row + (x + radius) * 3 + channel]
                    count += 1
                if x - radius - 1 >= 0:
                    window -= data[row + (x - radius - 1) * 3 + channel]
                    count -= 1
                if 0 <= x < width:
                    horizontal[row + x * 3 + channel] = window // max(1, count)
    out = bytearray(len(raster.data))
    for x in range(width):
        for channel in range(3):
            window = 0
            count = 0
            for y in range(-radius, height + radius):
                if y + radius < height and 0 <= y + radius:
                    window += horizontal[((y + radius) * width + x) * 3 + channel]
                    count += 1
                if y - radius - 1 >= 0:
                    window -= horizontal[((y - radius - 1) * width + x) * 3 + channel]
                    count -= 1
                if 0 <= y < height:
                    out[(y * width + x) * 3 + channel] = window // max(1, count)
    return Raster(width, height, out)


def _op_sharpen(raster: Raster, _params: Mapping[str, Any]) -> Raster:
    blurred = _box_blur(raster, 1)
    out = bytearray(len(raster.data))
    source = raster.data
    for offset in range(len(source)):
        value = source[offset] + int((source[offset] - blurred.data[offset]) * 0.8)
        out[offset] = max(0, min(255, value))
    return Raster(raster.width, raster.height, out)


_OPERATIONS: dict[str, Any] = {
    "rotate": _op_rotate,
    "flip_h": _op_flip_h,
    "flip_v": _op_flip_v,
    "resize": _op_resize,
    "crop": _op_crop,
    "crop_square": _op_crop_square,
    "brightness": _op_brightness,
    "contrast": _op_contrast,
    "saturation": _op_saturation,
    "grayscale": _op_grayscale,
    "invert": _op_invert,
    "sepia": _op_sepia,
    "warm": _op_warm,
    "cool": _op_cool,
    "threshold": _op_threshold,
    "blur": _op_blur,
    "sharpen": _op_sharpen,
}


__all__ = [
    "EditOp",
    "EditPlan",
    "ImageEditError",
    "Raster",
    "apply_op",
    "apply_plan",
    "decode_image",
    "decode_image_bytes",
    "plan_edits",
]
