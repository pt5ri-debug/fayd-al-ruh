"""Safe, dependency-free text decoding for imported files.

Imported files arrive in whatever encoding their author's editor used. Guessing
wrong is worse than failing: decoding UTF-8 bytes as Latin-1 turns every Arabic
word into mojibake, and decoding with ``errors="replace"`` silently swaps real
text for ``\ufffd``. So this module *detects* an encoding conservatively, decodes
strictly, and -- when no listed encoding fits -- raises
:class:`EncodingDetectionError` rather than returning corrupted text.

The detector is deliberately small and explainable (no ``chardet`` dependency):

1. A BOM, if present, decides the encoding outright.
2. UTF-8 is tried first, strictly. It is accepted only when it decodes *and* the
   result is plausible text -- so ASCII-range UTF-16 (interleaved NULs), which is
   also valid UTF-8, is not mistaken for UTF-8. A stream that ends mid-character
   is reported as truncated rather than guessed at.
3. Otherwise a small, ordered set of common legacy encodings is tried, each
   strictly, and the first that decodes *every* byte into plausible text wins
   (Windows Arabic ``cp1256`` first, then Western and ISO-8859 and KOI8-R, then
   BOM-less UTF-16).
4. If none fits, the caller is told which encodings were tried -- never handed a
   lossy result.

Every decode here is strict; there is no path that replaces bytes with ``\ufffd``.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass

#: Encodings tried, in order, when there is no BOM. UTF-8 first (dominant and
#: self-validating), then the legacy encodings a petroleum/engineering corpus is
#: most likely to contain (Windows Arabic and Western, then ISO-8859 and KOI8-R,
#: then BOM-less UTF-16).
CANDIDATE_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp1256",
    "cp1252",
    "iso-8859-6",
    "iso-8859-1",
    "koi8-r",
    "utf-16-le",
    "utf-16-be",
)

#: Legacy encodings tried only after UTF-8 has been ruled out (see below).
_LEGACY_ENCODINGS: tuple[str, ...] = (
    "cp1256",
    "cp1252",
    "iso-8859-6",
    "iso-8859-1",
    "koi8-r",
    "utf-16-le",
    "utf-16-be",
)

#: A BOM (or a UTF-16/32 signature) pins the encoding without guessing.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

#: Control characters (except tab/newline/carriage-return) signal a wrong guess.
#: C1 controls (U+0080-U+009F) are included: they are what a single-byte codec
#: produces for binary bytes, so their presence means the file is not text.
_BAD_CONTROL = frozenset(
    chr(code)
    for code in list(range(0x00, 0x09))
    + list(range(0x0B, 0x0D))
    + list(range(0x0E, 0x20))
    + list(range(0x7F, 0xA0))
)


class EncodingDetectionError(ValueError):
    """No candidate encoding could decode the bytes without corruption."""


@dataclass(frozen=True)
class DecodedText:
    """The decoded text of a byte string, plus how it was decoded."""

    text: str
    encoding: str
    used_bom: bool = False


def detect_encoding(raw: bytes, *, allow_truncated: bool = False) -> str:
    """The name of the encoding that decodes ``raw`` cleanly.

    ``allow_truncated`` tolerates a byte string that ends mid-character (a prefix
    read from a stream), so detection can run on a bounded sample rather than the
    whole file.

    UTF-8 is tried first and, when it fails, the failure is classified:

    * a *valid* UTF-8 prefix that ends in an incomplete character is a truncated
      UTF-8 stream, and is refused (decoding it as a single-byte codec would turn
      every non-ASCII byte into mojibake);
    * any other invalid UTF-8 means the file is not UTF-8, so the legacy
      encodings are tried.

    Raises :class:`EncodingDetectionError` when nothing fits, so a caller never
    silently accepts corrupted text.
    """
    if not raw:
        return "utf-8"  # an empty file decodes as anything; utf-8 is the safe default
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return encoding
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        if _is_truncated_utf8(raw, error):
            if allow_truncated:
                return "utf-8"
            raise EncodingDetectionError(
                "The file looks like UTF-8 but ends mid-character (it may be "
                "truncated). Re-save it as complete UTF-8 and try again."
            ) from error
    else:
        # Valid UTF-8, but ASCII-range UTF-16 (interleaved NULs) is *also* valid
        # UTF-8 -- and produces NUL-filled text. Only trust UTF-8 when the result
        # is plausible text; otherwise let the legacy pass try UTF-16.
        if _plausible_utf8(text):
            return "utf-8"
    for encoding in _LEGACY_ENCODINGS:
        if _decodes_cleanly(raw, encoding, allow_truncated=allow_truncated):
            return encoding
    raise EncodingDetectionError(
        "Could not decode the file as any of: "
        + ", ".join(CANDIDATE_ENCODINGS)
        + ". Re-save it as UTF-8 and try again."
    )


def decode_bytes(raw: bytes, *, encoding: str | None = None) -> DecodedText:
    """Decode ``raw`` strictly, detecting the encoding when ``encoding`` is None.

    ``encoding`` given by the caller is honoured but still strict: a wrong explicit
    encoding raises rather than producing replacement characters. A detected
    encoding that turns out not to fit (e.g. a truncated UTF-16 stream) raises the
    same way, so no path returns corrupted text.
    """
    chosen = encoding or detect_encoding(raw)
    try:
        text = raw.decode(chosen)
    except (UnicodeDecodeError, LookupError) as error:
        raise EncodingDetectionError(
            f"Could not decode the file as {chosen!r}: {error}"
        ) from error
    used_bom = chosen in {name for _bom, name in _BOMS}
    return DecodedText(text=text, encoding=chosen, used_bom=used_bom)


def _plausible_utf8(text: str) -> bool:
    """Whether UTF-8-decoded ``text`` is real text, not NUL-filled UTF-16 bytes."""
    return not any(char in _BAD_CONTROL for char in text)


def _is_truncated_utf8(raw: bytes, error: UnicodeDecodeError) -> bool:
    """Whether a UTF-8 failure is a valid prefix ending mid-character.

    The error must sit in the final three bytes (the most a UTF-8 character can
    span) and its "start" byte must introduce a multi-byte sequence whose bytes
    are all present-but-incomplete, so a genuinely invalid byte elsewhere is not
    mistaken for truncation.
    """
    if error.start < len(raw) - 3:
        return False
    return error.reason.startswith("unexpected end of data")


def _decodes_cleanly(raw: bytes, encoding: str, *, allow_truncated: bool = False) -> bool:
    """Whether ``encoding`` decodes ``raw`` strictly into plausible text.

    Plausibility rejects the tell-tale signs of a wrong guess: NUL bytes, stray
    control characters, and replacement characters. This keeps UTF-16 from
    "succeeding" on arbitrary ASCII-ish bytes, and keeps a legacy single-byte
    encoding from swallowing a UTF-8 file. With ``allow_truncated`` a trailing
    partial character is trimmed first, so a bounded stream prefix is judged on
    its complete characters only.
    """
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        if not allow_truncated:
            return False
        prefix = _decode_truncated_prefix(raw, encoding)
        if prefix is None:
            return False
        text = prefix
    if "\ufffd" in text:
        return False
    return not any(char in _BAD_CONTROL for char in text)


def _decode_truncated_prefix(raw: bytes, encoding: str) -> str | None:
    """Decode the longest clean prefix of ``raw`` under ``encoding``, or ``None``."""
    try:
        decoder = codecs.getincrementaldecoder(encoding)()
    except LookupError:
        return None
    try:
        # final=False keeps an incomplete trailing character out of the result
        # instead of raising, which is exactly the prefix case we want.
        return decoder.decode(raw, final=False)
    except UnicodeDecodeError:
        return None


__all__ = [
    "CANDIDATE_ENCODINGS",
    "DecodedText",
    "EncodingDetectionError",
    "decode_bytes",
    "detect_encoding",
]
