"""Read a model file's *facts* without loading it.

Every mouth in this project already knows how to find its own weights
(:func:`core.ultron.find_model_file` and its nine siblings), but nothing could
answer the question a model manager has to answer first: **what is this file,
and is it intact?** This module is that answer, and it deliberately reads as
little as possible:

* the ``GGUF`` magic, the header version, and the declared tensor/metadata
  counts (so a truncated download or a renamed file is *reported*, never handed
  to an engine that would then hang or crash);
* the metadata key/value block, walked only far enough to collect the handful of
  facts a chooser needs (name, architecture, quantisation, trained context
  length) -- the tensor data is never touched and the weights are never loaded;
* for the project's own trained artifact (``.npz``), the ``config`` array inside
  the zip, parsed with :mod:`struct` and :mod:`ast` so **numpy is not needed**
  to inspect a checkpoint (the base install keeps numpy, but an inspector must
  not depend on the very stack it is diagnosing).

Honest by construction, like the rest of the project:

* an unknown value stays **unknown** -- ``None`` for a number, ``""`` for a
  string -- rather than being guessed from a filename;
* a file that cannot be read is reported with the reason, not raised at a
  caller that only wanted to list a folder;
* nothing here writes, moves, copies, or deletes anything, and nothing here
  executes a byte of a model file.
"""

from __future__ import annotations

import ast
import hashlib
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

#: The magic every real GGUF model starts with.
GGUF_MAGIC = b"GGUF"
#: The magic every ``.npz`` (a zip) starts with -- the trained checkpoint's format.
NPZ_MAGIC = b"PK\x03\x04"

#: The model formats this build can recognise. Anything else is
#: :data:`FORMAT_UNKNOWN`, which is an honest answer, not a failure.
FORMAT_GGUF = "gguf"
FORMAT_FAYD_CHECKPOINT = "fayd-checkpoint"
FORMAT_UNKNOWN = "unknown"

#: Integrity verdicts. ``ok`` is the only one that licenses loading.
INTEGRITY_OK = "ok"
INTEGRITY_MISSING = "missing"
INTEGRITY_UNREADABLE = "unreadable"
INTEGRITY_MAGIC_MISMATCH = "magic-mismatch"
INTEGRITY_TRUNCATED = "truncated"
INTEGRITY_UNKNOWN = "unknown"

#: The default digest algorithm. ``blake2b`` is in the standard library, is fast,
#: and is what :mod:`ingest.dedup` already uses for content addressing.
DIGEST_ALGORITHM = "blake2b"

#: Sanity bounds on a GGUF header. A real model has thousands of tensors at most;
#: a header claiming a billion is a corrupt or hostile file, and the counts are
#: refused *before* anything is walked.
MAX_TENSORS = 1_000_000
MAX_METADATA_ENTRIES = 100_000
#: How much of the header to read while walking metadata. The block is normally a
#: few kilobytes; this bound stops a crafted file from making the inspector read
#: gigabytes.
MAX_HEADER_BYTES = 4 * 1024 * 1024
#: A string value longer than this is not a fact we want -- it is a blob.
MAX_STRING_BYTES = 4096

#: The GGUF value types, by their on-disk tag.
_GGUF_TYPES: dict[int, tuple[str, int]] = {
    0: ("<B", 1),  # uint8
    1: ("<b", 1),  # int8
    2: ("<H", 2),  # uint16
    3: ("<h", 2),  # int16
    4: ("<I", 4),  # uint32
    5: ("<i", 4),  # int32
    6: ("<f", 4),  # float32
    7: ("<B", 1),  # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9

#: ``general.file_type`` -> a readable quantisation. Only the well-known values
#: are mapped; anything else stays unknown rather than being labelled wrongly.
_FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
}

#: The metadata keys worth collecting, and the fact each one carries.
_WANTED_KEYS = (
    "general.name",
    "general.architecture",
    "general.file_type",
    "general.license",
    "general.license.name",
    "general.size_label",
    "general.parameter_count",
)


class ModelFileError(ValueError):
    """A path cannot be used as a model file, with the reason stated."""


def _is_probably_checkpoint_zip(path: Path) -> bool:
    """True when ``path`` looks like a numpy ``.npz`` package (a zip with config)."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return "config.npy" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def digest_of(path: Path, *, algorithm: str = DIGEST_ALGORITHM) -> str:
    """A content digest of ``path``, streamed so a multi-gigabyte file is fine.

    Reading the whole file is the point (it is what proves the bytes are the
    bytes), so this is deliberately opt-in: a caller that only wants to *list*
    models calls :func:`probe_file` without it.
    """
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class GgufInfo:
    """The facts read out of a GGUF header. Anything not found stays unknown."""

    version: int
    tensor_count: int
    metadata_count: int
    name: str = ""
    architecture: str = ""
    quantization: str = ""
    context_length: int | None = None
    parameter_count: int | None = None
    license: str = ""
    #: True when the header walk stopped early (the bound was hit, or the block
    #: was malformed). The facts collected so far are still reported.
    partial: bool = False

    def to_primitive(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata_count": self.metadata_count,
            "name": self.name,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "parameter_count": self.parameter_count,
            "license": self.license,
            "partial": self.partial,
        }


@dataclass(frozen=True)
class CheckpointInfo:
    """The facts read out of the project's own trained ``.npz`` checkpoint.

    Read with the standard library only, so inspecting a checkpoint does not need
    numpy (and therefore works on a machine where the runtime that *trains* it is
    not installed).
    """

    vocab_size: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None
    block_size: int | None = None
    dropout: float | None = None
    tie_weights: bool | None = None
    parameter_arrays: int = 0

    @property
    def parameter_count(self) -> int | None:
        """The transformer's parameter count, or ``None`` when the shape is unknown.

        An *estimate from the recorded shape* -- embedding, attention, and MLP
        weights at the dimensions the checkpoint itself declares. It is reported
        as a derived figure, never as a measured one.
        """
        v, layers, embd = self.vocab_size, self.n_layer, self.n_embd
        if v is None or layers is None or embd is None:
            return None
        # token + position embeddings, then per layer: 4 attention projections
        # plus a 2x MLP, then the (optional) output head.
        total = v * embd + (self.block_size or 0) * embd
        total += layers * (4 * embd * embd + 8 * embd * embd)
        if self.tie_weights is False:
            total += embd * v
        return int(total)

    def to_primitive(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights,
            "parameter_arrays": self.parameter_arrays,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class FileProbe:
    """Everything that can be learned about a model file without loading it."""

    path: Path
    exists: bool
    size_bytes: int = 0
    format: str = FORMAT_UNKNOWN
    integrity: str = INTEGRITY_MISSING
    detail: str = ""
    gguf: GgufInfo | None = None
    checkpoint: CheckpointInfo | None = None
    digest: str = ""
    digest_algorithm: str = ""

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def usable(self) -> bool:
        """True when the file is present, recognised, and intact."""
        return self.exists and self.format != FORMAT_UNKNOWN and self.integrity == INTEGRITY_OK

    def to_primitive(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "integrity": self.integrity,
            "detail": self.detail,
            "gguf": self.gguf.to_primitive() if self.gguf is not None else None,
            "checkpoint": (
                self.checkpoint.to_primitive() if self.checkpoint is not None else None
            ),
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
        }


def _read_exact(handle: object, count: int) -> bytes | None:
    """Read exactly ``count`` bytes, or ``None`` when the file ends first."""
    data = handle.read(count)  # type: ignore[attr-defined]
    return data if len(data) == count else None


def _read_gguf_string(handle: object) -> str | None:
    """Read a GGUF length-prefixed string, refusing an implausible length."""
    raw = _read_exact(handle, 8)
    if raw is None:
        return None
    (length,) = struct.unpack("<Q", raw)
    if length > MAX_STRING_BYTES:
        return None
    payload = _read_exact(handle, length)
    if payload is None:
        return None
    return payload.decode("utf-8", errors="replace")


def _skip_gguf_value(handle: object, value_type: int) -> bool:
    """Advance past one metadata value. ``False`` when the block is malformed."""
    if value_type == _GGUF_STRING:
        return _read_gguf_string(handle) is not None
    if value_type == _GGUF_ARRAY:
        raw = _read_exact(handle, 12)
        if raw is None:
            return False
        element_type, count = struct.unpack("<IQ", raw)
        if count > MAX_METADATA_ENTRIES:
            return False
        if element_type == _GGUF_STRING:
            for _ in range(count):
                if _read_gguf_string(handle) is None:
                    return False
            return True
        entry = _GGUF_TYPES.get(element_type)
        if entry is None:
            return False
        _, width = entry
        return _read_exact(handle, width * count) is not None
    entry = _GGUF_TYPES.get(value_type)
    if entry is None:
        return False
    _, width = entry
    return _read_exact(handle, width) is not None


def _read_gguf_value(handle: object, value_type: int) -> object:
    """Read one scalar metadata value, or ``None`` when it is not a scalar we keep."""
    if value_type == _GGUF_STRING:
        return _read_gguf_string(handle)
    entry = _GGUF_TYPES.get(value_type)
    if entry is None:
        return None
    fmt, width = entry
    raw = _read_exact(handle, width)
    if raw is None:
        return None
    return struct.unpack(fmt, raw)[0]


def _walk_gguf_metadata(handle: object, entries: int) -> tuple[dict[str, object], bool]:
    """Collect the wanted keys from a GGUF metadata block.

    Stops as soon as every wanted key is found, so an ordinary model is read in a
    few hundred bytes. A key that is not wanted is skipped by type, never parsed.

    Returns ``(found, truncated)``: ``truncated`` is ``True`` only when the block
    ended before it was finished, so a caller can tell "the file does not record a
    context length" from "the header could not be read".
    """
    found: dict[str, object] = {}
    remaining = set(_WANTED_KEYS)
    for _ in range(entries):
        if not remaining:
            # Every wanted key is in hand; the rest of the block is not our
            # business, and stopping here is a complete read, not a partial one.
            return found, False
        raw = _read_exact(handle, 8)
        if raw is None:
            return found, True
        (key_length,) = struct.unpack("<Q", raw)
        if key_length > MAX_STRING_BYTES:
            return found, True
        key_bytes = _read_exact(handle, key_length)
        if key_bytes is None:
            return found, True
        key = key_bytes.decode("utf-8", errors="replace")
        type_raw = _read_exact(handle, 4)
        if type_raw is None:
            return found, True
        (value_type,) = struct.unpack("<I", type_raw)
        if key in remaining:
            value = _read_gguf_value(handle, value_type)
            # A wanted key that is not a scalar is skipped, and the walk keeps
            # going: one odd entry must not cost us the rest of the block.
            if value is not None:
                found[key] = value
                remaining.discard(key)
            elif not _skip_gguf_value(handle, value_type):
                return found, True
        elif not _skip_gguf_value(handle, value_type):
            return found, True
    # The declared entries are exhausted; anything still missing is genuinely
    # absent from this file rather than unread.
    return found, False


def read_gguf_info(path: Path | str, *, max_bytes: int = MAX_HEADER_BYTES) -> GgufInfo | None:
    """Read a GGUF header, or ``None`` when the file is not a readable GGUF.

    The tensor data is never read: the metadata block is walked and the walk stops
    once the wanted keys are collected. ``GgufInfo.partial`` is ``True`` when the
    walk stopped early, so a caller can tell "no context length recorded" from
    "we could not look".
    """
    candidate = Path(path)
    try:
        with candidate.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                return None
            raw = _read_exact(handle, 4 + 8 + 8)
            if raw is None:
                return None
            version, tensor_count, metadata_count = struct.unpack("<IQQ", raw)
            if tensor_count > MAX_TENSORS or metadata_count > MAX_METADATA_ENTRIES:
                return None
            consumed = handle.tell()
            if consumed >= max_bytes:
                return GgufInfo(
                    version=version, tensor_count=tensor_count,
                    metadata_count=metadata_count, partial=True,
                )
            found, truncated = _walk_gguf_metadata(handle, int(metadata_count))
    except OSError:
        return None

    name = str(found.get("general.name") or "")
    architecture = str(found.get("general.architecture") or "")
    license_name = str(found.get("general.license") or found.get("general.license.name") or "")

    quantization = ""
    file_type = found.get("general.file_type")
    if isinstance(file_type, int):
        quantization = _FILE_TYPES.get(file_type, "")

    context_length: int | None = None
    if architecture:
        context_length = _read_architecture_context(candidate, architecture)

    parameter_count: int | None = None
    declared = found.get("general.parameter_count")
    if isinstance(declared, (int, float)) and declared > 0:
        parameter_count = int(declared)

    return GgufInfo(
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        name=name,
        architecture=architecture,
        quantization=quantization,
        context_length=context_length,
        parameter_count=parameter_count,
        license=license_name,
        partial=truncated,
    )


def _read_architecture_context(path: Path, architecture: str) -> int | None:
    """The ``<architecture>.context_length`` the GGUF declares, when it does.

    The trained context is recorded under an architecture-specific key (for
    example ``qwen2.context_length``), so it cannot be collected by the fixed key
    list. Only that one key is looked for; everything else is skipped by type.
    """
    wanted = f"{architecture}.context_length"
    try:
        with path.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                return None
            # The magic is already consumed, so the version and the two counts
            # are the next 20 bytes.
            raw = _read_exact(handle, 20)
            if raw is None:
                return None
            _, _, metadata_count = struct.unpack("<IQQ", raw)
            if metadata_count > MAX_METADATA_ENTRIES:
                return None
            for _ in range(int(metadata_count)):
                length_raw = _read_exact(handle, 8)
                if length_raw is None:
                    return None
                (key_length,) = struct.unpack("<Q", length_raw)
                if key_length > MAX_STRING_BYTES:
                    return None
                key_bytes = _read_exact(handle, key_length)
                if key_bytes is None:
                    return None
                key = key_bytes.decode("utf-8", errors="replace")
                type_raw = _read_exact(handle, 4)
                if type_raw is None:
                    return None
                (value_type,) = struct.unpack("<I", type_raw)
                if key == wanted:
                    value = _read_gguf_value(handle, value_type)
                    return int(value) if isinstance(value, (int, float)) else None
                if not _skip_gguf_value(handle, value_type):
                    return None
    except OSError:
        return None
    return None


def read_checkpoint_info(path: Path | str) -> CheckpointInfo | None:
    """Read the ``config`` array of a trained ``.npz`` checkpoint, or ``None``.

    Deliberately dependency-free: the numpy header is a small literal dict, so it
    is parsed with :mod:`ast` and the seven recorded numbers with :mod:`struct`.
    A checkpoint this build cannot understand returns ``None`` rather than a
    half-filled guess.
    """
    candidate = Path(path)
    try:
        with zipfile.ZipFile(candidate) as archive:
            names = archive.namelist()
            payload = archive.read("config.npy")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    parsed = _parse_npy_float64(payload)
    if parsed is None:
        return None
    values, _shape = parsed
    if len(values) < 5:
        return None
    parameter_arrays = max(0, len(names) - 1)
    return CheckpointInfo(
        vocab_size=int(values[0]),
        n_layer=int(values[1]),
        n_head=int(values[2]),
        n_embd=int(values[3]),
        block_size=int(values[4]),
        dropout=float(values[5]) if len(values) > 5 else None,
        tie_weights=bool(int(values[6])) if len(values) > 6 else None,
        parameter_arrays=parameter_arrays,
    )


def _parse_npy_float64(payload: bytes) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    """Parse a float64 ``.npy`` array into its values, or ``None``.

    Only the two layouts the trainer actually writes are accepted: a little-endian
    ``<f8`` array in C order. Anything else -- a different dtype, Fortran order, a
    malformed header -- is refused instead of misread.
    """
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        return None
    major = payload[6]
    if major == 1:
        header_length = struct.unpack_from("<H", payload, 8)[0]
        offset = 10
    elif major in (2, 3):
        header_length = struct.unpack_from("<I", payload, 8)[0]
        offset = 12
    else:
        return None
    header = payload[offset : offset + header_length]
    try:
        declared = ast.literal_eval(header.decode("latin1").strip())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(declared, dict):
        return None
    if declared.get("descr") != "<f8" or declared.get("fortran_order") is not False:
        return None
    shape = declared.get("shape")
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
        return None
    count = 1
    for dim in shape:
        count *= dim
    start = offset + header_length
    if count <= 0 or len(payload) < start + count * 8:
        return None
    return struct.unpack_from(f"<{count}d", payload, start), shape


def detect_format(path: Path | str) -> str:
    """The format of a model file: ``gguf``, ``fayd-checkpoint``, or ``unknown``.

    Decided by the file's **contents**, never by its name: a ``.gguf`` file that
    is really a zip is a checkpoint, and a renamed file is neither.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return FORMAT_UNKNOWN
    try:
        with candidate.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return FORMAT_UNKNOWN
    if magic == GGUF_MAGIC:
        return FORMAT_GGUF
    if magic == NPZ_MAGIC and _is_probably_checkpoint_zip(candidate):
        return FORMAT_FAYD_CHECKPOINT
    return FORMAT_UNKNOWN


def probe_file(path: Path | str, *, digest: bool = False) -> FileProbe:
    """Everything knowable about a model file without loading it. Never raises.

    A missing file, an unreadable file, a file of the wrong format, and a
    truncated header each produce a :class:`FileProbe` carrying the reason -- so a
    listing can report honestly about a folder that has drifted, and a caller that
    only wanted to enumerate never has to guard every entry.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return FileProbe(
            path=candidate, exists=False, integrity=INTEGRITY_MISSING,
            detail=f"There is no file at {candidate}.",
        )
    if candidate.is_dir():
        return FileProbe(
            path=candidate, exists=True, integrity=INTEGRITY_UNREADABLE,
            detail=f"{candidate} is a folder, not a model file.",
        )
    try:
        size = candidate.stat().st_size
    except OSError as error:
        return FileProbe(
            path=candidate, exists=True, integrity=INTEGRITY_UNREADABLE,
            detail=f"Could not read {candidate}: {error}",
        )

    digest_value = ""
    if digest:
        try:
            digest_value = digest_of(candidate)
        except OSError as error:
            return FileProbe(
                path=candidate, exists=True, size_bytes=size,
                integrity=INTEGRITY_UNREADABLE,
                detail=f"Could not read {candidate}: {error}",
            )

    try:
        with candidate.open("rb") as handle:
            magic = handle.read(4)
    except OSError as error:
        return FileProbe(
            path=candidate, exists=True, size_bytes=size,
            integrity=INTEGRITY_UNREADABLE, digest=digest_value,
            digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
            detail=f"Could not read {candidate}: {error}",
        )

    if magic == GGUF_MAGIC:
        info = read_gguf_info(candidate)
        if info is None:
            return FileProbe(
                path=candidate, exists=True, size_bytes=size, format=FORMAT_GGUF,
                integrity=INTEGRITY_TRUNCATED, digest=digest_value,
                digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
                detail=(
                    "The file starts with the GGUF magic but its header is "
                    "truncated or implausible, so it cannot be loaded. Re-download it."
                ),
            )
        return FileProbe(
            path=candidate, exists=True, size_bytes=size, format=FORMAT_GGUF,
            integrity=INTEGRITY_OK, gguf=info, digest=digest_value,
            digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
            detail="",
        )

    if magic == NPZ_MAGIC and _is_probably_checkpoint_zip(candidate):
        checkpoint_info = read_checkpoint_info(candidate)
        if checkpoint_info is None:
            return FileProbe(
                path=candidate, exists=True, size_bytes=size,
                format=FORMAT_FAYD_CHECKPOINT, integrity=INTEGRITY_TRUNCATED,
                digest=digest_value,
                digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
                detail=(
                    "This is a numpy archive but its config array could not be "
                    "read, so it is not a usable trained checkpoint."
                ),
            )
        return FileProbe(
            path=candidate, exists=True, size_bytes=size,
            format=FORMAT_FAYD_CHECKPOINT, integrity=INTEGRITY_OK,
            checkpoint=checkpoint_info, digest=digest_value,
            digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
            detail="",
        )

    return FileProbe(
        path=candidate, exists=True, size_bytes=size, format=FORMAT_UNKNOWN,
        integrity=INTEGRITY_MAGIC_MISMATCH, digest=digest_value,
        digest_algorithm=DIGEST_ALGORITHM if digest_value else "",
        detail=(
            "This is not a GGUF model and not a Fayd Al-Ruh checkpoint. Supported "
            "formats: .gguf (llama.cpp) and the trained .npz checkpoint."
        ),
    )


def validate_model_path(path: Path | str, *, digest: bool = False) -> FileProbe:
    """Probe ``path`` and raise :class:`ModelFileError` unless it is usable.

    The one place a path is validated for a *write* (a registration), so a bad
    choice is refused with the reason rather than stored and discovered later.
    """
    probe = probe_file(path, digest=digest)
    if not probe.usable:
        raise ModelFileError(probe.detail or f"{probe.path} is not a usable model file.")
    return probe


#: Folders that never hold a model and would make a scan crawl. The scan also
#: skips anything hidden, so ``.git`` and ``.pytest-tmp`` cost nothing.
_SKIP_DIRS = frozenset(
    {"node_modules", "site-packages", "__pycache__", ".pytest-tmp", "dist", "build"}
)

#: A scan stops after this many hits, so pointing it at a whole home folder cannot
#: make a window hang. The caller is told the listing was cut short.
DEFAULT_SCAN_LIMIT = 200


@dataclass(frozen=True)
class DiscoveryReport:
    """What a scan found, and whether it was cut short."""

    probes: tuple[FileProbe, ...] = field(default=())
    roots: tuple[Path, ...] = field(default=())
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.probes)

    def by_format(self, model_format: str) -> tuple[FileProbe, ...]:
        return tuple(probe for probe in self.probes if probe.format == model_format)

    def summary(self) -> str:
        if not self.probes:
            return "No model files were found in the searched folders."
        kinds = ", ".join(
            f"{count} {name}"
            for name, count in (
                ("GGUF", len(self.by_format(FORMAT_GGUF))),
                ("checkpoint", len(self.by_format(FORMAT_FAYD_CHECKPOINT))),
                ("unrecognised", len(self.by_format(FORMAT_UNKNOWN))),
            )
            if count
        )
        line = f"Found {len(self.probes)} model file(s): {kinds}."
        return line + (" The search stopped early." if self.truncated else "")


def discover_models(
    roots: object = None,
    *,
    limit: int = DEFAULT_SCAN_LIMIT,
    digest: bool = False,
) -> DiscoveryReport:
    """Find supported model files under ``roots``.

    A root may be a single folder or a sequence of folders. The walk is bounded
    (``limit`` hits) and skips hidden directories, so pointing it at a home folder
    is safe. Nothing is downloaded, opened for writing, or executed: this is a
    filesystem read and a header parse, nothing more.
    """
    candidates = _as_roots(roots)
    probes: list[FileProbe] = []
    truncated = False
    for root in candidates:
        if not root.is_dir():
            continue
        for probe in _scan_root(root, digest=digest):
            probes.append(probe)
            if len(probes) >= limit:
                truncated = True
                break
        if truncated:
            break
    return DiscoveryReport(probes=tuple(probes), roots=candidates, truncated=truncated)


def _as_roots(roots: object) -> tuple[Path, ...]:
    """Normalise the ``roots`` argument into a tuple of paths."""
    if roots is None:
        return ()
    if isinstance(roots, (str, Path)):
        return (Path(roots).expanduser(),)
    if isinstance(roots, (list, tuple, set, frozenset)):
        return tuple(Path(item).expanduser() for item in roots)
    return ()


def _scan_root(root: Path, *, digest: bool) -> list[FileProbe]:
    """Every supported model file under one root, in a stable order.

    The hidden/vendor check applies to each path **relative to the root**, so a
    root that itself sits under a dotted folder (``.pytest-tmp``, a cache
    directory) is still scanned -- only what the walk descends into is skipped.
    """
    found: list[Path] = []
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                relative_parts = path.parts
            if any(
                part.startswith(".") or part in _SKIP_DIRS
                for part in relative_parts[:-1]
            ):
                continue
            if path.suffix.lower() not in (".gguf", ".npz"):
                continue
            found.append(path)
    except OSError:
        return []
    return [probe_file(path, digest=digest) for path in found]


def is_supported_suffix(path: Path | str) -> bool:
    """True when the *name* suggests a supported model file.

    Only a hint for a file picker: the real decision is
    :func:`detect_format`, which reads the bytes.
    """
    return Path(path).suffix.lower() in (".gguf", ".npz")


__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "DIGEST_ALGORITHM",
    "FORMAT_FAYD_CHECKPOINT",
    "FORMAT_GGUF",
    "FORMAT_UNKNOWN",
    "GGUF_MAGIC",
    "INTEGRITY_MAGIC_MISMATCH",
    "INTEGRITY_MISSING",
    "INTEGRITY_OK",
    "INTEGRITY_TRUNCATED",
    "INTEGRITY_UNKNOWN",
    "INTEGRITY_UNREADABLE",
    "MAX_HEADER_BYTES",
    "NPZ_MAGIC",
    "CheckpointInfo",
    "DiscoveryReport",
    "FileProbe",
    "GgufInfo",
    "ModelFileError",
    "detect_format",
    "digest_of",
    "discover_models",
    "is_supported_suffix",
    "probe_file",
    "read_checkpoint_info",
    "read_gguf_info",
    "validate_model_path",
]
