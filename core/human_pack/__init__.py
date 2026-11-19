"""The shipped human-body pack: a real, artist-authored base mesh and its morphs.

The Body & shape studio's figures used to be *sculpted from implicit primitives*,
and no amount of smoothing made a pile of ellipsoids read as a person. This
module loads the honest alternative the project ships: MakeHuman's **CC0** base
mesh -- a 13k-vertex quad body with modelled hands, feet, face, eyes and teeth,
and a full joint-marker rig -- together with a curated set of MakeHuman's CC0
morph targets, which are the real, artist-authored deformation data behind its
sliders.

Why this is the right base
--------------------------
* **It is a real human mesh.** Every vertex was placed by a modeller, so the
  silhouette, the hands, the feet, the ears and the face are correct by
  construction rather than approximated by a field.
* **It is CC0.** MakeHuman's base mesh and targets were explicitly released into
  the public domain in September 2020, so shipping them is unencumbered.
* **It ships inside the program.** The pack is one committed, zlib-compressed
  binary (``human_pack.bin.z``, ~0.7 MB); nothing is downloaded, nothing is
  installed, and the studio works fully offline.
* **It is dependency-free to read.** This module parses the binary with
  ``struct`` + ``zlib`` only, and caches the decoded result in memory, so a
  second body costs nothing.

The pack holds
--------------
``vertices``
    The body's own points, in the base mesh's space (centimetres, ``+Y`` up,
    ``+Z`` forward, feet near ``y = -8.18``).
``faces``
    Quads and triangles, kept as authored (nothing is triangulated here).
``groups``
    Which vertices belong to the skin, the two eyes, and the two rows of teeth.
``joints``
    Named joint markers -- pelvis, four spine joints, neck, head, clavicles,
    shoulders, elbows, hands, knees, ankles, feet, toes, eyes -- which the rig
    builder uses to place real bones on *this* body.
``targets``
    Sparse ``incr``/``decr`` deltas per dial (``bust``, ``waist``, ``hips``,
    ``head_width``, ``nose_length``, ...), in the pack's vertex space.

The reader is read-only and side-effect free: it never writes, never downloads,
and never imports a third-party package.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MAGIC = b"FRHUMAN\x01"
SUPPORTED_VERSION = 1

#: The pack's filename inside this package.
PACK_FILENAME = "human_pack.bin.z"

#: The scale from the base mesh's own units to metres. MakeHuman models a body
#: roughly 16.4 units tall, which is 1.75 m of real height.
UNITS_PER_METRE = 16.4 / 1.75


class HumanPackError(RuntimeError):
    """The pack is missing, corrupt, or of an unsupported version."""


@dataclass(frozen=True)
class MorphTarget:
    """One dial's ``incr``/``decr`` deltas, as parallel index/vector lists.

    A dial value of ``0.5`` means "leave the base body alone"; above ``0.5``
    applies the ``incr`` target and below it the ``decr`` target, scaled by the
    distance from ``0.5`` -- which is exactly how MakeHuman's own sliders work.
    """

    key: str
    incr_indices: tuple[int, ...]
    incr_vectors: tuple[tuple[float, float, float], ...]
    decr_indices: tuple[int, ...]
    decr_vectors: tuple[tuple[float, float, float], ...]

    @property
    def vertex_count(self) -> int:
        return len(set(self.incr_indices) | set(self.decr_indices))


@dataclass(frozen=True)
class HumanPack:
    """The decoded pack: the base body plus its groups, joints, and morphs."""

    version: int
    source: str
    licence: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    groups: dict[str, tuple[int, ...]]
    joints: dict[str, tuple[float, float, float]]
    targets: dict[str, MorphTarget]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def triangle_count(self) -> int:
        return sum(max(0, len(face) - 2) for face in self.faces)

    def target_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.targets))

    def group(self, name: str) -> tuple[int, ...]:
        """The vertex indices of a group (``body``, ``helper-l-eye``, ...)."""
        return self.groups.get(name, ())

    def joint(self, name: str) -> tuple[float, float, float] | None:
        """One joint marker's position, or ``None`` when the pack has none."""
        return self.joints.get(name)

    def describe(self) -> str:
        return (
            f"Human pack v{self.version}: {self.vertex_count} vertices, "
            f"{self.face_count} faces ({self.triangle_count} triangles), "
            f"{len(self.joints)} joints, {len(self.targets)} morph targets. "
            f"{self.source} ({self.licence})."
        )


def pack_path() -> Path:
    """The shipped pack's path (resolved beside this module, in a wheel too)."""
    return Path(__file__).resolve().parent / PACK_FILENAME


def pack_available() -> bool:
    """Whether the pack is present on disk (a wheel without it degrades honestly)."""
    return pack_path().exists()


@lru_cache(maxsize=1)
def load_pack() -> HumanPack:
    """Read, verify, and decode the pack once; later calls return the same object."""
    path = pack_path()
    if not path.exists():
        raise HumanPackError(
            f"The human-body pack is missing ({path}). "
            "Reinstall the program, or rebuild it with tools/build_human_pack.py."
        )
    try:
        raw = zlib.decompress(path.read_bytes())
    except (OSError, zlib.error) as error:
        raise HumanPackError(f"The human-body pack could not be read: {error}") from error
    return _decode(raw)


def _decode(raw: bytes) -> HumanPack:
    """Parse the binary layout: magic, version, vertices, faces, targets, meta."""
    offset = 0

    def take(count: int) -> bytes:
        nonlocal offset
        chunk = raw[offset : offset + count]
        if len(chunk) != count:
            raise HumanPackError("The human-body pack is truncated.")
        offset += count
        return chunk

    if take(len(MAGIC)) != MAGIC:
        raise HumanPackError("This is not a Fayd al-Ruh human-body pack.")
    version = struct.unpack("<I", take(4))[0]
    if version != SUPPORTED_VERSION:
        raise HumanPackError(
            f"Human-body pack version {version} is not supported by this build "
            f"(expected {SUPPORTED_VERSION})."
        )

    vertex_count = struct.unpack("<I", take(4))[0]
    vertices = tuple(struct.unpack("<3f", take(12)) for _ in range(vertex_count))

    face_count = struct.unpack("<I", take(4))[0]
    faces: list[tuple[int, ...]] = []
    for _ in range(face_count):
        size = struct.unpack("<B", take(1))[0]
        faces.append(struct.unpack(f"<{size}H", take(size * 2)))

    target_count = struct.unpack("<I", take(4))[0]
    targets: dict[str, MorphTarget] = {}
    for _ in range(target_count):
        key_length = struct.unpack("<B", take(1))[0]
        key = take(key_length).decode("ascii")
        pair_count = struct.unpack("<B", take(1))[0]
        pair: dict[str, tuple[tuple[int, ...], tuple[tuple[float, float, float], ...]]] = {}
        for _ in range(pair_count):
            name_length = struct.unpack("<B", take(1))[0]
            name = take(name_length).decode("ascii")
            entries = struct.unpack("<I", take(4))[0]
            indices = struct.unpack(f"<{entries}H", take(entries * 2))
            vectors = tuple(struct.unpack("<3f", take(12)) for _ in range(entries))
            pair[name] = (indices, vectors)
        empty: tuple[tuple[int, ...], tuple[tuple[float, float, float], ...]] = ((), ())
        targets[key] = MorphTarget(
            key=key,
            incr_indices=pair.get("incr", empty)[0],
            incr_vectors=pair.get("incr", empty)[1],
            decr_indices=pair.get("decr", empty)[0],
            decr_vectors=pair.get("decr", empty)[1],
        )

    meta_length = struct.unpack("<I", take(4))[0]
    meta = json.loads(take(meta_length).decode("utf-8"))
    # The target keys are stored in the meta block in the same sorted order the
    # target block was written in, so the two can be zipped without storing the
    # name twice.
    return HumanPack(
        version=version,
        source=str(meta.get("source", "MakeHuman (CC0)")),
        licence=str(meta.get("licence", "CC0-1.0")),
        vertices=vertices,
        faces=tuple(faces),
        groups={name: tuple(indices) for name, indices in meta.get("groups", {}).items()},
        joints={name: tuple(value) for name, value in meta.get("joints", {}).items()},
        targets=targets,
    )
