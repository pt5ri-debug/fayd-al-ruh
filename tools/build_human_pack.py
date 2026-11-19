#!/usr/bin/env python3
"""Build the shipped human-body pack from MakeHuman's CC0 base mesh + targets.

The Body & shape studio used to sculpt a body from implicit primitives, which
never read as a person. The honest fix is an *artist-authored* base mesh, and
MakeHuman ships exactly that: a 13k-vertex quad body with eyes, teeth and a full
joint marker rig, explicitly released as **CC0** (public domain) in September
2020, together with ~200 CC0 morph targets that are the real deformation data
behind its sliders.

This script turns those files into one small, dependency-free binary the project
ships (``core/human_pack/human_pack.bin.z``): the body vertices/faces (quads kept,
so nothing is lost), the helper groups that give the body its eyes and teeth, the
joint markers as a named point cloud the rig builder reads, and the curated morph
targets as sparse float32 deltas.

Everything is read from a *local* copy of the MakeHuman repository: the pack is
built once, offline, and committed. Nothing here runs at program start-up and the
program itself never downloads anything.

Usage::

    python tools/build_human_pack.py --makehuman /path/to/makehuman/repo
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from collections import defaultdict
from pathlib import Path

MAGIC = b"FRHUMAN\x01"
VERSION = 1

#: The groups the body keeps. ``body`` is the skin itself; the rest give the
#: figure eyes and teeth, so a face reads as a face.
KEEP_GROUPS = (
    "body",
    "helper-l-eye",
    "helper-r-eye",
    "helper-upper-teeth",
    "helper-lower-teeth",
)

#: The morph targets the Body & shape dials drive. Each entry is the MakeHuman
#: target stem (without ``-incr``/``-decr``) and the pack key it is stored under.
TARGETS = (
    ("measure/measure-bust-circ", "bust"),
    ("measure/measure-hips-circ", "hips"),
    ("measure/measure-waist-circ", "waist"),
    ("measure/measure-shoulder-dist", "shoulder"),
    ("measure/measure-thigh-circ", "thigh_girth"),
    ("measure/measure-upperarm-circ", "arm_girth"),
    ("measure/measure-calf-circ", "calf_girth"),
    ("measure/measure-neck-circ", "neck_girth"),
    ("measure/measure-neck-height", "neck_length"),
    ("measure/measure-upperleg-height", "thigh_length"),
    ("measure/measure-lowerleg-height", "shin_length"),
    ("measure/measure-upperarm-length", "upper_arm_length"),
    ("measure/measure-lowerarm-length", "forearm_length"),
    ("measure/measure-wrist-circ", "wrist_girth"),
    ("measure/measure-ankle-circ", "ankle_girth"),
    ("torso/torso-scale-horiz", "torso_width"),
    ("torso/torso-scale-vert", "torso_length"),
    ("torso/torso-scale-depth", "chest_depth"),
    ("stomach/stomach-tone", "stomach_tone"),
    ("stomach/stomach-pregnant", "belly"),
    ("buttocks/buttocks-volume", "buttocks"),
    ("pelvis/pelvis-tone", "pelvis_tone"),
    ("hip/hip-scale-horiz", "hip_width"),
    ("hip/hip-scale-depth", "hip_depth"),
    ("head/head-scale-horiz", "head_width"),
    ("head/head-scale-vert", "head_height"),
    ("head/head-scale-depth", "head_depth"),
    ("forehead/forehead-scale-vert", "forehead"),
    ("chin/chin-width", "jaw_width"),
    ("chin/chin-height", "face_length"),
    ("chin/chin-prominent", "chin"),
    ("nose/nose-scale-horiz", "nose_width"),
    ("nose/nose-scale-vert", "nose_length"),
    ("nose/nose-scale-depth", "nose_depth"),
    ("eyes/l-eye-scale", "eye_size_l"),
    ("eyes/r-eye-scale", "eye_size_r"),
    ("mouth/mouth-scale-horiz", "mouth_width"),
    ("mouth/mouth-lowerlip-volume", "lower_lip"),
    ("mouth/mouth-upperlip-volume", "upper_lip"),
    ("cheek/l-cheek-volume", "cheek_l"),
    ("cheek/r-cheek-volume", "cheek_r"),
    ("ears/l-ear-scale", "ear_size_l"),
    ("ears/r-ear-scale", "ear_size_r"),
    ("armslegs/l-hand-scale", "hand_l"),
    ("armslegs/r-hand-scale", "hand_r"),
    ("armslegs/l-foot-scale", "foot_l"),
    ("armslegs/r-foot-scale", "foot_r"),
    ("armslegs/l-upperleg-scale-horiz", "thigh_l"),
    ("armslegs/r-upperleg-scale-horiz", "thigh_r"),
    ("armslegs/l-upperleg-scale-vert", "leg_length_l"),
    ("armslegs/r-upperleg-scale-vert", "leg_length_r"),
    ("armslegs/l-upperarm-scale-horiz", "upper_arm_l"),
    ("armslegs/r-upperarm-scale-horiz", "upper_arm_r"),
    ("armslegs/l-lowerarm-scale-horiz", "forearm_l"),
    ("armslegs/r-lowerarm-scale-horiz", "forearm_r"),
    ("armslegs/l-lowerleg-scale-horiz", "calf_l"),
    ("armslegs/r-lowerleg-scale-horiz", "calf_r"),
)

#: Every joint marker the rig builder reads, by MakeHuman's own group name.
JOINTS = (
    "joint-ground",
    "joint-pelvis",
    "joint-spine-1",
    "joint-spine-2",
    "joint-spine-3",
    "joint-spine-4",
    "joint-neck",
    "joint-head",
    "joint-head-2",
    "joint-jaw",
    "joint-l-clavicle",
    "joint-r-clavicle",
    "joint-l-shoulder",
    "joint-r-shoulder",
    "joint-l-elbow",
    "joint-r-elbow",
    "joint-l-hand",
    "joint-r-hand",
    "joint-l-hand-2",
    "joint-r-hand-2",
    "joint-l-hand-3",
    "joint-r-hand-3",
    "joint-l-upper-leg",
    "joint-r-upper-leg",
    "joint-l-knee",
    "joint-r-knee",
    "joint-l-ankle",
    "joint-r-ankle",
    "joint-l-foot-1",
    "joint-r-foot-1",
    "joint-l-foot-2",
    "joint-r-foot-2",
    "joint-l-eye",
    "joint-r-eye",
)


def read_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]], dict[str, set[int]]]:
    """Read the base OBJ into vertices, faces, and the vertex set of each group."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    groups: dict[str, set[int]] = defaultdict(set)
    group = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("g "):
            group = line.split(None, 1)[1].strip()
        elif line.startswith("f ") and group:
            indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            if len(indices) >= 3:
                faces.append(indices)
                groups[group].update(indices)
    return vertices, faces, dict(groups)


def read_target(path: Path) -> dict[int, tuple[float, float, float]]:
    """Read one MakeHuman ``.target`` file (``index dx dy dz`` per line)."""
    deltas: dict[int, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        deltas[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return deltas


def build(makehuman: Path) -> bytes:
    """Read the MakeHuman checkout and return the serialised pack."""
    base = makehuman / "makehuman" / "data" / "3dobjs" / "base.obj"
    if not base.exists():
        raise SystemExit(f"Not a MakeHuman checkout: {base} is missing")
    vertices, faces, groups = read_obj(base)

    keep = sorted({index for name in KEEP_GROUPS for index in groups.get(name, ())})
    remap = {old: new for new, old in enumerate(keep)}
    kept_vertices = [vertices[index] for index in keep]
    kept_faces = [
        [remap[index] for index in face]
        for face in faces
        if all(index in remap for index in face)
    ]

    kept_groups = {
        name: sorted(remap[index] for index in groups.get(name, ()) if index in remap)
        for name in KEEP_GROUPS
    }

    joints: dict[str, list[float]] = {}
    for name in JOINTS:
        members = groups.get(name, set())
        if not members:
            continue
        points = [vertices[index] for index in members]
        joints[name] = [
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
            sum(point[2] for point in points) / len(points),
        ]

    targets: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    targets_dir = makehuman / "makehuman" / "data" / "targets"
    for stem, key in TARGETS:
        pair: dict[str, dict] = {}
        for sign in ("incr", "decr"):
            path = targets_dir / f"{stem}-{sign}.target"
            if not path.exists():
                missing.append(path.name)
                continue
            deltas = read_target(path)
            pairs = sorted(
                (remap[index], delta) for index, delta in deltas.items() if index in remap
            )
            pair[sign] = {
                "indices": [item[0] for item in pairs],
                "vectors": [item[1] for item in pairs],
            }
        if pair:
            targets[key] = pair
    if missing:
        raise SystemExit("Missing targets: " + ", ".join(sorted(missing)))

    meta = {
        "version": VERSION,
        "source": "MakeHuman base.obj + morph targets (CC0, public domain, 2020)",
        "licence": "CC0-1.0",
        "groups": kept_groups,
        "joints": joints,
        "target_keys": sorted(targets),
    }
    chunks = [
        MAGIC,
        struct.pack("<I", VERSION),
        _pack_vertices(kept_vertices),
        _pack_faces(kept_faces),
        _pack_targets(targets),
        _pack_json(meta),
    ]
    return b"".join(chunks)


def _pack_vertices(points: list[tuple[float, float, float]]) -> bytes:
    return struct.pack("<I", len(points)) + b"".join(struct.pack("<3f", *point) for point in points)


def _pack_faces(faces: list[list[int]]) -> bytes:
    out: list[bytes] = [struct.pack("<I", len(faces))]
    for face in faces:
        out.append(struct.pack("<B", len(face)))
        out.append(b"".join(struct.pack("<H", index) for index in face))
    return b"".join(out)


def _pack_targets(targets: dict[str, dict[str, dict]]) -> bytes:
    out: list[bytes] = [struct.pack("<I", len(targets))]
    for key in sorted(targets):
        pair = targets[key]
        encoded_key = key.encode("ascii")
        out.append(struct.pack("<B", len(encoded_key)))
        out.append(encoded_key)
        out.append(struct.pack("<B", len(pair)))
        for sign in sorted(pair):
            entry = pair[sign]
            name = sign.encode("ascii")
            out.append(struct.pack("<B", len(name)))
            out.append(name)
            out.append(struct.pack("<I", len(entry["indices"])))
            out.append(b"".join(struct.pack("<H", index) for index in entry["indices"]))
            out.append(b"".join(struct.pack("<3f", *vector) for vector in entry["vectors"]))
    return b"".join(out)


def _pack_json(meta: dict) -> bytes:
    data = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<I", len(data)) + data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--makehuman", type=Path, required=True, help="path to a MakeHuman checkout")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "core" / "human_pack" / "human_pack.bin.z",
    )
    args = parser.parse_args()
    blob = zlib.compress(build(args.makehuman), 9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    print(f"wrote {args.output} ({len(blob) / 1e6:.2f} MB compressed)")


if __name__ == "__main__":
    main()
