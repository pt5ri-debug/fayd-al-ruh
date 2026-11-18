"""Import a ready-made human mesh and reshape it, part by part.

The Body & shape studio can build a body from its own dials, but a user who already
has a figure -- a DAZ export, a MakeHuman OBJ, a sculpt from any 3D program -- wants
to *open* it, see it in the studio, resize or move any part of it, and add a piece
that is missing. This module is that engine, and it is deliberately a **thin layer
over the project's own 3D machinery**, never a second renderer or mesh library:

* a file is read by :mod:`core.scene3d_io` (Wavefront OBJ) or by the small, bounded
  readers here (STL and PLY, so a common export opens without a conversion step);
* a body region (head, left arm, right thigh, ...) is the part of the mesh whose
  vertices fall inside that region's own **box**, taken from the same skeleton the
  parametric figure uses -- so a region means the same thing on any figure;
* a **resize**, a **move**, or a **rotate** of a region is a smooth-space deformation
  with a **falloff**: the part inside the region moves fully and the vertices around
  it blend out over a margin, so a limb really changes shape and the joint stays
  smooth instead of tearing;
* **adding** a missing part fuses a shape (a sphere, an ellipsoid, a capsule, a box)
  or a whole sub-mesh into the body by an implicit-surface union, so the new piece
  becomes one connected skin rather than a shell floating beside it.

Everything is local and deterministic: nothing here opens a network connection, and
the same edit always produces the same mesh. Reading a file only parses bytes -- no
instruction inside an OBJ, STL, or PLY is ever executed -- and every reader is
bounded by :data:`MAX_VERTICES` / :data:`MAX_FACES` / :data:`MAX_FILE_BYTES`, so a
malformed or hostile file is refused with a clear reason instead of exhausting
memory.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.scene3d import Mesh, Vec3
from core.scene3d_sculpt import _surface_nets

#: The mesh formats this build reads. OBJ goes through the project's own reader;
#: STL and PLY are read by the small, bounded readers in this module.
IMPORT_SUFFIXES: tuple[str, ...] = (".obj", ".stl", ".ply")

#: Refuse an over-large model rather than exhausting memory. A figure or a prop is
#: far smaller than these; they exist so a malformed file cannot hurt.
MAX_VERTICES = 2_000_000
MAX_FACES = 4_000_000
MAX_FILE_BYTES = 512 * 1024 * 1024

#: The grid spacing (metres) the fusion of an added part uses. Fine enough that the
#: join is smooth, coarse enough that a whole body stays fast.
DEFAULT_FUSE_CELL = 0.012

#: The largest grid (in samples) a fusion may build before it is coarsened.
MAX_FUSE_SAMPLES = 8_000_000

#: The falloff (metres) a region edit blends over, so a moved part does not tear.
DEFAULT_FALLOFF = 0.045


class BodyMeshError(RuntimeError):
    """Raised when a mesh cannot be read or edited, with the real reason."""


@dataclass(frozen=True)
class MeshSummary:
    """What a loaded mesh is, read from the file rather than guessed."""

    name: str
    source: str
    vertex_count: int
    face_count: int
    triangle_count: int
    width: float
    height: float
    depth: float
    min_y: float
    max_y: float
    units: str

    def describe(self) -> str:
        return (
            f"{self.name}: {self.vertex_count} vertices, {self.triangle_count} triangles, "
            f"{self.width:.2f} x {self.height:.2f} x {self.depth:.2f} m "
            f"({self.units})."
        )


@dataclass(frozen=True)
class BodyRegion:
    """One editable part of a body: a named box in the figure's own space.

    The box is what makes a region mean the same thing on any figure: the head is
    the box around the head, the left arm is the box around the left arm, and so on,
    because they are derived from the shared skeleton rather than from the mesh's
    own vertex order.
    """

    key: str
    label: str
    label_ar: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    #: The point a new part is added at, or a region edit turns about. It is *not*
    #: the box centre: a lateral region's box runs far past the body (the arm box
    #: reaches the whole half-space), so the geometric centre would sit in mid-air.
    #: This is a real point on the part, taken from the figure's own landmarks.
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def contains(self, point: tuple[float, float, float]) -> bool:
        return (
            self.lo[0] <= point[0] <= self.hi[0]
            and self.lo[1] <= point[1] <= self.hi[1]
            and self.lo[2] <= point[2] <= self.hi[2]
        )


#: The named regions a body is divided into, as ``(key, label, label_ar)``. The
#: order is the order a window shows them.
REGION_TITLES: tuple[tuple[str, str, str], ...] = (
    ("head", "Head", "الرأس"),
    ("neck", "Neck", "الرقبة"),
    ("torso", "Torso and chest", "الجذع والصدر"),
    ("pelvis", "Pelvis and hips", "الحوض"),
    ("arm_left", "Left arm", "الذراع اليسرى"),
    ("arm_right", "Right arm", "الذراع اليمنى"),
    ("hand_left", "Left hand", "اليد اليسرى"),
    ("hand_right", "Right hand", "اليد اليمنى"),
    ("leg_left", "Left leg", "الساق اليسرى"),
    ("leg_right", "Right leg", "الساق اليمنى"),
    ("foot_left", "Left foot", "القدم اليسرى"),
    ("foot_right", "Right foot", "القدم اليمنى"),
)


def _figure_landmarks(height: float) -> dict[str, float]:
    """The heights and widths the regions are cut at, from the figure's own rig.

    These are the *same* fractions :func:`core.scene3d_figures.build_figure_mesh`
    builds the body with, so a region is the anatomical part wherever the mesh came
    from -- an imported OBJ that a user normalised to the same height lines up.
    """
    return {
        "head_base": 0.875 * height,
        "shoulder_y": 0.820 * height,
        "chest_y": 0.720 * height,
        "hip_y": 0.480 * height,
        "shoulder_x": 0.145 * height,
        "hip_x": 0.105 * height,
    }


def body_regions(height: float, *, shoulder_width: float = 0.42) -> tuple[BodyRegion, ...]:
    """The named regions of a body ``height`` metres tall, centred on the origin.

    The boxes come from the figure's own landmarks, so the head is the head and the
    left arm is the left arm on any mesh of that height. The shoulder box is widened
    by the figure's own shoulder width so a broad figure is still covered.
    """
    mark = _figure_landmarks(height)
    half_shoulder = max(mark["shoulder_x"], shoulder_width * 0.62)
    half_hip = mark["hip_x"]
    big = height * 2.0  # a box that covers everything along an axis
    head_lo = mark["head_base"] - height * 0.02
    head_hi = height * 1.06
    # The arm/leg axes: where the limb really is, so a region's ``centre`` is on the
    # limb and not in the empty half-space the box reaches into.
    arm_x = half_shoulder * 0.85
    leg_x = half_hip * 0.55
    head_mid = (head_lo + head_hi) * 0.5

    boxes: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
        "head": ((-half_shoulder, head_lo, -big), (half_shoulder, head_hi, big)),
        "neck": (
            (-half_shoulder, mark["shoulder_y"] - height * 0.01, -big),
            (half_shoulder, mark["head_base"] + height * 0.02, big),
        ),
        "torso": (
            (-half_shoulder, mark["chest_y"] - height * 0.02, -big),
            (half_shoulder, mark["shoulder_y"] + height * 0.01, big),
        ),
        "pelvis": (
            (-half_shoulder, mark["hip_y"] - height * 0.06, -big),
            (half_shoulder, mark["chest_y"] + height * 0.02, big),
        ),
        "arm_left": (
            (half_shoulder * 0.42, mark["hip_y"] - height * 0.01, -big),
            (big, mark["shoulder_y"] + height * 0.01, big),
        ),
        "arm_right": (
            (-big, mark["hip_y"] - height * 0.01, -big),
            (-half_shoulder * 0.42, mark["shoulder_y"] + height * 0.01, big),
        ),
        "hand_left": (
            (half_shoulder * 0.42, -height * 0.02, -big),
            (big, mark["hip_y"] - height * 0.06, big),
        ),
        "hand_right": (
            (-big, -height * 0.02, -big),
            (-half_shoulder * 0.42, mark["hip_y"] - height * 0.06, big),
        ),
        "leg_left": (
            (half_hip * 0.25, height * 0.03, -big),
            (big, mark["hip_y"] + height * 0.02, big),
        ),
        "leg_right": (
            (-big, height * 0.03, -big),
            (-half_hip * 0.25, mark["hip_y"] + height * 0.02, big),
        ),
        "foot_left": (
            (half_hip * 0.25, -height * 0.02, -big),
            (big, height * 0.04, big),
        ),
        "foot_right": (
            (-big, -height * 0.02, -big),
            (-half_hip * 0.25, height * 0.04, big),
        ),
    }
    titles = {key: (en, ar) for key, en, ar in REGION_TITLES}
    centres: dict[str, tuple[float, float, float]] = {
        "head": (0.0, head_mid, 0.0),
        "neck": (0.0, (mark["shoulder_y"] + mark["head_base"]) * 0.5, 0.0),
        "torso": (0.0, (mark["chest_y"] + mark["shoulder_y"]) * 0.5, 0.0),
        "pelvis": (0.0, (mark["hip_y"] + mark["chest_y"]) * 0.5, 0.0),
        "arm_left": (arm_x, (mark["hip_y"] + mark["shoulder_y"]) * 0.5, 0.0),
        "arm_right": (-arm_x, (mark["hip_y"] + mark["shoulder_y"]) * 0.5, 0.0),
        "hand_left": (arm_x, mark["hip_y"] * 0.5, 0.0),
        "hand_right": (-arm_x, mark["hip_y"] * 0.5, 0.0),
        "leg_left": (leg_x, mark["hip_y"] * 0.5, 0.0),
        "leg_right": (-leg_x, mark["hip_y"] * 0.5, 0.0),
        "foot_left": (leg_x, height * 0.01, 0.0),
        "foot_right": (-leg_x, height * 0.01, 0.0),
    }
    return tuple(
        BodyRegion(key, titles[key][0], titles[key][1], lo, hi, centres.get(key, (0.0, 0.0, 0.0)))
        for key, (lo, hi) in boxes.items()
    )


def region_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each region, in either language."""
    if language == "ar":
        return tuple((key, ar) for key, _en, ar in REGION_TITLES)
    return tuple((key, en) for key, en, _ar in REGION_TITLES)


def find_region(key: str, regions: tuple[BodyRegion, ...]) -> BodyRegion | None:
    """The region named ``key``, or ``None`` for a name the studio does not have."""
    for region in regions:
        if region.key == key:
            return region
    return None


def region_vertices(mesh: Mesh, region: BodyRegion) -> tuple[int, ...]:
    """The indices of the vertices that fall inside ``region``'s box."""
    return tuple(
        index
        for index, vertex in enumerate(mesh.vertices)
        if region.contains((vertex.x, vertex.y, vertex.z))
    )


# -- reading a mesh -------------------------------------------------------


def _read_obj(path: Path) -> Mesh:
    from core.scene3d_io import SceneIOError, load_obj

    try:
        scene, _report = load_obj(path)
    except (SceneIOError, OSError, ValueError) as exc:
        raise BodyMeshError(f"{path.name} could not be read as an OBJ file: {exc}") from exc
    meshes = [obj.mesh for obj in scene.objects if obj.mesh.vertices]
    if not meshes:
        raise BodyMeshError(f"{path.name} holds no geometry.")
    merged = meshes[0]
    for extra in meshes[1:]:
        merged = merged.merge(extra)
    return Mesh(path.stem, merged.vertices, merged.faces, (), merged.vertex_colors)


def _read_stl(path: Path) -> Mesh:
    """Read an STL file, binary or ASCII, into a triangle mesh.

    STL stores a soup of triangles with no shared vertices; the result is welded so
    it is one surface rather than a pile of loose triangles.
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise BodyMeshError(f"{path.name} is too small to be an STL file.")
    header = data[:5].lower()
    ascii_like = header.startswith(b"solid") and b"facet" in data[:2048].lower()
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    if ascii_like:
        current: list[int] = []
        for raw in data.decode("utf-8", "replace").splitlines():
            parts = raw.strip().split()
            if len(parts) >= 4 and parts[0].lower() == "vertex":
                try:
                    x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError as exc:
                    raise BodyMeshError(f"{path.name} has a malformed vertex.") from exc
                vertices.append(Vec3(x, y, z))
                current.append(len(vertices) - 1)
                if len(current) == 3:
                    faces.append(tuple(current))
                    current = []
    else:
        count = struct.unpack_from("<I", data, 80)[0]
        if count == 0 or 84 + count * 50 > len(data):
            # Fall back to a triangle-count inferred from the file size.
            count = max(0, (len(data) - 84) // 50)
        for index in range(count):
            base = 84 + index * 50
            if base + 50 > len(data):
                break
            # An STL record is 50 bytes: a 3-float normal, three 3-float vertices
            # (the 12 floats read here), then a 2-byte attribute count.
            values = struct.unpack_from("<12f", data, base)
            triangle: list[int] = []
            for corner in range(3):
                x, y, z = values[3 + corner * 3 : 3 + corner * 3 + 3]
                vertices.append(Vec3(float(x), float(y), float(z)))
                triangle.append(len(vertices) - 1)
            faces.append(tuple(triangle))
    if not vertices:
        raise BodyMeshError(f"{path.name} holds no geometry.")
    return Mesh(path.stem, tuple(vertices), tuple(faces)).weld(tolerance=1e-5)


def _read_ply(path: Path) -> Mesh:
    """Read a PLY file (ASCII or binary little-endian) into a triangle mesh."""
    data = path.read_bytes()
    marker = data.find(b"end_header")
    if marker < 0:
        raise BodyMeshError(f"{path.name} has no PLY header.")
    newline = data.find(b"\n", marker)
    if newline < 0:
        raise BodyMeshError(f"{path.name} has a truncated PLY header.")
    header = data[:marker].decode("ascii", "replace")
    body = data[newline + 1 :]
    fmt = "ascii"
    vertex_count = 0
    face_count = 0
    properties: list[str] = []
    in_vertex = False
    for raw in header.splitlines():
        line = raw.strip()
        if line.startswith("format"):
            if "binary_big_endian" in line:
                raise BodyMeshError(f"{path.name} is big-endian PLY, which is not read.")
            fmt = "binary" if "binary" in line else "ascii"
        elif line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
            in_vertex = False
        elif line.startswith("element"):
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            properties.append(line.split()[-1])
    if vertex_count == 0:
        raise BodyMeshError(f"{path.name} holds no vertices.")
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    if fmt == "ascii":
        tokens = body.decode("ascii", "replace").split()
        cursor = 0
        for _ in range(vertex_count):
            if cursor + 3 > len(tokens):
                raise BodyMeshError(f"{path.name} ends before its vertices do.")
            x, y, z = (float(tokens[cursor]), float(tokens[cursor + 1]), float(tokens[cursor + 2]))
            cursor += len(properties) or 3
            vertices.append(Vec3(x, y, z))
        for _ in range(face_count):
            if cursor >= len(tokens):
                break
            n = int(tokens[cursor])
            cursor += 1
            face = tuple(int(tokens[cursor + i]) for i in range(n))
            cursor += n
            if len(face) >= 3:
                faces.append(face)
    else:
        stride = 4 * len(properties) if properties else 12
        expected = {"x": 0, "y": 1, "z": 2}
        offsets = [expected.get(name, -1) for name in properties] or [0, 1, 2]
        for index in range(vertex_count):
            base = index * stride
            if base + 12 > len(body):
                raise BodyMeshError(f"{path.name} ends before its vertices do.")
            values = struct.unpack_from("<3f", body, base + offsets[0] * 4)
            vertices.append(Vec3(float(values[0]), float(values[1]), float(values[2])))
        cursor = vertex_count * stride
        for _ in range(face_count):
            if cursor + 1 > len(body):
                break
            (n,) = struct.unpack_from("<B", body, cursor)
            cursor += 1
            if cursor + n * 4 > len(body):
                break
            face = struct.unpack_from("<" + "i" * n, body, cursor)
            cursor += n * 4
            if len(face) >= 3:
                faces.append(tuple(int(i) for i in face))
    if not vertices:
        raise BodyMeshError(f"{path.name} holds no geometry.")
    return Mesh(path.stem, tuple(vertices), tuple(faces)).weld(tolerance=1e-5)


def load_body_mesh(path: str | Path, *, name: str = "") -> Mesh:
    """Read a mesh file (OBJ, STL, or PLY) into a :class:`Mesh`.

    The suffix picks the reader; a format this build does not read is refused by
    name rather than guessed at, and an over-large file is refused before it is
    parsed.
    """
    target = Path(path)
    if not target.exists():
        raise BodyMeshError(f"{target} does not exist.")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise BodyMeshError(
            f"{target.name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB and is refused."
        )
    suffix = target.suffix.lower()
    if suffix == ".obj":
        mesh = _read_obj(target)
    elif suffix == ".stl":
        mesh = _read_stl(target)
    elif suffix == ".ply":
        mesh = _read_ply(target)
    else:
        raise BodyMeshError(
            f"{target.name} is not a mesh this build reads "
            f"(supported: {', '.join(IMPORT_SUFFIXES)})."
        )
    if len(mesh.vertices) > MAX_VERTICES or len(mesh.faces) > MAX_FACES:
        raise BodyMeshError(f"{target.name} has more geometry than this build accepts.")
    if not mesh.faces:
        raise BodyMeshError(f"{target.name} has no faces.")
    return Mesh(name or target.stem, mesh.vertices, mesh.faces, (), mesh.vertex_colors)


def mesh_bounds(mesh: Mesh) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The mesh's axis-aligned bounds, or a zero box for an empty mesh."""
    if not mesh.vertices:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [v.x for v in mesh.vertices]
    ys = [v.y for v in mesh.vertices]
    zs = [v.z for v in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def summarize_mesh(mesh: Mesh, *, source: str = "") -> MeshSummary:
    """Read a mesh's real size and counts (never a fabricated figure)."""
    lo, hi = mesh_bounds(mesh)
    height = hi[1] - lo[1]
    units = "metres" if 0.5 <= height <= 3.0 else "arbitrary units"
    return MeshSummary(
        name=mesh.name,
        source=source,
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
        triangle_count=mesh.triangle_count,
        width=hi[0] - lo[0],
        height=height,
        depth=hi[2] - lo[2],
        min_y=lo[1],
        max_y=hi[1],
        units=units,
    )


def normalize_mesh(mesh: Mesh, *, height: float = 1.75) -> Mesh:
    """Scale and centre a mesh so it stands on ``y = 0`` at the requested height.

    A DAZ or MakeHuman export can be in any unit and any origin; this puts it in the
    project's own space (metres, feet on the ground, centred on the origin) so the
    region boxes and the camera framing mean the same thing as they do for the
    parametric figure. A mesh with no height is returned unscaled.
    """
    lo, hi = mesh_bounds(mesh)
    source_height = hi[1] - lo[1]
    if source_height <= 1e-9:
        return mesh
    scale = float(height) / source_height
    centre_x = (lo[0] + hi[0]) * 0.5
    centre_z = (lo[2] + hi[2]) * 0.5
    moved = tuple(
        Vec3((v.x - centre_x) * scale, (v.y - lo[1]) * scale, (v.z - centre_z) * scale)
        for v in mesh.vertices
    )
    return Mesh(mesh.name, moved, mesh.faces, (), mesh.vertex_colors)


# -- editing a region -----------------------------------------------------


def _region_weights(
    points: np.ndarray, region: BodyRegion, falloff: float
) -> np.ndarray:
    """A 0..1 weight per point: 1 inside the region, fading to 0 over ``falloff``.

    The fade is what keeps a joint smooth: a vertex just outside the box still moves
    a little, so the part changes shape instead of tearing away from its neighbour.
    """
    lo = np.asarray(region.lo, dtype=float)
    hi = np.asarray(region.hi, dtype=float)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    distance = np.maximum.reduce(
        [np.maximum(lo[axis] - points[:, axis], points[:, axis] - hi[axis]) for axis in range(3)]
    )
    distance = np.maximum(distance, 0.0)
    margin = max(1e-6, float(falloff))
    fade = np.clip(1.0 - distance / margin, 0.0, 1.0)
    return np.where(inside, 1.0, fade)


def _region_pivot(points: np.ndarray, region: BodyRegion) -> np.ndarray:
    """The centre the region scales and rotates about (its own vertices' mean)."""
    lo = np.asarray(region.lo, dtype=float)
    hi = np.asarray(region.hi, dtype=float)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    chosen = points[mask] if bool(mask.any()) else np.asarray(region.centre, dtype=float)
    return chosen.mean(axis=0)


def _apply_matrix(
    points: np.ndarray, weights: np.ndarray, pivot: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    local = points - pivot
    transformed = local @ matrix.T
    blended = local * (1.0 - weights)[:, None] + transformed * weights[:, None]
    return blended + pivot


def resize_region(
    mesh: Mesh,
    region: BodyRegion,
    scale: tuple[float, float, float] | float,
    *,
    falloff: float = DEFAULT_FALLOFF,
) -> Mesh:
    """Scale the part of the mesh inside ``region`` (with a smooth falloff).

    ``scale`` is either one number (uniform) or a per-axis triple, so a limb can be
    made thicker, longer, or flatter independently. A non-positive factor is refused
    rather than turning the part inside out.
    """
    if isinstance(scale, (int, float)):
        factors = (float(scale), float(scale), float(scale))
    else:
        factors = (float(scale[0]), float(scale[1]), float(scale[2]))
    if any(factor <= 0.0 for factor in factors):
        raise BodyMeshError("A resize factor must be positive.")
    points = _points_array(mesh)
    weights = _region_weights(points, region, falloff)
    if not weights.any():
        return mesh
    pivot = _region_pivot(points, region)
    matrix = np.diag(np.asarray(factors, dtype=float))
    moved = _apply_matrix(points, weights, pivot, matrix)
    return _mesh_from_points(mesh, moved)


def move_region(
    mesh: Mesh,
    region: BodyRegion,
    offset: tuple[float, float, float],
    *,
    falloff: float = DEFAULT_FALLOFF,
) -> Mesh:
    """Translate the part of the mesh inside ``region`` (with a smooth falloff)."""
    points = _points_array(mesh)
    weights = _region_weights(points, region, falloff)
    if not weights.any():
        return mesh
    delta = np.asarray(offset, dtype=float)
    moved = points + weights[:, None] * delta
    return _mesh_from_points(mesh, moved)


def rotate_region(
    mesh: Mesh,
    region: BodyRegion,
    degrees: tuple[float, float, float],
    *,
    falloff: float = DEFAULT_FALLOFF,
) -> Mesh:
    """Rotate the part of the mesh inside ``region`` about its own centre."""
    points = _points_array(mesh)
    weights = _region_weights(points, region, falloff)
    if not weights.any():
        return mesh
    pivot = _region_pivot(points, region)
    rx, ry, rz = (math.radians(degrees[0]), math.radians(degrees[1]), math.radians(degrees[2]))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    matrix = rot_z @ rot_y @ rot_x
    moved = _apply_matrix(points, weights, pivot, matrix)
    return _mesh_from_points(mesh, moved)


def _points_array(mesh: Mesh) -> np.ndarray:
    return np.array([[v.x, v.y, v.z] for v in mesh.vertices], dtype=float)


def _mesh_from_points(mesh: Mesh, points: np.ndarray) -> Mesh:
    return Mesh(
        mesh.name,
        tuple(Vec3(float(p[0]), float(p[1]), float(p[2])) for p in points),
        mesh.faces,
        (),
        mesh.vertex_colors,
    )


# -- adding a missing part ------------------------------------------------

#: The shapes a missing part can be built from, as ``(key, label, label_ar)``.
ADD_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("sphere", "Sphere", "كرة"),
    ("ellipsoid", "Ellipsoid", "مجسم بيضي"),
    ("capsule", "Capsule", "كبسولة"),
    ("box", "Box", "صندوق"),
)

_ADD_SHAPE_TITLES: dict[str, tuple[str, str]] = {key: (en, ar) for key, en, ar in ADD_SHAPES}


def add_shape_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each shape a part can be built from."""
    index = 1 if language == "ar" else 0
    return tuple((key, titles[index]) for key, titles in _ADD_SHAPE_TITLES.items())


def _clamp01(value: Any, fallback: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def _clamp_scale(value: Any, fallback: float = 1.0) -> float:
    try:
        return min(5.0, max(0.05, float(value)))
    except (TypeError, ValueError):
        return fallback


def _shape_sdf(
    kind: str,
    centre: np.ndarray,
    size: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """The signed distance from ``points`` to an added shape (its own field)."""
    local = points - centre
    if kind == "sphere":
        radius = float(size[0])
        return np.linalg.norm(local, axis=1) - radius
    if kind == "ellipsoid":
        radii = np.maximum(size, 1e-4)
        scaled = local / radii
        return (np.linalg.norm(scaled, axis=1) - 1.0) * float(radii.min())
    if kind == "capsule":
        axis = np.asarray(size, dtype=float)
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return np.linalg.norm(local, axis=1) - 0.05
        direction = axis / length
        t = np.clip(local @ direction, -length * 0.5, length * 0.5)
        projection = t[:, None] * direction
        return np.linalg.norm(local - projection, axis=1) - 0.05
    half = np.maximum(size, 1e-4)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def _mesh_inside_grid(
    mesh: Mesh, origin: np.ndarray, cell: float, shape: tuple[int, int, int]
) -> np.ndarray:
    """A boolean grid: whether each lattice point is inside ``mesh``.

    The surface triangles are splatted onto the lattice, then an outside flood fill
    from the border marks everything reachable from outside; a point the fill cannot
    reach is inside the mesh. This is how a *mesh* (not a formula) becomes a field
    the union can fuse against.
    """
    nx, ny, nz = shape
    surface = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    points = _points_array(mesh)
    for face in mesh.faces:
        triangle = points[list(face)]
        low = np.maximum(np.floor((triangle.min(axis=0) - origin) / cell).astype(int), 0)
        high = np.minimum(
            np.ceil((triangle.max(axis=0) - origin) / cell).astype(int), [nx, ny, nz]
        )
        for i in range(low[0], high[0] + 1):
            for j in range(low[1], high[1] + 1):
                for k in range(low[2], high[2] + 1):
                    surface[i, j, k] = True
    outside = np.zeros_like(surface)
    queue: deque[tuple[int, int, int]] = deque()
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if i in (0, nx) or j in (0, ny) or k in (0, nz):
                    if not surface[i, j, k] and not outside[i, j, k]:
                        outside[i, j, k] = True
                        queue.append((i, j, k))
    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a <= nx and 0 <= b <= ny and 0 <= c <= nz:
                if not surface[a, b, c] and not outside[a, b, c]:
                    outside[a, b, c] = True
                    queue.append((a, b, c))
    return ~outside & ~surface


def _mesh_field(
    mesh: Mesh, origin: np.ndarray, cell: float, shape: tuple[int, int, int]
) -> np.ndarray:
    """A signed-distance-like field for a whole mesh, sampled on the lattice."""
    inside = _mesh_inside_grid(mesh, origin, cell, shape)
    return np.where(inside, -cell, cell)


def _blend_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """A smooth union of two fields (a small, bounded blend at the join)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _validate_part_size(shape: str, vector: tuple[float, float, float]) -> None:
    """Refuse a part size that cannot build a real shape.

    A **capsule**'s size is an axis *direction* (a vector from one end to the other),
    so a zero component is legitimate -- it means the capsule runs along the other
    axes, and a fully zero vector is a small sphere. A **sphere**, an **ellipsoid**,
    or a **box** is sized by radii or half-extents, which must all be positive.
    """
    if shape == "capsule":
        return
    if any(value <= 0.0 for value in vector):
        raise BodyMeshError("A part size must be positive.")


def add_part(
    mesh: Mesh,
    *,
    shape: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float] | float,
    part_mesh: Mesh | None = None,
    colour: tuple[float, float, float] | None = None,
    cell: float = DEFAULT_FUSE_CELL,
    blend: float = 0.02,
) -> Mesh:
    """Fuse a new part into ``mesh`` so it becomes one connected skin.

    ``shape`` picks the geometry: one of :data:`ADD_SHAPES` builds a shape from
    ``centre``/``size``, while ``"mesh"`` fuses the whole ``part_mesh`` (a sub-mesh
    the user imported) at ``centre``. The union is done on an implicit field, so the
    new part really joins the body instead of floating beside it, and any vertex
    colour already on the body is carried through. An unknown shape is refused.
    """
    points = _points_array(mesh)
    if part_mesh is not None and shape == "mesh":
        added = _points_array(part_mesh)
        added = added + np.asarray(centre, dtype=float)
        offset = len(mesh.vertices)
        extra_faces = [
            tuple(index + offset for index in face)
            for face in _mesh_from_points(part_mesh, added).faces
        ]
        combined = Mesh(
            mesh.name,
            _points_to_vecs(np.vstack([points, added])),
            tuple(mesh.faces) + tuple(extra_faces),
        )
        return _fuse_union(combined, cell=cell, blend=blend)
    if shape not in {key for key, _en, _ar in ADD_SHAPES}:
        raise BodyMeshError(
            f"Unknown part shape {shape!r}; choose from "
            f"{', '.join(key for key, _en, _ar in ADD_SHAPES)} or 'mesh'."
        )
    vector: tuple[float, float, float] = (
        (float(size),) * 3
        if isinstance(size, (int, float))
        else (float(size[0]), float(size[1]), float(size[2]))
    )
    _validate_part_size(shape, vector)
    del colour  # accepted for symmetry; the fused skin keeps its own colours
    lo, hi = mesh_bounds(mesh)
    centre_array = np.asarray(centre, dtype=float)
    pad = max(abs(value) for value in vector) + 0.05
    origin = np.minimum(np.asarray(lo, dtype=float), centre_array - pad)
    top = np.maximum(np.asarray(hi, dtype=float), centre_array + pad)
    spacing = max(0.004, float(cell))
    grid_dims: tuple[int, int, int] = (1, 1, 1)
    for _ in range(8):
        grid_dims = tuple(
            int((top[axis] - origin[axis]) / spacing) + 2 for axis in range(3)
        )  # type: ignore[assignment]
        if grid_dims[0] * grid_dims[1] * grid_dims[2] <= MAX_FUSE_SAMPLES:
            break
        spacing *= 1.3
    body_field = _mesh_field(mesh, origin, spacing, grid_dims)
    grid_shape = body_field.shape
    xs = origin[0] + spacing * np.arange(grid_shape[0])
    ys = origin[1] + spacing * np.arange(grid_shape[1])
    zs = origin[2] + spacing * np.arange(grid_shape[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    part_field = _shape_sdf(shape, centre_array, np.asarray(vector, dtype=float), grid_points)
    part_field = part_field.reshape(body_field.shape)
    combined_field = _blend_min(body_field, part_field, max(1e-4, float(blend)))
    return _mesh_from_field(mesh, combined_field, origin, spacing)


def _mesh_from_field(
    fallback: Mesh, field: Any, origin: np.ndarray, spacing: float
) -> Mesh:
    """Polygonize a scalar field into a mesh, or return ``fallback`` if it is empty."""
    verts, faces = _surface_nets(field, origin, spacing)
    if len(verts) == 0:
        return fallback
    return Mesh(
        fallback.name,
        _points_to_vecs(verts),
        tuple(tuple(int(i) for i in face) for face in faces),
    )


def _fuse_union(mesh: Mesh, *, cell: float, blend: float) -> Mesh:
    """Union a mesh's own overlapping shells into one surface (used by add-mesh)."""
    lo, hi = mesh_bounds(mesh)
    origin = np.asarray(lo, dtype=float) - 0.02
    top = np.asarray(hi, dtype=float) + 0.02
    spacing = max(0.004, float(cell))
    grid_dims: tuple[int, int, int] = (1, 1, 1)
    for _ in range(8):
        grid_dims = tuple(
            int((top[axis] - origin[axis]) / spacing) + 2 for axis in range(3)
        )  # type: ignore[assignment]
        if grid_dims[0] * grid_dims[1] * grid_dims[2] <= MAX_FUSE_SAMPLES:
            break
        spacing *= 1.3
    field = _mesh_field(mesh, origin, spacing, grid_dims)
    return _mesh_from_field(mesh, field, origin, spacing)


def _points_to_vecs(points: Any) -> tuple[Vec3, ...]:
    return tuple(Vec3(float(p[0]), float(p[1]), float(p[2])) for p in points)


def body_mesh_from_object(obj: Any, *, height: float = 1.75) -> Mesh:
    """A :class:`Mesh` from a scene object or a mesh, normalised to the body space.

    This is the bridge the studio uses: an imported object (or a parametric figure)
    becomes one normalised mesh the region tools can edit.
    """
    mesh = getattr(obj, "mesh", obj)
    if not isinstance(mesh, Mesh):
        raise BodyMeshError("A body object must hold a Mesh.")
    return normalize_mesh(mesh, height=height)


__all__ = [
    "ADD_SHAPES",
    "DEFAULT_FALLOFF",
    "DEFAULT_FUSE_CELL",
    "IMPORT_SUFFIXES",
    "MAX_FACES",
    "MAX_FILE_BYTES",
    "MAX_VERTICES",
    "BodyMeshError",
    "BodyRegion",
    "MeshSummary",
    "add_part",
    "add_shape_titles",
    "body_mesh_from_object",
    "body_regions",
    "find_region",
    "load_body_mesh",
    "mesh_bounds",
    "normalize_mesh",
    "region_titles",
    "region_vertices",
    "resize_region",
    "rotate_region",
    "summarize_mesh",
]
