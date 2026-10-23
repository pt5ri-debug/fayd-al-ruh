"""The local 3D figure studio's scene model: math, meshes, skeletons, skinning.

This is the *engine* half of the project's DAZ-style figure studio. It is a real
3D content model, not a picture of one:

* :class:`Vec3` / :class:`Mat4` -- the small, dependency-free linear algebra the
  rest of the studio is built from (points, transforms, view/projection).
* :class:`Mesh` -- vertices and polygonal faces, with computed vertex normals.
* :class:`Bone` / :class:`Skeleton` -- a real bone hierarchy with **forward
  kinematics**: rotating a bone moves everything below it, exactly as a figure
  rig does.
* **Linear blend skinning** (:func:`compute_skin_weights` / :func:`skin_mesh`) --
  each vertex is bound to the bones nearest it in the rest pose, so posing the
  skeleton actually deforms the mesh instead of moving rigid pieces.
* :class:`Scene`, :class:`SceneObject`, :class:`Camera`, :class:`Light`,
  :class:`Material` -- what a scene *is*: objects with transforms and optional
  rigs, a camera, and lights.

Everything here is deterministic, standard-library only, and Tk-free, so the
whole model is testable on a fresh clone with no display, no GPU, and no model
download. Rendering lives in :mod:`core.scene3d_render`, file I/O in
:mod:`core.scene3d_io`, and the built-in figures in :mod:`core.scene3d_figures`.

Honesty is a hard rule, as everywhere else in the project: this is a **local,
CPU, software** studio. It loads and poses real 3D content (OBJ/MTL) and renders
it with a real z-buffered rasteriser, but it is not a GPU renderer, it ships no
licensed figure content, and it makes no claim to reproduce any commercial
product's asset store or file formats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- math

_IDENTITY: tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


@dataclass(frozen=True)
class Vec3:
    """A 3-component vector, used for both points and directions."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vec3:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a Vec3 by zero.")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        """The unit vector in this direction; a zero vector is returned as-is."""
        magnitude = self.length()
        if magnitude < 1e-12:
            return Vec3()
        return self / magnitude

    def lerp(self, other: Vec3, t: float) -> Vec3:
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


ORIGIN = Vec3()
UP = Vec3(0.0, 1.0, 0.0)


def vec3(x: float, y: float, z: float) -> Vec3:
    """A short constructor, so scene literals read compactly."""
    return Vec3(float(x), float(y), float(z))


def _radians(degrees: float) -> float:
    return degrees * math.pi / 180.0


@dataclass(frozen=True)
class Mat4:
    """A 4x4 matrix stored **row-major** as a flat 16-tuple.

    Points are column vectors on the right (``M @ p``), which is the convention
    the view and projection matrices below are written for.
    """

    m: tuple[float, ...] = _IDENTITY

    def __post_init__(self) -> None:
        if len(self.m) != 16:
            raise ValueError(f"A Mat4 needs 16 components, got {len(self.m)}.")

    @staticmethod
    def identity() -> Mat4:
        return Mat4()

    @staticmethod
    def translation(offset: Vec3) -> Mat4:
        return Mat4(
            (
                1.0, 0.0, 0.0, offset.x,
                0.0, 1.0, 0.0, offset.y,
                0.0, 0.0, 1.0, offset.z,
                0.0, 0.0, 0.0, 1.0,
            )
        )

    @staticmethod
    def scaling(factor: Vec3) -> Mat4:
        return Mat4(
            (
                factor.x, 0.0, 0.0, 0.0,
                0.0, factor.y, 0.0, 0.0,
                0.0, 0.0, factor.z, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
        )

    @staticmethod
    def rotation_x(degrees: float) -> Mat4:
        c, s = math.cos(_radians(degrees)), math.sin(_radians(degrees))
        return Mat4(
            (
                1.0, 0.0, 0.0, 0.0,
                0.0, c, -s, 0.0,
                0.0, s, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
        )

    @staticmethod
    def rotation_y(degrees: float) -> Mat4:
        c, s = math.cos(_radians(degrees)), math.sin(_radians(degrees))
        return Mat4(
            (
                c, 0.0, s, 0.0,
                0.0, 1.0, 0.0, 0.0,
                -s, 0.0, c, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
        )

    @staticmethod
    def rotation_z(degrees: float) -> Mat4:
        c, s = math.cos(_radians(degrees)), math.sin(_radians(degrees))
        return Mat4(
            (
                c, -s, 0.0, 0.0,
                s, c, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
        )

    @staticmethod
    def euler(rotation: Vec3) -> Mat4:
        """Intrinsic X, then Y, then Z rotation (``Rz @ Ry @ Rx``), in degrees."""
        return (
            Mat4.rotation_z(rotation.z)
            @ Mat4.rotation_y(rotation.y)
            @ Mat4.rotation_x(rotation.x)
        )

    @staticmethod
    def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> Mat4:
        """A right-handed perspective projection looking down ``-Z``."""
        if not 0.0 < fov_degrees < 180.0:
            raise ValueError("Field of view must be in (0, 180) degrees.")
        if aspect <= 0:
            raise ValueError("Aspect ratio must be positive.")
        if not 0.0 < near < far:
            raise ValueError("Perspective needs 0 < near < far.")
        focal = 1.0 / math.tan(_radians(fov_degrees) / 2.0)
        return Mat4(
            (
                focal / aspect, 0.0, 0.0, 0.0,
                0.0, focal, 0.0, 0.0,
                0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far),
                0.0, 0.0, -1.0, 0.0,
            )
        )

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3 = UP) -> Mat4:
        """A right-handed view matrix that looks from ``eye`` at ``target``."""
        forward = (target - eye).normalized()
        if forward.length() < 1e-12:
            raise ValueError("The camera's eye and target must differ.")
        side = forward.cross(up).normalized()
        if side.length() < 1e-12:
            # The up vector is parallel to the view direction; pick another axis.
            side = forward.cross(Vec3(1.0, 0.0, 0.0)).normalized()
            if side.length() < 1e-12:
                side = forward.cross(Vec3(0.0, 0.0, 1.0)).normalized()
        true_up = side.cross(forward)
        return Mat4(
            (
                side.x, side.y, side.z, -side.dot(eye),
                true_up.x, true_up.y, true_up.z, -true_up.dot(eye),
                -forward.x, -forward.y, -forward.z, forward.dot(eye),
                0.0, 0.0, 0.0, 1.0,
            )
        )

    def __matmul__(self, other: Mat4) -> Mat4:
        a, b = self.m, other.m
        result = [0.0] * 16
        for row in range(4):
            base = row * 4
            for column in range(4):
                result[base + column] = (
                    a[base] * b[column]
                    + a[base + 1] * b[4 + column]
                    + a[base + 2] * b[8 + column]
                    + a[base + 3] * b[12 + column]
                )
        return Mat4(tuple(result))

    def transform_point(self, point: Vec3) -> Vec3:
        """Apply the full transform, dividing by ``w`` (a projective point)."""
        m = self.m
        x = m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3]
        y = m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7]
        z = m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11]
        w = m[12] * point.x + m[13] * point.y + m[14] * point.z + m[15]
        if abs(w) > 1e-12 and abs(w - 1.0) > 1e-12:
            return Vec3(x / w, y / w, z / w)
        return Vec3(x, y, z)

    def transform_vec4(self, point: Vec3) -> tuple[float, float, float, float]:
        """Apply the transform and return the *undivided* ``(x, y, z, w)``."""
        m = self.m
        return (
            m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3],
            m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7],
            m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11],
            m[12] * point.x + m[13] * point.y + m[14] * point.z + m[15],
        )

    def transform_direction(self, direction: Vec3) -> Vec3:
        """Rotate/scale a direction: the translation part is ignored."""
        m = self.m
        return Vec3(
            m[0] * direction.x + m[1] * direction.y + m[2] * direction.z,
            m[4] * direction.x + m[5] * direction.y + m[6] * direction.z,
            m[8] * direction.x + m[9] * direction.y + m[10] * direction.z,
        )

    def translation_part(self) -> Vec3:
        return Vec3(self.m[3], self.m[7], self.m[11])

    def inverse(self) -> Mat4:
        """The general 4x4 inverse, by Gauss-Jordan elimination with partial pivots."""
        size = 4
        augmented = [
            [self.m[row * 4 + column] for column in range(size)]
            + [1.0 if row == column else 0.0 for column in range(size)]
            for row in range(size)
        ]
        for column in range(size):
            pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
            if abs(augmented[pivot_row][column]) < 1e-12:
                raise ValueError("This matrix is singular and cannot be inverted.")
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )
            pivot = augmented[column][column]
            augmented[column] = [value / pivot for value in augmented[column]]
            for row in range(size):
                if row == column:
                    continue
                factor = augmented[row][column]
                if factor == 0.0:
                    continue
                augmented[row] = [
                    value - factor * augmented[column][index]
                    for index, value in enumerate(augmented[row])
                ]
        flat: list[float] = []
        for row in range(size):
            flat.extend(augmented[row][size:])
        return Mat4(tuple(flat))


IDENTITY_MATRIX = Mat4.identity()


# --------------------------------------------------------------------------- mesh


@dataclass(frozen=True)
class Mesh:
    """Polygonal geometry: vertices plus faces that index them.

    ``faces`` are tuples of vertex indices; a triangle is the usual case, and a
    quad or n-gon is triangulated by the renderer as a fan. ``normals`` are
    optional -- when absent :meth:`vertex_normals` computes them from the faces
    (area-weighted), so a mesh loaded from a file with no normals still shades
    correctly.
    """

    name: str
    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, ...], ...]
    normals: tuple[Vec3, ...] = ()

    def __post_init__(self) -> None:
        if self.normals and len(self.normals) != len(self.vertices):
            raise ValueError("A mesh's normals must match its vertex count.")
        count = len(self.vertices)
        for face in self.faces:
            if len(face) < 3:
                raise ValueError(f"Face {face!r} needs at least three vertices.")
            for index in face:
                if not -count <= index < count:
                    raise ValueError(f"Face {face!r} indexes a vertex outside the mesh.")

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def triangle_count(self) -> int:
        """How many triangles the faces fan out to (``n - 2`` per face)."""
        return sum(len(face) - 2 for face in self.faces)

    def bounds(self) -> tuple[Vec3, Vec3]:
        """The axis-aligned bounding box as ``(min, max)``; empty meshes give zeros."""
        if not self.vertices:
            return (Vec3(), Vec3())
        first = self.vertices[0]
        low = high = first
        for vertex in self.vertices[1:]:
            low = Vec3(min(low.x, vertex.x), min(low.y, vertex.y), min(low.z, vertex.z))
            high = Vec3(max(high.x, vertex.x), max(high.y, vertex.y), max(high.z, vertex.z))
        return (low, high)

    def center(self) -> Vec3:
        low, high = self.bounds()
        return (low + high) / 2.0

    def vertex_normals(self) -> tuple[Vec3, ...]:
        """Area-weighted vertex normals, computed from the faces.

        Every face contributes its (un-normalised) normal to each of its
        vertices, so a larger triangle influences the result more -- the standard
        cheap approach, and it smooths a faceted mesh well.
        """
        if self.normals:
            return self.normals
        accumulated = [Vec3() for _ in self.vertices]
        for face in self.faces:
            for index in range(1, len(face) - 1):
                a = self.vertices[face[0]]
                b = self.vertices[face[index]]
                c = self.vertices[face[index + 1]]
                normal = (b - a).cross(c - a)
                for vertex_index in (face[0], face[index], face[index + 1]):
                    current = accumulated[vertex_index]
                    accumulated[vertex_index] = current + normal
        return tuple(normal.normalized() for normal in accumulated)

    def with_normals(self) -> Mesh:
        """A copy carrying computed vertex normals (a no-op when they exist)."""
        if self.normals:
            return self
        return Mesh(self.name, self.vertices, self.faces, self.vertex_normals())

    def transformed(self, matrix: Mat4, *, name: str | None = None) -> Mesh:
        """A copy with every vertex moved by ``matrix`` (normals re-derived)."""
        return Mesh(
            name or self.name,
            tuple(matrix.transform_point(vertex) for vertex in self.vertices),
            self.faces,
        )

    def merge(self, other: Mesh, *, name: str | None = None) -> Mesh:
        """Combine two meshes into one, re-indexing the second mesh's faces."""
        offset = len(self.vertices)
        faces = tuple(self.faces) + tuple(
            tuple(index + offset for index in face) for face in other.faces
        )
        return Mesh(name or self.name, tuple(self.vertices) + tuple(other.vertices), faces)


# --------------------------------------------------------------------------- skeleton


@dataclass(frozen=True)
class Bone:
    """One bone in a figure's rig.

    ``offset`` places the bone's head relative to its parent's head, expressed in
    the parent's **local** frame (so a child of a bone whose length is 0.2 uses
    ``offset=(0, 0.2, 0)`` to sit at the parent's tail). ``rest_rotation`` orients
    the bone -- its local ``+Y`` is the bone axis -- and ``pose_rotation`` is the
    user's pose, applied after the rest orientation. ``length`` extends the bone
    along its local ``+Y`` and is what skinning measures a vertex's distance to.
    """

    name: str
    parent: str | None = None
    offset: Vec3 = Vec3()
    rest_rotation: Vec3 = Vec3()
    pose_rotation: Vec3 = Vec3()
    length: float = 0.0

    def with_pose(self, rotation: Vec3) -> Bone:
        return Bone(
            self.name, self.parent, self.offset, self.rest_rotation, rotation, self.length
        )


@dataclass(frozen=True)
class Skeleton:
    """An ordered bone hierarchy (parents always before their children)."""

    bones: tuple[Bone, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for bone in self.bones:
            if bone.name in seen:
                raise ValueError(f"Duplicate bone name: {bone.name!r}.")
            if bone.parent is not None and bone.parent not in seen:
                raise ValueError(
                    f"Bone {bone.name!r} names parent {bone.parent!r}, which is not "
                    "defined before it."
                )
            seen.add(bone.name)

    def names(self) -> tuple[str, ...]:
        return tuple(bone.name for bone in self.bones)

    def bone(self, name: str) -> Bone:
        for candidate in self.bones:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def with_pose(self, pose: dict[str, Vec3]) -> Skeleton:
        """A copy with ``pose``'s rotations applied (unknown names are ignored)."""
        return Skeleton(
            tuple(
                bone.with_pose(pose[bone.name]) if bone.name in pose else bone
                for bone in self.bones
            )
        )

    def cleared(self) -> Skeleton:
        """A copy at rest: every pose rotation reset to zero."""
        return Skeleton(tuple(bone.with_pose(Vec3()) for bone in self.bones))

    def local_matrices(self) -> dict[str, Mat4]:
        """Each bone's local matrix: offset, then rest orientation, then the pose."""
        return {
            bone.name: Mat4.translation(bone.offset)
            @ Mat4.euler(bone.rest_rotation)
            @ Mat4.euler(bone.pose_rotation)
            for bone in self.bones
        }

    def world_matrices(self) -> dict[str, Mat4]:
        """Forward kinematics: each bone's head frame in world space.

        Bones are stored parents-first, so one pass suffices -- and because a
        child's matrix is built from its parent's, rotating a bone necessarily
        moves everything below it.
        """
        local = self.local_matrices()
        world: dict[str, Mat4] = {}
        for bone in self.bones:
            parent = world[bone.parent] if bone.parent else IDENTITY_MATRIX
            world[bone.name] = parent @ local[bone.name]
        return world

    def segments(self) -> dict[str, tuple[Vec3, Vec3]]:
        """Each bone as a world-space segment ``(head, tail)`` in the current pose."""
        world = self.world_matrices()
        result: dict[str, tuple[Vec3, Vec3]] = {}
        for bone in self.bones:
            head = world[bone.name].translation_part()
            tail = head + world[bone.name].transform_direction(Vec3(0.0, bone.length, 0.0))
            result[bone.name] = (head, tail)
        return result


def _distance_to_segment(point: Vec3, start: Vec3, end: Vec3) -> float:
    """The shortest distance from ``point`` to the segment ``start``--``end``."""
    span = end - start
    length_sq = span.dot(span)
    if length_sq < 1e-12:
        return (point - start).length()
    t = max(0.0, min(1.0, (point - start).dot(span) / length_sq))
    return (point - (start + span * t)).length()


#: How many bones may influence one vertex. Two is the classic "smooth joint"
#: compromise: a knee blends between thigh and shin, while a distant limb never
#: drags a vertex with it.
MAX_INFLUENCES = 2


def compute_skin_weights(
    mesh: Mesh,
    skeleton: Skeleton,
    *,
    max_influences: int = MAX_INFLUENCES,
    falloff: float = 2.0,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Bind each vertex to the bones nearest it, in the skeleton's rest pose.

    The weight of a bone is ``1 / (distance + eps) ** falloff``, kept for the
    ``max_influences`` nearest bones and normalised to sum to one. That is real
    linear-blend skinning, and it is computed once at bind time -- posing never
    recomputes it, so a pose is cheap.
    """
    if max_influences < 1:
        raise ValueError("A vertex needs at least one influence.")
    segments = skeleton.segments()
    order = skeleton.names()
    if not order:
        return tuple(() for _ in mesh.vertices)
    bindings: list[tuple[tuple[int, float], ...]] = []
    for vertex in mesh.vertices:
        distances = [
            _distance_to_segment(vertex, segments[name][0], segments[name][1])
            for name in order
        ]
        ranked = sorted(range(len(order)), key=lambda index: distances[index])
        chosen = ranked[: max(1, min(max_influences, len(order)))]
        raw = [(index, 1.0 / ((distances[index] + 1e-4) ** falloff)) for index in chosen]
        total = sum(weight for _, weight in raw)
        if total <= 0.0:
            bindings.append(((chosen[0], 1.0),))
            continue
        bindings.append(tuple((index, weight / total) for index, weight in raw))
    return tuple(bindings)


def skin_mesh(
    mesh: Mesh,
    bind_matrices: dict[str, Mat4],
    pose_matrices: dict[str, Mat4],
    bindings: Sequence[Sequence[tuple[int, float]]],
    bone_order: Sequence[str],
) -> tuple[Vec3, ...]:
    """Deform a mesh with linear blend skinning and return its posed vertices.

    Each vertex is the weighted average of its influencing bones' transforms
    applied to its rest position, where a bone's transform is
    ``pose @ inverse(bind)`` -- so a vertex on a bone that has not moved stays
    exactly where it was.
    """
    deformation: dict[str, Mat4] = {}
    for name in bone_order:
        bind = bind_matrices.get(name)
        pose = pose_matrices.get(name)
        if bind is None or pose is None:
            continue
        try:
            deformation[name] = pose @ bind.inverse()
        except ValueError:
            deformation[name] = pose
    posed: list[Vec3] = []
    for index, vertex in enumerate(mesh.vertices):
        influences = bindings[index] if index < len(bindings) else ()
        if not influences:
            posed.append(vertex)
            continue
        accumulated = Vec3()
        for bone_index, weight in influences:
            if weight == 0.0:
                continue
            matrix = deformation.get(bone_order[bone_index])
            if matrix is None:
                accumulated = accumulated + vertex * weight
                continue
            accumulated = accumulated + matrix.transform_point(vertex) * weight
        posed.append(accumulated)
    return tuple(posed)


def skin_normals(
    normals: Sequence[Vec3],
    bind_matrices: dict[str, Mat4],
    pose_matrices: dict[str, Mat4],
    bindings: Sequence[Sequence[tuple[int, float]]],
    bone_order: Sequence[str],
) -> tuple[Vec3, ...]:
    """Rotate a mesh's normals with the same skinning, so shading follows the pose.

    A direction is transformed by the bone's rotation only (its translation does
    not move a normal), which is the usual approximation and is exact for the
    rigid, un-scaled bones this studio builds.
    """
    rotations: dict[str, Mat4] = {}
    for name in bone_order:
        pose = pose_matrices.get(name)
        if pose is not None:
            rotations[name] = pose
    posed: list[Vec3] = []
    for index, normal in enumerate(normals):
        influences = bindings[index] if index < len(bindings) else ()
        if not influences:
            posed.append(normal)
            continue
        accumulated = Vec3()
        for bone_index, weight in influences:
            matrix = rotations.get(bone_order[bone_index])
            if matrix is None:
                accumulated = accumulated + normal * weight
                continue
            accumulated = accumulated + matrix.transform_direction(normal) * weight
        posed.append(accumulated.normalized())
    return tuple(posed)


# --------------------------------------------------------------------------- scene


@dataclass(frozen=True)
class Material:
    """A simple Blinn-Phong surface: base colour, specular tint, and shininess."""

    name: str = "default"
    base_color: tuple[float, float, float] = (0.82, 0.66, 0.56)
    specular: float = 0.25
    shininess: float = 24.0
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("Material opacity must be in [0, 1].")
        if self.shininess <= 0.0:
            raise ValueError("Material shininess must be positive.")


@dataclass(frozen=True)
class Transform:
    """A translation, an XYZ euler rotation in degrees, and a scale."""

    translation: Vec3 = Vec3()
    rotation: Vec3 = Vec3()
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)

    def to_matrix(self) -> Mat4:
        return (
            Mat4.translation(self.translation)
            @ Mat4.euler(self.rotation)
            @ Mat4.scaling(self.scale)
        )


@dataclass(frozen=True)
class Camera:
    """A perspective camera aimed at a target, with a bounded depth range."""

    position: Vec3 = Vec3(0.0, 1.4, 3.6)
    target: Vec3 = Vec3(0.0, 1.0, 0.0)
    up: Vec3 = UP
    fov_degrees: float = 35.0
    near: float = 0.05
    far: float = 100.0

    def view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, self.up)

    def projection_matrix(self, aspect: float) -> Mat4:
        return Mat4.perspective(self.fov_degrees, aspect, self.near, self.far)


@dataclass(frozen=True)
class Light:
    """A directional or point light with a colour and an intensity."""

    name: str = "key"
    kind: str = "directional"  # "directional" or "point"
    direction: Vec3 = Vec3(-0.4, -0.8, -0.5)
    position: Vec3 = Vec3(2.0, 3.0, 2.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"directional", "point"}:
            raise ValueError(f"Unknown light kind: {self.kind!r}.")
        if self.intensity < 0.0:
            raise ValueError("A light's intensity cannot be negative.")

    def direction_to_light(self, point: Vec3) -> Vec3:
        """The unit vector from a surface point towards this light."""
        if self.kind == "point":
            return (self.position - point).normalized()
        return (-self.direction).normalized()


@dataclass
class SceneObject:
    """A mesh in the scene, with a material, a transform, and an optional rig.

    When ``skeleton`` is set the object is a *figure*: its vertices are bound to
    the skeleton once (``bindings``) and re-deformed from ``pose`` on every
    render, so a pose change moves the mesh.
    """

    name: str
    mesh: Mesh
    material: Material = Material()
    transform: Transform = Transform()
    skeleton: Skeleton | None = None
    pose: dict[str, Vec3] = field(default_factory=dict)
    bindings: tuple[tuple[tuple[int, float], ...], ...] = ()
    visible: bool = True

    def is_figure(self) -> bool:
        return self.skeleton is not None and bool(self.bindings)

    def bind(self, *, max_influences: int = MAX_INFLUENCES) -> SceneObject:
        """Bind the mesh to the skeleton at its current (rest) pose."""
        if self.skeleton is None:
            self.bindings = ()
            return self
        self.bindings = compute_skin_weights(
            self.mesh, self.skeleton, max_influences=max_influences
        )
        return self

    def deformed_vertices(self) -> tuple[Vec3, ...]:
        """The mesh's vertices after skinning the current pose (no object transform)."""
        if not self.is_figure() or self.skeleton is None:
            return tuple(self.mesh.vertices)
        posed = self.skeleton.with_pose(self.pose)
        return skin_mesh(
            self.mesh,
            self.skeleton.world_matrices(),
            posed.world_matrices(),
            self.bindings,
            self.skeleton.names(),
        )

    def deformed_normals(self) -> tuple[Vec3, ...]:
        """The mesh's vertex normals after skinning (shading follows the pose)."""
        normals = self.mesh.vertex_normals()
        if not self.is_figure() or self.skeleton is None:
            return normals
        posed = self.skeleton.with_pose(self.pose)
        return skin_normals(
            normals,
            self.skeleton.world_matrices(),
            posed.world_matrices(),
            self.bindings,
            self.skeleton.names(),
        )

    def world_vertices(self) -> tuple[Vec3, ...]:
        """The posed vertices with the object's own transform applied."""
        matrix = self.transform.to_matrix()
        return tuple(matrix.transform_point(vertex) for vertex in self.deformed_vertices())

    def world_normals(self) -> tuple[Vec3, ...]:
        """The posed normals rotated by the object's transform."""
        matrix = self.transform.to_matrix()
        return tuple(
            matrix.transform_direction(normal).normalized()
            for normal in self.deformed_normals()
        )

    def set_pose(self, bone: str, rotation: Vec3) -> None:
        """Set one bone's pose rotation (raising on an unknown bone name)."""
        if self.skeleton is None:
            raise ValueError(f"{self.name!r} has no skeleton to pose.")
        self.skeleton.bone(bone)  # validates the name
        self.pose[bone] = rotation

    def clear_pose(self) -> None:
        """Return the figure to its rest pose."""
        self.pose = {}


@dataclass
class Scene:
    """A complete scene: objects, one camera, lights, and a render size."""

    name: str = "Untitled scene"
    objects: list[SceneObject] = field(default_factory=list)
    camera: Camera = Camera()
    lights: list[Light] = field(default_factory=list)
    background: tuple[float, float, float] = (0.09, 0.11, 0.15)
    width: int = 800
    height: int = 600

    def add_object(self, obj: SceneObject) -> SceneObject:
        """Add an object, replacing any existing object with the same name."""
        self.remove_object(obj.name)
        self.objects.append(obj)
        return obj

    def remove_object(self, name: str) -> bool:
        before = len(self.objects)
        self.objects = [obj for obj in self.objects if obj.name != name]
        return len(self.objects) != before

    def find_object(self, name: str) -> SceneObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(name)

    def object_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects)

    def visible_objects(self) -> tuple[SceneObject, ...]:
        return tuple(obj for obj in self.objects if obj.visible)

    def bounds(self) -> tuple[Vec3, Vec3]:
        """The world-space bounding box of every visible object."""
        lows: list[Vec3] = []
        highs: list[Vec3] = []
        for obj in self.visible_objects():
            vertices = obj.world_vertices()
            if not vertices:
                continue
            low = high = vertices[0]
            for vertex in vertices[1:]:
                low = Vec3(min(low.x, vertex.x), min(low.y, vertex.y), min(low.z, vertex.z))
                high = Vec3(max(high.x, vertex.x), max(high.y, vertex.y), max(high.z, vertex.z))
            lows.append(low)
            highs.append(high)
        if not lows:
            return (Vec3(), Vec3())
        low = lows[0]
        high = highs[0]
        for candidate in lows[1:]:
            low = Vec3(min(low.x, candidate.x), min(low.y, candidate.y), min(low.z, candidate.z))
        for candidate in highs[1:]:
            high = Vec3(max(high.x, candidate.x), max(high.y, candidate.y), max(high.z, candidate.z))
        return (low, high)

    def center_and_radius(self) -> tuple[Vec3, float]:
        """The bounding sphere of the scene, used to frame the camera on it."""
        low, high = self.bounds()
        center = (low + high) / 2.0
        radius = (high - low).length() / 2.0
        return (center, radius if radius > 1e-6 else 1.0)

    def frame_camera(self, *, margin: float = 1.6, direction: Vec3 | None = None) -> None:
        """Point the camera at the scene from a distance that fits it.

        The camera keeps its current aim direction (or takes ``direction``) and is
        pulled back until the whole bounding sphere fits the field of view -- what
        a "frame all" button does in any 3D program.
        """
        center, radius = self.center_and_radius()
        aim = direction or (self.camera.position - self.camera.target)
        aim = aim.normalized() if aim.length() > 1e-9 else Vec3(0.0, 0.35, 1.0).normalized()
        half_fov = math.radians(max(1.0, self.camera.fov_degrees)) / 2.0
        distance = (radius * margin) / max(0.05, math.sin(half_fov))
        self.camera = Camera(
            position=center + aim * distance,
            target=center,
            up=self.camera.up,
            fov_degrees=self.camera.fov_degrees,
            near=max(0.01, distance - radius * 4.0),
            far=max(self.camera.far, distance + radius * 8.0),
        )

    def summary(self) -> str:
        """A short, honest description of what the scene holds."""
        figures = sum(1 for obj in self.objects if obj.is_figure())
        triangles = sum(obj.mesh.triangle_count for obj in self.objects)
        return (
            f"{self.name}: {len(self.objects)} object(s), {figures} figure(s), "
            f"{triangles} triangle(s), {len(self.lights)} light(s), "
            f"{self.width}x{self.height}."
        )


def make_default_scene(name: str = "Untitled scene") -> Scene:
    """A scene with a three-point light rig, ready to render."""
    return Scene(
        name=name,
        lights=[
            Light("key", "directional", Vec3(-0.45, -0.85, -0.55), intensity=1.0),
            Light("fill", "directional", Vec3(0.7, -0.3, -0.4), intensity=0.35),
            Light(
                "rim",
                "point",
                position=Vec3(-2.2, 2.6, -2.0),
                color=(0.85, 0.9, 1.0),
                intensity=0.55,
            ),
        ],
    )


def iter_triangles(mesh: Mesh) -> Iterable[tuple[int, int, int]]:
    """Yield a mesh's faces as triangles (fan-triangulating any n-gon)."""
    for face in mesh.faces:
        for index in range(1, len(face) - 1):
            yield (face[0], face[index], face[index + 1])


__all__ = [
    "IDENTITY_MATRIX",
    "MAX_INFLUENCES",
    "ORIGIN",
    "UP",
    "Bone",
    "Camera",
    "Light",
    "Mat4",
    "Material",
    "Mesh",
    "Scene",
    "SceneObject",
    "Skeleton",
    "Transform",
    "Vec3",
    "compute_skin_weights",
    "iter_triangles",
    "make_default_scene",
    "skin_mesh",
    "skin_normals",
    "vec3",
]
