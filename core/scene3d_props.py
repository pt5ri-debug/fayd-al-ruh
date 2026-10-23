"""Primitive props and a material palette, so a scene can hold more than a figure.

A figure studio needs a *set*: a floor to stand on, a box or a sphere to pose
against, a column, a platform. This module builds those primitives procedurally
(a cube, a sphere, a cylinder, a plane, a cone) with the same dependency-free,
deterministic approach as :mod:`core.scene3d_figures`, plus a small named
material palette and a **prop catalog** the menu can offer.

Everything is generated from parameters -- no asset files, no downloads, no
licensing -- and every primitive is an ordinary :class:`core.scene3d.SceneObject`
that can be transformed, duplicated, and rendered like anything else.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from core.scene3d import Material, Mesh, SceneObject, Transform, Vec3

#: Segments around a revolved primitive; the same compromise as the figures.
SEGMENTS = 16
#: Latitude bands on a sphere.
RINGS = 10


def make_cube(name: str = "Cube", *, size: float = 1.0) -> Mesh:
    """An axis-aligned cube centred on the origin."""
    half = size / 2.0
    vertices = tuple(
        Vec3(x * half, y * half, z * half)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    )
    # Indices follow the (x, y, z) sign ordering above.
    def index(x: int, y: int, z: int) -> int:
        return (x * 4) + (y * 2) + z

    faces = (
        (index(0, 0, 0), index(0, 1, 0), index(0, 1, 1), index(0, 0, 1)),
        (index(1, 0, 1), index(1, 1, 1), index(1, 1, 0), index(1, 0, 0)),
        (index(0, 0, 0), index(0, 0, 1), index(1, 0, 1), index(1, 0, 0)),
        (index(0, 1, 1), index(0, 1, 0), index(1, 1, 0), index(1, 1, 1)),
        (index(0, 0, 0), index(1, 0, 0), index(1, 1, 0), index(0, 1, 0)),
        (index(0, 0, 1), index(0, 1, 1), index(1, 1, 1), index(1, 0, 1)),
    )
    return Mesh(name, vertices, faces)


def make_sphere(name: str = "Sphere", *, radius: float = 0.5, rings: int = RINGS, segments: int = SEGMENTS) -> Mesh:
    """A UV sphere centred on the origin."""
    if radius <= 0:
        raise ValueError("A sphere needs a positive radius.")
    vertices: list[Vec3] = [Vec3(0.0, radius, 0.0)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        y = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                Vec3(ring_radius * math.cos(theta), y, ring_radius * math.sin(theta))
            )
    bottom = len(vertices)
    vertices.append(Vec3(0.0, -radius, 0.0))

    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((0, 1 + nxt, 1 + segment))
    for ring in range(rings - 2):
        base = 1 + ring * segments
        lower = base + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((base + segment, base + nxt, lower + nxt, lower + segment))
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((bottom, last + segment, last + nxt))
    return Mesh(name, tuple(vertices), tuple(faces))


def make_cylinder(
    name: str = "Cylinder",
    *,
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = SEGMENTS,
) -> Mesh:
    """A capped cylinder centred on the origin, its axis along ``Y``."""
    if radius <= 0 or height <= 0:
        raise ValueError("A cylinder needs a positive radius and height.")
    half = height / 2.0
    vertices: list[Vec3] = []
    for y in (-half, half):
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(Vec3(radius * math.cos(theta), y, radius * math.sin(theta)))
    bottom_center = len(vertices)
    vertices.append(Vec3(0.0, -half, 0.0))
    top_center = len(vertices)
    vertices.append(Vec3(0.0, half, 0.0))

    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((segment, nxt, segments + nxt, segments + segment))
        faces.append((bottom_center, nxt, segment))
        faces.append((top_center, segments + segment, segments + nxt))
    return Mesh(name, tuple(vertices), tuple(faces))


def make_cone(name: str = "Cone", *, radius: float = 0.5, height: float = 1.0, segments: int = SEGMENTS) -> Mesh:
    """A cone with its apex up, centred on the origin."""
    if radius <= 0 or height <= 0:
        raise ValueError("A cone needs a positive radius and height.")
    half = height / 2.0
    vertices = [
        Vec3(radius * math.cos(2.0 * math.pi * s / segments), -half, radius * math.sin(2.0 * math.pi * s / segments))
        for s in range(segments)
    ]
    apex = len(vertices)
    vertices.append(Vec3(0.0, half, 0.0))
    center = len(vertices)
    vertices.append(Vec3(0.0, -half, 0.0))
    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((apex, nxt, segment))
        faces.append((center, segment, nxt))
    return Mesh(name, tuple(vertices), tuple(faces))


def make_plane(name: str = "Ground", *, size: float = 10.0, subdivisions: int = 1) -> Mesh:
    """A flat plane on ``Y = 0``, useful as a floor.

    ``subdivisions`` splits it into a grid, which matters when a plane is used as
    a large ground surface with vertex shading.
    """
    if size <= 0:
        raise ValueError("A plane needs a positive size.")
    count = max(1, subdivisions)
    step = size / count
    start = -size / 2.0
    vertices: list[Vec3] = []
    for row in range(count + 1):
        for column in range(count + 1):
            vertices.append(Vec3(start + column * step, 0.0, start + row * step))
    faces: list[tuple[int, ...]] = []
    for row in range(count):
        for column in range(count):
            a = row * (count + 1) + column
            b = a + 1
            c = a + (count + 1) + 1
            d = a + (count + 1)
            faces.append((a, d, c, b))
    return Mesh(name, tuple(vertices), tuple(faces))


#: The primitive builders, keyed by the name the menu and the CLI use.
PRIMITIVES: dict[str, Callable[..., Mesh]] = {
    "cube": make_cube,
    "sphere": make_sphere,
    "cylinder": make_cylinder,
    "cone": make_cone,
    "plane": make_plane,
}


def primitive_names() -> tuple[str, ...]:
    return tuple(PRIMITIVES)


def build_primitive(kind: str, *, name: str | None = None, **options: float) -> Mesh:
    """Build one primitive by name, raising on an unknown kind."""
    try:
        builder = PRIMITIVES[kind]
    except KeyError as error:
        raise KeyError(
            f"Unknown primitive {kind!r}; choose from {', '.join(PRIMITIVES)}."
        ) from error
    return builder(name or kind.capitalize(), **options)


# --------------------------------------------------------------------------- materials

#: A small, named material palette, so a scene reads clearly without the user
#: picking numbers. Colours are linear RGB in ``[0, 1]``.
MATERIAL_PALETTE: dict[str, Material] = {
    "skin": Material("skin", (0.82, 0.66, 0.56), 0.18, 16.0),
    "cloth": Material("cloth", (0.28, 0.34, 0.55), 0.10, 12.0),
    "metal": Material("metal", (0.72, 0.74, 0.78), 0.75, 96.0),
    "gold": Material("gold", (0.83, 0.68, 0.21), 0.85, 128.0),
    "wood": Material("wood", (0.45, 0.31, 0.20), 0.12, 18.0),
    "stone": Material("stone", (0.55, 0.55, 0.54), 0.08, 10.0),
    "ground": Material("ground", (0.30, 0.32, 0.36), 0.05, 8.0),
    "glass": Material("glass", (0.70, 0.82, 0.88), 0.90, 160.0, 0.35),
    "red": Material("red", (0.72, 0.22, 0.20), 0.25, 32.0),
    "green": Material("green", (0.24, 0.60, 0.34), 0.25, 32.0),
    "blue": Material("blue", (0.24, 0.40, 0.75), 0.25, 32.0),
}


def material_names() -> tuple[str, ...]:
    return tuple(MATERIAL_PALETTE)


def material_for(key: str) -> Material:
    """Look up a palette material by name (``default`` gives a neutral surface)."""
    if key == "default":
        return Material()
    try:
        return MATERIAL_PALETTE[key]
    except KeyError as error:
        raise KeyError(
            f"Unknown material {key!r}; choose from {', '.join(MATERIAL_PALETTE)}."
        ) from error


# --------------------------------------------------------------------------- catalog

@dataclass(frozen=True)
class PropSpec:
    """A named prop the menu can add: a primitive plus a material and a placement."""

    key: str
    title: str
    primitive: str
    material: str
    translation: Vec3 = Vec3()
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)
    summary: str = ""


#: The props the studio offers by name. A floor and a few blocks are enough to
#: build a simple set for a figure to stand in.
PROP_CATALOG: tuple[PropSpec, ...] = (
    PropSpec(
        "ground",
        "Ground plane",
        "plane",
        "ground",
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 1.0, 1.0),
        "A 10 m floor at y = 0, for a figure to stand on.",
    ),
    PropSpec(
        "platform",
        "Platform",
        "cylinder",
        "stone",
        Vec3(0.0, 0.15, 0.0),
        Vec3(1.4, 0.3, 1.4),
        "A low round platform.",
    ),
    PropSpec(
        "cube",
        "Cube",
        "cube",
        "wood",
        Vec3(0.0, 0.5, 0.0),
        Vec3(1.0, 1.0, 1.0),
        "A 1 m cube.",
    ),
    PropSpec(
        "sphere",
        "Sphere",
        "sphere",
        "metal",
        Vec3(0.0, 0.5, 0.0),
        Vec3(1.0, 1.0, 1.0),
        "A 0.5 m radius sphere.",
    ),
    PropSpec(
        "column",
        "Column",
        "cylinder",
        "stone",
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.3, 2.0, 0.3),
        "A tall column.",
    ),
    PropSpec(
        "backdrop",
        "Backdrop",
        "plane",
        "blue",
        Vec3(0.0, 1.5, -1.5),
        Vec3(4.0, 3.0, 1.0),
        "A large flat backdrop.",
    ),
)


def prop_names() -> tuple[str, ...]:
    return tuple(spec.key for spec in PROP_CATALOG)


def prop_spec(key: str) -> PropSpec:
    for spec in PROP_CATALOG:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown prop {key!r}; choose from {', '.join(prop_names())}.")


def build_prop(key: str, *, name: str | None = None) -> SceneObject:
    """Build a catalog prop as a ready scene object."""
    spec = prop_spec(key)
    mesh = build_primitive(spec.primitive, name=name or spec.title)
    return SceneObject(
        name or spec.title,
        mesh,
        material_for(spec.material),
        Transform(spec.translation, Vec3(), spec.scale),
    )


def default_ground() -> SceneObject:
    """A ground plane, the one prop almost every scene wants."""
    return build_prop("ground", name="Ground")


__all__ = [
    "MATERIAL_PALETTE",
    "PRIMITIVES",
    "PROP_CATALOG",
    "RINGS",
    "SEGMENTS",
    "PropSpec",
    "build_primitive",
    "build_prop",
    "default_ground",
    "make_cone",
    "make_cube",
    "make_cylinder",
    "make_plane",
    "make_sphere",
    "material_for",
    "material_names",
    "primitive_names",
    "prop_names",
    "prop_spec",
]
