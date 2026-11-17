"""Built-in, procedurally generated figures -- so the studio works with no assets.

A DAZ-style studio is only useful if there is something to pose. Licensing real
figure content is not possible here, so this module **builds** the figures: a
parametric humanoid mesh (torso, neck, head, arms, hands, legs, feet, and a simple
face) generated from named measurements, and a matching **bone hierarchy** bound
to it with the real skinning in :mod:`core.scene3d`. The result is a genuinely
poseable figure that ships with the code, needs no download, and is honest about
being a *stylised* figure rather than a licensed likeness.

The generator is deterministic: the same parameters always produce the same
figure, so a saved scene reloads identically. The proportions come from named
parameters -- a **build** family (feminine / masculine / neutral), a **proportion**
style (average / athletic / slim / curvy / child), body measurements, and a set of
**shape** dials (bust, waist, hips, belly, muscle, shoulder slope, neck, arm and
leg thickness, hand and foot size, head and face features) -- which is what makes
the figures "customisable" the way a DAZ figure's morph dials are.

The hair is built as a **separate object** (see :func:`build_hair_object`) so it can
carry its own material: a figure's hair colour is then real rather than the skin
colour. :func:`build_figure` returns the body alone, exactly as before, and
:func:`build_figure_scene` adds the hair for you.

Nothing here is a claim about a commercial product: it is an original, simple,
parametric humanoid, and every figure it makes says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

from core.scene3d import (
    Bone,
    Material,
    Mesh,
    Scene,
    SceneObject,
    Skeleton,
    Vec3,
    make_default_scene,
)

#: Segments around a limb's circumference. Sixteen reads as a smooth silhouette at
#: the render sizes this studio uses while keeping the mesh small -- the rasteriser
#: is fill-bound, not triangle-bound, so the extra detail is nearly free and the
#: low-poly "wooden" look of eight is gone.
RADIAL_SEGMENTS = 16

#: The build families a figure can belong to.
BUILD_FEMININE = "feminine"
BUILD_MASCULINE = "masculine"
BUILD_NEUTRAL = "neutral"
BUILDS: tuple[str, ...] = (BUILD_FEMININE, BUILD_MASCULINE, BUILD_NEUTRAL)


@dataclass(frozen=True)
class FigureParameters:
    """The "morph dials": named measurements that shape one figure.

    All lengths are in metres, with the figure standing on ``y = 0`` and facing
    ``+Z``. The defaults describe an adult of average build.

    The first group (``height`` .. ``skin``) is the original torso/limb set. The
    second group (``build`` .. ``hair_color``) is the fuller **body-shape** set: a
    build family, a proportion style, the bust/waist/hip/belly/muscle shaping, the
    shoulder slope, neck and limb thicknesses, hand and foot size, the head and
    face features, and the hair colour. Every new field has a default that leaves
    the original figure's geometry unchanged, so existing callers and saved scenes
    keep working.
    """

    name: str = "Figure"
    height: float = 1.75
    shoulder_width: float = 0.42
    hip_width: float = 0.30
    chest_depth: float = 0.22
    waist: float = 0.26
    limb_thickness: float = 1.0
    head_scale: float = 1.0
    skin: tuple[float, float, float] = (0.82, 0.66, 0.56)

    # -- body shape ------------------------------------------------------
    build: str = BUILD_NEUTRAL
    proportion: str = "average"
    bust: float = 0.0
    waist_shape: float = 0.0
    hip_shape: float = 0.0
    belly: float = 0.0
    muscle: float = 0.0
    shoulder_slope: float = 0.0
    neck_thickness: float = 1.0
    arm_thickness: float = 1.0
    leg_thickness: float = 1.0
    hand_size: float = 1.0
    foot_size: float = 1.0
    # -- head, face and hair --------------------------------------------
    face_length: float = 1.0
    face_width: float = 1.0
    jaw_width: float = 1.0
    chin_shape: float = 0.0
    brow_ridge: float = 0.0
    nose_size: float = 1.0
    eye_size: float = 1.0
    eye_spacing: float = 1.0
    lips: float = 1.0
    cheek: float = 0.0
    ear_size: float = 1.0
    hair: float = 1.0
    hair_color: tuple[float, float, float] = (0.20, 0.14, 0.10)
    # -- per-part shape and colour --------------------------------------
    #: The chest's areola size (a dial), and the colour of the lips, the irises,
    #: the areolae, the fingernails/toenails, and the genitalia. Each is a real
    #: per-vertex colour on the one skinned mesh, so it follows a pose.
    areola: float = 1.0
    lip_color: tuple[float, float, float] = (0.62, 0.30, 0.30)
    eye_color: tuple[float, float, float] = (0.22, 0.34, 0.42)
    nipple_color: tuple[float, float, float] = (0.60, 0.34, 0.32)
    nail_color: tuple[float, float, float] = (0.90, 0.82, 0.80)
    genitalia: float = 1.0
    genital_size: float = 1.0
    genital_color: tuple[float, float, float] = (0.60, 0.40, 0.38)
    #: How many Catmull-Clark subdivision passes smooth the mesh (0 = the raw
    #: primitives, 1 = the default fine, round skin, 2 = very fine). The mesh is
    #: welded first so the pass smooths *across* the seams between primitives
    #: rather than leaving a crease, which is what turns a faceted figure into a
    #: smooth, human-looking one.
    subdivisions: int = 1
    #: Laplacian smoothing passes applied after subdivision (a cheap relax that
    #: removes the last facets a single pass can leave).
    smoothing: int = 1
    #: The figure's pose, kept with the body so a saved body remembers how it stood.
    pose: str = "rest"

    def __post_init__(self) -> None:
        if not 0.8 <= self.height <= 2.6:
            raise ValueError("Figure height must be between 0.8 and 2.6 metres.")
        for field_name in (
            "shoulder_width",
            "hip_width",
            "chest_depth",
            "waist",
            "limb_thickness",
            "head_scale",
        ):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ValueError(f"Figure {field_name} must be positive.")
        if self.build not in BUILDS:
            raise ValueError(f"Unknown build {self.build!r}; choose from {', '.join(BUILDS)}.")
        for field_name in (
            "neck_thickness",
            "arm_thickness",
            "leg_thickness",
            "hand_size",
            "foot_size",
            "face_length",
            "face_width",
            "jaw_width",
            "nose_size",
            "eye_size",
            "eye_spacing",
            "lips",
            "ear_size",
            "areola",
            "genital_size",
        ):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ValueError(f"Figure {field_name} must be positive.")
        if not 0.0 <= self.genitalia <= 1.0:
            raise ValueError("The genitalia dial must be in [0, 1].")
        if not 0 <= self.subdivisions <= 2:
            raise ValueError("Subdivisions must be 0, 1, or 2.")
        if not 0 <= self.smoothing <= 4:
            raise ValueError("Smoothing passes must be between 0 and 4.")

    # -- convenience -----------------------------------------------------

    def with_changes(self, **changes: object) -> FigureParameters:
        """A copy with the named dials changed (used by the body studio)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def describe(self) -> str:
        """A one-line summary of the shape."""
        return f"{self.build} build, {self.proportion} proportions, {self.height:.2f} m"


class _Body:
    """A growing mesh that carries a **per-vertex colour** as it is built.

    Every primitive appends vertices together with the colour they should be shaded
    with, so one skinned mesh can hold the skin tone *and* the lips, the irises, the
    areolae, the nails, and the genitalia -- a separate object per colour would not
    follow a pose, because only the body mesh is bound to the skeleton. The colour is
    a plain RGB triple the rasteriser interpolates across each triangle, which is
    what turns a flat "wooden" figure into one with real tonal detail.
    """

    def __init__(self, default_color: tuple[float, float, float]) -> None:
        self.vertices: list[Vec3] = []
        self.faces: list[tuple[int, ...]] = []
        self.colors: list[tuple[float, float, float]] = []
        self.default_color = default_color

    # -- primitives -------------------------------------------------------

    def _add(self, points: Sequence[Vec3], color: tuple[float, float, float]) -> list[int]:
        indices: list[int] = []
        for point in points:
            indices.append(len(self.vertices))
            self.vertices.append(point)
            self.colors.append(color)
        return indices

    def ring(
        self,
        center: Vec3,
        radius_x: float,
        radius_z: float,
        color: tuple[float, float, float],
        *,
        segments: int = RADIAL_SEGMENTS,
    ) -> list[int]:
        """An ellipse of points in the XZ plane; returns their vertex indices."""
        points = [
            Vec3(
                center.x + radius_x * math.cos(2.0 * math.pi * index / segments),
                center.y,
                center.z + radius_z * math.sin(2.0 * math.pi * index / segments),
            )
            for index in range(segments)
        ]
        return self._add(points, color)

    def loft(self, rings: Sequence[Sequence[int]]) -> None:
        """Stitch consecutive rings (each already added) into a tube of quads."""
        segments = len(rings[0])
        for lower, upper in pairwise(rings):
            for index in range(segments):
                nxt = (index + 1) % segments
                self.faces.append((lower[index], lower[nxt], upper[nxt], upper[index]))

    def cap(self, ring: Sequence[int], center: Vec3, color: tuple[float, float, float]) -> None:
        """Close a ring with a fan to ``center``."""
        apex = self._add([center], color)[0]
        count = len(ring)
        for index in range(count):
            nxt = (index + 1) % count
            self.faces.append((apex, ring[nxt], ring[index]))

    def box(self, center: Vec3, size: Vec3, color: tuple[float, float, float]) -> None:
        """An axis-aligned box (a palm, a foot)."""
        half = Vec3(size.x * 0.5, size.y * 0.5, size.z * 0.5)
        corners = [
            Vec3(center.x + sx * half.x, center.y + sy * half.y, center.z + sz * half.z)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        base = self._add(corners, color)

        def corner(sx: int, sy: int, sz: int) -> int:
            return base[sx * 4 + sy * 2 + sz]

        self.faces.extend(
            [
                (corner(0, 0, 0), corner(0, 0, 1), corner(0, 1, 1), corner(0, 1, 0)),
                (corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, 0, 1)),
                (corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)),
                (corner(0, 1, 0), corner(0, 1, 1), corner(1, 1, 1), corner(1, 1, 0)),
                (corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 0), corner(1, 0, 0)),
                (corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)),
            ]
        )

    def ellipsoid(
        self,
        center: Vec3,
        radii: Vec3,
        color: tuple[float, float, float],
        *,
        segments: int = RADIAL_SEGMENTS,
        latitude: int = 6,
    ) -> None:
        """A smooth ellipsoid (a nose, an eye, a breast)."""
        if min(radii.x, radii.y, radii.z) <= 0.0:
            return
        rings: list[list[int]] = []
        for step in range(latitude, -1, -1):
            phi = math.pi * step / latitude
            ring_radius = math.sin(phi)
            y = center.y + radii.y * math.cos(phi)
            points = [
                Vec3(
                    center.x + radii.x * ring_radius * math.cos(2.0 * math.pi * i / segments),
                    y,
                    center.z + radii.z * ring_radius * math.sin(2.0 * math.pi * i / segments),
                )
                for i in range(segments)
            ]
            rings.append(self._add(points, color))
        self.loft(rings)

    def tube(
        self,
        centers: Sequence[Vec3],
        radii: Sequence[tuple[float, float]],
        color: tuple[float, float, float],
        *,
        segments: int = RADIAL_SEGMENTS,
    ) -> None:
        """A stack of rings along a path, one ring per ``(radius_x, radius_z)``."""
        rings = [
            self.ring(center, rx, rz, color, segments=segments)
            for center, (rx, rz) in zip(centers, radii, strict=True)
        ]
        self.loft(rings)

    def to_mesh(self, name: str) -> Mesh:
        return Mesh(name, tuple(self.vertices), tuple(self.faces), (), tuple(self.colors))


def _build_torso(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """A torso as stacked elliptical rings from hips to shoulders.

    The rings carry the build and the shape dials: the bust widens the chest and
    pushes it forward, the waist and hip dials pinch or widen, the belly pushes the
    lower rings forward, and the muscle dial thickens the chest.
    """
    height = params.height
    scale = params.limb_thickness
    bust = params.bust
    hips = params.hip_width * (1.0 + 0.45 * params.hip_shape)
    waist_r = params.waist * (1.0 - 0.30 * params.waist_shape)
    chest_r = params.shoulder_width * (0.42 + 0.10 * params.muscle)
    chest_z = params.chest_depth * (0.55 + 0.30 * bust)
    belly_z = params.chest_depth * (0.45 + 0.35 * params.belly)

    rings = [
        body.ring(Vec3(0, height * 0.50, 0.0), hips * 0.5, params.chest_depth * 0.60, skin),
        body.ring(
            Vec3(0, height * 0.60, belly_z * 0.25),
            waist_r * 0.5,
            params.chest_depth * 0.42 + belly_z * 0.4,
            skin,
        ),
        body.ring(Vec3(0, height * 0.70, chest_z * 0.22), chest_r, chest_z, skin),
        body.ring(
            Vec3(0, height * 0.81, 0.0),
            params.shoulder_width * 0.5 * scale,
            params.chest_depth * 0.50,
            skin,
        ),
    ]
    body.loft(rings)
    _build_chest(body, params, skin)


def _build_chest(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """Breasts (with areolae) when the bust dial is raised, else a flat chest."""
    if params.bust <= 0.02:
        return
    height = params.height
    radius = 0.030 + 0.075 * params.bust
    forward = params.chest_depth * 0.55 + 0.05 * params.bust
    for side in (-1.0, 1.0):
        center = Vec3(side * params.shoulder_width * 0.20, height * 0.695, forward)
        body.ellipsoid(center, Vec3(radius * 0.85, radius * 0.80, radius), skin)
        body.ellipsoid(
            Vec3(center.x, center.y + radius * 0.10, center.z + radius * 0.72),
            Vec3(radius * 0.34 * params.areola, radius * 0.34 * params.areola, radius * 0.16),
            params.nipple_color,
        )


def _build_neck(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """A short neck between the shoulders and the jaw."""
    height = params.height
    radius = 0.052 * params.neck_thickness
    body.tube(
        [Vec3(0, height * 0.815, 0.0), Vec3(0, height * 0.855, 0.0)],
        [(radius * 1.25, radius * 1.15), (radius, radius * 0.95)],
        skin,
    )


def _build_limb(
    body: _Body,
    params: FigureParameters,
    skin: tuple[float, float, float],
    *,
    top: Vec3,
    joints: Sequence[tuple[float, float]],
    end_radius: float,
) -> None:
    """A limb as rings around ``top`` descending along ``-Y``.

    ``joints`` is a list of ``(length, radius)`` pairs, one per bone segment, so
    the limb tapers the way the skeleton bends. Rings are emitted **bottom-first**
    so the loft's winding faces outwards, which is what the rasteriser's backface
    culling expects.
    """
    positions: list[Vec3] = []
    position = top
    for length, _radius in joints:
        positions.append(position)
        position = Vec3(position.x, position.y - length, position.z)
    positions.append(position)
    radii = [radius for _length, radius in joints]
    radii.append(end_radius)
    body.tube(
        list(reversed(positions)),
        [(radius, radius) for radius in reversed(radii)],
        skin,
        segments=RADIAL_SEGMENTS,
    )


def _build_head(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """A head as stacked rings from the jaw to the crown, shaped by the face dials."""
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    jaw = width * params.jaw_width
    chin = 1.0 - 0.35 * params.chin_shape

    rings = [
        body.ring(Vec3(0, base_y, 0.0), jaw * 0.62 * chin, depth * 0.60, skin),
        body.ring(Vec3(0, base_y + length * 0.16, 0.006), jaw * 0.95, depth * 0.95, skin),
        body.ring(
            Vec3(0, base_y + length * 0.42, 0.010),
            width * (1.0 + 0.10 * params.cheek),
            depth * 1.02,
            skin,
        ),
        body.ring(Vec3(0, base_y + length * 0.68, 0.004), width * 0.98, depth * 0.98, skin),
        body.ring(Vec3(0, base_y + length * 0.88, 0.0), width * 0.72, depth * 0.72, skin),
    ]
    body.loft(rings)
    body.cap(rings[-1], Vec3(0, base_y + length, 0.0), skin)


def _build_face(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """The face features: eyes, a nose, lips, ears, and eyebrows.

    Low-poly ellipsoids in the head's front half, each carrying its own colour --
    so the irises, the lips, and the brows read as real features rather than the
    same flat tone as the face.
    """
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    front = depth * 0.62

    eye_y = base_y + length * 0.60
    eye_x = width * 0.40 * params.eye_spacing
    eye_r = 0.013 * params.head_scale * params.eye_size
    for side in (-1.0, 1.0):
        # The white of the eye, then the iris just in front of it.
        body.ellipsoid(Vec3(side * eye_x, eye_y, front * 0.70), Vec3(eye_r, eye_r * 0.9, eye_r), (0.94, 0.94, 0.95))
        body.ellipsoid(
            Vec3(side * eye_x, eye_y, front * 0.70 + eye_r * 0.72),
            Vec3(eye_r * 0.55, eye_r * 0.55, eye_r * 0.30),
            params.eye_color,
        )

    body.ellipsoid(
        Vec3(0.0, base_y + length * 0.47, front * 0.90),
        Vec3(
            0.011 * params.head_scale * params.nose_size,
            0.014 * params.head_scale * params.nose_size,
            0.013 * params.head_scale * params.nose_size,
        ),
        skin,
    )
    body.ellipsoid(
        Vec3(0.0, base_y + length * 0.35, front * 0.78),
        Vec3(0.017 * params.head_scale * params.lips, 0.008 * params.head_scale * params.lips, 0.010),
        params.lip_color,
    )
    ear_r = 0.020 * params.head_scale * params.ear_size
    for side in (-1.0, 1.0):
        body.ellipsoid(
            Vec3(side * width * 0.98, base_y + length * 0.55, 0.0),
            Vec3(ear_r * 0.45, ear_r, ear_r * 0.85),
            skin,
        )
    _build_brow(body, params, skin)


def _build_brow(
    body: _Body, params: FigureParameters, skin: tuple[float, float, float]
) -> None:
    """Eyebrows across the eyes: always present, raised by the brow-ridge dial."""
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    thickness = 0.004 * params.head_scale * (0.6 + 0.8 * params.brow_ridge)
    for side in (-1.0, 1.0):
        body.ellipsoid(
            Vec3(side * width * 0.40 * params.eye_spacing, base_y + length * 0.685, depth * 0.60),
            Vec3(width * 0.42, thickness, 0.010),
            params.hair_color,
            latitude=3,
        )


def _build_hair_mass(
    params: FigureParameters, color: tuple[float, float, float]
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """The hair as a smooth dome over the crown and down the back of the head."""
    body = _Body(color)
    if params.hair <= 0.0:
        return [], []
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    crown = base_y + length * 0.86
    scale = 1.06 + 0.14 * params.hair
    # A larger ellipsoid than the skull, cut back at the face so the face is clear.
    body.ellipsoid(
        Vec3(0.0, crown - length * 0.10, -depth * 0.18),
        Vec3(width * 1.12 * scale, length * 0.62 * scale, depth * 1.15 * scale),
        color,
        latitude=6,
    )
    # A fall down the back, when the volume dial is up.
    if params.hair >= 0.35:
        fall = (0.10 + 0.22 * params.hair) * params.head_scale
        body.ellipsoid(
            Vec3(0.0, crown - length * 0.55 - fall * 0.5, -depth * 0.95),
            Vec3(width * 1.05 * scale, fall * 0.7, depth * 0.55),
            color,
            latitude=5,
        )
    return body.vertices, body.faces


def build_hair_object(params: FigureParameters, parent_name: str) -> SceneObject | None:
    """The hair as its own object (its own material), or ``None`` when there is none.

    It is a *separate* object on purpose: the hair is not skinned to the body, so it
    keeps its own material and a hair colour is real rather than the skin colour.
    """
    vertices, faces = _build_hair_mass(params, params.hair_color)
    if not faces:
        return None
    mesh = Mesh(f"{parent_name} hair", tuple(vertices), tuple(faces))
    material = Material(
        name=f"{parent_name} hair", base_color=params.hair_color, specular=0.4, shininess=56.0
    )
    return SceneObject(f"{parent_name} hair", mesh, material)


def _build_hand(
    body: _Body,
    params: FigureParameters,
    skin: tuple[float, float, float],
    *,
    center: Vec3,
    radius: float,
) -> None:
    """A palm with four fingers and a thumb, each tipped with a nail colour."""
    body.ellipsoid(center, Vec3(radius * 0.75, radius * 1.05, radius * 0.40), skin)
    finger_len = radius * 0.95
    finger_r = radius * 0.16
    for index in range(4):
        offset = (index - 1.5) * radius * 0.40
        tip = Vec3(center.x + offset, center.y - radius * 1.05 - finger_len * 0.5, center.z)
        body.ellipsoid(tip, Vec3(finger_r, finger_len * 0.55, finger_r), skin, latitude=4)
        body.ellipsoid(
            Vec3(tip.x, tip.y - finger_len * 0.55, tip.z),
            Vec3(finger_r * 0.9, finger_r * 0.5, finger_r * 0.9),
            params.nail_color,
            latitude=3,
        )
    thumb = Vec3(center.x - radius * 0.9, center.y - radius * 0.15, center.z + radius * 0.1)
    body.ellipsoid(thumb, Vec3(finger_r * 1.1, finger_len * 0.5, finger_r * 1.1), skin, latitude=4)


def _build_foot(
    body: _Body,
    params: FigureParameters,
    skin: tuple[float, float, float],
    *,
    center: Vec3,
    size: Vec3,
) -> None:
    """A foot as a tapered block, with a nail colour on the toes."""
    body.box(center, size, skin)
    toe_r = size.x * 0.20
    for index in range(4):
        offset = (index - 1.5) * size.x * 0.28
        body.ellipsoid(
            Vec3(center.x + offset, center.y, center.z + size.z * 0.5 + toe_r * 0.4),
            Vec3(toe_r, toe_r * 0.7, toe_r),
            skin,
            latitude=3,
        )


def _build_genitalia(
    body: _Body, params: FigureParameters
) -> None:
    """The genital region, drawn from the build and shaped by its own dials.

    Kept simple and honest -- a smooth shape whose *size* and *colour* are the
    user's own dials -- so the body reads as anatomically complete without the
    geometry pretending to be a photograph.
    """
    if params.genitalia <= 0.0:
        return
    height = params.height
    scale = params.genitalia * params.genital_size
    color = params.genital_color
    center_y = height * 0.495
    forward = params.chest_depth * 0.34
    if params.build == BUILD_FEMININE:
        body.ellipsoid(Vec3(0.0, center_y, forward * 0.6), Vec3(0.026 * scale, 0.012 * scale, 0.030 * scale), color, latitude=4)
        body.ellipsoid(Vec3(0.0, center_y - 0.012 * scale, forward * 0.5), Vec3(0.010 * scale, 0.012 * scale, 0.014 * scale), color, latitude=4)
    else:
        shaft_len = 0.045 * scale
        body.ellipsoid(
            Vec3(0.0, center_y - shaft_len * 0.4, forward * 0.7),
            Vec3(0.012 * scale, shaft_len * 0.6, 0.012 * scale),
            color,
            latitude=4,
        )
        body.ellipsoid(
            Vec3(0.0, center_y - shaft_len * 0.9, forward * 0.8),
            Vec3(0.016 * scale, 0.016 * scale, 0.016 * scale),
            color,
            latitude=4,
        )
        for side in (-1.0, 1.0):
            body.ellipsoid(
                Vec3(side * 0.014 * scale, center_y + 0.004, forward * 0.5),
                Vec3(0.012 * scale, 0.014 * scale, 0.012 * scale),
                color,
                latitude=4,
            )


def build_figure_mesh(params: FigureParameters) -> Mesh:
    """Generate the humanoid mesh: torso, neck, head, arms, hands, legs, feet, face.

    The figure stands on ``y = 0``, centred on the origin, facing ``+Z``. It is a
    single skinned mesh carrying per-vertex colours for the skin, the lips, the
    irises, the areolae, the nails, and the genitalia. The hair is a separate object
    (:func:`build_hair_object`).
    """
    skin = params.skin
    body = _Body(skin)

    _build_torso(body, params, skin)
    _build_neck(body, params, skin)
    _build_head(body, params, skin)
    _build_face(body, params, skin)

    thickness = params.limb_thickness
    shoulder_y = params.height * 0.80
    hip_y = params.height * 0.50
    arm_radius = 0.045 * thickness * params.arm_thickness
    leg_radius = 0.075 * thickness * params.leg_thickness
    hand_r = 0.055 * thickness * params.hand_size
    foot_l = 0.20 * params.foot_size

    for side in (-1.0, 1.0):
        shoulder_x = side * params.shoulder_width * 0.5
        arm_length = params.height * 0.155
        fore_length = params.height * 0.150
        _build_limb(
            body,
            params,
            skin,
            top=Vec3(shoulder_x, shoulder_y, 0),
            joints=[(arm_length, arm_radius), (fore_length, arm_radius * 0.85)],
            end_radius=arm_radius * 0.62,
        )
        _build_hand(
            body,
            params,
            skin,
            center=Vec3(shoulder_x, shoulder_y - arm_length - fore_length - hand_r, 0.0),
            radius=hand_r,
        )

        hip_x = side * params.hip_width * 0.28
        thigh_length = params.height * 0.24
        shin_length = params.height * 0.235
        _build_limb(
            body,
            params,
            skin,
            top=Vec3(hip_x, hip_y, 0),
            joints=[(thigh_length, leg_radius), (shin_length, leg_radius * 0.85)],
            end_radius=leg_radius * 0.62,
        )
        _build_foot(
            body,
            params,
            skin,
            center=Vec3(hip_x, hip_y - thigh_length - shin_length - 0.02, foot_l * 0.28),
            size=Vec3(leg_radius * 1.5, 0.045, foot_l),
        )

    _build_genitalia(body, params)
    mesh = body.to_mesh(f"{params.name} body")
    if params.subdivisions > 0:
        # Weld first so the pass smooths across the seams between primitives
        # (a shoulder ring touching an arm ring), then subdivide, then relax.
        mesh = mesh.weld(tolerance=1e-5).subdivide(params.subdivisions)
        if params.smoothing > 0:
            mesh = mesh.smoothed(iterations=params.smoothing, factor=0.35)
    return mesh



def build_figure_skeleton(params: FigureParameters) -> Skeleton:
    """The bone hierarchy that matches :func:`build_figure_mesh`.

    Bones are named for what they are (``hips``, ``spine``, ``chest``, ``neck``,
    ``head``, ``upper_arm.L``, ``thigh.R``, ...) and ordered parents-first, so
    forward kinematics is a single pass. Each bone's ``offset`` is expressed in
    its parent's local frame and its ``length`` matches the mesh segment it drives.
    """
    height = params.height
    shoulder = params.shoulder_width * 0.5
    hip = params.hip_width * 0.28

    bones: list[Bone] = [
        Bone("hips", None, Vec3(0, height * 0.50, 0), Vec3(), Vec3(), height * 0.10),
        Bone("spine", "hips", Vec3(0, height * 0.10, 0), Vec3(), Vec3(), height * 0.10),
        Bone("chest", "spine", Vec3(0, height * 0.10, 0), Vec3(), Vec3(), height * 0.11),
        Bone("neck", "chest", Vec3(0, height * 0.11, 0), Vec3(), Vec3(), height * 0.035),
        Bone("head", "neck", Vec3(0, height * 0.035, 0), Vec3(), Vec3(), height * 0.14),
    ]
    for side, suffix in ((1.0, "L"), (-1.0, "R")):
        bones.extend(
            [
                Bone(
                    f"shoulder.{suffix}",
                    "chest",
                    Vec3(side * shoulder * 0.55, height * 0.10, 0),
                    Vec3(),
                    Vec3(),
                    shoulder * 0.45,
                ),
                Bone(
                    f"upper_arm.{suffix}",
                    f"shoulder.{suffix}",
                    Vec3(side * shoulder * 0.45, 0, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.155,
                ),
                Bone(
                    f"forearm.{suffix}",
                    f"upper_arm.{suffix}",
                    Vec3(0, -height * 0.155, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.150,
                ),
                Bone(
                    f"hand.{suffix}",
                    f"forearm.{suffix}",
                    Vec3(0, -height * 0.150, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.055,
                ),
                Bone(
                    f"thigh.{suffix}",
                    "hips",
                    Vec3(side * hip, 0, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.24,
                ),
                Bone(
                    f"shin.{suffix}",
                    f"thigh.{suffix}",
                    Vec3(0, -height * 0.24, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.235,
                ),
                Bone(
                    f"foot.{suffix}",
                    f"shin.{suffix}",
                    Vec3(0, -height * 0.235, 0),
                    Vec3(),
                    Vec3(),
                    height * 0.06,
                ),
            ]
        )
    return Skeleton(tuple(bones))


def build_figure(
    params: FigureParameters | None = None,
    *,
    name: str | None = None,
    material: Material | None = None,
) -> SceneObject:
    """Build a complete, bound, poseable figure as a scene object.

    The mesh is bound to the skeleton with real skin weights at construction, so
    ``obj.set_pose("forearm.L", Vec3(0, 0, -90))`` immediately bends the arm. The
    hair is a separate object -- see :func:`build_hair_object`.
    """
    settings = params or FigureParameters()
    object_name = name or settings.name
    mesh = build_figure_mesh(settings)
    skeleton = build_figure_skeleton(settings)
    skin = material or Material(
        name=f"{object_name} skin",
        base_color=settings.skin,
        specular=0.18,
        shininess=16.0,
        wrap=0.35,
    )
    obj = SceneObject(object_name, mesh, skin, skeleton=skeleton)
    obj.bind()
    if settings.pose and settings.pose != "rest":
        obj.pose = pose_rotations(settings.pose)
    return obj


#: Ready-made parameter presets, so the menu can offer named figures.
FIGURE_PRESETS: dict[str, FigureParameters] = {
    "adult": FigureParameters("Adult"),
    "athletic": FigureParameters(
        "Athletic",
        shoulder_width=0.48,
        hip_width=0.30,
        waist=0.24,
        chest_depth=0.24,
        limb_thickness=1.12,
        proportion="athletic",
        muscle=0.5,
        arm_thickness=1.08,
        leg_thickness=1.10,
    ),
    "slim": FigureParameters(
        "Slim",
        shoulder_width=0.38,
        hip_width=0.27,
        waist=0.22,
        chest_depth=0.19,
        limb_thickness=0.86,
        proportion="slim",
        arm_thickness=0.92,
        leg_thickness=0.92,
    ),
    "child": FigureParameters(
        "Child",
        height=1.25,
        shoulder_width=0.32,
        hip_width=0.24,
        chest_depth=0.18,
        waist=0.22,
        limb_thickness=0.9,
        head_scale=1.18,
        proportion="child",
        hand_size=0.82,
        foot_size=0.82,
    ),
    "feminine": FigureParameters(
        "Feminine",
        build=BUILD_FEMININE,
        proportion="curvy",
        shoulder_width=0.36,
        hip_width=0.35,
        waist=0.22,
        chest_depth=0.21,
        bust=0.5,
        waist_shape=0.6,
        hip_shape=0.5,
        arm_thickness=0.92,
        leg_thickness=1.02,
        lips=1.15,
    ),
    "masculine": FigureParameters(
        "Masculine",
        build=BUILD_MASCULINE,
        proportion="athletic",
        shoulder_width=0.50,
        hip_width=0.28,
        waist=0.26,
        chest_depth=0.25,
        muscle=0.6,
        shoulder_slope=0.2,
        jaw_width=1.12,
        brow_ridge=0.6,
        arm_thickness=1.14,
        leg_thickness=1.10,
        hair=0.35,
        hair_color=(0.14, 0.10, 0.07),
    ),
    "curvy": FigureParameters(
        "Curvy",
        build=BUILD_FEMININE,
        proportion="curvy",
        hip_width=0.37,
        waist=0.30,
        bust=0.62,
        waist_shape=0.7,
        hip_shape=0.7,
        belly=0.25,
        leg_thickness=1.08,
    ),
}


def figure_preset_names() -> tuple[str, ...]:
    return tuple(FIGURE_PRESETS)


#: Ready-made poses a figure can be put into, as named bone rotations (degrees).
#: They live here, beside the rig, so the Scene Studio and the body studio share one
#: set and cannot drift apart. Each entry is ``bone -> (x, y, z)`` euler degrees.
POSE_PRESETS: dict[str, dict[str, Vec3]] = {
    "rest": {},
    "contrapposto": {
        "hips": Vec3(0.0, 0.0, 6.0),
        "spine": Vec3(0.0, 0.0, -3.0),
        "thigh.L": Vec3(-4.0, 0.0, 4.0),
        "shin.L": Vec3(6.0, 0.0, 0.0),
        "thigh.R": Vec3(2.0, 0.0, -2.0),
    },
    "wave": {
        "shoulder.L": Vec3(0.0, 0.0, -20.0),
        "upper_arm.L": Vec3(0.0, 0.0, -125.0),
        "forearm.L": Vec3(0.0, 0.0, -35.0),
    },
    "walk": {
        "thigh.L": Vec3(-28.0, 0.0, 0.0),
        "shin.L": Vec3(22.0, 0.0, 0.0),
        "thigh.R": Vec3(24.0, 0.0, 0.0),
        "shin.R": Vec3(8.0, 0.0, 0.0),
        "upper_arm.L": Vec3(22.0, 0.0, 0.0),
        "upper_arm.R": Vec3(-22.0, 0.0, 0.0),
    },
    "run": {
        "hips": Vec3(6.0, 0.0, 0.0),
        "thigh.L": Vec3(-48.0, 0.0, 0.0),
        "shin.L": Vec3(58.0, 0.0, 0.0),
        "thigh.R": Vec3(34.0, 0.0, 0.0),
        "shin.R": Vec3(22.0, 0.0, 0.0),
        "upper_arm.L": Vec3(48.0, 0.0, 0.0),
        "forearm.L": Vec3(-70.0, 0.0, 0.0),
        "upper_arm.R": Vec3(-48.0, 0.0, 0.0),
        "forearm.R": Vec3(-70.0, 0.0, 0.0),
    },
    "sit": {
        "thigh.L": Vec3(-85.0, 0.0, 0.0),
        "shin.L": Vec3(85.0, 0.0, 0.0),
        "thigh.R": Vec3(-85.0, 0.0, 0.0),
        "shin.R": Vec3(85.0, 0.0, 0.0),
    },
    "sit_ground": {
        "hips": Vec3(0.0, 0.0, 0.0),
        "thigh.L": Vec3(-92.0, 12.0, 0.0),
        "shin.L": Vec3(60.0, 0.0, 0.0),
        "thigh.R": Vec3(-92.0, -12.0, 0.0),
        "shin.R": Vec3(60.0, 0.0, 0.0),
        "spine": Vec3(-6.0, 0.0, 0.0),
    },
    "tpose": {
        "shoulder.L": Vec3(0.0, 0.0, -90.0),
        "shoulder.R": Vec3(0.0, 0.0, 90.0),
        "upper_arm.L": Vec3(0.0, 0.0, -90.0),
        "upper_arm.R": Vec3(0.0, 0.0, 90.0),
    },
    "reach": {
        "shoulder.L": Vec3(0.0, 0.0, -40.0),
        "upper_arm.L": Vec3(-60.0, 0.0, -100.0),
        "forearm.L": Vec3(-25.0, 0.0, 0.0),
        "chest": Vec3(-8.0, 0.0, 0.0),
    },
    "arms_crossed": {
        "shoulder.L": Vec3(0.0, 0.0, -20.0),
        "upper_arm.L": Vec3(-70.0, 30.0, -35.0),
        "forearm.L": Vec3(0.0, 0.0, -95.0),
        "shoulder.R": Vec3(0.0, 0.0, 20.0),
        "upper_arm.R": Vec3(-70.0, -30.0, 35.0),
        "forearm.R": Vec3(0.0, 0.0, 95.0),
    },
    "hands_hips": {
        "upper_arm.L": Vec3(0.0, 0.0, -45.0),
        "forearm.L": Vec3(-80.0, 0.0, -45.0),
        "upper_arm.R": Vec3(0.0, 0.0, 45.0),
        "forearm.R": Vec3(-80.0, 0.0, 45.0),
    },
    "squat": {
        "hips": Vec3(20.0, 0.0, 0.0),
        "thigh.L": Vec3(-70.0, 8.0, 0.0),
        "shin.L": Vec3(80.0, 0.0, 0.0),
        "foot.L": Vec3(-10.0, 0.0, 0.0),
        "thigh.R": Vec3(-70.0, -8.0, 0.0),
        "shin.R": Vec3(80.0, 0.0, 0.0),
        "foot.R": Vec3(-10.0, 0.0, 0.0),
        "upper_arm.L": Vec3(-40.0, 0.0, -20.0),
        "upper_arm.R": Vec3(-40.0, 0.0, 20.0),
    },
    "lying": {
        "hips": Vec3(-90.0, 0.0, 0.0),
        "thigh.L": Vec3(6.0, 0.0, 0.0),
        "thigh.R": Vec3(6.0, 0.0, 0.0),
        "upper_arm.L": Vec3(0.0, 0.0, -20.0),
        "upper_arm.R": Vec3(0.0, 0.0, 20.0),
    },
}


def pose_preset_names() -> tuple[str, ...]:
    return tuple(POSE_PRESETS)


def pose_rotations(preset: str) -> dict[str, Vec3]:
    """The bone rotations of a named pose, or an empty pose for ``rest``/unknown."""
    return dict(POSE_PRESETS.get(preset, {}))



def build_figure_from_preset(key: str = "adult", *, name: str | None = None) -> SceneObject:
    """Build one of the named presets (``adult`` is the default)."""
    try:
        params = FIGURE_PRESETS[key]
    except KeyError as error:
        raise KeyError(
            f"Unknown figure preset {key!r}; choose from {', '.join(FIGURE_PRESETS)}."
        ) from error
    return build_figure(params, name=name)


def build_figure_scene(key: str = "adult", *, name: str | None = None) -> Scene:
    """A complete scene holding one figure, framed and lit.

    The figure is the **body alone**, exactly as before, so a caller that hides
    ``objects[0]`` or counts the scene's objects is unaffected. The hair is a
    separate object a caller adds itself when it wants one (the body studio does,
    through :func:`build_hair_object`).
    """
    figure = build_figure_from_preset(key, name=name)
    scene = make_default_scene(f"{figure.name} scene")
    scene.add_object(figure)
    scene.frame_camera()
    return scene


def morph_figure(obj: SceneObject, params: FigureParameters) -> SceneObject:
    """Rebuild a figure's mesh from new parameters, keeping its pose.

    This is the "morph" operation: the mesh is regenerated and re-bound, while
    the pose the user has already applied is carried over.
    """
    pose = dict(obj.pose)
    rebuilt = build_figure(params, name=obj.name, material=obj.material)
    rebuilt.pose = pose
    return rebuilt


__all__ = [
    "BUILDS",
    "BUILD_FEMININE",
    "BUILD_MASCULINE",
    "BUILD_NEUTRAL",
    "FIGURE_PRESETS",
    "POSE_PRESETS",
    "RADIAL_SEGMENTS",
    "FigureParameters",
    "build_figure",
    "build_figure_from_preset",
    "build_figure_mesh",
    "build_figure_scene",
    "build_figure_skeleton",
    "build_hair_object",
    "figure_preset_names",
    "morph_figure",
    "pose_preset_names",
    "pose_rotations",
]
