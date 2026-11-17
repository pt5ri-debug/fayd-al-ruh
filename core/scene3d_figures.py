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
from dataclasses import dataclass, replace

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

#: Segments around a limb's circumference. Eight is enough for a smooth silhouette
#: at the render sizes this studio uses, and keeps the mesh small and fast.
RADIAL_SEGMENTS = 8

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
        ):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ValueError(f"Figure {field_name} must be positive.")

    # -- convenience -----------------------------------------------------

    def with_changes(self, **changes: object) -> FigureParameters:
        """A copy with the named dials changed (used by the body studio)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def describe(self) -> str:
        """A one-line summary of the shape."""
        return f"{self.build} build, {self.proportion} proportions, {self.height:.2f} m"


def _ring(center: Vec3, radius_x: float, radius_z: float, segments: int) -> list[Vec3]:
    """A horizontal ellipse of points around ``center``, in the XZ plane.

    Points are generated counter-clockwise seen from above, which is the winding
    the rasteriser's backface culling treats as the *outside* of a lofted tube.
    """
    points = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        points.append(
            Vec3(
                center.x + radius_x * math.cos(angle),
                center.y,
                center.z + radius_z * math.sin(angle),
            )
        )
    return points


def _bridge_rings(rings: list[list[Vec3]], vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """Stitch consecutive rings into a closed tube of quads (bottom ring first)."""
    offsets = []
    for ring in rings:
        offsets.append(len(vertices))
        vertices.extend(ring)
    segments = len(rings[0])
    for ring_index in range(len(rings) - 1):
        lower = offsets[ring_index]
        upper = offsets[ring_index + 1]
        for index in range(segments):
            nxt = (index + 1) % segments
            faces.append((lower + index, lower + nxt, upper + nxt, upper + index))


def _cap_top(ring: list[Vec3], center: Vec3, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """Close the *top* of a tube with a fan to ``center``."""
    base = len(vertices)
    vertices.append(center)
    count = len(ring)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((base, base + 1 + nxt, base + 1 + index))


def _box(center: Vec3, size: Vec3, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """Add an axis-aligned box (a hand's palm, a foot, and a hair mass)."""
    half = Vec3(size.x * 0.5, size.y * 0.5, size.z * 0.5)
    base = len(vertices)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                vertices.append(
                    Vec3(center.x + sx * half.x, center.y + sy * half.y, center.z + sz * half.z)
                )

    def corner(sx: int, sy: int, sz: int) -> int:
        return base + sx * 4 + sy * 2 + sz

    faces.extend(
        [
            (corner(0, 0, 0), corner(0, 0, 1), corner(0, 1, 1), corner(0, 1, 0)),
            (corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, 0, 1)),
            (corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)),
            (corner(0, 1, 0), corner(0, 1, 1), corner(1, 1, 1), corner(1, 1, 0)),
            (corner(0, 0, 0), corner(0, 1, 0), corner(1, 1, 0), corner(1, 0, 0)),
            (corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)),
        ]
    )


def _sphere(center: Vec3, radius: float, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A low-poly sphere, used for a face feature (bottom ring first, so it is solid)."""
    if radius <= 0.0:
        return
    latitude = 4
    rings: list[list[Vec3]] = []
    for step in range(latitude, -1, -1):
        phi = math.pi * step / latitude
        ring_radius = radius * math.sin(phi)
        y = center.y + radius * math.cos(phi)
        rings.append(
            [
                Vec3(
                    center.x + ring_radius * math.cos(2.0 * math.pi * index / RADIAL_SEGMENTS),
                    y,
                    center.z + ring_radius * math.sin(2.0 * math.pi * index / RADIAL_SEGMENTS),
                )
                for index in range(RADIAL_SEGMENTS)
            ]
        )
    _bridge_rings(rings, vertices, faces)


def _build_torso(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
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
        _ring(Vec3(0, height * 0.50, 0.0), hips * 0.5, params.chest_depth * 0.60, RADIAL_SEGMENTS),
        _ring(
            Vec3(0, height * 0.60, belly_z * 0.25),
            waist_r * 0.5,
            params.chest_depth * 0.42 + belly_z * 0.4,
            RADIAL_SEGMENTS,
        ),
        _ring(Vec3(0, height * 0.70, chest_z * 0.22), chest_r, chest_z, RADIAL_SEGMENTS),
        _ring(
            Vec3(0, height * 0.81, 0.0),
            params.shoulder_width * 0.5 * scale,
            params.chest_depth * 0.50,
            RADIAL_SEGMENTS,
        ),
    ]
    _bridge_rings(rings, vertices, faces)


def _build_neck(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A short neck between the shoulders and the jaw."""
    height = params.height
    radius = 0.052 * params.neck_thickness
    rings = [
        _ring(Vec3(0, height * 0.815, 0.0), radius * 1.25, radius * 1.15, RADIAL_SEGMENTS),
        _ring(Vec3(0, height * 0.855, 0.0), radius, radius * 0.95, RADIAL_SEGMENTS),
    ]
    _bridge_rings(rings, vertices, faces)


def _build_limb(
    params: FigureParameters,
    vertices: list[Vec3],
    faces: list[tuple[int, ...]],
    *,
    top: Vec3,
    joints: list[tuple[float, float]],
) -> None:
    """A limb as rings around ``top`` descending along ``-Y``.

    ``joints`` is a list of ``(length, radius)`` pairs, one per bone segment, so
    the limb tapers and bends the way the skeleton does. Rings are emitted
    **bottom-first** so the loft's winding faces outwards, which is what the
    rasteriser's backface culling expects.
    """
    positions: list[Vec3] = []
    position = top
    for length, _radius in joints:
        positions.append(position)
        position = Vec3(position.x, position.y - length, position.z)
    positions.append(position)

    radii = [radius for _length, radius in joints]
    radii.append(radii[-1] * 0.7)
    rings = [
        _ring(point, radius, radius, RADIAL_SEGMENTS)
        for point, radius in reversed(list(zip(positions, radii, strict=True)))
    ]
    _bridge_rings(rings, vertices, faces)


def _build_head(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A head as stacked rings from the jaw to the crown, shaped by the face dials."""
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    jaw = width * params.jaw_width
    chin = 1.0 - 0.35 * params.chin_shape

    rings = [
        _ring(Vec3(0, base_y, 0.0), jaw * 0.62 * chin, depth * 0.60, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + length * 0.16, 0.006), jaw * 0.95, depth * 0.95, RADIAL_SEGMENTS),
        _ring(
            Vec3(0, base_y + length * 0.42, 0.010),
            width * (1.0 + 0.10 * params.cheek),
            depth * 1.02,
            RADIAL_SEGMENTS,
        ),
        _ring(Vec3(0, base_y + length * 0.68, 0.004), width * 0.98, depth * 0.98, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + length * 0.88, 0.0), width * 0.72, depth * 0.72, RADIAL_SEGMENTS),
    ]
    _bridge_rings(rings, vertices, faces)
    _cap_top(rings[-1], Vec3(0, base_y + length, 0.0), vertices, faces)


def _build_face(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """The face features: eyes, a nose, lips, and ears.

    Deliberately simple -- low-poly spheres in the head's front half -- so a figure
    reads as a *person* at the studio's render sizes rather than a featureless
    mannequin, while staying cheap to build and honest about being a stylised
    likeness.
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
        _sphere(Vec3(side * eye_x, eye_y, front * 0.72), eye_r, vertices, faces)

    _sphere(
        Vec3(0.0, base_y + length * 0.47, front * 0.92),
        0.011 * params.head_scale * params.nose_size,
        vertices,
        faces,
    )
    _sphere(
        Vec3(0.0, base_y + length * 0.36, front * 0.80),
        0.012 * params.head_scale * params.lips,
        vertices,
        faces,
    )
    ear_r = 0.020 * params.head_scale * params.ear_size
    for side in (-1.0, 1.0):
        _sphere(Vec3(side * width * 0.98, base_y + length * 0.55, 0.0), ear_r, vertices, faces)


def _build_brow(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A brow ridge across the eyes, present only when the dial is raised."""
    if params.brow_ridge <= 0.0:
        return
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    _box(
        Vec3(0.0, base_y + length * 0.66, depth * 0.58),
        Vec3(width * 1.5, 0.012 * params.head_scale * (1.0 + params.brow_ridge), 0.02),
        vertices,
        faces,
    )


def _build_hair(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A hair mass over the crown and down the back, when the hair dial is up."""
    if params.hair <= 0.0:
        return
    base_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    width = 0.088 * params.head_scale * params.face_width
    depth = 0.100 * params.head_scale * params.face_length
    crown = base_y + length * 0.86
    height = (0.05 + 0.10 * params.hair) * params.head_scale
    _box(
        Vec3(0.0, crown + height * 0.30, -depth * 0.28),
        Vec3(width * 2.0, height, depth * (1.2 + 0.8 * params.hair)),
        vertices,
        faces,
    )


def build_hair_object(params: FigureParameters, parent_name: str) -> SceneObject | None:
    """The hair as its own object (its own material), or ``None`` when there is none.

    It is a *separate* object on purpose: the renderer takes one material per
    object, so a hair colour can only be real if the hair is not part of the body
    mesh. A studio adds it beside the figure; :func:`build_figure_scene` does.
    """
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    _build_hair(params, vertices, faces)
    if not faces:
        return None
    mesh = Mesh(f"{parent_name} hair", tuple(vertices), tuple(faces))
    material = Material(
        name=f"{parent_name} hair", base_color=params.hair_color, specular=0.35, shininess=48.0
    )
    return SceneObject(f"{parent_name} hair", mesh, material)


def build_figure_mesh(params: FigureParameters) -> Mesh:
    """Generate the humanoid mesh: torso, neck, head, arms, hands, legs, feet.

    The figure stands on ``y = 0``, centred on the origin, facing ``+Z``. It is a
    single mesh with the given name, so it skins as one body. The hair is a
    separate object (:func:`build_hair_object`).
    """
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []

    _build_torso(params, vertices, faces)
    _build_neck(params, vertices, faces)
    _build_head(params, vertices, faces)
    _build_brow(params, vertices, faces)
    _build_face(params, vertices, faces)

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
            params,
            vertices,
            faces,
            top=Vec3(shoulder_x, shoulder_y, 0),
            joints=[(arm_length, arm_radius), (fore_length, arm_radius * 0.85)],
        )
        hand_center = Vec3(shoulder_x, shoulder_y - arm_length - fore_length - hand_r, 0.0)
        _box(hand_center, Vec3(hand_r * 1.4, hand_r * 1.9, hand_r * 0.7), vertices, faces)

        hip_x = side * params.hip_width * 0.28
        thigh_length = params.height * 0.24
        shin_length = params.height * 0.235
        _build_limb(
            params,
            vertices,
            faces,
            top=Vec3(hip_x, hip_y, 0),
            joints=[(thigh_length, leg_radius), (shin_length, leg_radius * 0.85)],
        )
        foot_center = Vec3(hip_x, hip_y - thigh_length - shin_length - 0.02, foot_l * 0.28)
        _box(foot_center, Vec3(leg_radius * 1.5, 0.045, foot_l), vertices, faces)

    return Mesh(f"{params.name} body", tuple(vertices), tuple(faces))


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
        name=f"{object_name} skin", base_color=settings.skin, specular=0.18, shininess=16.0
    )
    obj = SceneObject(object_name, mesh, skin, skeleton=skeleton)
    obj.bind()
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
]
