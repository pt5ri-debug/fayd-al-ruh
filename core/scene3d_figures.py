"""Built-in, procedurally generated figures -- so the studio works with no assets.

A DAZ-style studio is only useful if there is something to pose. Licensing real
figure content is not possible here, so this module **builds** the figures: a
parametric humanoid mesh (torso, head, arms, legs, hands, feet) generated from a
handful of measurements, and a matching **bone hierarchy** bound to it with the
real skinning in :mod:`core.scene3d`. The result is a genuinely poseable figure
that ships with the code, needs no download, and is honest about being a
*stylised* figure rather than a licensed likeness.

The generator is deterministic: the same parameters always produce the same
figure, so a saved scene reloads identically. The proportions come from named
parameters (height, shoulder width, limb thickness), which is what makes the
figures "customisable" the way a DAZ figure's morph dials are.

Nothing here is a claim about a commercial product: it is an original, simple,
parametric humanoid, and every figure it makes says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class FigureParameters:
    """The "morph dials": named measurements that shape one figure.

    All lengths are in metres, with the figure standing on ``y = 0`` and facing
    ``+Z``. The defaults describe an adult of average build.
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


def _ring(center: Vec3, radius_x: float, radius_z: float, segments: int) -> list[Vec3]:
    """A horizontal ellipse of points around ``center``, in the XZ plane."""
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
    """Stitch consecutive rings into a closed tube of quads."""
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


def _build_torso(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A torso as stacked elliptical rings from hips to shoulders."""
    hip_y = params.height * 0.50
    waist_y = params.height * 0.60
    chest_y = params.height * 0.70
    shoulder_y = params.height * 0.81
    scale = params.limb_thickness
    rings = [
        _ring(Vec3(0, hip_y, 0), params.hip_width * 0.5, params.chest_depth * 0.6, RADIAL_SEGMENTS),
        _ring(Vec3(0, waist_y, 0), params.waist * 0.5, params.chest_depth * 0.45, RADIAL_SEGMENTS),
        _ring(Vec3(0, chest_y, 0), params.shoulder_width * 0.42, params.chest_depth * 0.55, RADIAL_SEGMENTS),
        _ring(Vec3(0, shoulder_y, 0), params.shoulder_width * 0.5 * scale, params.chest_depth * 0.5, RADIAL_SEGMENTS),
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
    the limb tapers and bends the way the skeleton does.
    """
    rings: list[list[Vec3]] = []
    position = top
    for length, radius in joints:
        rings.append(
            _ring(position, radius, radius, RADIAL_SEGMENTS)
        )
        position = Vec3(position.x, position.y - length, position.z)
    rings.append(_ring(position, joints[-1][1] * 0.7, joints[-1][1] * 0.7, RADIAL_SEGMENTS))
    _bridge_rings(rings, vertices, faces)


def _build_head(params: FigureParameters, vertices: list[Vec3], faces: list[tuple[int, ...]]) -> None:
    """A head as stacked rings from the neck to the crown."""
    base_y = params.height * 0.82
    radius = 0.085 * params.head_scale
    height = 0.23 * params.head_scale
    rings = [
        _ring(Vec3(0, base_y, 0), radius * 0.7, radius * 0.7, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + height * 0.25, 0), radius, radius * 0.95, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + height * 0.60, 0), radius * 1.05, radius * 1.0, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + height * 0.85, 0), radius * 0.75, radius * 0.75, RADIAL_SEGMENTS),
        _ring(Vec3(0, base_y + height, 0), radius * 0.25, radius * 0.25, RADIAL_SEGMENTS),
    ]
    _bridge_rings(rings, vertices, faces)


def build_figure_mesh(params: FigureParameters) -> Mesh:
    """Generate the humanoid mesh: torso, head, two arms, two legs, hands, feet.

    The figure stands on ``y = 0``, centred on the origin, facing ``+Z``. It is a
    single mesh with the given name, so it skins as one body.
    """
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []

    _build_torso(params, vertices, faces)
    _build_head(params, vertices, faces)

    thickness = params.limb_thickness
    shoulder_y = params.height * 0.80
    hip_y = params.height * 0.50
    arm_radius = 0.045 * thickness
    leg_radius = 0.075 * thickness

    for side in (-1.0, 1.0):
        shoulder_x = side * params.shoulder_width * 0.5
        # Arm: upper arm, forearm, then a simple hand block.
        _build_limb(
            params,
            vertices,
            faces,
            top=Vec3(shoulder_x, shoulder_y, 0),
            joints=[
                (params.height * 0.155, arm_radius),
                (params.height * 0.150, arm_radius * 0.85),
                (params.height * 0.055, arm_radius * 0.75),
            ],
        )
        # Leg: thigh, shin, then a simple foot.
        hip_x = side * params.hip_width * 0.28
        _build_limb(
            params,
            vertices,
            faces,
            top=Vec3(hip_x, hip_y, 0),
            joints=[
                (params.height * 0.24, leg_radius),
                (params.height * 0.235, leg_radius * 0.85),
                (params.height * 0.035, leg_radius * 0.95),
            ],
        )

    mesh = Mesh(f"{params.name} body", tuple(vertices), tuple(faces))
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
    ``obj.set_pose("forearm.L", Vec3(0, 0, -90))`` immediately bends the arm.
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
    ),
    "slim": FigureParameters(
        "Slim",
        shoulder_width=0.38,
        hip_width=0.27,
        waist=0.22,
        chest_depth=0.19,
        limb_thickness=0.86,
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
    """A complete scene holding one figure, framed and lit, ready to render."""
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
    "FIGURE_PRESETS",
    "RADIAL_SEGMENTS",
    "FigureParameters",
    "build_figure",
    "build_figure_from_preset",
    "build_figure_mesh",
    "build_figure_scene",
    "build_figure_skeleton",
    "figure_preset_names",
    "morph_figure",
]
