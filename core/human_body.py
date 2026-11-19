"""The real human body: a CC0 base mesh morphed by artist-authored targets.

This is the module that replaces the old "sculpt a body from implicit
primitives" figure with a genuine human mesh. It is deliberately thin and honest:

* the **geometry** is MakeHuman's CC0 base mesh (13,660 vertices with modelled
  hands, feet, face, eyes and teeth) -- see :mod:`core.human_pack`;
* every **dial** applies a real, artist-authored morph target in both directions
  (``incr``/``decr``), so moving a slider moves vertices the way a modeller
  intended rather than scaling an ellipsoid;
* the **rig** is built from the base mesh's own joint markers, so a bone sits
  where the anatomy puts it and stays there when the body is morphed.

Everything is deterministic and dependency-free, and the body is scaled into the
project's own space (metres, feet on ``y = 0``, centred on the origin, facing
``+Z``) so it drops straight into the existing renderer, camera and studio.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from core.human_pack import HumanPack, MorphTarget, load_pack
from core.scene3d import Bone, Mesh, SceneObject, Skeleton, Vec3

#: The dial value that leaves the base body alone. Every dial is ``0..1`` and
#: ``0.5`` is "no change", which is how MakeHuman's own sliders work.
NEUTRAL = 0.5

#: The default height (metres) a body is scaled to.
DEFAULT_HEIGHT = 1.75

#: Per-vertex colours: skin, lips, irises, teeth, and the nail tone.
SKIN = (0.86, 0.71, 0.62)
LIPS = (0.62, 0.30, 0.30)
IRIS = (0.22, 0.34, 0.42)
TEETH = (0.94, 0.92, 0.88)
NAIL = (0.90, 0.82, 0.80)

#: The dials the studio exposes, each mapped to its pack target. A dial that is
#: ``None`` means "no artist-authored target exists for it"; the value is still
#: accepted and applied as a *uniform* body-scale change, so a slider never does
#: nothing, and the caller can report exactly which dials are real morphs.
DIAL_TARGETS: dict[str, str] = {
    "height": "",
    "overall_scale": "",
    "shoulder_width": "shoulder",
    "hip_width": "hip_width",
    "chest_depth": "chest_depth",
    "waist": "waist",
    "bust": "bust",
    "belly": "belly",
    "muscle": "stomach_tone",
    "buttocks": "buttocks",
    "pelvis": "pelvis_tone",
    "neck_thickness": "neck_girth",
    "neck_length": "neck_length",
    "arm_thickness": "arm_girth",
    "forearm_thickness": "forearm_l",
    "thigh_thickness": "thigh_girth",
    "calf_thickness": "calf_girth",
    "wrist_thickness": "wrist_girth",
    "ankle_thickness": "ankle_girth",
    "arm_length": "upper_arm_length",
    "forearm_length": "forearm_length",
    "thigh_length": "thigh_length",
    "lower_leg_length": "shin_length",
    "hand_size": "hand_l",
    "foot_size": "foot_l",
    "head_scale": "head_height",
    "head_width": "head_width",
    "head_depth": "head_depth",
    "face_length": "face_length",
    "jaw_width": "jaw_width",
    "chin_shape": "chin",
    "nose_size": "nose_length",
    "nose_width": "nose_width",
    "eye_size": "eye_size_l",
    "lips": "lower_lip",
    "mouth_width": "mouth_width",
    "cheek": "cheek_l",
    "ear_size": "ear_size_l",
    "forehead": "forehead",
}

#: Dials that move only the left side, so their twin can be applied too.
SIDE_TARGETS: dict[str, str] = {
    "thigh_l": "thigh_l",
    "thigh_r": "thigh_r",
    "calf_l": "calf_l",
    "calf_r": "calf_r",
    "upper_arm_l": "upper_arm_l",
    "upper_arm_r": "upper_arm_r",
    "forearm_l": "forearm_l",
    "forearm_r": "forearm_r",
    "hand_l": "hand_l",
    "hand_r": "hand_r",
    "foot_l": "foot_l",
    "foot_r": "foot_r",
    "leg_length_l": "leg_length_l",
    "leg_length_r": "leg_length_r",
    "eye_size_l": "eye_size_l",
    "eye_size_r": "eye_size_r",
    "cheek_l": "cheek_l",
    "cheek_r": "cheek_r",
    "ear_size_l": "ear_size_l",
    "ear_size_r": "ear_size_r",
}

#: A body's proportions, as dial values in ``0..1`` (``0.5`` is the base body).
Proportions = dict[str, float]


@dataclass(frozen=True)
class HumanBody:
    """One morphed body, ready to render or to bind to a rig."""

    name: str
    mesh: Mesh
    height: float
    dials: dict[str, float]
    moved_vertices: int
    max_displacement: float
    base_vertices: tuple[tuple[float, float, float], ...] = ()
    joints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    _scale: float = 1.0
    _offset: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    def describe(self) -> str:
        return (
            f"{self.name}: {self.mesh.vertex_count} vertices, "
            f"{self.mesh.triangle_count} triangles, {self.height:.2f} m, "
            f"{self.moved_vertices} vertices morphed (max {self.max_displacement * 100:.1f} cm)."
        )


def pack() -> HumanPack:
    """The loaded pack (a thin wrapper, so callers need not import the module)."""
    return load_pack()


def base_mesh(name: str = "Body") -> Mesh:
    """The base body as a :class:`core.scene3d.Mesh`, in the pack's own units."""
    data = load_pack()
    colors = _base_colors(data)
    return Mesh(
        name=name,
        vertices=tuple(Vec3(*point) for point in data.vertices),
        faces=data.faces,
        vertex_colors=colors,
    )


def _base_colors(data: HumanPack) -> tuple[tuple[float, float, float], ...]:
    """Per-vertex colours: skin, lips, irises, teeth and nails.

    One skinned mesh carries every colour, so the lips and the eyes follow a pose
    instead of being separate objects that would slide off the face. The regions
    are found from the pack's own joint markers, so they sit where the anatomy
    puts them on *this* body rather than at a guessed offset.
    """
    colors = [SKIN] * data.vertex_count
    for index in data.group("helper-upper-teeth"):
        colors[index] = TEETH
    for index in data.group("helper-lower-teeth"):
        colors[index] = TEETH
    _tint(colors, data, LIPS, _lips_region(data))
    _tint(colors, data, IRIS, _iris_region(data))
    return tuple(colors)


def _tint(
    colors: list[tuple[float, float, float]],
    data: HumanPack,
    colour: tuple[float, float, float],
    indices: set[int],
) -> None:
    for index in indices:
        if 0 <= index < len(colors):
            colors[index] = colour


def _lips_region(data: HumanPack) -> set[int]:
    """The mouth's own vertices: just in front of the jaw marker and near it."""
    jaw = data.joint("joint-jaw")
    if jaw is None:
        return set()
    jx, jy, jz = jaw
    radius = 0.55
    return {
        index
        for index in data.group("body")
        if abs(data.vertices[index][0] - jx) < radius
        and abs(data.vertices[index][1] - jy) < radius * 0.75
        and data.vertices[index][2] > jz - radius * 1.4
    }


def _iris_region(data: HumanPack) -> set[int]:
    """The eyeball groups themselves, which the pack models as spheres."""
    return set(data.group("helper-l-eye")) | set(data.group("helper-r-eye"))


def apply_morph(
    vertices: list[tuple[float, float, float]],
    dials: dict[str, float],
    *,
    joints: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[tuple[float, float, float]], int, float]:
    """Apply every dial to ``vertices`` in place.

    A dial with an artist-authored target applies that target's sparse deltas. A
    dial with **no** target (a measurement MakeHuman does not morph, such as
    ``overall_scale`` or ``hand_width``) is applied as a *regional scale* about
    the matching joint markers, so a slider still moves real vertices instead of
    doing nothing -- and :data:`DIAL_TARGETS` tells the caller which is which.

    Returns the mutated list, how many vertices moved, and the largest single
    displacement (pack units), so a caller can prove a slider changed geometry.
    """
    data = load_pack()
    markers = joints if joints is not None else data.joints
    moved: set[int] = set()
    largest = 0.0
    for key, target_name in _dial_targets().items():
        weight = (dials.get(key, NEUTRAL) - NEUTRAL) * 2.0
        if abs(weight) < 1e-6:
            continue
        target = data.targets.get(target_name) if target_name else None
        if target is not None:
            indices, vectors = _target_side(target, weight)
            if indices:
                sign = 1.0 if weight > 0 else -1.0
                amount = abs(weight)
                for index, vector in zip(indices, vectors, strict=False):
                    x, y, z = vertices[index]
                    dx, dy, dz = sign * vector[0] * amount, sign * vector[1] * amount, sign * vector[2] * amount
                    vertices[index] = (x + dx, y + dy, z + dz)
                    moved.add(index)
                    largest = max(largest, abs(dx), abs(dy), abs(dz))
                continue
        region = REGIONAL_SCALES.get(key)
        if region is None:
            continue
        center = _region_center(markers, region.anchors)
        if center is None:
            continue
        factor = 1.0 + weight * REGION_STRENGTH
        if abs(factor - 1.0) < 1e-6:
            continue
        _changed, biggest = _scale_region(vertices, center, factor, region, moved)
        largest = max(largest, biggest)
    return vertices, len(moved), largest


#: How far a regional dial moves its vertices at full deflection (a fraction).
REGION_STRENGTH = 0.35


@dataclass(frozen=True)
class RegionScale:
    """A dial with no pack target, applied as a smooth scale about joint markers.

    ``count`` selects the nearest vertices to the anchors -- a count rather than a
    radius, because a joint marker sits *inside* the body and the mesh's surface is
    a variable distance away, so a radius in pack units is not portable between a
    child and an adult. ``count=None`` means every vertex (the whole-body scale).
    ``axes`` decides which of x/y/z the scale applies to.
    """

    anchors: tuple[str, ...]
    count: int | None = 900
    axes: tuple[bool, bool, bool] = (True, True, True)


#: Dials the pack has no target for. Each is scaled about its own joint markers,
#: so the slider moves the region it names and nothing else.
REGIONAL_SCALES: dict[str, RegionScale] = {
    # The whole body, about the pelvis.
    "overall_scale": RegionScale(("joint-pelvis",), count=None),
    # Hands and feet.
    "hand_width": RegionScale(("joint-l-hand",), count=140, axes=(True, False, True)),
    "hand_length": RegionScale(("joint-l-hand",), count=140, axes=(False, True, True)),
    "foot_width": RegionScale(("joint-l-foot-1",), count=160, axes=(True, False, False)),
    "foot_length": RegionScale(("joint-l-foot-1",), count=160, axes=(False, False, True)),
    # Limb girth.
    "limb_thickness": RegionScale(("joint-l-elbow", "joint-l-knee"), count=700),
    "leg_thickness": RegionScale(("joint-l-knee",), count=420),
    # Face features with no target of their own.
    "face_width": RegionScale(("joint-head",), count=700, axes=(True, False, False)),
    "eye_spacing": RegionScale(("joint-head",), count=300, axes=(True, False, False)),
    "brow_ridge": RegionScale(("joint-l-eye", "joint-r-eye"), count=90, axes=(False, False, True)),
    "waist_shape": RegionScale(("joint-spine-3",), count=600, axes=(True, False, True)),
    "hip_shape": RegionScale(("joint-pelvis",), count=700, axes=(True, False, True)),
    "shoulder_slope": RegionScale(("joint-l-shoulder", "joint-r-shoulder"), count=420),
    # The crotch: between the two upper legs, which is where the detail sits.
    "areola": RegionScale(("joint-spine-4",), count=120, axes=(False, False, True)),
    "genitalia": RegionScale(("joint-l-upper-leg", "joint-r-upper-leg"), count=220),
    "genital_size": RegionScale(("joint-l-upper-leg", "joint-r-upper-leg"), count=220),
}


def _region_center(
    markers: dict[str, tuple[float, float, float]],
    anchors: tuple[str, ...],
) -> tuple[float, float, float] | None:
    """The mean of the named joint markers, or ``None`` when none are present."""
    points = [markers[name] for name in anchors if name in markers]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _scale_region(
    vertices: list[tuple[float, float, float]],
    center: tuple[float, float, float],
    factor: float,
    region: RegionScale,
    moved: set[int],
) -> tuple[int, float]:
    """Scale the ``region.count`` vertices nearest ``center`` with a smooth falloff."""
    cx, cy, cz = center
    axes = region.axes
    if region.count is None or region.count >= len(vertices):
        chosen: list[int] | range = range(len(vertices))
        furthest = 0.0
    else:
        distances = [
            (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 for x, y, z in vertices
        ]
        chosen = heapq.nsmallest(region.count, range(len(vertices)), key=distances.__getitem__)
        furthest = math.sqrt(max(distances[index] for index in chosen)) or 1.0
    count = 0
    largest = 0.0
    for index in chosen:
        x, y, z = vertices[index]
        dx, dy, dz = x - cx, y - cy, z - cz
        if furthest > 0.0:
            # A cosine falloff: full effect at the centre, none at the rim of the
            # selected set, so the scaled region blends into the untouched skin.
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            blend = 0.5 * (1.0 + math.cos(math.pi * min(distance / furthest, 1.0)))
        else:
            blend = 1.0
        amount = 1.0 + (factor - 1.0) * blend
        nx = cx + dx * (amount if axes[0] else 1.0)
        ny = cy + dy * (amount if axes[1] else 1.0)
        nz = cz + dz * (amount if axes[2] else 1.0)
        vertices[index] = (nx, ny, nz)
        moved.add(index)
        largest = max(largest, abs(nx - x), abs(ny - y), abs(nz - z))
        count += 1
    return count, largest


def _target_side(target: MorphTarget, weight: float) -> tuple[tuple[int, ...], tuple[tuple[float, float, float], ...]]:
    """The ``incr`` deltas for a positive weight and the ``decr`` deltas for a negative one."""
    if weight > 0:
        return target.incr_indices, target.incr_vectors
    return target.decr_indices, target.decr_vectors


def _dial_targets() -> dict[str, str]:
    """Every dial mapped to its target, with ``""`` for a regionally scaled one."""
    merged = dict(DIAL_TARGETS)
    merged.update(SIDE_TARGETS)
    for key in REGIONAL_SCALES:
        merged.setdefault(key, "")
    return merged


def dials_from_parameters(parameters: object) -> dict[str, float]:
    """Translate a :class:`core.scene3d_figures.FigureParameters` into pack dials.

    The studio's dials are *measurements* (metres, multipliers, named styles); the
    pack's targets are artist-authored morphs. This is the one place the two
    vocabularies meet, and it converts a measurement into the ``0..1`` value that
    applies the matching target: a measurement above its neutral value moves the
    ``incr`` target, below it the ``decr`` one.

    The mapping is deliberately *relative to the parameter's own default*, so a
    figure that was never edited (every dial at its default) comes out as the base
    body -- the honest "no morph" state -- and an edited figure moves from there.
    """

    def around(value: float, neutral: float, reach: float) -> float:
        """A multiplier's dial value: 1.0 is neutral, ``1 +- reach`` is the extreme."""
        if reach <= 0:
            return NEUTRAL
        return max(0.0, min(1.0, NEUTRAL + (float(value) - neutral) / reach * NEUTRAL))

    def get(name: str, fallback: float) -> float:
        value = getattr(parameters, name, fallback)
        return fallback if value is None else float(value)

    def _unit(value: float) -> float:
        return max(0.0, min(1.0, value))


    dials: dict[str, float] = {
        # Measurements in metres, each neutral at the base body's own value, so a
        # figure that was never edited comes out as the base body exactly.
        "shoulder_width": around(get("shoulder_width", 0.42), 0.42, 0.16),
        "hip_width": around(get("hip_width", 0.30), 0.30, 0.10),
        "waist": around(get("waist", 0.26), 0.26, 0.10),
        "chest_depth": around(get("chest_depth", 0.22), 0.22, 0.08),
        # Shape dials that are *amounts*: 0 is "none", so they only ever move
        # upward and a default figure stays the base body.
        "bust": NEUTRAL + 0.5 * _unit(get("bust", 0.0)),
        "belly": NEUTRAL + 0.5 * _unit(get("belly", 0.0)),
        "muscle": NEUTRAL + 0.5 * _unit(get("muscle", 0.0)),
        "cheek": NEUTRAL + 0.5 * _unit(get("cheek", 0.0)),
        "chin_shape": NEUTRAL + 0.5 * _unit(get("chin_shape", 0.0)),
        # Multipliers around 1.0.
        "overall_scale": around(get("overall_scale", 1.0), 1.0, 0.40),
        "neck_thickness": around(get("neck_thickness", 1.0), 1.0, 0.45),
        "neck_length": around(get("neck_length", 1.0), 1.0, 0.45),
        "arm_thickness": around(get("arm_thickness", 1.0), 1.0, 0.45),
        "forearm_thickness": around(get("arm_thickness", 1.0), 1.0, 0.45),
        "thigh_thickness": around(get("thigh_thickness", 1.0), 1.0, 0.45),
        "calf_thickness": around(get("calf_thickness", 1.0), 1.0, 0.45),
        "arm_length": around(get("arm_length", 1.0), 1.0, 0.40),
        "forearm_length": around(get("forearm_length", 1.0), 1.0, 0.40),
        "thigh_length": around(get("thigh_length", 1.0), 1.0, 0.40),
        "lower_leg_length": around(get("lower_leg_length", 1.0), 1.0, 0.40),
        "hand_size": around(get("hand_size", 1.0), 1.0, 0.40),
        "foot_size": around(get("foot_size", 1.0), 1.0, 0.40),
        "hand_width": around(get("hand_width", 1.0), 1.0, 0.35),
        "hand_length": around(get("hand_length", 1.0), 1.0, 0.35),
        "foot_width": around(get("foot_width", 1.0), 1.0, 0.35),
        "foot_length": around(get("foot_length", 1.0), 1.0, 0.35),
        "limb_thickness": around(get("limb_thickness", 1.0), 1.0, 0.40),
        "leg_thickness": around(get("leg_thickness", 1.0), 1.0, 0.40),
        "face_width": around(get("face_width", 1.0), 1.0, 0.25),
        "eye_spacing": around(get("eye_spacing", 1.0), 1.0, 0.35),
        "brow_ridge": NEUTRAL + 0.5 * _unit(get("brow_ridge", 0.0)),
        "areola": NEUTRAL + 0.5 * _unit(get("areola", 0.0)),
        "genital_size": NEUTRAL + 0.5 * _unit(get("genital_size", 0.0) - 1.0),
        "waist_shape": NEUTRAL + 0.5 * _unit(get("waist_shape", 0.0)),
        "hip_shape": NEUTRAL + 0.5 * _unit(get("hip_shape", 0.0)),
        "shoulder_slope": NEUTRAL + 0.5 * _unit(get("shoulder_slope", 0.0)),
        # A head-scale dial of 1.0 is the adult ratio. A child carries a larger
        # head for its height, so the dial is mapped onto the *proportion*: below
        # 1.0 shrinks it, above 1.0 grows it, and the region scale below makes the
        # change a real proportion change rather than a uniform resize.
        "head_scale": around(get("head_scale", 1.0), 1.0, 0.55),
        "head_depth": around(get("head_depth", 1.0), 1.0, 0.30),
        "face_length": around(get("face_length", 1.0), 1.0, 0.25),
        "jaw_width": around(get("jaw_width", 1.0), 1.0, 0.30),
        "nose_size": around(get("nose_size", 1.0), 1.0, 0.45),
        "eye_size": around(get("eye_size", 1.0), 1.0, 0.45),
        "lips": around(get("lips", 1.0), 1.0, 0.45),
        "ear_size": around(get("ear_size", 1.0), 1.0, 0.45),
    }
    # The intimate switch: the base body is the "present" state, so 0 flattens the
    # region and 1 leaves it exactly as the modeller built it.
    genitalia = _unit(get("genitalia", 1.0))
    dials["genitalia"] = NEUTRAL + 0.5 * (genitalia - 1.0)
    # A named proportion nudges the whole figure the way its name implies, so the
    # Templates section still means what it says.
    style = str(getattr(parameters, "proportion", "average") or "average")
    shifts = {
        "athletic": {"shoulder_width": 0.10, "waist": -0.06, "muscle": 0.18},
        "slim": {"overall_scale": -0.10, "shoulder_width": -0.04},
        "curvy": {"hip_width": 0.12, "bust": 0.12, "waist": -0.04},
        "child": {"head_scale": 0.16, "overall_scale": -0.05},
    }.get(style, {})
    for key, delta in shifts.items():
        dials[key] = max(0.0, min(1.0, dials.get(key, NEUTRAL) + delta))
    return dials


def build_body(
    dials: dict[str, float] | None = None,
    *,
    height: float = DEFAULT_HEIGHT,
    name: str = "Body",
) -> HumanBody:
    """Morph the base body with ``dials`` and scale it into the project's space.

    ``dials`` are ``0..1`` values (``0.5`` leaves the base body alone). ``height``
    is the finished figure's height in metres, so the studio's height dial and the
    real body height agree.
    """
    data = load_pack()
    values = dict.fromkeys(_dial_targets(), NEUTRAL)
    values.update({key: float(value) for key, value in (dials or {}).items()})
    points: list[tuple[float, float, float]] = [
        (float(point[0]), float(point[1]), float(point[2])) for point in data.vertices
    ]
    markers: dict[str, tuple[float, float, float]] = {
        joint: (float(point[0]), float(point[1]), float(point[2]))
        for joint, point in data.joints.items()
    }
    points, moved, largest = apply_morph(points, values, joints=markers)
    # The joint markers are moved by the *same* dials, so a bone stays where the
    # morphed anatomy puts it instead of being left at the base body's position.
    markers = _morphed_markers(markers, values, points, data)
    points, scale, offset = normalize(points, height=height)
    # A head-scale dial is a *proportion*, not a resize: growing the head and then
    # re-normalising to the requested height would cancel it. It is therefore
    # applied after normalisation, about the neck, so a child's head really is
    # large for its body (the ratio the eye judges a child by) rather than the
    # whole figure being scaled up.
    head_dial = values.get("head_scale", NEUTRAL)
    if abs(head_dial - NEUTRAL) > 1e-6:
        neck = _joint_in_space("joint-neck", markers, scale, offset)
        if neck is not None:
            _scale_head(points, neck, 1.0 + (head_dial - NEUTRAL) * 2.0 * HEAD_STRENGTH)
    mesh = Mesh(
        name=f"{name} body",
        vertices=tuple(Vec3(*point) for point in points),
        faces=data.faces,
        vertex_colors=_scaled_colors(data, offset, scale),
    )
    return HumanBody(
        name=name,
        mesh=mesh,
        height=height,
        dials=values,
        moved_vertices=moved,
        max_displacement=largest,
        base_vertices=tuple(points),
        joints=markers,
        _scale=scale,
        _offset=offset,
    )


#: How far the head-scale dial changes the head's size at full deflection.
HEAD_STRENGTH = 0.45


def _joint_in_space(
    name: str,
    markers: dict[str, tuple[float, float, float]],
    scale: float,
    offset: Vec3,
) -> tuple[float, float, float] | None:
    """A joint marker mapped into the finished body's space."""
    point = markers.get(name)
    if point is None:
        return None
    return (
        (point[0] - offset.x) * scale,
        (point[1] - offset.y) * scale,
        (point[2] - offset.z) * scale,
    )


def _scale_head(
    points: list[tuple[float, float, float]],
    neck: tuple[float, float, float],
    factor: float,
) -> None:
    """Scale the head about the neck, blending in smoothly from the shoulders."""
    nx, ny, nz = neck
    # The head sits above the neck; a band below it blends the change into the
    # neck and shoulders so nothing tears, and the top of the head moves most.
    band = 0.10
    for index, (x, y, z) in enumerate(points):
        rise = y - ny
        if rise <= -band:
            continue
        # 0 at the bottom of the band, 1 from the neck upward.
        blend = 0.0 if rise <= 0.0 else min(rise / band, 1.0)
        if rise < 0.0:
            blend = 0.5 * (1.0 + math.cos(math.pi * (-rise) / band)) * 0.5
        amount = 1.0 + (factor - 1.0) * blend
        points[index] = (
            nx + (x - nx) * amount,
            ny + (y - ny) * amount,
            nz + (z - nz) * amount,
        )


def normalize(
    points: list[tuple[float, float, float]],
    *,
    height: float,
) -> tuple[list[tuple[float, float, float]], float, Vec3]:
    """Scale a body to ``height`` metres, feet on ``y = 0`` and centred on ``x``.

    The body's own width is preserved (only the whole figure is scaled), so a
    tall figure is the same *shape*, just bigger -- which is what a height dial
    should mean. The returned ``scale`` and ``offset`` are what a caller needs to
    map a joint marker from the pack's space into the finished body's space.
    """
    ys = [point[1] for point in points]
    low, high = min(ys), max(ys)
    span = high - low
    if span <= 1e-9:
        return points, 1.0, Vec3(0.0, 0.0, 0.0)
    scale = height / span
    center_x = sum(point[0] for point in points) / len(points)
    # The feet stay on the ground; the body is centred on x and on z.
    center_z = 0.0
    xs = sorted(point[0] for point in points)
    center_x = (xs[0] + xs[-1]) * 0.5
    zs = sorted(point[2] for point in points)
    center_z = (zs[0] + zs[-1]) * 0.5
    offset = Vec3(center_x, low, center_z)
    scaled = [
        ((point[0] - offset.x) * scale, (point[1] - offset.y) * scale, (point[2] - offset.z) * scale)
        for point in points
    ]
    return scaled, scale, offset


def _scaled_colors(
    data: HumanPack,
    offset: Vec3,
    scale: float,
) -> tuple[tuple[float, float, float], ...]:
    """Colours in the finished space: the pack's own colours, unchanged by scaling."""
    del offset, scale
    return _base_colors(data)


def _morphed_markers(
    markers: dict[str, tuple[float, float, float]],
    values: dict[str, float],
    points: list[tuple[float, float, float]],
    data: HumanPack,
) -> dict[str, tuple[float, float, float]]:
    """Move each joint marker the way the dials moved the body around it.

    A joint marker is a point *on the body*, so it must travel with the vertices
    that surround it. This takes the nearest few base vertices of each marker,
    measures how the morph moved them, and applies that same average displacement
    to the marker -- so the elbow follows an arm-length dial and the hip follows a
    hip-width dial, without a second hand-written rule per bone.
    """
    # The base body's own points (before any morph), so the displacement is real.
    base: list[tuple[float, float, float]] = [
        (float(point[0]), float(point[1]), float(point[2])) for point in data.vertices
    ]
    original: dict[str, tuple[float, float, float]] = dict(markers)
    result: dict[str, tuple[float, float, float]] = {}
    # A small neighbourhood is enough: the nearest few vertices sit on the joint.
    # The lookup is bucketed on a coarse grid, so this stays linear in the mesh
    # size instead of sorting 13k points per marker.
    cell = 0.5
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, (x, y, z) in enumerate(base):
        key = (round(x / cell), round(y / cell), round(z / cell))
        buckets.setdefault(key, []).append(index)
    for name, point in original.items():
        key = (round(point[0] / cell), round(point[1] / cell), round(point[2] / cell))
        candidates: list[int] = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    candidates.extend(buckets.get((key[0] + ox, key[1] + oy, key[2] + oz), ()))
        if not candidates:
            result[name] = point
            continue
        nearest = heapq.nsmallest(
            min(12, len(candidates)),
            candidates,
            key=lambda index: (
                (base[index][0] - point[0]) ** 2
                + (base[index][1] - point[1]) ** 2
                + (base[index][2] - point[2]) ** 2
            ),
        )
        dx = sum(points[index][0] - base[index][0] for index in nearest) / len(nearest)
        dy = sum(points[index][1] - base[index][1] for index in nearest) / len(nearest)
        dz = sum(points[index][2] - base[index][2] for index in nearest) / len(nearest)
        result[name] = (point[0] + dx, point[1] + dy, point[2] + dz)
    del values
    return result


def joint_positions(
    body: HumanBody,
    data: HumanPack | None = None,
) -> dict[str, Vec3]:
    """The pack's joint markers mapped into the finished body's space."""
    data = data or load_pack()
    scale = body._scale
    offset = body._offset
    source = body.joints or data.joints
    positions: dict[str, Vec3] = {}
    for name, point in source.items():
        positions[name] = Vec3(
            (point[0] - offset.x) * scale,
            (point[1] - offset.y) * scale,
            (point[2] - offset.z) * scale,
        )
    return positions


def build_skeleton(body: HumanBody, *, data: HumanPack | None = None) -> Skeleton:
    """A humanoid rig placed on *this* body's own joint markers.

    The bones are named the way the rest of the project names them (``hips``,
    ``spine``, ``chest``, ``neck``, ``head``, ``shoulder.L``, ``upper_arm.L``,
    ``forearm.L``, ``hand.L``, ``thigh.L``, ``shin.L``, ``foot.L``, ``toe.L``),
    and every one is placed from a MakeHuman joint marker, so it sits where the
    anatomy puts it and stays there when the body is morphed.
    """
    points = joint_positions(body, data)
    height = body.height

    def point(name: str, fallback: tuple[float, float, float]) -> Vec3:
        return points.get(name) or Vec3(*fallback)

    pelvis = point("joint-pelvis", (0.0, height * 0.52, 0.0))
    spine = point("joint-spine-2", (0.0, height * 0.60, 0.0))
    chest = point("joint-spine-4", (0.0, height * 0.70, 0.0))
    neck = point("joint-neck", (0.0, height * 0.80, 0.0))
    head = point("joint-head", (0.0, height * 0.86, 0.0))
    head_top = point("joint-head-2", (0.0, height * 0.97, 0.0))

    bones: list[Bone] = [
        Bone("hips", None, pelvis, Vec3(), Vec3(), _length(pelvis, spine)),
        Bone("spine", "hips", spine - pelvis, Vec3(), Vec3(), _length(spine, chest)),
        Bone("chest", "spine", chest - spine, Vec3(), Vec3(), _length(chest, neck)),
        Bone("neck", "chest", neck - chest, Vec3(), Vec3(), _length(neck, head)),
        Bone("head", "neck", head - neck, Vec3(), Vec3(), _length(head, head_top)),
    ]

    for side, suffix in (("l", "L"), ("r", "R")):
        clavicle = point(f"joint-{side}-clavicle", (0.0, height * 0.78, 0.0))
        shoulder = point(f"joint-{side}-shoulder", (0.0, height * 0.77, 0.0))
        elbow = point(f"joint-{side}-elbow", (0.0, height * 0.62, 0.0))
        hand = point(f"joint-{side}-hand", (0.0, height * 0.50, 0.0))
        hand_end = point(f"joint-{side}-hand-3", (0.0, height * 0.45, 0.0))
        hip = point(f"joint-{side}-upper-leg", (0.0, height * 0.50, 0.0))
        knee = point(f"joint-{side}-knee", (0.0, height * 0.28, 0.0))
        ankle = point(f"joint-{side}-ankle", (0.0, height * 0.06, 0.0))
        heel = point(f"joint-{side}-foot-1", (0.0, height * 0.03, 0.0))
        toe = point(f"joint-{side}-foot-2", (0.0, height * 0.02, 0.10))

        bones.extend(
            [
                Bone(f"shoulder.{suffix}", "chest", clavicle - chest, Vec3(), Vec3(), _length(clavicle, shoulder)),
                Bone(f"upper_arm.{suffix}", f"shoulder.{suffix}", shoulder - clavicle, Vec3(), Vec3(), _length(shoulder, elbow)),
                Bone(f"forearm.{suffix}", f"upper_arm.{suffix}", elbow - shoulder, Vec3(), Vec3(), _length(elbow, hand)),
                Bone(f"hand.{suffix}", f"forearm.{suffix}", hand - elbow, Vec3(), Vec3(), _length(hand, hand_end)),
                Bone(f"thigh.{suffix}", "hips", hip - pelvis, Vec3(), Vec3(), _length(hip, knee)),
                Bone(f"shin.{suffix}", f"thigh.{suffix}", knee - hip, Vec3(), Vec3(), _length(knee, ankle)),
                Bone(f"foot.{suffix}", f"shin.{suffix}", ankle - knee, Vec3(), Vec3(), _length(ankle, heel)),
                Bone(f"toe.{suffix}", f"foot.{suffix}", heel - ankle, Vec3(), Vec3(), _length(heel, toe)),
            ]
        )
    return Skeleton(tuple(bones))


def _length(start: Vec3, end: Vec3) -> float:
    """A bone's length, never zero (skinning divides by it)."""
    return max(1e-3, (end - start).length())


def build_figure(
    dials: dict[str, float] | None = None,
    *,
    height: float = DEFAULT_HEIGHT,
    name: str = "Body",
    material: object | None = None,
) -> SceneObject:
    """A morphed, bound, poseable human figure -- the studio's one body builder."""
    body = build_body(dials, height=height, name=name)
    skeleton = build_skeleton(body)
    skin = material or _skin_material(name)
    obj = SceneObject(name, body.mesh, skin, skeleton=skeleton)  # type: ignore[arg-type]
    obj.bind()
    return obj


def _skin_material(name: str) -> object:
    from core.scene3d import Material

    return Material(
        name=f"{name} skin",
        base_color=SKIN,
        specular=0.14,
        shininess=12.0,
        wrap=0.45,
    )


def apply_pose(obj: SceneObject, pose: dict[str, Vec3]) -> SceneObject:
    """Set a named pose on a bound figure (bone rotations in degrees)."""
    obj.pose = dict(pose)
    return obj


def _round(value: float, places: int = 6) -> float:
    return float(round(value, places))


__all__ = [
    "DIAL_TARGETS",
    "NEUTRAL",
    "SIDE_TARGETS",
    "HumanBody",
    "apply_morph",
    "apply_pose",
    "base_mesh",
    "build_body",
    "build_figure",
    "build_skeleton",
    "dials_from_parameters",
    "joint_positions",
    "normalize",
    "pack",
]
