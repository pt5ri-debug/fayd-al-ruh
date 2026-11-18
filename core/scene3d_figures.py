"""Built-in, procedurally generated figures -- so the studio works with no assets.

A DAZ-style studio is only useful if there is something to pose. Licensing real
figure content is not possible here, so this module **builds** the figures.

The body is a **single, connected skin**, not a bundle of separate parts. It is
built the way a sculptor builds: every limb, the torso, the head, and the face
features are smooth volumetric parts, they are **fused** where they meet by a
smooth-union (so a shoulder blends into the arm with no seam and no gap), and the
resulting surface is polygonized once, in :mod:`core.scene3d_sculpt`, into one
closed mesh. A viewer looking at the figure therefore sees one human body rather
than a set of disjoint pieces -- which is the whole point of a body.

The figure is parametric: a **build** family (feminine / masculine / neutral), a
**proportion** style (average / athletic / slim / curvy / child), named body
measurements, and a set of **shape** dials (bust, waist, hips, belly, muscle,
shoulder slope, neck, arm and leg thickness, hand and foot size, head and face
features) shape the body the way a DAZ figure's morph dials do. Every dial reaches
the geometry: it moves the volumetric parts the skin is sculpted from.

The generator is deterministic: the same parameters always produce the same figure,
so a saved scene reloads identically. The hair is built as a **separate object**
(see :func:`build_hair_object`) so it can carry its own material -- a figure's hair
colour is then real rather than the skin colour. :func:`build_figure` returns the
body alone, exactly as before, and :func:`build_figure_scene` adds the hair for you.

Nothing here is a claim about a commercial product: it is an original, parametric
humanoid, and every figure it makes says so.
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
from core.scene3d_sculpt import (
    DEFAULT_CELL,
    _Blob,
    capsule,
    ellipsoid,
    sculpt,
)

#: Kept for callers that import it; the sculptor no longer uses a fixed radial count.
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
    #: How finely the sculptor samples the body's surface: 0 is coarse (a fast,
    #: faceted body), 1 is the smooth default, 2 is very fine. Internally each
    #: level is a grid spacing in metres -- smaller is smoother and heavier.
    subdivisions: int = 1
    #: Extra Laplacian relaxation passes applied after the surface is polygonized
    #: (a cheap polish that removes the last facets). ``0`` is the raw sculpt.
    smoothing: int = 0
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


#: The grid spacing (metres) each subdivision level sculpts at. Level 1 is the
#: studio's smooth default; level 0 is a fast, coarser body and level 2 is very fine.
_DETAIL_SPACING: dict[int, float] = {0: 0.018, 1: DEFAULT_CELL, 2: 0.007}


def _proportion_scale(params: FigureParameters) -> dict[str, float]:
    """Per-proportion tweaks the shape dials multiply into the base measurements.

    A proportion is a *style*, so it nudges several measurements at once rather than
    one: ``athletic`` broadens the shoulders and thins the waist, ``slim`` narrows
    everything, ``curvy`` widens the hips, and ``child`` shortens the limbs.
    """
    style = params.proportion
    if style == "athletic":
        return {"shoulder": 1.10, "waist": 0.92, "hip": 0.96, "limb": 1.05}
    if style == "slim":
        return {"shoulder": 0.94, "waist": 0.90, "hip": 0.92, "limb": 0.90}
    if style == "curvy":
        return {"shoulder": 0.98, "waist": 0.94, "hip": 1.14, "limb": 1.02}
    if style == "child":
        return {"shoulder": 0.86, "waist": 1.02, "hip": 0.92, "limb": 0.96}
    return {"shoulder": 1.0, "waist": 1.0, "hip": 1.0, "limb": 1.0}


def _body_parts(params: FigureParameters) -> list[_Blob]:
    """Every volumetric part of the body, ready to be fused into one skin.

    The list is the body, read top to bottom -- pelvis, waist, chest, shoulders,
    neck, head, face, ears, arms, hands, legs, feet, and the coloured details --
    each a smooth field. The sculptor's smooth union fuses them into one body
    rather than a pile of shapes, so a viewer sees a single skin with no gap and
    no seam.

    The proportions are the ones a figure is judged by, and every length here is a
    **fraction of the total height** (so a tall figure is the same shape, only
    bigger): a head-to-height ratio of about eight, the shoulder line at 0.82 of
    the height, the hip line at 0.48, and arms and legs of natural length. The
    shoulders and the hips are wide, flattened masses and the arms and legs are
    kept **clear of the torso**, so the armpit and the inner thigh stay open the
    way they do on a real body. That is what makes the result read as a person
    rather than a blob.
    """
    skin = params.skin
    H = params.height
    s = H / 1.75  # scales the metre dials (shoulder width, etc.) to this height
    style = _proportion_scale(params)
    limb = params.limb_thickness * style["limb"]

    shoulder = params.shoulder_width * style["shoulder"] * s
    hips = params.hip_width * style["hip"] * (1.0 + 0.35 * params.hip_shape) * s
    waist = params.waist * style["waist"] * (1.0 - 0.28 * params.waist_shape) * s
    chest_depth = params.chest_depth * (0.9 + 0.25 * params.muscle) * s
    belly = params.belly

    parts: list[_Blob] = []

    # -- torso: a chain of blended masses from the pelvis up to the shoulders ---
    hip_y = 0.480 * H
    waist_y = 0.600 * H
    chest_y = 0.720 * H
    shoulder_y = 0.820 * H
    pelvis_half = hips * 0.5
    waist_half = waist * 0.5
    # The chest is *narrower* than the shoulder line, so the arms hang clear of it
    # and the armpit stays open rather than fusing into the torso.
    chest_half = shoulder * 0.5 * 0.72

    parts.append(
        capsule((0.0, hip_y, 0.0), (0.0, waist_y, 0.0),
                pelvis_half * 0.95, waist_half, skin, blend=0.05)
    )
    parts.append(
        capsule((0.0, waist_y, belly * 0.015 * H), (0.0, chest_y, 0.004 * s),
                waist_half, chest_half, skin, blend=0.05)
    )
    parts.append(
        capsule((0.0, chest_y, 0.004 * s), (0.0, shoulder_y, 0.0),
                chest_half, chest_half * 1.05, skin, blend=0.05)
    )
    # The chest is only a little deeper than it is wide, so the body is not a tube.
    parts.append(
        ellipsoid((0.0, chest_y - 0.005 * H, chest_depth * 0.14),
                  (chest_half * 1.02, 0.032 * H, chest_depth * 0.40), skin, blend=0.05)
    )
    if belly > 0.02:
        parts.append(
            ellipsoid((0.0, waist_y - 0.005 * H, chest_depth * 0.10),
                      (waist_half * 1.02, 0.030 * H, chest_depth * (0.36 + 0.30 * belly)),
                      skin, blend=0.05)
        )
    # Hip masses: two flattened, wide masses on the sides of the pelvis.
    for side in (-1.0, 1.0):
        parts.append(
            ellipsoid((side * pelvis_half * 0.55, hip_y + 0.010 * H, -chest_depth * 0.04),
                      (pelvis_half * 0.42, 0.032 * H, chest_depth * 0.40), skin, blend=0.05)
        )

    # -- shoulders: a wide flattened mass each side, so the shoulder line reads --
    shoulder_x = shoulder * 0.5
    for side in (-1.0, 1.0):
        drop = params.shoulder_slope * 0.02 * H
        parts.append(
            ellipsoid((side * shoulder_x * 0.80, shoulder_y - drop, 0.0),
                      (shoulder * 0.22, 0.040 * H, chest_depth * 0.40), skin, blend=0.05)
        )

    # -- bust ------------------------------------------------------------
    if params.bust > 0.02:
        radius = (0.012 + 0.024 * params.bust) * H
        forward = chest_depth * 0.32 + 0.02 * params.bust
        for side in (-1.0, 1.0):
            centre = (side * shoulder * 0.150, chest_y - 0.015 * H, forward)
            parts.append(
                ellipsoid(centre, (radius * 1.05, radius * 0.92, radius * 0.95), skin, blend=0.03)
            )
            parts.append(
                ellipsoid((centre[0], centre[1] + radius * 0.05, centre[2] + radius * 0.80),
                          (radius * 0.30 * params.areola, radius * 0.30 * params.areola,
                           radius * 0.16),
                          params.nipple_color, blend=0.010)
            )

    # -- neck and head ---------------------------------------------------
    neck_radius = 0.034 * H * params.neck_thickness
    neck_y = shoulder_y + 0.025 * H
    head_base = 0.875 * H
    parts.append(
        capsule((0.0, shoulder_y + 0.005 * H, -0.004 * s), (0.0, neck_y, -0.008 * s),
                neck_radius * 1.15, neck_radius, skin, blend=0.04)
    )
    # A head is about 0.125 of the total height tall and about 0.72 as wide as it
    # is tall: the classic proportions, so the head reads as a head.
    head_length = 0.125 * H * params.head_scale * params.face_length
    head_width = head_length * 0.36 * params.face_width
    head_depth = head_length * 0.42 * params.face_length
    head_centre = head_base + head_length * 0.5
    parts.append(
        ellipsoid((0.0, head_centre, 0.006 * s),
                  (head_width, head_length * 0.50, head_depth), skin, blend=0.030)
    )
    jaw_width = head_width * params.jaw_width
    chin_drop = (1.0 - 0.30 * params.chin_shape) * head_length * 0.12
    parts.append(
        ellipsoid((0.0, head_base + head_length * 0.17 - chin_drop, 0.012 * s),
                  (jaw_width * 0.82, head_length * 0.18, head_depth * 0.78), skin, blend=0.025)
    )
    if params.cheek > 0.0:
        for side in (-1.0, 1.0):
            parts.append(
                ellipsoid((side * head_width * 0.58, head_base + head_length * 0.44, 0.026 * s),
                          (head_width * 0.24 * params.cheek, head_length * 0.13,
                           head_depth * 0.36 * params.cheek),
                          skin, blend=0.018)
            )

    # -- ears ------------------------------------------------------------
    ear = 0.012 * H * params.head_scale * params.ear_size
    for side in (-1.0, 1.0):
        parts.append(
            ellipsoid((side * head_width * 0.98, head_base + head_length * 0.52, -0.006 * s),
                      (ear * 0.34, ear * 0.78, ear * 0.62), skin, blend=0.018)
        )

    # -- face ------------------------------------------------------------
    # The face features sit **on the head's own front surface**, computed from the
    # head ellipsoid rather than guessed, so the eyes, the nose, and the lips really
    # stand proud of the skin instead of being buried inside it.
    head_z = 0.006 * s

    def _surface_z(y: float) -> float:
        dy = (y - head_centre) / (head_length * 0.50)
        return head_z + head_depth * math.sqrt(max(0.0, 1.0 - dy * dy))

    eye_y = head_base + head_length * 0.58
    eye_x = head_width * 0.36 * params.eye_spacing
    eye_r = 0.006 * H * params.head_scale * params.eye_size
    eye_z = _surface_z(eye_y)
    for side in (-1.0, 1.0):
        parts.append(
            ellipsoid((side * eye_x, eye_y, eye_z - eye_r * 0.35),
                      (eye_r, eye_r * 0.92, eye_r * 0.75), (0.94, 0.94, 0.95), blend=0.006)
        )
        parts.append(
            ellipsoid((side * eye_x, eye_y, eye_z + eye_r * 0.12),
                      (eye_r * 0.55, eye_r * 0.55, eye_r * 0.45), params.eye_color, blend=0.005)
        )
        thickness = 0.002 * H * (0.5 + 0.7 * params.brow_ridge)
        brow_y = eye_y + eye_r * 2.2
        parts.append(
            ellipsoid((side * eye_x, brow_y, _surface_z(brow_y) - thickness * 0.2),
                      (head_width * 0.38, thickness, head_depth * 0.12),
                      params.hair_color, blend=0.006)
        )
    nose_y = head_base + head_length * 0.44
    nose = 0.006 * H * params.head_scale * params.nose_size
    parts.append(
        ellipsoid((0.0, nose_y, _surface_z(nose_y) + nose * 0.25),
                  (nose, nose * 1.5, nose * 1.25), skin, blend=0.010)
    )
    lips_y = head_base + head_length * 0.30
    parts.append(
        ellipsoid((0.0, lips_y, _surface_z(lips_y) - 0.002 * H),
                  (0.010 * H * params.head_scale * params.lips,
                   0.0045 * H * params.head_scale * params.lips,
                   head_depth * 0.12),
                  params.lip_color, blend=0.008)
    )

    # -- arms: kept clear of the torso so the armpit stays open --------------
    upper_arm_len = 0.170 * H
    fore_arm_len = 0.145 * H
    arm_radius = 0.026 * H * limb * params.arm_thickness
    arm_centre = shoulder * 0.5 * 0.96
    for side in (-1.0, 1.0):
        top = Vec3(side * arm_centre, shoulder_y - 0.010 * H, 0.0)
        elbow = Vec3(top.x, top.y - upper_arm_len, 0.006 * s)
        wrist = Vec3(top.x, elbow.y - fore_arm_len, 0.012 * s)
        parts.append(capsule(top, elbow, arm_radius * 1.10, arm_radius * 0.78, skin, blend=0.04))
        parts.append(capsule(elbow, wrist, arm_radius * 0.78, arm_radius * 0.58, skin, blend=0.035))
        _add_hand(parts, params, skin, wrist, arm_radius, side, s)

    # -- legs: the feet sit on the ground and the legs stay clear of each other --
    # Hip joint at the hip line, knee and ankle a natural distance below it, so the
    # ankle lands about 0.04 of the height above the ground and the feet rest on it.
    thigh_len = 0.240 * H
    shin_len = 0.200 * H
    leg_radius = 0.040 * H * limb * params.leg_thickness
    for side in (-1.0, 1.0):
        hip_x = side * hips * 0.5 * 0.62
        hip_top = Vec3(hip_x, hip_y, 0.0)
        knee = Vec3(hip_x, hip_top.y - thigh_len, 0.006 * s)
        ankle = Vec3(hip_x, knee.y - shin_len, 0.0)
        parts.append(capsule(hip_top, knee, leg_radius * 1.10, leg_radius * 0.56, skin, blend=0.055))
        parts.append(capsule(knee, ankle, leg_radius * 0.56, leg_radius * 0.38, skin, blend=0.035))
        _add_foot(parts, params, skin, ankle, leg_radius, side, s)

    _add_genitalia(parts, params, skin, hip_y, hips, s)
    return parts


def _add_hand(
    parts: list[_Blob],
    params: FigureParameters,
    skin: tuple[float, float, float],
    wrist: Vec3,
    arm_radius: float,
    side: float,
    s: float,
) -> None:
    """A palm, four fingers, and a thumb, each tipped with a nail."""
    size = params.hand_size * s
    hand_r = arm_radius * 0.85 * size
    palm = Vec3(wrist.x, wrist.y - hand_r * 1.15, wrist.z + 0.004 * s)
    parts.append(
        ellipsoid((palm.x, palm.y, palm.z), (hand_r * 0.85, hand_r * 1.05, hand_r * 0.40),
                  skin, blend=0.018)
    )
    finger_len = hand_r * 1.0
    finger_r = hand_r * 0.17
    for index in range(4):
        offset = (index - 1.5) * hand_r * 0.44
        base = Vec3(palm.x + offset, palm.y - hand_r * 0.95, palm.z)
        tip = Vec3(base.x, base.y - finger_len, base.z)
        parts.append(capsule(base, tip, finger_r, finger_r * 0.9, skin, blend=0.006))
        # The nail is a flattened disc *on* the fingertip, its centre well inside
        # the finger's rounded end so the smooth union always fuses the two.
        parts.append(
            ellipsoid((tip.x, tip.y - finger_r * 0.15, tip.z + finger_r * 0.5),
                      (finger_r * 0.9, finger_r * 0.6, finger_r * 0.9),
                      params.nail_color, blend=0.009)
        )
    thumb_base = Vec3(palm.x - side * hand_r * 0.85, palm.y + hand_r * 0.1,
                      palm.z + hand_r * 0.2)
    thumb_tip = Vec3(palm.x - side * hand_r * 1.25, palm.y - hand_r * 0.5,
                     palm.z + hand_r * 0.35)
    parts.append(capsule(thumb_base, thumb_tip, finger_r * 1.1, finger_r * 0.95, skin, blend=0.006))


def _add_foot(
    parts: list[_Blob],
    params: FigureParameters,
    skin: tuple[float, float, float],
    ankle: Vec3,
    leg_radius: float,
    side: float,
    s: float,
) -> None:
    """An ankle, a foot, and five toes, each tipped with a nail.

    The foot lies flat on the ground (``y = 0``): its sole is a flattened capsule
    from the heel to the ball, the toes fan out in front, and the ankle rises into
    the shin, so the leg really stands on the ground. Its length is a natural
    fraction of the height (about ``0.15``), so it reads as a foot rather than a stub.
    """
    size = params.foot_size * s
    width = leg_radius * 0.95 * size
    length = 0.105 * params.height * size
    sole_r = width * 0.58
    parts.append(
        capsule((ankle.x, ankle.y, ankle.z),
                (ankle.x, ankle.y - 0.020 * s, ankle.z + 0.012 * s),
                leg_radius * 0.46, width * 0.52, skin, blend=0.025)
    )
    # The sole is a flattened capsule from the heel to the ball, raised by its own
    # radius so its underside rests on the ground (``y = 0``); the ball is wide
    # enough that every toe overlaps it, so the toes fuse into the foot rather than
    # detaching as separate specks.
    heel = Vec3(ankle.x, sole_r, -length * 0.34)
    ball = Vec3(ankle.x, sole_r, length * 0.58)
    parts.append(capsule(heel, ball, sole_r * 0.86, sole_r, skin, blend=0.030))
    for index in range(5):
        offset = (index - 2.0) * width * 0.44
        toe_r = width * (0.30 if index == 0 else 0.24)
        # Each toe starts *inside* the ball and runs forward, resting on the ground
        # (raised by its own radius), so its base is buried in the sole and the two
        # are always one surface.
        base = Vec3(ankle.x + offset * 0.75, toe_r, ball.z - length * 0.20)
        tip = Vec3(ankle.x + offset, toe_r, ball.z + length * 0.22)
        parts.append(capsule(base, tip, toe_r * 1.05, toe_r, skin, blend=0.020))
        # The nail sits *on* the toe (its centre inside the toe's surface) so the
        # smooth union fuses it; a nail floating past the tip would detach.
        parts.append(
            ellipsoid((tip.x, tip.y + toe_r * 0.5, tip.z + toe_r * 0.35),
                      (toe_r * 0.8, toe_r * 0.6, toe_r * 0.8), params.nail_color, blend=0.009)
        )


def _add_genitalia(
    parts: list[_Blob],
    params: FigureParameters,
    skin: tuple[float, float, float],
    hip_y: float,
    hips: float,
    s: float,
) -> None:
    """The genital region, drawn from the build and shaped by its own dials.

    Kept simple and honest -- a smooth shape whose *size* and *colour* are the
    user's own dials -- so the body reads as anatomically complete without the
    geometry pretending to be a photograph. It sits on the **pelvis's own front
    surface** (the pelvis is a capsule of radius ``hips/2``), so it really protrudes
    and the dial changes the silhouette, and it is fused into the body like every
    other part, so it is not a separate piece.
    """
    if params.genitalia <= 0.0:
        return
    scale = params.genitalia * params.genital_size * s
    colour = params.genital_color
    centre_y = hip_y - 0.020 * s
    front = hips * 0.5 * 0.92
    if params.build == BUILD_FEMININE:
        parts.append(
            ellipsoid((0.0, centre_y, front),
                      (0.022 * scale, 0.016 * scale, 0.020 * scale), colour, blend=0.02)
        )
        parts.append(
            ellipsoid((0.0, centre_y - 0.008 * scale, front * 1.01),
                      (0.010 * scale, 0.011 * scale, 0.014 * scale), colour, blend=0.015)
        )
    else:
        shaft = 0.034 * scale
        parts.append(
            ellipsoid((0.0, centre_y - shaft * 0.5, front * 1.01),
                      (0.011 * scale, shaft * 0.6, 0.012 * scale), colour, blend=0.02)
        )
        parts.append(
            ellipsoid((0.0, centre_y - shaft * 1.05, front * 1.02),
                      (0.014 * scale, 0.014 * scale, 0.014 * scale), colour, blend=0.015)
        )
        for side in (-1.0, 1.0):
            parts.append(
                ellipsoid((side * 0.014 * scale, centre_y - 0.005 * s, front * 0.98),
                          (0.012 * scale, 0.013 * scale, 0.012 * scale), colour, blend=0.015)
            )


def build_figure_mesh(params: FigureParameters) -> Mesh:
    """Sculpt the body into one connected, closed, skinned mesh.

    The figure stands on ``y = 0``, centred on the origin, facing ``+Z``. It is a
    **single** mesh -- the torso, neck, head, face, arms, hands, legs, feet, and the
    coloured details are one fused skin, so there are no separate pieces and no gaps
    between them. It carries per-vertex colours for the skin, the lips, the irises,
    the areolae, the nails, and the genitalia. The hair is a separate object
    (:func:`build_hair_object`).
    """
    parts = _body_parts(params)
    cell = _DETAIL_SPACING.get(params.subdivisions, DEFAULT_CELL)
    mesh = sculpt(parts, cell=cell, name=f"{params.name} body")
    # Keep the one connected shell: a tiny feature (a nail) can polygonize as a few
    # stray triangles, and those specks are dropped so the body is a single skin.
    mesh = mesh.largest_shell()
    if params.smoothing > 0:
        mesh = mesh.smoothed(iterations=params.smoothing, factor=0.30)
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


def _hair_parts(params: FigureParameters, colour: tuple[float, float, float]) -> list[_Blob]:
    """The hair as volumetric parts: a cap over the crown, and a fall down the back."""
    if params.hair <= 0.0:
        return []
    s = params.height / 1.75
    head_length = 0.235 * params.head_scale * params.face_length * s
    head_width = 0.088 * params.head_scale * params.face_width * s
    head_depth = 0.100 * params.head_scale * params.face_length * s
    shoulder_y = 1.44 * s
    neck_y = shoulder_y + 0.04 * s
    head_base = neck_y + 0.05 * s
    crown = head_base + head_length * 0.62
    volume = 1.06 + 0.16 * params.hair
    parts: list[_Blob] = [
        # A larger mass than the skull, shifted back, so the face stays clear.
        ellipsoid((0.0, crown, -head_depth * 0.28),
                  (head_width * 1.18 * volume, head_length * 0.62 * volume,
                   head_depth * 1.22 * volume),
                  colour, blend=0.03)
    ]
    if params.hair >= 0.35:
        fall = (0.10 + 0.24 * params.hair) * params.head_scale * s
        parts.append(
            ellipsoid((0.0, crown - head_length * 0.5 - fall * 0.45, -head_depth * 1.0),
                      (head_width * 1.08 * volume, fall * 0.7, head_depth * 0.6),
                      colour, blend=0.05)
        )
    return parts


def build_hair_object(params: FigureParameters, parent_name: str) -> SceneObject | None:
    """The hair as its own object (its own material), or ``None`` when there is none.

    It is a *separate* object on purpose: the hair is not skinned to the body, so it
    keeps its own material and a hair colour is real rather than the skin colour.
    """
    parts = _hair_parts(params, params.hair_color)
    if not parts:
        return None
    mesh = sculpt(parts, cell=DEFAULT_CELL, name=f"{parent_name} hair")
    if not mesh.faces:
        return None
    material = Material(
        name=f"{parent_name} hair", base_color=params.hair_color, specular=0.4, shininess=56.0
    )
    return SceneObject(f"{parent_name} hair", mesh, material)


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
    # -- the four age-and-gender figures the Body & shape studio shows by default.
    # Each is a *real* body at that age and sex, on the proportions a figure is
    # judged by (the head-to-height ratio and the limb lengths are what make a
    # child read as a child), and every one fuses into a single connected skin.
    "woman": FigureParameters(
        "Woman",
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
    "man": FigureParameters(
        "Man",
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
    "girl": FigureParameters(
        "Girl",
        build=BUILD_FEMININE,
        height=1.32,
        proportion="child",
        shoulder_width=0.30,
        hip_width=0.25,
        chest_depth=0.17,
        waist=0.21,
        head_scale=1.12,
        hand_size=0.84,
        foot_size=0.86,
        arm_thickness=0.90,
        leg_thickness=0.94,
        bust=0.0,
        waist_shape=0.15,
        hip_shape=0.15,
        lips=1.05,
        hair=1.35,
        hair_color=(0.22, 0.15, 0.09),
    ),
    "boy": FigureParameters(
        "Boy",
        build=BUILD_MASCULINE,
        height=1.38,
        proportion="child",
        shoulder_width=0.33,
        hip_width=0.26,
        chest_depth=0.18,
        waist=0.23,
        head_scale=1.08,
        hand_size=0.88,
        foot_size=0.90,
        arm_thickness=0.94,
        leg_thickness=0.96,
        muscle=0.1,
        jaw_width=1.02,
        brow_ridge=0.15,
        hair=0.75,
        hair_color=(0.20, 0.14, 0.09),
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
