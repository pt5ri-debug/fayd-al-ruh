"""The body studio: build a character's whole body, shape by shape.

The Romance & Companion Studio's managers can already give a character a name, a
personality, and a **picture** -- but the picture was either a file the user
uploaded or an abstract generated image, never a *body*. This module is the engine
behind the **Body & shape** window: it renders the character's own body from a
parametric figure and lets every measurement be dialled.

It is a **thin layer over the project's existing 3D engine** -- never a second
figure builder:

* the geometry and the morph dials are :class:`core.scene3d_figures.FigureParameters`;
* the figure, its skeleton, and its skinning are :mod:`core.scene3d_figures` /
  :mod:`core.scene3d`;
* the pixels are the CPU software rasteriser in :mod:`core.scene3d_render`;
* the picture is stored through the Studio's own avatar manager, so a rendered body
  lands exactly where an uploaded picture does.

Everything is local and deterministic: nothing here opens a network connection or
downloads an asset, and the same dials always produce the same body.

**Honest by construction.** The figure is a *stylised, parametric* humanoid, not a
licensed likeness and not a photograph; the render is a rasteriser, not a
commercial render engine. The module says so, the window says so, and no dial
promises more than the geometry can do. A dial the current figure has no geometry
for (the hair, when the hair dial is at zero) is simply not drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.scene3d import Camera, Material, Scene, SceneObject, Vec3, make_default_scene
from core.scene3d_figures import (
    BUILD_FEMININE,
    BUILD_MASCULINE,
    BUILD_NEUTRAL,
    BUILDS,
    FIGURE_PRESETS,
    FigureParameters,
    build_figure,
    build_hair_object,
    pose_preset_names,
)
from core.scene3d_render import render_scene

#: The ready-made poses the body studio offers, taken from the figure rig so the
#: body studio and the Scene Studio share one set.
POSE_PRESET_KEYS: tuple[str, ...] = pose_preset_names()

#: What this engine is, said plainly, so a surface can show it.
HONESTY_LINE = (
    "A stylised, parametric figure drawn by the project's own CPU rasteriser -- "
    "not a licensed likeness, a photograph, or a commercial render."
)

#: The size a full-body preview is rendered at (portrait), and the square an
#: avatar is rendered at. Both are real files the user can open.
PREVIEW_WIDTH = 420
PREVIEW_HEIGHT = 620
AVATAR_WIDTH = 512
AVATAR_HEIGHT = 512

#: The two camera framings the studio offers.
VIEW_FULL = "full"
VIEW_FACE = "face"
VIEWS: tuple[str, ...] = (VIEW_FULL, VIEW_FACE)
VIEW_TITLES: dict[str, str] = {VIEW_FULL: "Full body", VIEW_FACE: "Face and head"}

#: Ready-made camera angles, as ``(key, title, azimuth, elevation)`` degrees. The
#: camera tool offers these as one-click viewpoints beside the free orbit sliders.
CAMERA_PRESETS: tuple[tuple[str, str, float, float], ...] = (
    ("front", "Front", 0.0, 0.0),
    ("back", "Back", 180.0, 0.0),
    ("left", "Left side", -90.0, 0.0),
    ("right", "Right side", 90.0, 0.0),
    ("three_quarter", "Three-quarter", 35.0, 8.0),
    ("top", "From above", 0.0, 70.0),
    ("low", "From below", 0.0, -35.0),
)

#: The neutral camera an orbit starts from.
CAMERA_DEFAULT_AZIMUTH = 0.0
CAMERA_DEFAULT_ELEVATION = 0.0
CAMERA_DEFAULT_ZOOM = 1.0
#: The zoom range the camera tool allows (a distance multiplier).
CAMERA_MIN_ZOOM = 0.35
CAMERA_MAX_ZOOM = 3.0

#: The fallback skin a mesh that carries no per-vertex colours is drawn with.
SKIN_COLOUR: tuple[float, float, float] = (0.82, 0.66, 0.56)


def camera_preset_titles() -> tuple[tuple[str, str], ...]:
    """``(key, title)`` for each ready-made camera angle."""
    return tuple((key, title) for key, title, _azimuth, _elevation in CAMERA_PRESETS)


def camera_preset(key: str) -> tuple[float, float]:
    """The ``(azimuth, elevation)`` a named camera angle asks for.

    An unknown key is refused with a ``KeyError`` rather than silently returning
    the front view, so a typo in a window or a CLI call is reported.
    """
    for preset_key, _title, azimuth, elevation in CAMERA_PRESETS:
        if preset_key == key:
            return azimuth, elevation
    raise KeyError(f"Unknown camera preset {key!r}.")


def clamp_zoom(value: float) -> float:
    """Bring a zoom multiplier inside the camera tool's own range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return CAMERA_DEFAULT_ZOOM
    return min(CAMERA_MAX_ZOOM, max(CAMERA_MIN_ZOOM, number))

#: The dial groups, in the order the window shows them.
DIAL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("proportions", "Proportions", "القياسات"),
    ("torso", "Torso and shape", "الجذع والقوام"),
    ("breasts", "Breasts", "الثديان"),
    ("limbs", "Arms and legs", "الذراعان والساقان"),
    ("measurements", "Fine measurements", "القياسات الدقيقة"),
    ("head", "Head and face", "الرأس والوجه"),
    ("intimate", "Intimate detail", "التفاصيل الحميمة"),
    ("detail", "Mesh and detail", "الشبكة والتفاصيل"),
    ("hair", "Hair", "الشعر"),
    ("colour", "Skin and hair colour", "لون البشرة والشعر"),
    ("pose", "Pose", "الوضعية"),
)

_GROUP_TITLES: dict[str, str] = {key: en for key, en, _ar in DIAL_GROUPS}
_GROUP_TITLES_AR: dict[str, str] = {key: ar for key, _en, ar in DIAL_GROUPS}


@dataclass(frozen=True)
class DialSpec:
    """One editable measurement: what it is called and the range it accepts.

    ``kind`` is ``float`` (a slider), ``int`` (a whole number), ``choice`` (one of
    ``choices``), or ``colour`` (an RGB triple). ``minimum``/``maximum`` bound a
    numeric dial so a hand-edited value can never be stored out of range.
    """

    key: str
    label: str
    label_ar: str
    group: str
    kind: str = "float"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    choices: tuple[str, ...] = ()
    hint: str = ""

    def clamp(self, value: Any) -> Any:
        """Return ``value`` brought inside this dial's own range.

        A choice dial **refuses** a value it does not offer rather than silently
        substituting one, so a typo is reported instead of quietly changing the
        character's body.
        """
        if self.kind == "choice":
            text = str(value)
            if text not in self.choices:
                raise ValueError(
                    f"{value!r} is not a {self.key}; choose from {', '.join(self.choices)}."
                )
            return text
        if self.kind == "colour":
            return as_colour(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.minimum
        number = min(self.maximum, max(self.minimum, number))
        return round(number) if self.kind == "int" else round(number, 4)


def as_colour(value: Any) -> tuple[float, float, float]:
    """Coerce anything colour-shaped into a clamped ``(r, g, b)`` triple."""
    try:
        parts = tuple(float(component) for component in value)[:3]
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    return tuple(min(1.0, max(0.0, component)) for component in parts)  # type: ignore[return-value]


#: The proportion styles a figure can be given (the figure builder's own set).
PROPORTION_STYLES: tuple[str, ...] = ("average", "athletic", "slim", "curvy", "child")

#: Every dial the studio can edit, grouped. The keys are the parameter field names
#: themselves, so setting a dial is a direct write to the figure's own parameters.
DIALS: tuple[DialSpec, ...] = (
    DialSpec(
        "build", "Build family", "نمط الجسم", "proportions",
        kind="choice", choices=BUILDS, hint="Feminine / masculine / neutral",
    ),
    DialSpec(
        "proportion", "Proportion style", "نمط القوام", "proportions",
        kind="choice", choices=PROPORTION_STYLES, hint="Average / athletic / slim / curvy / child",
    ),
    DialSpec(
        "height", "Height (metres)", "الطول (متر)", "proportions",
        minimum=1.20, maximum=2.10, step=0.01, hint="Total standing height",
    ),
    DialSpec("shoulder_width", "Shoulder width", "عرض الكتفين", "proportions", minimum=0.26, maximum=0.62, step=0.01),
    DialSpec("hip_width", "Hip width", "عرض الحوض", "proportions", minimum=0.20, maximum=0.46, step=0.01),
    DialSpec("chest_depth", "Chest depth", "عمق الصدر", "proportions", minimum=0.14, maximum=0.36, step=0.01),
    DialSpec("waist", "Waist width", "عرض الخصر", "proportions", minimum=0.16, maximum=0.42, step=0.01),
    DialSpec("bust", "Bust fullness", "امتلاء الصدر", "breasts", hint="Base chest fullness and forward shape"),
    DialSpec(
        "breast_size", "Breast size", "حجم الثدي", "breasts",
        minimum=0.30, maximum=2.50, step=0.01,
        hint="The overall size of the breast (a multiplier; 1 is the reference)",
    ),
    DialSpec(
        "breast_size_left", "Left breast size", "حجم الثدي الأيسر", "breasts",
        minimum=0.30, maximum=2.50, step=0.01,
        hint="The left breast's own size, independent of the right one",
    ),
    DialSpec(
        "breast_size_right", "Right breast size", "حجم الثدي الأيمن", "breasts",
        minimum=0.30, maximum=2.50, step=0.01,
        hint="The right breast's own size, independent of the left one",
    ),
    DialSpec(
        "chest_size", "Chest size (ribcage)", "حجم الصدر (القفص الصدري)", "breasts",
        minimum=0.60, maximum=1.60, step=0.01,
        hint="The whole chest the breasts sit on, separate from the breast volume",
    ),
    DialSpec(
        "breast_projection", "Breast projection", "بروز الثدي", "breasts",
        minimum=0.30, maximum=2.50, step=0.01,
        hint="How far the breast projects forward from the chest",
    ),
    DialSpec(
        "breast_fullness", "Breast fullness", "امتلاء الثدي", "breasts",
        minimum=0.30, maximum=2.50, step=0.01,
        hint="How round and full the breast is",
    ),
    DialSpec(
        "breast_separation", "Breast separation", "تباعد الثديين", "breasts",
        minimum=0.50, maximum=1.80, step=0.01,
        hint="How far apart the two breasts sit",
    ),
    DialSpec(
        "breast_height", "Breast height", "ارتفاع الثدي", "breasts",
        minimum=0.60, maximum=1.40, step=0.01,
        hint="How high on the chest the breasts sit",
    ),
    DialSpec(
        "breast_sag", "Breast sag", "ترهل الثدي", "breasts",
        minimum=0.0, maximum=1.0, step=0.01,
        hint="How much the lower half hangs (0 is perky, 1 is a full droop)",
    ),
    DialSpec("waist_shape", "Waist curve", "انحناء الخصر", "torso"),
    DialSpec("hip_shape", "Hip curve", "انحناء الحوض", "torso"),
    DialSpec("belly", "Belly", "البطن", "torso"),
    DialSpec("muscle", "Muscle", "العضلات", "torso"),
    DialSpec("shoulder_slope", "Shoulder slope", "ميل الكتفين", "torso"),
    DialSpec("limb_thickness", "Overall limb thickness", "سماكة الأطراف", "limbs", minimum=0.70, maximum=1.45, step=0.01),
    DialSpec("neck_thickness", "Neck thickness", "سماكة الرقبة", "limbs", minimum=0.60, maximum=1.60, step=0.01),
    DialSpec("arm_thickness", "Arm thickness", "سماكة الذراع", "limbs", minimum=0.60, maximum=1.70, step=0.01),
    DialSpec("leg_thickness", "Leg thickness", "سماكة الساق", "limbs", minimum=0.60, maximum=1.70, step=0.01),
    DialSpec("hand_size", "Hand size", "حجم اليد", "limbs", minimum=0.60, maximum=1.50, step=0.01),
    DialSpec("foot_size", "Foot size", "حجم القدم", "limbs", minimum=0.60, maximum=1.50, step=0.01),
    # -- fine measurements (FR-CCE): an independent control per part, on top of the
    #    coarse dials above. Each 1.0 default leaves the body exactly as it was.
    DialSpec(
        "overall_scale", "Overall build (girth)", "البنية العامة", "measurements",
        minimum=0.70, maximum=1.40, step=0.01,
        hint="A single multiplier on every girth, so the whole body is heavier or lighter",
    ),
    DialSpec("head_depth", "Head depth", "عمق الرأس", "measurements", minimum=0.70, maximum=1.35, step=0.01),
    DialSpec("neck_length", "Neck length", "طول الرقبة", "measurements", minimum=0.60, maximum=1.60, step=0.01),
    DialSpec("arm_length", "Upper-arm length", "طول العضد", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("forearm_length", "Forearm length", "طول الساعد", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("hand_length", "Hand length", "طول اليد", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("hand_width", "Hand width", "عرض اليد", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("thigh_thickness", "Thigh thickness", "سماكة الفخذ", "measurements", minimum=0.60, maximum=1.70, step=0.01),
    DialSpec("thigh_length", "Thigh length", "طول الفخذ", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("calf_thickness", "Calf thickness", "سماكة الساق", "measurements", minimum=0.60, maximum=1.70, step=0.01),
    DialSpec("lower_leg_length", "Lower-leg length", "طول الساق السفلى", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("foot_length", "Foot length", "طول القدم", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("foot_width", "Foot width", "عرض القدم", "measurements", minimum=0.70, maximum=1.40, step=0.01),
    DialSpec("head_scale", "Head size", "حجم الرأس", "head", minimum=0.80, maximum=1.30, step=0.01),
    DialSpec("face_length", "Face length", "طول الوجه", "head", minimum=0.80, maximum=1.25, step=0.01),
    DialSpec("face_width", "Face width", "عرض الوجه", "head", minimum=0.80, maximum=1.25, step=0.01),
    DialSpec("jaw_width", "Jaw width", "عرض الفك", "head", minimum=0.70, maximum=1.35, step=0.01),
    DialSpec("chin_shape", "Chin shape", "شكل الذقن", "head", hint="Rounded to pointed"),
    DialSpec("brow_ridge", "Brow", "الحاجبان", "head"),
    DialSpec("nose_size", "Nose size", "حجم الأنف", "head", minimum=0.60, maximum=1.60, step=0.01),
    DialSpec("eye_size", "Eye size", "حجم العين", "head", minimum=0.60, maximum=1.60, step=0.01),
    DialSpec("eye_spacing", "Eye spacing", "تباعد العينين", "head", minimum=0.60, maximum=1.50, step=0.01),
    DialSpec("lips", "Lips", "الشفتان", "head", minimum=0.50, maximum=1.80, step=0.01),
    DialSpec("cheek", "Cheeks", "الوجنتان", "head"),
    DialSpec("ear_size", "Ears", "الأذنان", "head", minimum=0.50, maximum=1.80, step=0.01),
    DialSpec(
        "genitalia", "Intimate detail", "التفاصيل الحميمة", "intimate",
        hint="Shows the genital set (adults only); 0 hides it",
    ),
    DialSpec(
        "genital_size", "Intimate size", "حجم التفاصيل الحميمة", "intimate",
        minimum=0.30, maximum=2.20, step=0.01,
        hint="The whole genital set, scaled",
    ),
    DialSpec(
        "genital_length", "Intimate length", "طول التفاصيل الحميمة", "intimate",
        minimum=0.30, maximum=2.20, step=0.01,
        hint="How far the set extends forward",
    ),
    DialSpec(
        "genital_fullness", "Intimate fullness", "امتلاء التفاصيل الحميمة", "intimate",
        minimum=0.30, maximum=2.20, step=0.01,
        hint="Labia for a feminine body; shaft and scrotum for a masculine one",
    ),
    DialSpec(
        "genital_direction", "Intimate direction", "اتجاه التفاصيل الحميمة", "intimate",
        minimum=-85.0, maximum=85.0, step=1.0,
        hint="Tilt of the set's axis in the vertical plane (degrees, + is up)",
    ),
    DialSpec(
        "genital_erection", "Erection (male)", "الانتصاب (للذكور)", "intimate",
        minimum=0.0, maximum=1.0, step=0.01,
        hint="The male organ: flaccid at 0, fully erect at 1 (adults only)",
    ),
    DialSpec(
        "genital_glans", "Glans size", "حجم الحشفة", "intimate",
        minimum=0.40, maximum=2.00, step=0.01,
        hint="The glans's own size, relative to the shaft",
    ),
    DialSpec(
        "genital_curve", "Shaft curve", "انحناء العضو", "intimate",
        minimum=0.0, maximum=1.0, step=0.01,
        hint="A gentle upward curve along the shaft (0 is straight)",
    ),
    DialSpec(
        "genital_sex", "Intimate set", "نوع التفاصيل الحميمة", "intimate",
        kind="choice", choices=("auto", "female", "male", "none"),
        hint="Which set the body carries: auto (from its build), female, male, or none",
    ),
    DialSpec("areola", "Areola size", "حجم الهالة", "intimate", minimum=0.25, maximum=2.50, step=0.01),
    DialSpec(
        "areola_raise", "Areola raise", "بروز الهالة", "intimate",
        minimum=0.0, maximum=1.50, step=0.01,
        hint="How far the areola stands proud of the breast; 0 lies flat",
    ),
    DialSpec(
        "nipple_size", "Nipple size", "حجم الحلمة", "intimate",
        minimum=0.05, maximum=1.20, step=0.01,
        hint="The nipple's radius, as a fraction of the areola's",
    ),
    DialSpec(
        "nipple_length", "Nipple length", "طول الحلمة", "intimate",
        minimum=0.0, maximum=2.50, step=0.01,
        hint="How far the nipple protrudes",
    ),
    DialSpec(
        "nipple_direction", "Nipple direction", "اتجاه الحلمة", "intimate",
        minimum=-85.0, maximum=85.0, step=1.0,
        hint="Tilt of the protrusion in the vertical plane (degrees, + is up)",
    ),
    DialSpec(
        "nipple_spacing", "Nipple spacing", "تباعد الحلمتين", "intimate",
        minimum=0.50, maximum=1.80, step=0.01,
        hint="How far apart the two are, as a multiplier of their natural spacing",
    ),
    DialSpec(
        "subdivisions", "Mesh detail (smoothness)", "دقة الشبكة (النعومة)", "detail",
        kind="int", minimum=0, maximum=2, step=1,
        hint="How finely the skin mesh is subdivided: 0 is the raw mesh, 1 is the "
             "smooth default, 2 is very fine",
    ),
    DialSpec(
        "smoothing", "Surface smoothing", "نعومة السطح", "detail",
        kind="int", minimum=0, maximum=4, step=1,
        hint="Relaxation passes that remove the last facets after subdivision",
    ),
    DialSpec("hair", "Hair volume", "كثافة الشعر", "hair"),
    DialSpec(
        "hair_length", "Hair length", "طول الشعر", "hair",
        minimum=0.0, maximum=1.0, step=0.01,
        hint="How far the hair falls down the back (0 is a close cap)",
    ),
    DialSpec(
        "hairline", "Hairline", "خط الشعر", "hair",
        minimum=0.0, maximum=1.0, step=0.01,
        hint="Where the hairline sits (0 is low on the brow, 1 is high)",
    ),
    DialSpec("skin", "Skin colour", "لون البشرة", "colour", kind="colour"),
    DialSpec("hair_color", "Hair colour", "لون الشعر", "colour", kind="colour"),
    DialSpec("lip_color", "Lip colour", "لون الشفتين", "colour", kind="colour"),
    DialSpec("eye_color", "Eye colour", "لون العين", "colour", kind="colour"),
    DialSpec("nipple_color", "Areola colour", "لون الهالة", "colour", kind="colour"),
    DialSpec("nipple_tip_color", "Nipple colour", "لون الحلمة", "colour", kind="colour"),
    DialSpec("nail_color", "Nail colour", "لون الأظافر", "colour", kind="colour"),
    DialSpec("genital_color", "Intimate colour", "لون التفاصيل الحميمة", "colour", kind="colour"),
    DialSpec(
        "pose", "Pose", "الوضعية", "pose",
        kind="choice", choices=POSE_PRESET_KEYS, hint="A ready-made body position",
    ),
)

_DIAL_BY_KEY: dict[str, DialSpec] = {dial.key: dial for dial in DIALS}

#: The choices a figure's build family offers.
BUILD_CHOICES: tuple[str, ...] = BUILDS
BUILD_TITLES: dict[str, str] = {
    BUILD_FEMININE: "Feminine",
    BUILD_MASCULINE: "Masculine",
    BUILD_NEUTRAL: "Neutral",
}


def dial_keys() -> tuple[str, ...]:
    return tuple(dial.key for dial in DIALS)


def dial_for(key: str) -> DialSpec | None:
    """The dial named ``key``, or ``None`` for a name the studio does not have."""
    return _DIAL_BY_KEY.get(key)


def dials_in_group(group: str) -> tuple[DialSpec, ...]:
    return tuple(dial for dial in DIALS if dial.group == group)


def group_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(group key, readable title)`` for each group, in the window's order."""
    if language == "ar":
        return tuple((key, _GROUP_TITLES_AR[key]) for key, _en, _ar in DIAL_GROUPS)
    return tuple((key, _GROUP_TITLES[key]) for key, _en, _ar in DIAL_GROUPS)


#: The ready-made bodies a user can start from and then edit. Each is a full set of
#: dials, so applying one really changes the body.
BODY_PRESETS: tuple[tuple[str, str, str, FigureParameters], ...] = (
    ("woman", "Woman", "امرأة", FIGURE_PRESETS["woman"]),
    ("girl", "Girl", "بنت", FIGURE_PRESETS["girl"]),
    ("man", "Man", "رجل", FIGURE_PRESETS["man"]),
    ("boy", "Boy", "ولد", FIGURE_PRESETS["boy"]),
    ("average", "Average adult", "بالغ معتاد", FIGURE_PRESETS["adult"]),
    ("athletic", "Athletic", "رياضي", FIGURE_PRESETS["athletic"]),
    ("slim", "Slim", "نحيف", FIGURE_PRESETS["slim"]),
    ("feminine", "Feminine", "أنثوي", FIGURE_PRESETS["feminine"]),
    ("masculine", "Masculine", "ذكوري", FIGURE_PRESETS["masculine"]),
    ("curvy", "Curvy", "ممتلئ", FIGURE_PRESETS["curvy"]),
    ("youthful", "Youthful", "شاب", FIGURE_PRESETS["child"]),
)

_BODY_PRESET_BY_KEY: dict[str, FigureParameters] = {
    key: params for key, _en, _ar, params in BODY_PRESETS
}


def body_preset_keys() -> tuple[str, ...]:
    return tuple(key for key, _en, _ar, _params in BODY_PRESETS)


def body_preset_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each ready-made body, in either language."""
    if language == "ar":
        return tuple((key, ar) for key, _en, ar, _params in BODY_PRESETS)
    return tuple((key, en) for key, en, _ar, _params in BODY_PRESETS)


#: Ready-made breast shapes, as one click on the *Breasts* group. Each is a set of
#: the breast dials applied together, so a user who wants a size picks a word
#: rather than guessing six numbers -- and every dial stays individually editable
#: afterwards, because a preset only sets them.
BREAST_PRESETS: tuple[tuple[str, str, str, dict[str, float]], ...] = (
    (
        "flat",
        "Flat",
        "مسطح",
        {"breast_size": 0.35, "bust": 0.10, "breast_projection": 0.50,
         "breast_fullness": 0.50, "breast_sag": 0.30},
    ),
    (
        "small",
        "Small",
        "صغير",
        {"breast_size": 0.70, "bust": 0.30, "breast_projection": 0.80,
         "breast_fullness": 0.55, "breast_sag": 0.35},
    ),
    (
        "medium",
        "Medium",
        "متوسط",
        {"breast_size": 1.00, "bust": 0.50, "breast_projection": 1.00,
         "breast_fullness": 0.65, "breast_sag": 0.40},
    ),
    (
        "large",
        "Large",
        "كبير",
        {"breast_size": 1.40, "bust": 0.70, "breast_projection": 1.20,
         "breast_fullness": 0.75, "breast_sag": 0.45},
    ),
    (
        "full",
        "Full",
        "ممتلئ",
        {"breast_size": 1.80, "bust": 0.85, "breast_projection": 1.35,
         "breast_fullness": 0.85, "breast_sag": 0.50},
    ),
)

_BREAST_PRESET_BY_KEY: dict[str, dict[str, float]] = {
    key: changes for key, _en, _ar, changes in BREAST_PRESETS
}


def breast_preset_keys() -> tuple[str, ...]:
    return tuple(key for key, _en, _ar, _changes in BREAST_PRESETS)


def breast_preset_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each ready-made breast shape."""
    if language == "ar":
        return tuple((key, ar) for key, _en, ar, _changes in BREAST_PRESETS)
    return tuple((key, en) for key, en, _ar, _changes in BREAST_PRESETS)


def breast_preset_changes(key: str) -> dict[str, float]:
    """The dial changes a named breast shape asks for.

    An unknown key is refused with a ``KeyError`` rather than silently returning
    nothing, so a typo in a window or a CLI call is reported.
    """
    try:
        changes = dict(_BREAST_PRESET_BY_KEY[key])
    except KeyError:
        raise KeyError(f"Unknown breast preset {key!r}.") from None
    # A named shape is symmetric: it clears any left/right asymmetry the user
    # had dialled in, so "Large" really means both.
    changes.setdefault("breast_size_left", 1.0)
    changes.setdefault("breast_size_right", 1.0)
    return changes


def apply_breast_preset(params: FigureParameters, key: str) -> FigureParameters:
    """A copy of ``params`` with a named breast shape applied."""
    return set_dials(params, breast_preset_changes(key))


def preset_parameters(key: str) -> FigureParameters | None:
    """The parameters a named preset starts from, or ``None`` for an unknown key."""
    return _BODY_PRESET_BY_KEY.get(key)


def _character_field(character: Any, name: str) -> Any:
    """Read a field off a character *or* the bundle that wraps it.

    The Studio's own ``character()`` returns a :class:`CharacterBundle` whose inner
    record carries the profile, while the CLI and the tests often pass the record
    itself, so both shapes are accepted rather than one being silently ignored.
    """
    if character is None:
        return None
    if hasattr(character, name):
        return getattr(character, name)
    inner = getattr(character, "character", None)
    return getattr(inner, name, None) if inner is not None else None


def build_for_gender(gender: str) -> str:
    """The build family a character's own recorded gender implies.

    A feminine or masculine record maps to that build; anything else -- including an
    unrecorded gender, or a non-binary one -- maps to the **neutral** build, because
    the studio must not guess a body from a blank field.
    """
    text = (gender or "").strip().casefold()
    if any(word in text for word in ("woman", "female", "feminine", "أنثى", "امرأة")):
        return BUILD_FEMININE
    if any(word in text for word in ("man", "male", "masculine", "ذكر", "رجل")):
        return BUILD_MASCULINE
    return BUILD_NEUTRAL


def default_parameters(character: Any = None, *, preset: str = "") -> FigureParameters:
    """The parameters a character's body starts from.

    With no explicit ``preset`` the studio opens on one of the **four ready-made
    figures** -- a woman, a girl, a man, or a boy -- chosen from the character's own
    recorded **gender and age** (see :mod:`core.romance.figures`), so opening the
    studio for a woman shows a woman's body and for a boy a boy's, rather than a
    single generic figure a user had reshaped. A named ``preset`` still wins, so the
    Templates section can put any figure or template on the body.
    """
    if not preset:
        from core.romance.figures import slot_parameters_for_character

        _slot, params = slot_parameters_for_character(character)
        return params
    base = preset_parameters(preset) or FIGURE_PRESETS["adult"]
    gender = str(_character_field(character, "gender") or "")
    build = build_for_gender(gender) if gender else base.build
    name = str(_character_field(character, "name") or "") or base.name
    return base.with_changes(name=name, build=build)


def values_of(params: FigureParameters) -> dict[str, Any]:
    """Every dial's current value on ``params`` (what the window's controls show)."""
    return {dial.key: getattr(params, dial.key) for dial in DIALS}


def set_dial(params: FigureParameters, key: str, value: Any) -> FigureParameters:
    """A copy of ``params`` with one dial set (clamped to that dial's range).

    An unknown dial name is refused with a ``KeyError`` rather than silently
    ignored, so a typo in a window or a CLI call is reported.
    """
    dial = _DIAL_BY_KEY.get(key)
    if dial is None:
        raise KeyError(f"Unknown body dial {key!r}.")
    value = dial.clamp(value)
    if isinstance(value, str):
        current = getattr(params, key, None)
        if isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
    return params.with_changes(**{key: value})


def set_dials(params: FigureParameters, changes: dict[str, Any]) -> FigureParameters:
    """A copy of ``params`` with several dials set at once."""
    updated = params
    for key, value in changes.items():
        updated = set_dial(updated, key, value)
    return updated


def reset_to_preset(params: FigureParameters, key: str) -> FigureParameters:
    """Return to a named preset's measurements, keeping the figure's own name."""
    preset = preset_parameters(key)
    if preset is None:
        raise KeyError(f"Unknown body preset {key!r}.")
    return preset.with_changes(name=params.name)


#: Human-readable titles for the ready-made poses.
POSE_TITLES: dict[str, str] = {
    "rest": "Rest (standing)",
    "contrapposto": "Contrapposto",
    "wave": "Wave",
    "walk": "Walk",
    "run": "Run",
    "sit": "Sit (chair)",
    "sit_ground": "Sit (ground)",
    "tpose": "T-pose",
    "reach": "Reach up",
    "arms_crossed": "Arms crossed",
    "hands_hips": "Hands on hips",
    "squat": "Squat",
    "lying": "Lying down",
}


def pose_keys() -> tuple[str, ...]:
    """The pose presets the studio offers."""
    return POSE_PRESET_KEYS


def pose_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each pose, in either language."""
    titles = {
        "ar": {
            "rest": "وقوف",
            "contrapposto": "وقفة مائلة",
            "wave": "تلويح",
            "walk": "مشي",
            "run": "جري",
            "sit": "جلوس (كرسي)",
            "sit_ground": "جلوس (أرض)",
            "tpose": "وضع T",
            "reach": "مدّ اليد",
            "arms_crossed": "ذراعان متقاطعتان",
            "hands_hips": "اليدان على الخصر",
            "squat": "قرفصاء",
            "lying": "استلقاء",
        }
    }.get(language, {})
    return tuple((key, titles.get(key) or POSE_TITLES.get(key, key)) for key in POSE_PRESET_KEYS)


def set_pose(params: FigureParameters, key: str) -> FigureParameters:
    """A copy of ``params`` in the named pose (an unknown pose is refused)."""
    if key not in POSE_PRESET_KEYS:
        raise KeyError(f"Unknown pose {key!r}; choose from {', '.join(POSE_PRESET_KEYS)}.")
    return params.with_changes(pose=key)


#: Cues that map a plain-language instruction onto a ready-made pose. The keys are
#: the words a user would type; the values are the pose they mean. Both languages
#: are covered, so "اجلس" and "sit" reach the same pose.
_POSE_CUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("wave", "waving", "تلويح", "لوّح", "تحية"), "wave"),
    (("walk", "walking", "مشي", "يمشي", "امش"), "walk"),
    (("run", "running", "jog", "جري", "يجري"), "run"),
    (("sit on the floor", "sit on the ground", "cross-legged", "أرض", "الأرض"), "sit_ground"),
    (("sit", "sitting", "chair", "اجلس", "جلوس", "كرسي"), "sit"),
    (("t-pose", "tpose", "t pose", "arms out", "وضع t"), "tpose"),
    (("reach", "reaching", "raise", "رفع", "مدّ", "يمد"), "reach"),
    (("arms crossed", "cross your arms", "ذراعان متقاطعتان", "تقاطع"), "arms_crossed"),
    (("hands on hips", "akimbo", "اليدان على الخصر", "الخصر"), "hands_hips"),
    (("squat", "crouch", "kneel", "قرفصاء", "اقرفص"), "squat"),
    (("lying", "lie down", "lay down", "استلقاء", "استلق"), "lying"),
    (("contrapposto", "posing", "model", "وقفة", "موديل"), "contrapposto"),
    (("stand", "standing", "rest", "قف", "وقوف"), "rest"),
)


def pose_from_instruction(text: str) -> str | None:
    """The pose a plain-language instruction asks for, or ``None`` if none matches.

    A small, deterministic keyword matcher -- not a model -- so the studio can turn
    a typed or spoken "sit down" into the sitting pose without guessing: an
    instruction that names no pose returns ``None`` and changes nothing.
    """
    lowered = (text or "").strip().casefold()
    if not lowered:
        return None
    for cues, key in _POSE_CUES:
        if any(cue in lowered for cue in cues):
            return key
    return None


def apply_pose_instruction(
    params: FigureParameters, text: str
) -> tuple[FigureParameters, str]:
    """Set the pose a plain-language instruction asks for.

    Returns ``(params, message)``: the new parameters and a readable note. When the
    instruction names no pose the parameters are returned unchanged and the message
    says so, so the studio can show an honest answer rather than pretend.
    """
    key = pose_from_instruction(text)
    if key is None:
        return params, (
            "No pose matched that instruction. Try a word like sit, wave, walk, run, "
            "reach, squat, arms crossed, hands on hips, or lying."
        )
    return params.with_changes(pose=key), f"Pose set to {POSE_TITLES.get(key, key)}."


def changed_keys(params: FigureParameters) -> tuple[str, ...]:
    """The dials that differ from the neutral default (so the window can mark them)."""
    base = FIGURE_PRESETS["adult"]
    return tuple(
        dial.key for dial in DIALS if getattr(params, dial.key) != getattr(base, dial.key)
    )


# -- rendering ------------------------------------------------------------


def preview_scene(
    params: FigureParameters,
    *,
    view: str = VIEW_FULL,
    name: str = "Body",
    azimuth: float = 0.0,
    elevation: float = 0.0,
    zoom: float = 1.0,
    body_mesh: Any = None,
) -> Scene:
    """A scene holding the figure (and its hair), framed for the requested view.

    ``azimuth`` (degrees around the vertical axis), ``elevation`` (degrees above
    the horizon), and ``zoom`` (a distance multiplier) place the camera on an
    orbit around the figure's own framing target, so the same body can be viewed
    from any direction without rebuilding it.

    When ``body_mesh`` is given, that mesh *is* the body -- an imported or edited
    figure the studio is working on -- so the window can show a model the dials did
    not build. The hair is only added for the parametric figure, because an
    imported model already carries its own.
    """
    scene = make_default_scene(f"{name} body")
    if body_mesh is not None:
        scene.add_object(
            SceneObject(
                name=f"{name} body",
                mesh=body_mesh,
                material=Material("body", SKIN_COLOUR),
            )
        )
    else:
        figure = build_figure(params, name=name)
        scene.add_object(figure)
        # The hair is fitted to this body's own skull and bound to its rig, so it
        # leaves the face clear and follows the head when the figure is posed.
        hair = build_hair_object(params, name, body=figure)
        if hair is not None:
            scene.add_object(hair)
    scene.background = (0.13, 0.15, 0.19)
    if view == VIEW_FACE:
        _frame_head(scene, params)
    else:
        # A close margin so the figure fills the frame and the finer mesh is
        # visible, but with enough room that the head and the feet are both
        # inside it -- a cropped foot reads as a broken body.
        scene.frame_camera(margin=1.30)
    if azimuth or elevation or zoom != 1.0:
        _orbit(scene, azimuth=azimuth, elevation=elevation, zoom=zoom)
    return scene


def _orbit(scene: Scene, *, azimuth: float, elevation: float, zoom: float) -> None:
    """Place the camera on an orbit around its current target.

    The azimuth rotates about the world vertical axis; the elevation lifts the
    camera above (positive) or below (negative) the horizon, clamped short of the
    poles so the up vector never flips; the zoom scales the distance. Only the
    camera moves, so the figure and its pose are untouched.
    """
    camera = scene.camera
    offset = camera.position - camera.target
    radius = offset.length()
    if radius <= 1e-6:
        return
    elevation = max(-89.0, min(89.0, elevation))
    polar = math.radians(90.0 - elevation)
    azimuth = math.radians(azimuth)
    distance = max(0.05, radius * max(0.05, zoom))
    horizontal = math.sin(polar)
    position = camera.target + Vec3(
        distance * horizontal * math.sin(azimuth),
        distance * math.cos(polar),
        distance * horizontal * math.cos(azimuth),
    )
    scene.camera = Camera(
        position=position,
        target=camera.target,
        up=camera.up,
        fov_degrees=camera.fov_degrees,
        near=camera.near,
        far=camera.far,
    )


def _frame_head(scene: Scene, params: FigureParameters) -> None:
    """Aim the camera at the head, close enough that the face fills the frame."""
    # The head is a *fraction of the total height*: its base sits at 0.875 of the
    # height and it is about 0.125 tall, so its centre is at 0.9375 of the height.
    head_y = params.height * 0.9375
    head_length = 0.125 * params.height * params.head_scale * params.face_length
    target = Vec3(0.0, head_y, 0.0)
    # Pull back just enough that the head (with its hair) fills the frame.
    distance = max(0.22, 3.2 * head_length)
    scene.camera = Camera(
        position=Vec3(0.0, target.y, distance),
        target=target,
        fov_degrees=30.0,
        near=0.05,
        far=50.0,
    )


def render_preview(
    params: FigureParameters,
    *,
    view: str = VIEW_FULL,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    zoom: float = 1.0,
    body_mesh: Any = None,
) -> bytes:
    """Render the body and return the PNG bytes (the window shows them directly).

    ``azimuth``/``elevation``/``zoom`` orbit the camera around the figure, so the
    window's camera tool can show the same body from any direction. ``body_mesh``
    renders an imported or edited mesh instead of the parametric figure.
    """
    if view not in VIEWS:
        view = VIEW_FULL
    scene = preview_scene(
        params,
        view=view,
        azimuth=azimuth,
        elevation=elevation,
        zoom=zoom,
        body_mesh=body_mesh,
    )
    buffer, _stats = render_scene(scene, width=int(width), height=int(height))
    return buffer.to_png()


def render_avatar(
    params: FigureParameters,
    *,
    width: int = AVATAR_WIDTH,
    height: int = AVATAR_HEIGHT,
    body_mesh: Any = None,
) -> bytes:
    """Render a square **portrait** of the body, for use as a character's picture."""
    scene = preview_scene(params, view=VIEW_FULL, body_mesh=body_mesh)
    buffer, _stats = render_scene(scene, width=int(width), height=int(height))
    return buffer.to_png()


# -- an imported or edited mesh (the body is a model, not only the dials) --


def load_model_mesh(path: str) -> Any:
    """Read a ready-made human model (OBJ, STL, or PLY) into a normalised mesh.

    The mesh is put in the project's own space (metres, feet on the ground, centred
    on the origin) so the region boxes, the camera framing, and the renderer all
    mean the same thing as they do for the parametric figure. A file this build
    cannot read is refused by :class:`core.body_editor.BodyMeshError` with the real
    reason.
    """
    from core.body_editor import load_body_mesh, normalize_mesh

    return normalize_mesh(load_body_mesh(path))


def figure_mesh(params: FigureParameters) -> Any:
    """The parametric figure's own mesh, in the studio's normalised body space."""
    from core.body_editor import normalize_mesh
    from core.scene3d_figures import build_figure_mesh

    return normalize_mesh(build_figure_mesh(params), height=params.height)


def model_regions(params: FigureParameters) -> tuple[Any, ...]:
    """The editable regions of a body of this figure's height (from its own rig)."""
    from core.body_editor import body_regions

    return body_regions(params.height, shoulder_width=params.shoulder_width)


def model_region_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each editable body region, in either language."""
    from core.body_editor import region_titles

    return region_titles(language)


def add_part_titles(language: str = "en") -> tuple[tuple[str, str], ...]:
    """``(key, readable title)`` for each shape a missing part can be built from."""
    from core.body_editor import add_shape_titles

    return add_shape_titles(language)


def model_summary(mesh: Any, *, source: str = "") -> Any:
    """A :class:`core.body_editor.MeshSummary` of a body mesh (its real figures)."""
    from core.body_editor import summarize_mesh

    return summarize_mesh(mesh, source=source)


def edit_model_region(
    mesh: Any,
    region_key: str,
    *,
    params: FigureParameters,
    scale: tuple[float, float, float] | None = None,
    move: tuple[float, float, float] | None = None,
    rotate: tuple[float, float, float] | None = None,
    falloff: float | None = None,
) -> Any:
    """Resize, move, and/or rotate one region of ``mesh`` (a smooth-space edit).

    The region is resolved from the figure's own rig (so ``"arm_left"`` is the left
    arm on any mesh), and each edit blends out over ``falloff`` so the joint stays
    smooth. An unknown region is refused.
    """
    from core.body_editor import (
        DEFAULT_FALLOFF,
        BodyMeshError,
        find_region,
        move_region,
        resize_region,
        rotate_region,
    )

    region = find_region(region_key, model_regions(params))
    if region is None:
        raise BodyMeshError(f"Unknown body region {region_key!r}.")
    blend = DEFAULT_FALLOFF if falloff is None else float(falloff)
    result = mesh
    if scale is not None:
        result = resize_region(result, region, scale, falloff=blend)
    if move is not None:
        result = move_region(result, region, move, falloff=blend)
    if rotate is not None:
        result = rotate_region(result, region, rotate, falloff=blend)
    return result


def add_model_part(
    mesh: Any,
    *,
    shape: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float] | float,
    part_mesh: Any = None,
) -> Any:
    """Fuse a new part (a shape or a sub-mesh) into ``mesh`` as one connected skin."""
    from core.body_editor import add_part

    return add_part(mesh, shape=shape, centre=centre, size=size, part_mesh=part_mesh)


def body_summary(params: FigureParameters) -> str:
    """A short, readable description of the body the dials currently describe."""
    parts = [f"{BUILD_TITLES.get(params.build, params.build)} build", params.proportion]
    parts.append(f"{params.height:.2f} m")
    if params.bust >= 0.4:
        parts.append("fuller chest")
    if params.muscle >= 0.4:
        parts.append("defined muscle")
    if params.belly >= 0.4:
        parts.append("fuller middle")
    if params.hair <= 0.0:
        parts.append("no hair")
    return ", ".join(parts) + "."


__all__ = [
    "AVATAR_HEIGHT",
    "AVATAR_WIDTH",
    "BODY_PRESETS",
    "BUILD_CHOICES",
    "BUILD_TITLES",
    "CAMERA_DEFAULT_AZIMUTH",
    "CAMERA_DEFAULT_ELEVATION",
    "CAMERA_DEFAULT_ZOOM",
    "CAMERA_MAX_ZOOM",
    "CAMERA_MIN_ZOOM",
    "CAMERA_PRESETS",
    "DIALS",
    "DIAL_GROUPS",
    "HONESTY_LINE",
    "POSE_PRESET_KEYS",
    "POSE_TITLES",
    "PREVIEW_HEIGHT",
    "PREVIEW_WIDTH",
    "SKIN_COLOUR",
    "VIEWS",
    "VIEW_FACE",
    "VIEW_FULL",
    "VIEW_TITLES",
    "DialSpec",
    "add_model_part",
    "add_part_titles",
    "apply_pose_instruction",
    "as_colour",
    "body_preset_keys",
    "body_preset_titles",
    "body_summary",
    "build_for_gender",
    "camera_preset",
    "camera_preset_titles",
    "changed_keys",
    "clamp_zoom",
    "default_parameters",
    "dial_for",
    "dial_keys",
    "dials_in_group",
    "edit_model_region",
    "figure_mesh",
    "group_titles",
    "load_model_mesh",
    "model_region_titles",
    "model_regions",
    "model_summary",
    "pose_from_instruction",
    "pose_keys",
    "pose_titles",
    "preset_parameters",
    "preview_scene",
    "render_avatar",
    "render_preview",
    "reset_to_preset",
    "set_dial",
    "set_dials",
    "set_pose",
    "values_of",
]
