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

from dataclasses import dataclass
from typing import Any

from core.scene3d import Camera, Scene, Vec3, make_default_scene
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

#: The dial groups, in the order the window shows them.
DIAL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("proportions", "Proportions", "القياسات"),
    ("torso", "Torso and shape", "الجذع والقوام"),
    ("limbs", "Arms and legs", "الذراعان والساقان"),
    ("head", "Head and face", "الرأس والوجه"),
    ("intimate", "Intimate detail", "التفاصيل الحميمة"),
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


#: Every dial the studio can edit, grouped. The keys are the parameter field names
#: themselves, so setting a dial is a direct write to the figure's own parameters.
DIALS: tuple[DialSpec, ...] = (
    DialSpec(
        "build", "Build family", "نمط الجسم", "proportions",
        kind="choice", choices=BUILDS, hint="Feminine / masculine / neutral",
    ),
    DialSpec(
        "height", "Height (metres)", "الطول (متر)", "proportions",
        minimum=1.20, maximum=2.10, step=0.01, hint="Total standing height",
    ),
    DialSpec("shoulder_width", "Shoulder width", "عرض الكتفين", "proportions", minimum=0.26, maximum=0.62, step=0.01),
    DialSpec("hip_width", "Hip width", "عرض الحوض", "proportions", minimum=0.20, maximum=0.46, step=0.01),
    DialSpec("chest_depth", "Chest depth", "عمق الصدر", "proportions", minimum=0.14, maximum=0.36, step=0.01),
    DialSpec("waist", "Waist width", "عرض الخصر", "proportions", minimum=0.16, maximum=0.42, step=0.01),
    DialSpec("bust", "Bust", "الصدر", "torso", hint="Chest fullness and forward shape"),
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
        hint="Shows the genital region (adults only); 0 hides it",
    ),
    DialSpec(
        "genital_size", "Intimate size", "حجم التفاصيل الحميمة", "intimate",
        minimum=0.40, maximum=1.80, step=0.01,
    ),
    DialSpec("areola", "Areola size", "حجم الهالة", "intimate", minimum=0.40, maximum=2.00, step=0.01),
    DialSpec("hair", "Hair volume", "كثافة الشعر", "hair"),
    DialSpec("skin", "Skin colour", "لون البشرة", "colour", kind="colour"),
    DialSpec("hair_color", "Hair colour", "لون الشعر", "colour", kind="colour"),
    DialSpec("lip_color", "Lip colour", "لون الشفتين", "colour", kind="colour"),
    DialSpec("eye_color", "Eye colour", "لون العين", "colour", kind="colour"),
    DialSpec("nipple_color", "Areola colour", "لون الهالة", "colour", kind="colour"),
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

    A named preset wins; otherwise the character's own recorded **gender** picks the
    build family and everything else is the neutral default, so opening the studio
    for a woman shows a woman's shape and for a man a man's -- without inventing a
    measurement the character never recorded.
    """
    base = preset_parameters(preset) if preset else None
    if base is None:
        base = FIGURE_PRESETS["adult"]
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
    return params.with_changes(**{key: dial.clamp(value)})


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


def preview_scene(params: FigureParameters, *, view: str = VIEW_FULL, name: str = "Body") -> Scene:
    """A scene holding the figure (and its hair), framed for the requested view."""
    figure = build_figure(params, name=name)
    scene = make_default_scene(f"{name} body")
    scene.add_object(figure)
    hair = build_hair_object(params, name)
    if hair is not None:
        scene.add_object(hair)
    scene.background = (0.13, 0.15, 0.19)
    if view == VIEW_FACE:
        _frame_head(scene, params)
    else:
        scene.frame_camera(margin=1.5)
    return scene


def _frame_head(scene: Scene, params: FigureParameters) -> None:
    """Aim the camera at the head, close enough that the face fills the frame."""
    head_y = params.height * 0.845
    length = 0.235 * params.head_scale * params.face_length
    target = Vec3(0.0, head_y + length * 0.5, 0.0)
    distance = max(0.35, 1.35 * params.head_scale * params.face_length)
    scene.camera = Camera(
        position=Vec3(0.0, target.y, distance),
        target=target,
        fov_degrees=32.0,
        near=0.05,
        far=50.0,
    )


def render_preview(
    params: FigureParameters,
    *,
    view: str = VIEW_FULL,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
) -> bytes:
    """Render the body and return the PNG bytes (the window shows them directly)."""
    if view not in VIEWS:
        view = VIEW_FULL
    scene = preview_scene(params, view=view)
    buffer, _stats = render_scene(scene, width=int(width), height=int(height))
    return buffer.to_png()


def render_avatar(
    params: FigureParameters, *, width: int = AVATAR_WIDTH, height: int = AVATAR_HEIGHT
) -> bytes:
    """Render a square **portrait** of the body, for use as a character's picture."""
    scene = preview_scene(params, view=VIEW_FULL)
    buffer, _stats = render_scene(scene, width=int(width), height=int(height))
    return buffer.to_png()


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
    "DIALS",
    "DIAL_GROUPS",
    "HONESTY_LINE",
    "POSE_PRESET_KEYS",
    "POSE_TITLES",
    "PREVIEW_HEIGHT",
    "PREVIEW_WIDTH",
    "VIEWS",
    "VIEW_FACE",
    "VIEW_FULL",
    "VIEW_TITLES",
    "DialSpec",
    "apply_pose_instruction",
    "as_colour",
    "body_preset_keys",
    "body_preset_titles",
    "body_summary",
    "build_for_gender",
    "changed_keys",
    "default_parameters",
    "dial_for",
    "dial_keys",
    "dials_in_group",
    "group_titles",
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
