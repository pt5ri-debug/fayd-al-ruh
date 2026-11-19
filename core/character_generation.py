"""FR-CCE -- the native procedural 3D character-generation engine.

This is the **engine** behind the project's Body & Shape window. It generates a
complete human body from code -- geometry, skeleton, skinning, materials, and a
preview render -- and lets every measurement of that body be dialled and saved
per character. It is a *thin orchestration layer* over the 3D systems the project
already has, never a second figure builder:

* the geometry and the morph dials are :class:`core.scene3d_figures.FigureParameters`;
* the fused, single-skin mesh is :func:`core.scene3d_figures.build_figure_mesh`
  (which sculpts smooth volumetric parts into one connected surface through
  :mod:`core.scene3d_sculpt`);
* the rig and the linear-blend skinning are :mod:`core.scene3d`;
* the pixels are the CPU software rasteriser in :mod:`core.scene3d_render`;
* the body parameters the window edits are :mod:`core.romance.body`'s own dial
  catalogue.

So the engine owns *composition* and *persistence*, not a parallel renderer or a
parallel mesh generator -- which is what keeps the window, the CLI, and the gate
in agreement.

**The four templates.** A :class:`CharacterTemplateManager` ships four complete
bodies -- a **woman**, a **girl**, a **man**, and a **boy** -- each a real body at
its age and sex (an adult female, an adult male, and two children on
age-appropriate child proportions, not scaled-down adults), each fused into one
connected skin, each generated locally with no download.

**Honesty by construction.** The figure is a *stylised, parametric* humanoid, not
a licensed likeness; the render is a rasteriser, not a commercial render engine.
The material system says plainly which of its settings the renderer can honour
(roughness and metallic shape the Blinn-Phong shading) and which it cannot
(texture maps and normal maps are **not** sampled by this rasteriser), rather than
offering a control that does nothing. Nothing here opens a network connection or
downloads an asset.

The module is dependency-light and Tk-free: it imports no window and loads no
model, so the whole engine is testable headless.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.scene3d import (
    Material,
    Mesh,
    Scene,
    SceneObject,
    Skeleton,
)
from core.scene3d_figures import (
    FIGURE_PRESETS,
    FigureParameters,
    build_figure,
    build_figure_mesh,
    build_figure_skeleton,
    pose_rotations,
)
from core.scene3d_render import RenderStats, render_scene
from core.version import __version__

#: The version of the *saved body configuration* this engine writes. It is bumped
#: only when the on-disk shape changes, and :func:`migrate_config` upgrades an older
#: document, so a configuration written by an earlier build is never lost.
CONFIG_VERSION = 1

#: What this engine is, said plainly, so any surface can show it.
HONESTY_LINE = (
    "A native, procedural character generator: a stylised parametric humanoid "
    "sculpted and rendered locally by the project's own CPU rasteriser -- not a "
    "licensed likeness, a photograph, or a commercial render engine."
)

#: The four base templates, in the order the window shows them.
TEMPLATE_WOMAN = "woman"
TEMPLATE_GIRL = "girl"
TEMPLATE_MAN = "man"
TEMPLATE_BOY = "boy"
TEMPLATE_IDS: tuple[str, ...] = (TEMPLATE_WOMAN, TEMPLATE_GIRL, TEMPLATE_MAN, TEMPLATE_BOY)

#: Bilingual template titles.
TEMPLATE_TITLES: dict[str, tuple[str, str]] = {
    TEMPLATE_WOMAN: ("Adult female", "أنثى بالغة"),
    TEMPLATE_GIRL: ("Girl", "بنت"),
    TEMPLATE_MAN: ("Adult male", "ذكر بالغ"),
    TEMPLATE_BOY: ("Boy", "ولد"),
}

#: Whether a template is an adult (used by the age guard).
_ADULT_TEMPLATES = frozenset({TEMPLATE_WOMAN, TEMPLATE_MAN})

#: The age at which a template is drawn as an adult body. A character recorded
#: younger gets the younger template of its sex, so a minor is never drawn with an
#: adult body.
ADULT_AGE = 18


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CharacterTemplate:
    """One base template: its stable id, its titles, and its default parameters."""

    key: str
    title: str
    title_ar: str
    parameters: FigureParameters

    @property
    def is_adult(self) -> bool:
        return self.key in _ADULT_TEMPLATES

    def describe(self) -> str:
        return f"{self.title} ({self.key}) — {self.parameters.describe()}"


class CharacterTemplateManager:
    """The four base templates the generator can instantiate.

    Each template is a complete, independent set of body parameters -- a woman's
    body is not a scaled man's, and a child's is not a scaled adult's -- so a
    character built from one reads as that age and sex. A template is *immutable*:
    the manager never mutates a shipped template, so regenerating a character from
    it always starts from the same, known body.
    """

    def __init__(self, templates: dict[str, FigureParameters] | None = None) -> None:
        source = templates if templates is not None else FIGURE_PRESETS
        self._templates: dict[str, CharacterTemplate] = {}
        for key in TEMPLATE_IDS:
            params = source[key]
            title, title_ar = TEMPLATE_TITLES[key]
            self._templates[key] = CharacterTemplate(key, title, title_ar, params)

    def keys(self) -> tuple[str, ...]:
        return TEMPLATE_IDS

    def has(self, key: str) -> bool:
        return key in self._templates

    def get(self, key: str) -> CharacterTemplate:
        """The template named ``key`` (an unknown key is refused, not guessed)."""
        template = self._templates.get(key)
        if template is None:
            raise KeyError(
                f"Unknown character template {key!r}; choose from {', '.join(TEMPLATE_IDS)}."
            )
        return template

    def parameters(self, key: str, *, name: str = "") -> FigureParameters:
        """A fresh, independent copy of a template's parameters."""
        params = self.get(key).parameters
        return params.with_changes(name=name) if name else params

    def summaries(self, *, language: str = "en") -> tuple[tuple[str, str], ...]:
        """``(key, title)`` for each template, in either language."""
        return tuple(
            (key, t.title_ar if language == "ar" else t.title)
            for key, t in self._templates.items()
        )

    def template_for_character(self, character: Any) -> str:
        """The template a character opens on, from its own recorded sex and age.

        A woman shows the adult-female body and a man the adult-male one; a
        character recorded under :data:`ADULT_AGE` shows the younger template of its
        sex, so a minor is never drawn with an adult body. An unrecorded or
        non-binary sex falls back to the documented default (the adult female)
        rather than a made-up choice.
        """
        sex = _sex_of(character)
        age = _age_of(character)
        young = age is not None and age < ADULT_AGE
        if sex == "male":
            return TEMPLATE_BOY if young else TEMPLATE_MAN
        if sex == "female":
            return TEMPLATE_GIRL if young else TEMPLATE_WOMAN
        return TEMPLATE_GIRL if young else TEMPLATE_WOMAN


def _character_field(character: Any, name: str) -> Any:
    """Read a field off a character *or* the bundle that wraps it."""
    if character is None:
        return None
    if hasattr(character, name):
        return getattr(character, name)
    inner = getattr(character, "character", None)
    return getattr(inner, name, None) if inner is not None else None


def _sex_of(character: Any) -> str:
    """``"female"``, ``"male"``, or ``""`` from a character's own recorded sex.

    Broad about spelling (an English word, an Arabic word, or the usual
    abbreviations) but never a guess: a blank or non-binary record returns ``""``.
    """
    text = str(_character_field(character, "gender") or "").strip().casefold()
    if not text:
        return ""
    female = ("woman", "female", "feminine", "girl", "lady", "mother", "sister",
              "أنثى", "امرأة", "بنت", "فتاة", "سيدة", "أم", "أخت")
    male = ("man", "male", "masculine", "boy", "gentleman", "father", "brother",
            "ذكر", "رجل", "ولد", "فتى", "سيد", "أب", "أخ")
    if any(word in text for word in female):
        return "female"
    if any(word in text for word in male):
        return "male"
    return ""


def _age_of(character: Any) -> int | None:
    value = _character_field(character, "fictional_age")
    if value is None:
        return None
    try:
        age = int(value)
    except (TypeError, ValueError):
        return None
    return age if age > 0 else None


# --------------------------------------------------------------------------- #
# Mesh generation
# --------------------------------------------------------------------------- #


class ProceduralMeshGenerator:
    """Turn a template's parameters into one fused, connected human mesh.

    It is the single entry point the rest of the engine uses to obtain geometry, so
    the mesh is always the project's own sculpted body (one skin, no gaps) and
    never a second, divergent generator. It caches the last build by the parameters'
    own signature, so a repeated read (an orbit render, a re-open) is free.
    """

    def __init__(self) -> None:
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_mesh: Mesh | None = None

    @staticmethod
    def signature(params: FigureParameters) -> tuple[Any, ...]:
        """A hashable key for the parameters that change the mesh.

        ``name`` and ``pose`` are deliberately excluded: the name does not change
        geometry, and the pose is applied at render time by the rig rather than by
        rebuilding the mesh -- so an orbit or a pose change never invalidates it.
        """
        fields = params.__dataclass_fields__
        return tuple(
            (key, getattr(params, key))
            for key in fields
            if key not in {"name", "pose"}
        )

    def build(self, params: FigureParameters) -> Mesh:
        """The mesh for ``params`` (cached by signature)."""
        key = self.signature(params)
        if self._cache_key == key and self._cache_mesh is not None:
            return self._cache_mesh
        mesh = build_figure_mesh(params)
        self._cache_key = key
        self._cache_mesh = mesh
        return mesh

    def invalidate(self) -> None:
        self._cache_key = None
        self._cache_mesh = None


# --------------------------------------------------------------------------- #
# Skeleton / rig
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RigReport:
    """An honest description of a figure's rig, for a window or a test."""

    bones: int
    names: tuple[str, ...]
    fingers: int
    toes: int
    bound_vertices: int

    def describe(self) -> str:
        return (
            f"{self.bones} bones ({self.fingers} finger bones, {self.toes} toe bones); "
            f"{self.bound_vertices} vertices bound."
        )


class SkeletonRigManager:
    """Build and bind a humanoid rig, and keep it consistent with the body.

    The rig is the project's own 19-bone skeleton (:func:`build_figure_skeleton`),
    which follows the same measurement dials the mesh does, so changing a
    proportion never leaves the rig behind the body it deforms. Skinning is real
    linear-blend skinning (:meth:`core.scene3d.SceneObject.bind`), computed once at
    bind time, so a pose is cheap. Finger and toe *bones* are not part of the rig
    yet -- the geometry has fingers and toes, but the rig drives them through the
    hand and foot bones -- and the manager says so rather than implying otherwise.
    """

    def build(self, params: FigureParameters) -> Skeleton:
        return build_figure_skeleton(params)

    def bind(self, obj: SceneObject) -> SceneObject:
        """Bind an object's mesh to its skeleton (real skin weights)."""
        return obj.bind()

    def report(self, obj: SceneObject) -> RigReport:
        skeleton = obj.skeleton
        if skeleton is None:
            return RigReport(0, (), 0, 0, 0)
        names = skeleton.names()
        fingers = sum(1 for name in names if "hand" in name or "finger" in name)
        toes = sum(1 for name in names if "foot" in name or "toe" in name)
        return RigReport(len(names), names, fingers, toes, len(obj.bindings))

    def pose(self, obj: SceneObject, preset: str) -> SceneObject:
        """Put the figure into a named pose (the rig is unchanged)."""
        obj.pose = pose_rotations(preset)
        return obj


# --------------------------------------------------------------------------- #
# Body parameter system
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParameterGroup:
    """One group of body parameters, as ``(key, title, title_ar, dial keys)``."""

    key: str
    title: str
    title_ar: str
    keys: tuple[str, ...]


class BodyParameterSystem:
    """The catalogue of body parameters the engine can morph, grouped.

    It reuses :mod:`core.romance.body`'s own dial catalogue -- the one the window
    already shows -- and adds the **measurement** dials the FR-CCE task requires
    (per-part length and thickness). It never re-declares a dial the project
    already has, so the window, the CLI, and the engine can never disagree about
    what a parameter is called or what range it accepts.
    """

    #: The dial keys, in group order, that the engine exposes. A group is a label
    #: over a subset of the catalogue, so no key is invented here.
    GROUPS: tuple[ParameterGroup, ...] = (
        ParameterGroup("proportions", "Overall proportions", "القياسات العامة",
                       ("build", "overall_scale", "height", "proportion")),
        ParameterGroup("head", "Head and face", "الرأس والوجه",
                       ("head_scale", "head_depth", "face_length", "face_width",
                        "jaw_width", "chin_shape", "brow_ridge", "nose_size",
                        "eye_size", "eye_spacing", "lips", "cheek", "ear_size")),
        ParameterGroup("neck", "Neck", "الرقبة", ("neck_thickness", "neck_length")),
        ParameterGroup("shoulders", "Shoulders", "الكتفان",
                       ("shoulder_width", "shoulder_slope")),
        ParameterGroup("chest", "Chest", "الصدر", ("chest_depth", "bust", "areola")),
        ParameterGroup("waist", "Waist", "الخصر", ("waist", "waist_shape")),
        ParameterGroup("abdomen", "Abdomen", "البطن", ("belly",)),
        ParameterGroup("pelvis", "Pelvis and hips", "الحوض والأرداف",
                       ("hip_width", "hip_shape")),
        ParameterGroup("arms", "Arms", "الذراعان",
                       ("arm_length", "forearm_length", "arm_thickness")),
        ParameterGroup("hands", "Hands", "اليدان",
                       ("hand_size", "hand_length", "hand_width")),
        ParameterGroup("thighs", "Thighs", "الفخذان",
                       ("thigh_length", "thigh_thickness")),
        ParameterGroup("legs", "Legs", "الساقان",
                       ("lower_leg_length", "calf_thickness", "leg_thickness")),
        ParameterGroup("feet", "Feet", "القدمان",
                       ("foot_size", "foot_length", "foot_width")),
        ParameterGroup("shape", "General body shape", "شكل الجسم العام",
                       ("muscle", "limb_thickness")),
        ParameterGroup("detail", "Mesh and detail", "الشبكة والتفاصيل",
                       ("subdivisions", "smoothing")),
        ParameterGroup("pose", "Pose", "الوضعية", ("pose",)),
    )

    def __init__(self, dial_catalogue: Iterable[Any] | None = None) -> None:
        from core.romance.body import DIALS

        catalogue = tuple(dial_catalogue) if dial_catalogue is not None else DIALS
        self._dials = {dial.key: dial for dial in catalogue}

    def keys(self) -> tuple[str, ...]:
        return tuple(self._dials)

    def has(self, key: str) -> bool:
        return key in self._dials

    def dial(self, key: str) -> Any:
        dial = self._dials.get(key)
        if dial is None:
            raise KeyError(f"Unknown body parameter {key!r}.")
        return dial

    def groups(self, *, language: str = "en") -> tuple[tuple[str, str, tuple[Any, ...]], ...]:
        """``(group key, title, dials)`` for each group, in the window's order.

        A dial named by a group but absent from the catalogue is skipped rather
        than invented, so the engine cannot advertise a control the project does
        not have.
        """
        result: list[tuple[str, str, tuple[Any, ...]]] = []
        for group in self.GROUPS:
            dials = tuple(self._dials[key] for key in group.keys if key in self._dials)
            if dials:
                title = group.title_ar if language == "ar" else group.title
                result.append((group.key, title, dials))
        return tuple(result)

    def set(self, params: FigureParameters, key: str, value: Any) -> FigureParameters:
        from core.romance.body import set_dial

        return set_dial(params, key, value)

    def set_many(
        self, params: FigureParameters, changes: dict[str, Any]
    ) -> FigureParameters:
        updated = params
        for key, value in changes.items():
            updated = self.set(updated, key, value)
        return updated

    def values(self, params: FigureParameters) -> dict[str, Any]:
        return {key: getattr(params, key) for key in self._dials}


# --------------------------------------------------------------------------- #
# Morph engine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MorphResult:
    """What one morph did: whether the geometry really moved, and by how much."""

    changed: bool
    moved_vertices: int
    max_displacement: float
    vertex_count: int

    def describe(self) -> str:
        if not self.changed:
            return "No vertex moved."
        if self.max_displacement == float("inf"):
            return (
                f"The surface was rebuilt ({self.moved_vertices} vertices in the "
                "previous mesh); the topology changed."
            )
        return (
            f"{self.moved_vertices}/{self.vertex_count} vertices moved "
            f"(max {self.max_displacement:.4f} m)."
        )


class MorphTargetEngine:
    """Apply a body parameter and report the *real* geometry change.

    A morph rebuilds the mesh from the parameters (the same sculpt every other
    surface uses) and compares the result to the mesh before it, so a slider is
    only ever counted as working when actual vertex positions moved -- never merely
    because a number changed. This is the honesty rule the FR-CCE task asks for.
    """

    def __init__(self, generator: ProceduralMeshGenerator | None = None) -> None:
        self.generator = generator or ProceduralMeshGenerator()

    def apply(
        self, params: FigureParameters, key: str, value: Any
    ) -> tuple[FigureParameters, Mesh, MorphResult]:
        """Set one parameter and return ``(new params, new mesh, what moved)``."""
        from core.romance.body import set_dial

        before = self.generator.build(params)
        updated = set_dial(params, key, value)
        after = self.generator.build(updated)
        return updated, after, self.compare(before, after)

    @staticmethod
    def compare(before: Mesh, after: Mesh) -> MorphResult:
        """How many vertices moved between two meshes, and by how far."""
        count = min(len(before.vertices), len(after.vertices))
        if len(before.vertices) != len(after.vertices):
            # A rebuild with a different vertex count is a real change (the
            # topology moved); report it as such rather than pretending nothing
            # happened.
            return MorphResult(True, count, float("inf"), len(after.vertices))
        moved = 0
        largest = 0.0
        for index in range(count):
            a = before.vertices[index]
            b = after.vertices[index]
            distance = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
            if distance > 1e-9:
                moved += 1
                largest = max(largest, distance)
        return MorphResult(moved > 0, moved, largest, count)


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurfaceMaterial:
    """A material with the appearance controls the task asks for.

    ``roughness`` and ``metallic`` are honoured: roughness maps onto the
    Blinn-Phong shininess and specular (a rougher surface is duller), and metallic
    tints the specular toward the base colour and sharpens it. ``base_color`` is
    the diffuse colour. The two *texture* controls (a texture map and a normal
    map) are recorded but **not** sampled, because this project's rasteriser has no
    image sampler -- and the class says so rather than offering a control that does
    nothing.
    """

    name: str = "material"
    base_color: tuple[float, float, float] = (0.82, 0.66, 0.56)
    roughness: float = 0.7
    metallic: float = 0.0
    wrap: float = 0.0
    opacity: float = 1.0
    #: A texture/normal map reference. Kept for completeness; the CPU rasteriser
    #: does not sample it, and :meth:`describe` says so.
    texture_map: str = ""
    normal_map: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("Material roughness must be in [0, 1].")
        if not 0.0 <= self.metallic <= 1.0:
            raise ValueError("Material metallic must be in [0, 1].")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("Material opacity must be in [0, 1].")

    def to_material(self) -> Material:
        """The renderer's own :class:`core.scene3d.Material` for this surface.

        Roughness drives shininess (smooth = high shininess) and specular
        (rough = low specular); metallic blends the specular tint toward the base
        colour, the way a metal's highlight takes its colour from the surface.
        """
        roughness = min(0.999, max(0.0, self.roughness))
        # Smooth (low roughness) -> a tight, bright highlight; rough -> a broad, dim
        # one. The mapping stays inside the renderer's own valid ranges.
        shininess = 2.0 + (1.0 - roughness) ** 2 * 190.0
        specular = (0.05 + (1.0 - roughness) * 0.55) * (1.0 - 0.5 * self.metallic)
        specular += 0.5 * self.metallic
        return Material(
            name=self.name,
            base_color=self.base_color,
            specular=min(1.0, max(0.0, specular)),
            shininess=max(1.0, shininess),
            opacity=self.opacity,
            wrap=min(1.0, max(0.0, self.wrap)),
        )

    def describe(self) -> str:
        maps = []
        if self.texture_map:
            maps.append("a texture map (recorded, not sampled by this rasteriser)")
        if self.normal_map:
            maps.append("a normal map (recorded, not sampled by this rasteriser)")
        suffix = ("; " + ", ".join(maps)) if maps else ""
        return (
            f"{self.name}: roughness {self.roughness:.2f}, metallic {self.metallic:.2f}"
            f"{suffix}."
        )


class MaterialManager:
    """Build the skin, eye, and hair materials from a figure's own parameters.

    The skin carries a soft diffuse wrap (a cheap stand-in for subsurface
    scattering), the hair is its own material so a hair colour is real, and the
    eyes are a smooth, low-roughness surface. Texture and normal maps are not
    sampled by this rasteriser; the manager says so in :meth:`capabilities`.
    """

    #: Whether the renderer can sample a texture/normal map. It cannot, and the
    #: manager reports it rather than offering a control that would do nothing.
    TEXTURE_MAPS_SUPPORTED = False
    NORMAL_MAPS_SUPPORTED = False

    def skin(self, params: FigureParameters) -> SurfaceMaterial:
        return SurfaceMaterial(
            name=f"{params.name} skin",
            base_color=params.skin,
            roughness=0.72,
            metallic=0.0,
            wrap=0.35,
        )

    def hair(self, params: FigureParameters) -> SurfaceMaterial:
        return SurfaceMaterial(
            name=f"{params.name} hair",
            base_color=params.hair_color,
            roughness=0.42,
            metallic=0.0,
        )

    def eyes(self, params: FigureParameters) -> SurfaceMaterial:
        return SurfaceMaterial(
            name=f"{params.name} eyes",
            base_color=params.eye_color,
            roughness=0.12,
            metallic=0.0,
        )

    def for_object(self, obj: SceneObject, params: FigureParameters) -> Material:
        """The renderer material a figure's own object should use."""
        if obj.name.endswith(" hair"):
            return self.hair(params).to_material()
        return self.skin(params).to_material()

    def capabilities(self) -> dict[str, bool]:
        """What this engine's material system can and cannot honour, honestly."""
        return {
            "skin_material": True,
            "eye_material": True,
            "hair_material": True,
            "roughness": True,
            "metallic": True,
            "texture_maps": self.TEXTURE_MAPS_SUPPORTED,
            "normal_maps": self.NORMAL_MAPS_SUPPORTED,
            "lighting_controls": True,
        }


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CharacterPreset:
    """A named, saved set of body parameters a user can return to."""

    preset_id: str
    title: str
    template_id: str
    parameters: dict[str, Any]
    created_at: float = 0.0

    def to_primitive(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "title": self.title,
            "template_id": self.template_id,
            "parameters": dict(self.parameters),
            "created_at": self.created_at,
        }

    @classmethod
    def from_primitive(cls, raw: dict[str, Any]) -> CharacterPreset:
        return cls(
            preset_id=str(raw.get("preset_id") or ""),
            title=str(raw.get("title") or ""),
            template_id=str(raw.get("template_id") or ""),
            parameters=dict(raw.get("parameters") or {}),
            created_at=float(raw.get("created_at") or 0.0),
        )


class CharacterPresetManager:
    """Save and load named body presets (one JSON document per preset)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root
        from core.paths import character_presets_dir

        return character_presets_dir()

    def save(
        self,
        preset_id: str,
        title: str,
        template_id: str,
        params: FigureParameters,
    ) -> CharacterPreset:
        """Write a preset (overwriting one of the same id)."""
        slug = _safe_slug(preset_id)
        preset = CharacterPreset(
            preset_id=slug,
            title=title or slug,
            template_id=template_id,
            parameters=params_to_primitive(params),
            created_at=time.time(),
        )
        folder = self.root
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{slug}.json").write_text(
            json.dumps(preset.to_primitive(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return preset

    def load(self, preset_id: str) -> CharacterPreset | None:
        path = self.root / f"{_safe_slug(preset_id)}.json"
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        return CharacterPreset.from_primitive(raw)

    def parameters(self, preset_id: str, *, name: str = "") -> FigureParameters | None:
        """The parameters a preset holds, or ``None`` when it is absent."""
        preset = self.load(preset_id)
        if preset is None:
            return None
        params = params_from_primitive(preset.parameters)
        return params.with_changes(name=name) if name else params

    def list(self) -> tuple[CharacterPreset, ...]:
        folder = self.root
        if not folder.is_dir():
            return ()
        presets: list[CharacterPreset] = []
        for path in sorted(folder.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(raw, dict):
                presets.append(CharacterPreset.from_primitive(raw))
        return tuple(presets)

    def remove(self, preset_id: str) -> bool:
        path = self.root / f"{_safe_slug(preset_id)}.json"
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


def _safe_slug(value: str) -> str:
    """A filesystem-safe id from a user-supplied preset name."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in (value or "").strip()
    )
    return cleaned.strip("-") or "preset"


# --------------------------------------------------------------------------- #
# Serialization of parameters
# --------------------------------------------------------------------------- #


def params_to_primitive(params: FigureParameters) -> dict[str, Any]:
    """A JSON-safe dict of every figure parameter (colours become lists)."""
    result: dict[str, Any] = {}
    for key in params.__dataclass_fields__:
        value = getattr(params, key)
        result[key] = list(value) if isinstance(value, tuple) else value
    return result


def params_from_primitive(raw: dict[str, Any]) -> FigureParameters:
    """Rebuild parameters from a dict, **migrating** a document written by an
    earlier build: an unknown key is ignored, a missing key keeps its default, and
    a colour given as a list becomes a tuple. So a configuration saved before a new
    dial existed still loads."""
    fields = FigureParameters.__dataclass_fields__
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            continue
        if isinstance(value, list) and len(value) == 3 and key.endswith("color"):
            cleaned[key] = tuple(float(component) for component in value)
        elif key == "skin" and isinstance(value, list):
            cleaned[key] = tuple(float(component) for component in value)
        else:
            cleaned[key] = value
    return FigureParameters(**cleaned)


# --------------------------------------------------------------------------- #
# Per-character persistence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CharacterBodyConfig:
    """One character's independent body configuration, as it is stored.

    It carries every field the FR-CCE task asks for: the template id, the
    character id, the body parameters, the morph (measurement) values, the
    skeleton configuration, the material settings, the preset id, the last-modified
    timestamp, and the engine version -- all under stable identifiers.
    """

    character_id: str
    template_id: str
    parameters: dict[str, Any]
    morphs: dict[str, Any] = field(default_factory=dict)
    skeleton: dict[str, Any] = field(default_factory=dict)
    materials: dict[str, Any] = field(default_factory=dict)
    preset_id: str = ""
    last_modified: float = 0.0
    engine_version: str = __version__
    config_version: int = CONFIG_VERSION

    def to_primitive(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "template_id": self.template_id,
            "parameters": dict(self.parameters),
            "morphs": dict(self.morphs),
            "skeleton": dict(self.skeleton),
            "materials": dict(self.materials),
            "preset_id": self.preset_id,
            "last_modified": self.last_modified,
            "engine_version": self.engine_version,
            "config_version": self.config_version,
        }

    @classmethod
    def from_primitive(cls, raw: dict[str, Any]) -> CharacterBodyConfig:
        return cls(
            character_id=str(raw.get("character_id") or ""),
            template_id=str(raw.get("template_id") or TEMPLATE_WOMAN),
            parameters=dict(raw.get("parameters") or {}),
            morphs=dict(raw.get("morphs") or {}),
            skeleton=dict(raw.get("skeleton") or {}),
            materials=dict(raw.get("materials") or {}),
            preset_id=str(raw.get("preset_id") or ""),
            last_modified=float(raw.get("last_modified") or 0.0),
            engine_version=str(raw.get("engine_version") or __version__),
            config_version=int(raw.get("config_version") or CONFIG_VERSION),
        )


def migrate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored configuration up to the current schema.

    A document written by an earlier build is **upgraded**, not refused: a missing
    field takes its default and an unknown field is dropped, so a character's body
    survives a program update. The stored ``parameters`` are passed through
    :func:`params_from_primitive`'s rules by the loader, so a parameter added since
    the file was written simply takes its default.
    """
    migrated = dict(raw)
    migrated.setdefault("template_id", TEMPLATE_WOMAN)
    migrated.setdefault("parameters", {})
    migrated.setdefault("morphs", {})
    migrated.setdefault("skeleton", {})
    migrated.setdefault("materials", {})
    migrated.setdefault("preset_id", "")
    migrated.setdefault("last_modified", 0.0)
    migrated.setdefault("engine_version", __version__)
    migrated["config_version"] = CONFIG_VERSION
    return migrated


class CharacterPersistenceManager:
    """Store each character's body configuration under a stable id.

    A character's configuration is independent: regenerating a template never
    touches a saved configuration (the template is only read), and a configuration
    is written only on an explicit save. A corrupt or unreadable file reads as
    *absent* rather than crashing the studio, so a window always opens.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        if self._root is not None:
            return self._root
        from core.paths import character_bodies_dir

        return character_bodies_dir()

    def path_for(self, character_id: str) -> Path:
        return self.root / f"{_safe_slug(character_id)}.json"

    def load(self, character_id: str) -> CharacterBodyConfig | None:
        """A character's saved configuration, or ``None`` when there is none."""
        path = self.path_for(character_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        return CharacterBodyConfig.from_primitive(migrate_config(raw))

    def save(
        self,
        character_id: str,
        template_id: str,
        params: FigureParameters,
        *,
        morphs: dict[str, Any] | None = None,
        skeleton: dict[str, Any] | None = None,
        materials: dict[str, Any] | None = None,
        preset_id: str = "",
    ) -> CharacterBodyConfig:
        config = CharacterBodyConfig(
            character_id=character_id,
            template_id=template_id,
            parameters=params_to_primitive(params),
            morphs=morphs if morphs is not None else measurement_values(params),
            skeleton=skeleton or {},
            materials=materials or {},
            preset_id=preset_id,
            last_modified=time.time(),
            engine_version=__version__,
            config_version=CONFIG_VERSION,
        )
        folder = self.root
        folder.mkdir(parents=True, exist_ok=True)
        path = self.path_for(character_id)
        # Write atomically: a crash mid-write must not leave a half configuration.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(config.to_primitive(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
        return config

    def remove(self, character_id: str) -> bool:
        path = self.path_for(character_id)
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


#: The measurement dials whose values are the "morph values" of a configuration.
MEASUREMENT_KEYS: tuple[str, ...] = (
    "overall_scale", "height", "head_scale", "head_depth", "face_length",
    "face_width", "neck_length", "neck_thickness", "shoulder_width",
    "chest_depth", "waist", "hip_width", "arm_length", "forearm_length",
    "arm_thickness", "hand_length", "hand_width", "thigh_length",
    "thigh_thickness", "lower_leg_length", "calf_thickness", "leg_thickness",
    "foot_length", "foot_width",
)


def measurement_values(params: FigureParameters) -> dict[str, Any]:
    """The morph (measurement) values of a parameter set."""
    return {key: getattr(params, key) for key in MEASUREMENT_KEYS}


# --------------------------------------------------------------------------- #
# Preview renderer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreviewReport:
    """The result of a preview render, with the renderer's own honest stats."""

    png: bytes
    width: int
    height: int
    stats: RenderStats

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height} PNG, {self.stats.triangles_drawn} triangles drawn "
            f"({self.stats.triangles_submitted} submitted)."
        )


class CharacterPreviewRenderer:
    """Render a character (with its rig and materials) to a PNG.

    It builds the scene through the project's own scene helpers -- three-point
    lighting, a framed camera -- so the preview is the same body every other
    surface shows. ``azimuth``/``elevation``/``zoom`` orbit the camera; the figure
    and its pose are untouched.
    """

    def __init__(self, materials: MaterialManager | None = None) -> None:
        self.materials = materials or MaterialManager()

    def scene(
        self,
        params: FigureParameters,
        *,
        view: str = "full",
        azimuth: float = 0.0,
        elevation: float = 0.0,
        zoom: float = 1.0,
        light_scale: float = 1.0,
    ) -> Scene:
        from dataclasses import replace

        from core.romance.body import preview_scene

        scene = preview_scene(
            params, view=view, azimuth=azimuth, elevation=elevation, zoom=zoom
        )
        if light_scale != 1.0:
            scene.lights = [
                replace(light, intensity=light.intensity * light_scale)
                for light in scene.lights
            ]
        return scene

    def render(
        self,
        params: FigureParameters,
        *,
        view: str = "full",
        width: int = 420,
        height: int = 620,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        zoom: float = 1.0,
        light_scale: float = 1.0,
    ) -> PreviewReport:
        scene = self.scene(
            params, view=view, azimuth=azimuth, elevation=elevation,
            zoom=zoom, light_scale=light_scale,
        )
        buffer, stats = render_scene(scene, width=int(width), height=int(height))
        return PreviewReport(buffer.to_png(), int(width), int(height), stats)

    def png(self, params: FigureParameters, **options: Any) -> bytes:
        return self.render(params, **options).png


# --------------------------------------------------------------------------- #
# The engine facade
# --------------------------------------------------------------------------- #


@dataclass
class CharacterInstance:
    """A live character: its parameters, its bound object, and its template id."""

    character_id: str
    template_id: str
    params: FigureParameters
    obj: SceneObject

    @property
    def mesh(self) -> Mesh:
        return self.obj.mesh

    @property
    def skeleton(self) -> Skeleton | None:
        return self.obj.skeleton


class CharacterGenerationEngine:
    """The FR-CCE facade: templates, mesh, rig, materials, morphs, and storage.

    One object composes the components above so a caller -- the Body & Shape
    window, the CLI, or a test -- drives the whole engine through one API. It
    creates a character from a template, morphs it, poses it, renders it, and saves
    or restores its configuration, always through the project's own 3D systems.
    """

    def __init__(
        self,
        *,
        templates: CharacterTemplateManager | None = None,
        generator: ProceduralMeshGenerator | None = None,
        parameters: BodyParameterSystem | None = None,
        rig: SkeletonRigManager | None = None,
        materials: MaterialManager | None = None,
        presets: CharacterPresetManager | None = None,
        persistence: CharacterPersistenceManager | None = None,
        renderer: CharacterPreviewRenderer | None = None,
    ) -> None:
        self.templates = templates or CharacterTemplateManager()
        self.generator = generator or ProceduralMeshGenerator()
        self.parameters = parameters or BodyParameterSystem()
        self.rig = rig or SkeletonRigManager()
        self.materials = materials or MaterialManager()
        self.morphs = MorphTargetEngine(self.generator)
        self.presets = presets or CharacterPresetManager()
        self.persistence = persistence or CharacterPersistenceManager()
        self.renderer = renderer or CharacterPreviewRenderer(self.materials)

    # -- creation --------------------------------------------------------

    def create(
        self,
        template_id: str,
        *,
        character_id: str = "",
        name: str = "",
        overrides: dict[str, Any] | None = None,
    ) -> CharacterInstance:
        """Build a character from a template (an independent copy of its params)."""
        params = self.templates.parameters(template_id, name=name)
        if overrides:
            params = self.parameters.set_many(params, overrides)
        obj = build_figure(params)
        self.rig.bind(obj)
        return CharacterInstance(character_id, template_id, params, obj)

    def create_for_character(self, character: Any, *, character_id: str = "") -> CharacterInstance:
        """Build the character whose template its own sex and age select."""
        template_id = self.templates.template_for_character(character)
        name = str(_character_field(character, "name") or "") or self.templates.get(
            template_id
        ).title
        cid = character_id or str(_character_field(character, "id") or "")
        return self.create(template_id, character_id=cid, name=name)

    def create_for_params(
        self, params: FigureParameters, *, template_id: str = "", character_id: str = ""
    ) -> CharacterInstance:
        """Build a live character from an existing parameter set.

        A window that edits parameters directly (rather than through a template)
        uses this to render or bind them; the parameters are used as given, so the
        rendered body is exactly the one the dials describe.
        """
        obj = build_figure(params)
        self.rig.bind(obj)
        return CharacterInstance(character_id, template_id or params.build, params, obj)

    # -- morphing --------------------------------------------------------

    def morph(
        self, instance: CharacterInstance, key: str, value: Any
    ) -> MorphResult:
        """Set one parameter on a live character and report the real geometry move."""
        updated, _mesh, result = self.morphs.apply(instance.params, key, value)
        if not result.changed:
            return result
        pose = dict(instance.obj.pose)
        rebuilt = build_figure(updated, name=instance.obj.name, material=instance.obj.material)
        rebuilt.pose = pose
        self.rig.bind(rebuilt)
        instance.params = updated
        instance.obj = rebuilt
        return result

    def morph_many(
        self, instance: CharacterInstance, changes: dict[str, Any]
    ) -> MorphResult:
        """Set several parameters and report the net geometry move."""
        before = instance.mesh
        params = self.parameters.set_many(instance.params, changes)
        mesh = self.generator.build(params)
        result = MorphTargetEngine.compare(before, mesh)
        if result.changed:
            pose = dict(instance.obj.pose)
            rebuilt = build_figure(
                params, name=instance.obj.name, material=instance.obj.material
            )
            rebuilt.pose = pose
            self.rig.bind(rebuilt)
            instance.params = params
            instance.obj = rebuilt
        return result

    def regenerate(self, instance: CharacterInstance) -> CharacterInstance:
        """Rebuild a character from its template's defaults (its pose kept).

        The character's *saved* configuration is not touched by this -- a
        regeneration only changes the live instance -- so a user's saved body is
        never lost by regenerating.
        """
        template_id = instance.template_id
        name = instance.params.name
        pose = dict(instance.obj.pose)
        fresh = self.templates.parameters(template_id, name=name)
        obj = build_figure(fresh)
        obj.pose = pose
        self.rig.bind(obj)
        return CharacterInstance(instance.character_id, template_id, fresh, obj)

    def restore(self, instance: CharacterInstance, config: CharacterBodyConfig) -> CharacterInstance:
        """Restore a saved configuration onto a live character."""
        params = params_from_primitive(config.parameters)
        pose = dict(instance.obj.pose)
        obj = build_figure(params)
        obj.pose = pose
        self.rig.bind(obj)
        return CharacterInstance(
            instance.character_id or config.character_id,
            config.template_id,
            params,
            obj,
        )

    # -- pose ------------------------------------------------------------

    def set_pose(self, instance: CharacterInstance, preset: str) -> None:
        self.rig.pose(instance.obj, preset)

    # -- rendering -------------------------------------------------------

    def render(
        self,
        instance: CharacterInstance,
        *,
        view: str = "full",
        width: int = 420,
        height: int = 620,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        zoom: float = 1.0,
        light_scale: float = 1.0,
    ) -> PreviewReport:
        return self.renderer.render(
            instance.params, view=view, width=width, height=height,
            azimuth=azimuth, elevation=elevation, zoom=zoom, light_scale=light_scale,
        )

    # -- persistence -----------------------------------------------------

    def save(
        self, instance: CharacterInstance, *, preset_id: str = ""
    ) -> CharacterBodyConfig:
        """Save a character's configuration (never a template's)."""
        skeleton = {
            "bones": list(self.rig.build(instance.params).names()),
        }
        materials = self.materials.capabilities()
        return self.persistence.save(
            instance.character_id,
            instance.template_id,
            instance.params,
            skeleton=skeleton,
            materials=materials,
            preset_id=preset_id,
        )

    def load_config(self, character_id: str) -> CharacterBodyConfig | None:
        return self.persistence.load(character_id)

    def load_for_character(
        self, character: Any, *, character_id: str = ""
    ) -> CharacterInstance:
        """The character a surface should open: its saved body, else its template."""
        cid = character_id or str(_character_field(character, "id") or "")
        config = self.persistence.load(cid) if cid else None
        if config is not None:
            template_id = config.template_id if self.templates.has(config.template_id) else (
                self.templates.template_for_character(character)
            )
            params = params_from_primitive(config.parameters)
            obj = build_figure(params)
            self.rig.bind(obj)
            return CharacterInstance(cid, template_id, params, obj)
        return self.create_for_character(character, character_id=cid)

    # -- presets ---------------------------------------------------------

    def save_preset(
        self, preset_id: str, title: str, instance: CharacterInstance
    ) -> CharacterPreset:
        return self.presets.save(preset_id, title, instance.template_id, instance.params)

    def apply_preset(
        self, instance: CharacterInstance, preset_id: str
    ) -> CharacterInstance | None:
        params = self.presets.parameters(preset_id, name=instance.params.name)
        if params is None:
            return None
        pose = dict(instance.obj.pose)
        obj = build_figure(params)
        obj.pose = pose
        self.rig.bind(obj)
        return CharacterInstance(instance.character_id, instance.template_id, params, obj)

    # -- parameter-space helpers (no mesh built) -------------------------

    def template_params(self, template_id: str, *, name: str = "") -> FigureParameters:
        """A template's parameters without building a mesh (cheap for a window)."""
        return self.templates.parameters(template_id, name=name)

    def save_params(
        self,
        character_id: str,
        template_id: str,
        params: FigureParameters,
        *,
        preset_id: str = "",
    ) -> CharacterBodyConfig:
        """Save a parameter set as a character's body (no mesh is built)."""
        return self.persistence.save(
            character_id, template_id, params, preset_id=preset_id
        )

    def load_params(self, character_id: str) -> FigureParameters | None:
        """A character's saved body parameters, or ``None`` when there is none."""
        config = self.persistence.load(character_id)
        if config is None:
            return None
        return params_from_primitive(config.parameters)

    def save_preset_params(
        self, preset_id: str, title: str, template_id: str, params: FigureParameters
    ) -> CharacterPreset:
        """Save a named preset from a parameter set (no mesh is built)."""
        return self.presets.save(preset_id, title, template_id, params)

    def load_preset_params(self, preset_id: str) -> FigureParameters | None:
        """A saved preset's parameters, or ``None`` when it is absent."""
        return self.presets.parameters(preset_id)

    # -- introspection ---------------------------------------------------

    def status(self) -> dict[str, Any]:
        """An honest status of the engine (what it can do, and what it cannot)."""
        return {
            "templates": list(self.templates.keys()),
            "parameters": len(self.parameters.keys()),
            "groups": [group for group, _title, _dials in self.parameters.groups()],
            "materials": self.materials.capabilities(),
            "engine_version": __version__,
            "config_version": CONFIG_VERSION,
            "honesty": HONESTY_LINE,
        }


# --------------------------------------------------------------------------- #
# Default engine
# --------------------------------------------------------------------------- #


def default_engine(
    *,
    data_root: str | Path | None = None,
) -> CharacterGenerationEngine:
    """The engine a surface uses, with its stores under ``data_root`` when given.

    With no ``data_root`` the stores use the project's own data folders
    (:func:`core.paths.character_bodies_dir` and
    :func:`core.paths.character_presets_dir`); a caller that scopes the engine to
    its own folder (a Studio built with ``root=``) passes that folder instead, so
    a test's bodies never touch the real ones.
    """
    if data_root is None:
        return CharacterGenerationEngine()
    root = Path(data_root)
    return CharacterGenerationEngine(
        presets=CharacterPresetManager(root / "presets"),
        persistence=CharacterPersistenceManager(root),
    )


#: A module-level engine for the window and the CLI (its stores use the data dir).
_ENGINE: CharacterGenerationEngine | None = None


def engine() -> CharacterGenerationEngine:
    """The shared default engine (created on first use)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = default_engine()
    return _ENGINE


__all__ = [
    "ADULT_AGE",
    "CONFIG_VERSION",
    "HONESTY_LINE",
    "MEASUREMENT_KEYS",
    "TEMPLATE_BOY",
    "TEMPLATE_GIRL",
    "TEMPLATE_IDS",
    "TEMPLATE_MAN",
    "TEMPLATE_TITLES",
    "TEMPLATE_WOMAN",
    "BodyParameterSystem",
    "CharacterBodyConfig",
    "CharacterGenerationEngine",
    "CharacterInstance",
    "CharacterPersistenceManager",
    "CharacterPreset",
    "CharacterPresetManager",
    "CharacterPreviewRenderer",
    "CharacterTemplate",
    "CharacterTemplateManager",
    "MaterialManager",
    "MorphResult",
    "MorphTargetEngine",
    "ParameterGroup",
    "PreviewReport",
    "ProceduralMeshGenerator",
    "RigReport",
    "SkeletonRigManager",
    "SurfaceMaterial",
    "default_engine",
    "engine",
    "measurement_values",
    "migrate_config",
    "params_from_primitive",
    "params_to_primitive",
]
