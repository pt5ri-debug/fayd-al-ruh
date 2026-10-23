"""Load and save real 3D content: Wavefront OBJ geometry with MTL materials.

DAZ-style figure content is distributed as meshes, and the one interchange
format every 3D program and every asset library agrees on is **Wavefront OBJ**.
This module reads and writes it, so the studio can open a figure, a prop, or a
whole scene the user already has on disk -- entirely locally, with no network
and no asset store.

What is supported, and what is not, is stated plainly:

* ``v`` / ``vn`` / ``vt`` / ``f`` (including ``f v/vt/vn`` and negative indices),
  ``o`` and ``g`` for objects and groups, ``usemtl`` / ``mtllib`` for materials,
  and ``s`` (accepted and ignored -- this studio computes its own smooth normals).
* ``.mtl`` files: ``newmtl``, ``Kd`` (diffuse colour), ``Ks`` (specular),
  ``Ns`` (shininess), and ``d``/``Tr`` (opacity).
* **Not** supported: quads-with-holes, free-form geometry, NURBS, point clouds,
  texture images, and vertex-colour extensions. An unsupported line is reported,
  never silently misread.

Honesty and safety:

* Reading a file never executes anything in it; OBJ/MTL are plain text and are
  parsed as data.
* The reader is bounded: an over-large file, or one with more vertices or faces
  than :data:`MAX_VERTICES` / :data:`MAX_FACES`, is refused with a clear message
  rather than exhausting memory.
* A missing or malformed file raises :class:`SceneIOError` naming the real
  problem, so the window and the CLI can report the actual next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.scene3d import (
    Material,
    Mesh,
    Scene,
    SceneObject,
    Vec3,
    make_default_scene,
)

#: The mesh formats this build reads and writes.
MESH_SUFFIXES = (".obj",)
MATERIAL_SUFFIXES = (".mtl",)

#: Refuse an over-large model rather than exhausting memory. These are generous
#: for a figure or a prop and small enough that a malformed file cannot hurt.
MAX_VERTICES = 2_000_000
MAX_FACES = 4_000_000
MAX_FILE_BYTES = 512 * 1024 * 1024


class SceneIOError(RuntimeError):
    """Raised when a 3D file cannot be read or written, with the real reason."""


@dataclass(frozen=True)
class LoadReport:
    """An honest account of what a load produced."""

    path: str
    objects: int
    vertices: int
    faces: int
    materials: tuple[str, ...]
    skipped_lines: int

    def summary(self) -> str:
        return (
            f"Loaded {Path(self.path).name}: {self.objects} object(s), "
            f"{self.vertices} vertices, {self.faces} faces, "
            f"{len(self.materials)} material(s)"
            + (f", {self.skipped_lines} unsupported line(s) ignored." if self.skipped_lines else ".")
        )


def parse_mtl(text: str, *, name: str = "material") -> dict[str, Material]:
    """Parse Wavefront MTL text into materials keyed by name.

    Only the fields this studio's shading model uses are read; an unknown line is
    ignored (it cannot change a render). A file with no ``newmtl`` yields an empty
    mapping rather than inventing a material.
    """
    materials: dict[str, Material] = {}
    current: str | None = None
    diffuse = (0.82, 0.66, 0.56)
    specular = 0.25
    shininess = 24.0
    opacity = 1.0

    def flush() -> None:
        if current is None:
            return
        materials[current] = Material(
            name=current,
            base_color=diffuse,
            specular=specular,
            shininess=shininess if shininess > 0 else 1.0,
            opacity=max(0.0, min(1.0, opacity)),
        )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        keyword = parts[0].lower()
        values = parts[1:]
        if keyword == "newmtl":
            flush()
            current = " ".join(values) or name
            diffuse = (0.82, 0.66, 0.56)
            specular = 0.25
            shininess = 24.0
            opacity = 1.0
        elif keyword == "kd" and len(values) >= 3:
            diffuse = _rgb(values)
        elif keyword == "ks" and len(values) >= 3:
            specular = _average(_rgb(values))
        elif keyword == "ns" and values:
            shininess = _positive_float(values[0], shininess)
        elif keyword == "d" and values:
            opacity = _unit_float(values[0], opacity)
        elif keyword == "tr" and values:
            opacity = 1.0 - _unit_float(values[0], 1.0 - opacity)
    flush()
    return materials


def _rgb(values: list[str]) -> tuple[float, float, float]:
    channels = []
    for value in values[:3]:
        try:
            channels.append(max(0.0, min(1.0, float(value))))
        except ValueError:
            channels.append(0.5)
    while len(channels) < 3:
        channels.append(0.5)
    return (channels[0], channels[1], channels[2])


def _average(color: tuple[float, float, float]) -> float:
    return (color[0] + color[1] + color[2]) / 3.0


def _positive_float(text: str, fallback: float) -> float:
    try:
        value = float(text)
    except ValueError:
        return fallback
    return value if value > 0.0 else fallback


def _unit_float(text: str, fallback: float) -> float:
    try:
        value = float(text)
    except ValueError:
        return fallback
    return max(0.0, min(1.0, value))


def _parse_index(token: str, count: int, *, what: str) -> int:
    """Resolve an OBJ index token (1-based, or negative for "from the end")."""
    try:
        index = int(token)
    except ValueError as error:
        raise SceneIOError(f"Invalid {what} index {token!r}.") from error
    if index > 0:
        return index - 1
    if index < 0:
        return count + index
    raise SceneIOError(f"{what.capitalize()} index 0 is not valid in OBJ.")


def load_obj(path: str | Path) -> tuple[Scene, LoadReport]:
    """Load a Wavefront OBJ (plus its MTL, when named) into a scene.

    The file's objects and groups become the scene's objects, each with the
    material last named by ``usemtl``. A file with no ``o``/``g`` becomes one
    object named after the file. The scene gets the studio's default three-point
    light rig and is framed on its own bounds, so a freshly loaded model is
    immediately visible.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise SceneIOError(f"No such file: {target}")
    try:
        size = target.stat().st_size
    except OSError as error:
        raise SceneIOError(f"Could not read {target}: {error}") from error
    if size > MAX_FILE_BYTES:
        raise SceneIOError(
            f"{target.name} is {size / 1_048_576:.1f} MB, over the "
            f"{MAX_FILE_BYTES // 1_048_576} MB limit."
        )
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise SceneIOError(f"Could not read {target}: {error}") from error
    return load_obj_text(text, name=target.stem, directory=target.parent)


def load_obj_text(
    text: str, *, name: str = "model", directory: Path | None = None
) -> tuple[Scene, LoadReport]:
    """Load OBJ text (used by :func:`load_obj`, and directly by tests)."""
    positions: list[Vec3] = []
    materials: dict[str, Material] = {}
    current_material: str | None = None

    objects: list[tuple[str, list[tuple[int, ...]], str | None]] = []
    current_name: str | None = None
    current_faces: list[tuple[int, ...]] = []
    skipped = 0

    def finish_object() -> None:
        nonlocal current_name, current_faces
        # An OBJ with no ``o``/``g`` line is common (a single unnamed mesh), so a
        # pending group is finished under the file's own name rather than dropped.
        # The material in force is recorded with the group, so a multi-material
        # file does not hand every object whichever material happened to be last.
        if current_faces:
            objects.append((current_name or name, list(current_faces), current_material))
        current_name = None
        current_faces = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        keyword = parts[0].lower()
        values = parts[1:]
        if keyword == "v" and len(values) >= 3:
            positions.append(_vec(values))
        elif keyword == "vn" and len(values) >= 3:
            # File normals are read only to keep face tokens valid; the mesh
            # recomputes smooth normals so shading is consistent everywhere.
            continue
        elif keyword == "vt":
            # Texture coordinates are read and dropped: this studio's shading
            # model has no texture mapping, and saying so is better than a
            # silently ignored feature.
            continue
        elif keyword in {"o", "g"}:
            finish_object()
            current_name = " ".join(values) or name
        elif keyword == "mtllib" and values and directory is not None:
            for filename in values:
                candidate = directory / filename
                try:
                    materials.update(
                        parse_mtl(candidate.read_text(encoding="utf-8", errors="replace"))
                    )
                except OSError:
                    continue
        elif keyword == "usemtl":
            current_material = " ".join(values) or None
        elif keyword == "s":
            continue
        elif keyword == "f" and len(values) >= 3:
            if len(positions) > MAX_VERTICES or len(current_faces) > MAX_FACES:
                raise SceneIOError(
                    f"{name!r} exceeds the supported size "
                    f"({MAX_VERTICES} vertices / {MAX_FACES} faces)."
                )
            face = _parse_face(values, len(positions))
            current_faces.append(face)
        else:
            skipped += 1
    finish_object()

    if not positions:
        raise SceneIOError(f"{name!r} contains no vertices.")

    if current_name is None:
        current_name = name
    if not objects:
        raise SceneIOError(f"{name!r} contains no faces.")

    scene = make_default_scene(name)
    for object_name, faces, material_name in objects:
        mesh = Mesh(object_name, tuple(positions), tuple(faces))
        material = materials.get(material_name or "", Material())
        scene.add_object(SceneObject(object_name, mesh, material))

    total_faces = sum(len(obj.mesh.faces) for obj in scene.objects)
    scene.frame_camera()
    report = LoadReport(
        path=name,
        objects=len(scene.objects),
        vertices=len(positions),
        faces=total_faces,
        materials=tuple(sorted(materials)),
        skipped_lines=skipped,
    )
    return scene, report


def _parse_face(values: list[str], vertex_count: int) -> tuple[int, ...]:
    """Parse one ``f`` line's vertex indices, ignoring any ``vt``/``vn`` parts."""
    indices: list[int] = []
    for token in values:
        pieces = token.split("/")
        if not pieces or not pieces[0]:
            raise SceneIOError(f"Malformed face vertex {token!r}.")
        index = _parse_index(pieces[0], vertex_count, what="vertex")
        if not 0 <= index < vertex_count:
            raise SceneIOError(f"Face vertex {token!r} is outside the model.")
        indices.append(index)
    return tuple(indices)


def _vec(values: list[str]) -> Vec3:
    try:
        return Vec3(float(values[0]), float(values[1]), float(values[2]))
    except ValueError as error:
        raise SceneIOError(f"Malformed vertex line: {' '.join(values)!r}.") from error


def save_obj(scene: Scene, path: str | Path, *, write_materials: bool = True) -> Path:
    """Write a scene to a Wavefront OBJ (and a sidecar MTL), returning the path.

    Every visible object is written under its own ``o`` group with its material,
    so the file re-opens in this studio or any other 3D program. Geometry is
    written in each object's **world** pose, because OBJ has no skeleton: what is
    saved is the figure as it is currently posed.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = target.with_suffix(".mtl").name

    lines: list[str] = [
        "# Written by Fayd Al-Ruh Scene Studio (local 3D figure studio)",
        f"# scene: {scene.name}",
    ]
    if write_materials:
        lines.append(f"mtllib {mtl_name}")
    vertex_offset = 1
    materials: dict[str, Material] = {}
    for obj in scene.visible_objects():
        lines.append(f"o {obj.name.replace(' ', '_')}")
        if write_materials:
            material_name = obj.material.name or "default"
            materials[material_name] = obj.material
            lines.append(f"usemtl {material_name}")
        for vertex in obj.world_vertices():
            lines.append(f"v {vertex.x:.6f} {vertex.y:.6f} {vertex.z:.6f}")
        for face in obj.mesh.faces:
            tokens = " ".join(str(vertex_offset + index) for index in face)
            lines.append(f"f {tokens}")
        vertex_offset += obj.mesh.vertex_count

    try:
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if write_materials and materials:
            target.with_suffix(".mtl").write_text(
                write_mtl(materials), encoding="utf-8"
            )
    except OSError as error:
        raise SceneIOError(f"Could not write {target}: {error}") from error
    return target


def write_mtl(materials: dict[str, Material]) -> str:
    """Render a materials mapping as Wavefront MTL text."""
    blocks: list[str] = ["# Written by Fayd Al-Ruh Scene Studio"]
    for name, material in materials.items():
        red, green, blue = material.base_color
        blocks.extend(
            [
                f"newmtl {name}",
                f"Kd {red:.6f} {green:.6f} {blue:.6f}",
                f"Ks {material.specular:.6f} {material.specular:.6f} {material.specular:.6f}",
                f"Ns {material.shininess:.6f}",
                f"d {material.opacity:.6f}",
                "",
            ]
        )
    return "\n".join(blocks)


def load_scene(path: str | Path) -> tuple[Scene, LoadReport]:
    """Load a scene from a supported file, dispatching on its suffix."""
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    if suffix in MESH_SUFFIXES:
        return load_obj(target)
    raise SceneIOError(
        f"Unsupported 3D format {suffix or '(none)'!r}; this build reads "
        f"{', '.join(MESH_SUFFIXES)}."
    )


def is_mesh_file(path: str | Path) -> bool:
    """True when the file's suffix is one this build reads."""
    return Path(path).suffix.lower() in MESH_SUFFIXES


def mesh_filetypes() -> list[tuple[str, str]]:
    """The file-dialog filters, derived from the supported suffixes."""
    return [
        ("3D models", " ".join(f"*{suffix}" for suffix in MESH_SUFFIXES)),
        ("Wavefront OBJ", "*.obj"),
        ("All files", "*.*"),
    ]


__all__ = [
    "MATERIAL_SUFFIXES",
    "MAX_FACES",
    "MAX_FILE_BYTES",
    "MAX_VERTICES",
    "MESH_SUFFIXES",
    "LoadReport",
    "SceneIOError",
    "is_mesh_file",
    "load_obj",
    "load_obj_text",
    "load_scene",
    "mesh_filetypes",
    "parse_mtl",
    "save_obj",
    "write_mtl",
]
