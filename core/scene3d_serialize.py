"""Save and reload a scene as a small, readable JSON document.

OBJ (in :mod:`core.scene3d_io`) is the *interchange* format: geometry any 3D
program can read, but with no rig. This module is the *native* format: a scene
saved here keeps everything the studio knows -- each object's transform and
material, the figure's **skeleton**, and the **pose** the user applied -- so a
saved scene reopens exactly as it was left, ready to keep posing.

The format is plain JSON, deliberately: a saved scene is text the user can read,
diff, and hand-edit, and it needs no library to parse. Skinning bindings are
**not** stored -- they are recomputed deterministically on load from the mesh and
skeleton, so the file stays small and can never drift out of sync with the rig.

Honesty and safety, as everywhere else: loading never executes anything (the file
is data), an unknown format version is refused with a clear message rather than
misread, and a malformed field raises :class:`SceneIOError` naming the problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.scene3d import (
    Bone,
    Camera,
    Light,
    Material,
    Mesh,
    Scene,
    SceneObject,
    Skeleton,
    Transform,
    Vec3,
)
from core.scene3d_io import SceneIOError

#: The native scene format's version. A file with another version is refused.
SCENE_FORMAT_VERSION = 1

#: The suffix of a native scene file.
SCENE_SUFFIXES = (".faydscene", ".json")


def _vec_to_list(vector: Vec3) -> list[float]:
    # Full precision: JSON round-trips a Python float exactly, so the geometry --
    # and therefore the recomputed skin weights -- is identical after a reload.
    # Rounding here would perturb a vertex slightly, and skin weights amplify that
    # near a joint, which showed up as a visible millimetre-scale pose shift.
    return [vector.x, vector.y, vector.z]


def _vec_from_list(values: Any, *, field: str) -> Vec3:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise SceneIOError(f"Field {field!r} must be three numbers.")
    try:
        return Vec3(float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError) as error:
        raise SceneIOError(f"Field {field!r} must be three numbers.") from error


def _mesh_to_dict(mesh: Mesh) -> dict[str, Any]:
    return {
        "name": mesh.name,
        "vertices": [_vec_to_list(vertex) for vertex in mesh.vertices],
        "faces": [list(face) for face in mesh.faces],
    }


def _mesh_from_dict(data: Any) -> Mesh:
    if not isinstance(data, dict):
        raise SceneIOError("A mesh must be an object.")
    name = str(data.get("name", "mesh"))
    raw_vertices = data.get("vertices", [])
    raw_faces = data.get("faces", [])
    if not isinstance(raw_vertices, list) or not isinstance(raw_faces, list):
        raise SceneIOError(f"Mesh {name!r} needs a vertex list and a face list.")
    vertices = tuple(
        _vec_from_list(vertex, field=f"{name}.vertices") for vertex in raw_vertices
    )
    faces: list[tuple[int, ...]] = []
    for face in raw_faces:
        if not isinstance(face, list) or len(face) < 3:
            raise SceneIOError(f"Mesh {name!r} has a malformed face.")
        faces.append(tuple(int(index) for index in face))
    return Mesh(name, vertices, tuple(faces))


def _material_to_dict(material: Material) -> dict[str, Any]:
    return {
        "name": material.name,
        "base_color": list(material.base_color),
        "specular": material.specular,
        "shininess": material.shininess,
        "opacity": material.opacity,
    }


def _material_from_dict(data: Any) -> Material:
    if not isinstance(data, dict):
        return Material()
    color = data.get("base_color", [0.82, 0.66, 0.56])
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        color = [0.82, 0.66, 0.56]
    return Material(
        name=str(data.get("name", "default")),
        base_color=(float(color[0]), float(color[1]), float(color[2])),
        specular=float(data.get("specular", 0.25)),
        shininess=float(data.get("shininess", 24.0)),
        opacity=float(data.get("opacity", 1.0)),
    )


def _skeleton_to_dict(skeleton: Skeleton) -> dict[str, Any]:
    return {
        "bones": [
            {
                "name": bone.name,
                "parent": bone.parent,
                "offset": _vec_to_list(bone.offset),
                "rest_rotation": _vec_to_list(bone.rest_rotation),
                "length": bone.length,
            }
            for bone in skeleton.bones
        ]
    }


def _skeleton_from_dict(data: Any) -> Skeleton:
    if not isinstance(data, dict):
        raise SceneIOError("A skeleton must be an object.")
    raw_bones = data.get("bones", [])
    if not isinstance(raw_bones, list):
        raise SceneIOError("A skeleton needs a bone list.")
    bones: list[Bone] = []
    for entry in raw_bones:
        if not isinstance(entry, dict):
            raise SceneIOError("A bone must be an object.")
        parent = entry.get("parent")
        bones.append(
            Bone(
                name=str(entry.get("name", "")),
                parent=str(parent) if parent else None,
                offset=_vec_from_list(entry.get("offset", [0, 0, 0]), field="bone.offset"),
                rest_rotation=_vec_from_list(
                    entry.get("rest_rotation", [0, 0, 0]), field="bone.rest_rotation"
                ),
                length=float(entry.get("length", 0.0)),
            )
        )
    return Skeleton(tuple(bones))


def _transform_to_dict(transform: Transform) -> dict[str, Any]:
    return {
        "translation": _vec_to_list(transform.translation),
        "rotation": _vec_to_list(transform.rotation),
        "scale": _vec_to_list(transform.scale),
    }


def _transform_from_dict(data: Any) -> Transform:
    if not isinstance(data, dict):
        return Transform()
    return Transform(
        translation=_vec_from_list(data.get("translation", [0, 0, 0]), field="translation"),
        rotation=_vec_from_list(data.get("rotation", [0, 0, 0]), field="rotation"),
        scale=_vec_from_list(data.get("scale", [1, 1, 1]), field="scale"),
    )


def object_to_dict(obj: SceneObject) -> dict[str, Any]:
    """A scene object as a JSON-ready mapping."""
    data: dict[str, Any] = {
        "name": obj.name,
        "mesh": _mesh_to_dict(obj.mesh),
        "material": _material_to_dict(obj.material),
        "transform": _transform_to_dict(obj.transform),
        "visible": obj.visible,
    }
    if obj.skeleton is not None:
        data["skeleton"] = _skeleton_to_dict(obj.skeleton)
        data["pose"] = {
            bone: _vec_to_list(rotation) for bone, rotation in obj.pose.items()
        }
    return data


def object_from_dict(data: Any) -> SceneObject:
    """Rebuild a scene object, re-binding a figure's skin deterministically."""
    if not isinstance(data, dict):
        raise SceneIOError("A scene object must be an object.")
    mesh = _mesh_from_dict(data.get("mesh", {}))
    skeleton_data = data.get("skeleton")
    skeleton = _skeleton_from_dict(skeleton_data) if skeleton_data is not None else None
    pose: dict[str, Vec3] = {}
    raw_pose = data.get("pose", {})
    if isinstance(raw_pose, dict):
        for bone, rotation in raw_pose.items():
            pose[str(bone)] = _vec_from_list(rotation, field=f"pose.{bone}")
    obj = SceneObject(
        name=str(data.get("name", mesh.name)),
        mesh=mesh,
        material=_material_from_dict(data.get("material")),
        transform=_transform_from_dict(data.get("transform")),
        skeleton=skeleton,
        pose=pose,
        visible=bool(data.get("visible", True)),
    )
    if skeleton is not None:
        obj.bind()
    return obj


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    """A whole scene as a JSON-ready mapping."""
    return {
        "format": "fayd-al-ruh.scene",
        "version": SCENE_FORMAT_VERSION,
        "name": scene.name,
        "width": scene.width,
        "height": scene.height,
        "background": list(scene.background),
        "camera": {
            "position": _vec_to_list(scene.camera.position),
            "target": _vec_to_list(scene.camera.target),
            "up": _vec_to_list(scene.camera.up),
            "fov_degrees": scene.camera.fov_degrees,
            "near": scene.camera.near,
            "far": scene.camera.far,
        },
        "lights": [
            {
                "name": light.name,
                "kind": light.kind,
                "direction": _vec_to_list(light.direction),
                "position": _vec_to_list(light.position),
                "color": list(light.color),
                "intensity": light.intensity,
            }
            for light in scene.lights
        ],
        "objects": [object_to_dict(obj) for obj in scene.objects],
    }


def scene_from_dict(data: Any) -> Scene:
    """Rebuild a scene from a mapping, validating the format version."""
    if not isinstance(data, dict):
        raise SceneIOError("A scene file must contain an object.")
    version = data.get("version")
    if version != SCENE_FORMAT_VERSION:
        raise SceneIOError(
            f"Unsupported scene format version {version!r}; this build reads "
            f"version {SCENE_FORMAT_VERSION}."
        )
    scene = Scene(
        name=str(data.get("name", "Untitled scene")),
        width=int(data.get("width", 800)),
        height=int(data.get("height", 600)),
    )
    background = data.get("background", [0.09, 0.11, 0.15])
    if isinstance(background, (list, tuple)) and len(background) >= 3:
        scene.background = (float(background[0]), float(background[1]), float(background[2]))

    camera = data.get("camera")
    if isinstance(camera, dict):
        scene.camera = Camera(
            position=_vec_from_list(camera.get("position", [0, 1.4, 3.6]), field="camera.position"),
            target=_vec_from_list(camera.get("target", [0, 1, 0]), field="camera.target"),
            up=_vec_from_list(camera.get("up", [0, 1, 0]), field="camera.up"),
            fov_degrees=float(camera.get("fov_degrees", 35.0)),
            near=float(camera.get("near", 0.05)),
            far=float(camera.get("far", 100.0)),
        )

    lights = data.get("lights", [])
    if isinstance(lights, list):
        for light in lights:
            if not isinstance(light, dict):
                continue
            color = light.get("color", [1, 1, 1])
            scene.lights.append(
                Light(
                    name=str(light.get("name", "light")),
                    kind=str(light.get("kind", "directional")),
                    direction=_vec_from_list(light.get("direction", [-0.4, -0.8, -0.5]), field="light.direction"),
                    position=_vec_from_list(light.get("position", [2, 3, 2]), field="light.position"),
                    color=(float(color[0]), float(color[1]), float(color[2])),
                    intensity=float(light.get("intensity", 1.0)),
                )
            )

    objects = data.get("objects", [])
    if isinstance(objects, list):
        for entry in objects:
            scene.objects.append(object_from_dict(entry))
    return scene


def save_scene(scene: Scene, path: str | Path) -> Path:
    """Write a scene to a native scene file (JSON), returning the path."""
    target = Path(path).expanduser()
    if target.suffix.lower() not in SCENE_SUFFIXES:
        target = target.with_suffix(".faydscene")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(
            json.dumps(scene_to_dict(scene), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise SceneIOError(f"Could not write {target}: {error}") from error
    return target


def load_scene_file(path: str | Path) -> Scene:
    """Read a native scene file, raising :class:`SceneIOError` on any problem."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise SceneIOError(f"No such scene: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise SceneIOError(f"Could not read {target}: {error}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise SceneIOError(f"{target.name} is not valid JSON: {error}") from error
    return scene_from_dict(data)


def scene_filename(name: str) -> str:
    """A filesystem-safe scene filename derived from a scene name."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name.strip()
    )
    # Collapse the runs a separator or a punctuation mark left behind, and trim
    # them, so "My Scene: v2!" reads as "My_Scene_v2" rather than "_My_Scene__v2_".
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return f"{cleaned or 'scene'}.faydscene"


__all__ = [
    "SCENE_FORMAT_VERSION",
    "SCENE_SUFFIXES",
    "load_scene_file",
    "object_from_dict",
    "object_to_dict",
    "save_scene",
    "scene_filename",
    "scene_from_dict",
    "scene_to_dict",
]
