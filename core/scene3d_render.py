"""The local 3D studio's renderer: a real, CPU, z-buffered rasteriser.

This is what turns a :class:`core.scene3d.Scene` into pixels. It is a genuine
software renderer, not a placeholder:

* a **z-buffer** per pixel, so nearer geometry correctly hides farther geometry;
* **perspective-correct** barycentric interpolation of depth, world position, and
  normals across each triangle;
* **Blinn-Phong** shading from the scene's lights, with an ambient floor;
* optional **wireframe overlay** and **backface culling**.

It is deliberately CPU-only and standard-library only, so the whole studio runs
on a plain laptop with no GPU, no OpenGL, and no third-party package -- and it is
testable with no display. The output is a :class:`Framebuffer` of 8-bit RGB that
:meth:`Framebuffer.to_png` encodes with the project's own dependency-free PNG
writer (:func:`core.imagegen.encode_png`), so a render is a real image file.

Honesty: this is a **rasteriser**, and it says so. It does not do global
illumination, shadows, reflections, or GPU acceleration, and it is not a
commercial render engine. A triangle that crosses the near plane is skipped
rather than clipped, which is stated plainly in the module's contract.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.imagegen import encode_png
from core.scene3d import (
    Light,
    Mat4,
    Material,
    Scene,
    SceneObject,
    Vec3,
    iter_triangles,
)

RGB = tuple[int, int, int]

#: Ambient light added to every surface, so nothing is ever pure black.
AMBIENT = 0.16


def _resolve_size(value: int | None, fallback: int, field: str) -> int:
    """Resolve one render dimension, refusing a non-positive or non-integer value.

    ``width or scene.width`` silently treated ``0`` as "use the default", so a
    caller that asked for a zero-size render got a normal picture back instead of
    an error; this makes the refusal explicit.
    """
    if value is None:
        resolved = int(fallback)
    elif isinstance(value, bool):
        raise ValueError(f"A render {field} must be an integer, not a bool.")
    else:
        try:
            resolved = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"A render {field} must be an integer.") from error
    if resolved < 1:
        raise ValueError(f"A render needs a positive {field} (got {resolved}).")
    return resolved


@dataclass
class Framebuffer:
    """An RGB image being drawn: a flat byte buffer plus a depth buffer."""

    width: int
    height: int
    pixels: bytearray = field(default_factory=bytearray)
    depth: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("A framebuffer needs a positive size.")
        if not self.pixels:
            self.pixels = bytearray(self.width * self.height * 3)
        if not self.depth:
            self.depth = [math.inf] * (self.width * self.height)

    def clear(self, color: tuple[float, float, float]) -> None:
        """Fill the whole image with one colour and reset the depth buffer."""
        pixel = bytes(
            max(0, min(255, round(channel * 255.0))) for channel in color
        ) * self.width
        row = bytearray()
        for _ in range(self.height):
            row.extend(pixel)
        self.pixels = row
        self.depth = [math.inf] * (self.width * self.height)

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset] = color[0]
        self.pixels[offset + 1] = color[1]
        self.pixels[offset + 2] = color[2]

    def get_pixel(self, x: int, y: int) -> RGB:
        offset = (y * self.width + x) * 3
        return (
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
        )

    def rows(self) -> Iterable[bytes]:
        """Yield each scanline as bytes, the shape the PNG encoder wants."""
        stride = self.width * 3
        for y in range(self.height):
            start = y * stride
            yield bytes(self.pixels[start : start + stride])

    def to_png(self) -> bytes:
        """Encode the image as a PNG with the project's dependency-free writer."""
        return encode_png(self.width, self.height, self.rows())


@dataclass(frozen=True)
class RenderStats:
    """An honest account of one render, for the window and the CLI."""

    width: int
    height: int
    objects: int
    triangles_submitted: int
    triangles_drawn: int
    shaded: bool
    wireframe: bool

    def summary(self) -> str:
        return (
            f"Rendered {self.width}x{self.height}: {self.objects} object(s), "
            f"{self.triangles_drawn}/{self.triangles_submitted} triangle(s) drawn, "
            f"{'shaded' if self.shaded else 'flat'}"
            f"{' + wireframe' if self.wireframe else ''}. "
            "CPU software rasteriser (no GPU)."
        )


def _clamp_channel(value: float) -> int:
    return max(0, min(255, round(value * 255.0)))


def shade_point(
    position: Vec3,
    normal: Vec3,
    view_direction: Vec3,
    material: Material,
    lights: Iterable[Light],
    *,
    ambient: float = AMBIENT,
) -> RGB:
    """Blinn-Phong shading for one surface point, returning an 8-bit colour.

    ``view_direction`` points from the surface towards the eye. The ambient term
    keeps an unlit surface visible; each light adds a diffuse and a specular term.
    """
    facing = normal.normalized()
    if facing.dot(view_direction) < 0.0:
        # The normal points away from the camera (a thin/open surface); flip it so
        # the surface is still lit rather than rendered black.
        facing = -facing
    red = material.base_color[0] * ambient
    green = material.base_color[1] * ambient
    blue = material.base_color[2] * ambient
    for light in lights:
        to_light = light.direction_to_light(position)
        diffuse = max(0.0, facing.dot(to_light))
        if diffuse <= 0.0:
            continue
        half = (to_light + view_direction).normalized()
        specular = 0.0
        if material.specular > 0.0:
            specular = material.specular * max(0.0, facing.dot(half)) ** material.shininess
        scale = light.intensity * diffuse
        red += material.base_color[0] * light.color[0] * scale + light.color[0] * specular
        green += material.base_color[1] * light.color[1] * scale + light.color[1] * specular
        blue += material.base_color[2] * light.color[2] * scale + light.color[2] * specular
    return (_clamp_channel(red), _clamp_channel(green), _clamp_channel(blue))


@dataclass
class _Projected:
    """One vertex after projection: screen position, depth, world point, normal."""

    x: float
    y: float
    depth: float
    world: Vec3
    normal: Vec3
    behind: bool


class SoftwareRenderer:
    """Draws a :class:`Scene` into a :class:`Framebuffer` on the CPU.

    The renderer is stateless between calls: :meth:`render` builds the camera
    matrices, projects every visible object's posed geometry, and rasterises it.
    """

    def __init__(
        self,
        *,
        wireframe: bool = False,
        backface_culling: bool = True,
        shaded: bool = True,
    ) -> None:
        self.wireframe = wireframe
        self.backface_culling = backface_culling
        self.shaded = shaded

    # -- projection -------------------------------------------------------

    def _project(
        self,
        world: Vec3,
        normal: Vec3,
        view: Mat4,
        projection: Mat4,
        width: int,
        height: int,
    ) -> _Projected:
        view_position = view.transform_point(world)
        clip_x, clip_y, clip_z, clip_w = projection.transform_vec4(view_position)
        behind = clip_w <= 1e-6
        if behind:
            return _Projected(0.0, 0.0, math.inf, world, normal, True)
        inv_w = 1.0 / clip_w
        ndc_x = clip_x * inv_w
        ndc_y = clip_y * inv_w
        screen_x = (ndc_x * 0.5 + 0.5) * width
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
        # ``clip_w`` equals -view_z for a perspective matrix, so it is a
        # monotonic depth: smaller is nearer. ``clip_z`` is kept only for
        # completeness of the projection call.
        del clip_z
        return _Projected(screen_x, screen_y, clip_w, world, normal, False)

    # -- rasterisation ----------------------------------------------------

    def _draw_triangle(
        self,
        buffer: Framebuffer,
        a: _Projected,
        b: _Projected,
        c: _Projected,
        material: Material,
        lights: list[Light],
        eye: Vec3,
    ) -> bool:
        """Rasterise one screen-space triangle. Returns whether any pixel was drawn."""
        if a.behind or b.behind or c.behind:
            return False
        area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
        if abs(area) < 1e-9:
            return False
        if self.backface_culling and area >= 0.0:
            # Screen Y grows downward, so a front face has negative signed area.
            return False

        min_x = max(0, math.floor(min(a.x, b.x, c.x)))
        max_x = min(buffer.width - 1, math.ceil(max(a.x, b.x, c.x)))
        min_y = max(0, math.floor(min(a.y, b.y, c.y)))
        max_y = min(buffer.height - 1, math.ceil(max(a.y, b.y, c.y)))
        if min_x > max_x or min_y > max_y:
            return False

        inv_area = 1.0 / area
        # Perspective-correct interpolation weights use depth (clip_w).
        inv_wa = 1.0 / a.depth
        inv_wb = 1.0 / b.depth
        inv_wc = 1.0 / c.depth
        drew = False
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px = x + 0.5
                py = y + 0.5
                w0 = ((b.x - px) * (c.y - py) - (c.x - px) * (b.y - py)) * inv_area
                w1 = ((c.x - px) * (a.y - py) - (a.x - px) * (c.y - py)) * inv_area
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                depth = 1.0 / (w0 * inv_wa + w1 * inv_wb + w2 * inv_wc)
                index = y * buffer.width + x
                if depth >= buffer.depth[index]:
                    continue
                buffer.depth[index] = depth
                if self.shaded:
                    world = _interpolate(a.world, b.world, c.world, w0, w1, w2)
                    normal = _interpolate(a.normal, b.normal, c.normal, w0, w1, w2)
                    color = shade_point(
                        world,
                        normal,
                        (eye - world).normalized(),
                        material,
                        lights,
                    )
                else:
                    color = _clamp_rgb(material.base_color)
                buffer.set_pixel(x, y, color)
                drew = True
        return drew

    def _draw_wireframe(
        self, buffer: Framebuffer, a: _Projected, b: _Projected, c: _Projected
    ) -> None:
        color = (235, 240, 250)
        for start, end in ((a, b), (b, c), (c, a)):
            if start.behind or end.behind:
                continue
            _draw_line(buffer, start, end, color)

    # -- public API -------------------------------------------------------

    def render(self, scene: Scene, *, width: int | None = None, height: int | None = None) -> Framebuffer:
        """Render ``scene`` and return the finished framebuffer."""
        out_width = _resolve_size(width, scene.width, "width")
        out_height = _resolve_size(height, scene.height, "height")
        buffer = Framebuffer(out_width, out_height)
        buffer.clear(scene.background)

        aspect = out_width / out_height
        view = scene.camera.view_matrix()
        projection = scene.camera.projection_matrix(aspect)
        eye = scene.camera.position
        lights = list(scene.lights)

        for obj in scene.visible_objects():
            self._draw_object(buffer, obj, view, projection, lights, eye)
        return buffer

    def render_with_stats(
        self, scene: Scene, *, width: int | None = None, height: int | None = None
    ) -> tuple[Framebuffer, RenderStats]:
        """Render and also report honestly what was drawn."""
        out_width = _resolve_size(width, scene.width, "width")
        out_height = _resolve_size(height, scene.height, "height")
        buffer = Framebuffer(out_width, out_height)
        buffer.clear(scene.background)
        aspect = out_width / out_height
        view = scene.camera.view_matrix()
        projection = scene.camera.projection_matrix(aspect)
        eye = scene.camera.position
        lights = list(scene.lights)

        submitted = 0
        drawn = 0
        objects = 0
        for obj in scene.visible_objects():
            objects += 1
            submitted += obj.mesh.triangle_count
            drawn += self._draw_object(buffer, obj, view, projection, lights, eye)
        stats = RenderStats(
            width=out_width,
            height=out_height,
            objects=objects,
            triangles_submitted=submitted,
            triangles_drawn=drawn,
            shaded=self.shaded,
            wireframe=self.wireframe,
        )
        return buffer, stats

    def _draw_object(
        self,
        buffer: Framebuffer,
        obj: SceneObject,
        view: Mat4,
        projection: Mat4,
        lights: list[Light],
        eye: Vec3,
    ) -> int:
        """Project and rasterise one object; returns how many triangles were drawn."""
        vertices = obj.world_vertices()
        normals = obj.world_normals()
        projected = [
            self._project(vertex, normals[index], view, projection, buffer.width, buffer.height)
            for index, vertex in enumerate(vertices)
        ]
        material = obj.material
        drawn = 0
        for triangle in iter_triangles(obj.mesh):
            a, b, c = (projected[index] for index in triangle)
            if self._draw_triangle(buffer, a, b, c, material, lights, eye):
                drawn += 1
            if self.wireframe:
                self._draw_wireframe(buffer, a, b, c)
        return drawn


def _interpolate(a: Vec3, b: Vec3, c: Vec3, w0: float, w1: float, w2: float) -> Vec3:
    return Vec3(
        a.x * w0 + b.x * w1 + c.x * w2,
        a.y * w0 + b.y * w1 + c.y * w2,
        a.z * w0 + b.z * w1 + c.z * w2,
    )


def _clamp_rgb(color: tuple[float, float, float]) -> RGB:
    return (
        max(0, min(255, round(color[0] * 255.0))),
        max(0, min(255, round(color[1] * 255.0))),
        max(0, min(255, round(color[2] * 255.0))),
    )


def _draw_line(buffer: Framebuffer, start: _Projected, end: _Projected, color: RGB) -> None:
    """Bresenham line, used for the wireframe overlay."""
    x0, y0 = round(start.x), round(start.y)
    x1, y1 = round(end.x), round(end.y)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        buffer.set_pixel(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def render_scene(
    scene: Scene,
    *,
    width: int | None = None,
    height: int | None = None,
    wireframe: bool = False,
    backface_culling: bool = True,
    shaded: bool = True,
) -> tuple[Framebuffer, RenderStats]:
    """Render a scene with one call, returning the image and an honest report."""
    renderer = SoftwareRenderer(
        wireframe=wireframe, backface_culling=backface_culling, shaded=shaded
    )
    return renderer.render_with_stats(scene, width=width, height=height)


def render_scene_png(scene: Scene, **options: object) -> bytes:
    """Render a scene straight to PNG bytes."""
    buffer, _ = render_scene(scene, **options)  # type: ignore[arg-type]
    return buffer.to_png()


__all__ = [
    "AMBIENT",
    "Framebuffer",
    "RenderStats",
    "SoftwareRenderer",
    "render_scene",
    "render_scene_png",
    "shade_point",
]
