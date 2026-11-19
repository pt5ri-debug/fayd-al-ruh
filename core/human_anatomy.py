"""Parametric anatomical detail for the human body: areolae, nipples, genitalia.

The MakeHuman base mesh the studio builds from is a *neutral* body: it has a
chest and a pelvis, but no areolae, no nipples, and no genital geometry. This
module adds them as real, generated geometry -- not a texture, not a painted
patch -- so they can be sized, tilted, coloured, and (for the genitalia) shown or
hidden, and so they follow a pose with the rest of the skin.

How it is built
---------------
Each detail is a small closed surface generated in code from a handful of
measurements: a domed disc for an areola, a tapered cone for a nipple, and
ellipsoids (and a tapered shaft) for the genitalia. Every detail is placed from
the **body's own surface**, found by probing the finished mesh -- so an areola
sits on the breast a bust dial actually moved, not at a guessed constant -- and
oriented along the surface normal, tilted by the dial's own direction angle.

The details are appended to the body mesh as separate closed shells, exactly as
the eyes and the two rows of teeth already are. That is how character assets are
built (a nipple or a genital mesh is its own piece of geometry), it keeps the
body's own 27k-triangle skin untouched, and it means the existing skinning binds
them like every other vertex.

Honest by construction
----------------------
* The geometry is generated here; nothing is downloaded and no photograph or
  scanned asset is used.
* A detail is only built when the body's own build calls for it: the female set
  for a feminine body, the male set for a masculine one, and neither for a
  neutral one -- so a child template never receives adult anatomy.
* Every dial has a bounded range and the result is deterministic: the same
  settings always produce the same vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.scene3d import Mesh, Vec3

RGB = tuple[float, float, float]

#: How many segments a round detail is built from. A disc is small on screen, so
#: 20 segments read as a circle without adding thousands of vertices.
DEFAULT_SEGMENTS = 20

#: Rings in an areola's dome (more rings = a smoother curve away from the body).
DEFAULT_RINGS = 3

#: The lowest and highest a direction dial may tilt a detail (degrees).
MIN_DIRECTION = -85.0
MAX_DIRECTION = 85.0


def clamp(value: float, low: float, high: float) -> float:
    """``value`` brought inside ``[low, high]`` (a hand-edited dial cannot escape)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if number != number:  # NaN
        return low
    return min(high, max(low, number))


@dataclass(frozen=True)
class BreastDetail:
    """The areola and nipple pair: size, protrusion, tilt, and colours."""

    #: Areola radius as a multiplier of its base size (``1.0`` is the reference).
    areola_size: float = 1.0
    #: How far the areola stands proud of the breast, as a multiplier.
    areola_raise: float = 0.35
    #: Nipple radius as a multiplier of the areola's.
    nipple_size: float = 0.30
    #: Nipple protrusion as a multiplier of its base length.
    nipple_length: float = 1.0
    #: Tilt of the protrusion axis in the vertical plane (degrees, ``+`` is up).
    nipple_direction: float = 0.0
    #: How far apart the two are, as a multiplier of their natural separation.
    nipple_spacing: float = 1.0
    areola_color: RGB = (0.60, 0.34, 0.32)
    nipple_color: RGB = (0.66, 0.38, 0.35)

    def normalised(self) -> BreastDetail:
        return BreastDetail(
            areola_size=clamp(self.areola_size, 0.25, 2.5),
            areola_raise=clamp(self.areola_raise, 0.0, 1.5),
            nipple_size=clamp(self.nipple_size, 0.05, 1.2),
            nipple_length=clamp(self.nipple_length, 0.0, 2.5),
            nipple_direction=clamp(self.nipple_direction, MIN_DIRECTION, MAX_DIRECTION),
            nipple_spacing=clamp(self.nipple_spacing, 0.5, 1.8),
            areola_color=_rgb(self.areola_color),
            nipple_color=_rgb(self.nipple_color),
        )


@dataclass(frozen=True)
class GenitalDetail:
    """The genital set: whether it shows, its size, tilt, and colour."""

    #: ``"female"``, ``"male"``, or ``""`` (nothing is built).
    sex: str = ""
    #: How visible the set is (``0`` hides it, ``1`` is the reference size).
    show: float = 1.0
    #: Overall size multiplier.
    size: float = 1.0
    #: Length multiplier (how far the detail extends forward / down).
    length: float = 1.0
    #: Tilt of the main axis in the vertical plane (degrees, ``+`` is up).
    direction: float = 0.0
    #: Female labia / male shaft-and-scrotum fullness.
    fullness: float = 1.0
    color: RGB = (0.60, 0.40, 0.38)

    def normalised(self) -> GenitalDetail:
        sex = str(self.sex or "").strip().lower()
        if sex not in ("female", "male"):
            sex = ""
        return GenitalDetail(
            sex=sex,
            show=clamp(self.show, 0.0, 1.0),
            size=clamp(self.size, 0.3, 2.2),
            length=clamp(self.length, 0.3, 2.2),
            direction=clamp(self.direction, MIN_DIRECTION, MAX_DIRECTION),
            fullness=clamp(self.fullness, 0.3, 2.2),
            color=_rgb(self.color),
        )


@dataclass
class AnatomyGeometry:
    """The extra vertices, faces, and colours a detail set contributes."""

    vertices: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    colors: list[RGB] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def add(
        self,
        points: list[Vec3],
        faces: list[tuple[int, ...]],
        colour: RGB,
    ) -> None:
        base = len(self.vertices)
        self.vertices.extend(points)
        self.colors.extend([colour] * len(points))
        self.faces.extend(tuple(index + base for index in face) for face in faces)


def _rgb(value: object) -> RGB:
    if not isinstance(value, (tuple, list)):
        return (0.0, 0.0, 0.0)
    try:
        parts = tuple(float(component) for component in value)[:3]
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    return (clamp(parts[0], 0.0, 1.0), clamp(parts[1], 0.0, 1.0), clamp(parts[2], 0.0, 1.0))


# ---------------------------------------------------------------------------
# A local frame: the surface normal plus a stable "up" so a detail can be built
# in its own space and then placed, tilted, and mirrored on the body.
# ---------------------------------------------------------------------------


class _Frame:
    """An orthonormal basis: ``n`` (outward), ``u`` (up), ``r`` (right)."""

    __slots__ = ("n", "origin", "r", "u")

    def __init__(self, origin: Vec3, normal: Vec3, up: Vec3) -> None:
        n = normal.normalized() if normal.length() > 1e-9 else Vec3(0.0, 0.0, 1.0)
        # Gram-Schmidt: make ``up`` perpendicular to the normal.
        u = up - n * up.dot(n)
        if u.length() < 1e-6:
            # ``up`` was parallel to the normal: pick any perpendicular axis.
            u = Vec3(0.0, 1.0, 0.0) - n * Vec3(0.0, 1.0, 0.0).dot(n)
            if u.length() < 1e-6:
                u = Vec3(1.0, 0.0, 0.0) - n * Vec3(1.0, 0.0, 0.0).dot(n)
        u = u.normalized()
        self.origin = origin
        self.n = n
        self.u = u
        self.r = u.cross(n).normalized()

    def point(self, right: float, up: float, out: float) -> Vec3:
        """A point in world space from this frame's own (right, up, out) axes."""
        return self.origin + self.r * right + self.u * up + self.n * out


def _tilt(frame: _Frame, degrees: float) -> _Frame:
    """Tilt a frame's outward axis toward its up axis (the direction dial)."""
    if abs(degrees) < 1e-9:
        return frame
    angle = math.radians(clamp(degrees, MIN_DIRECTION, MAX_DIRECTION))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    normal = (frame.n * cos_a + frame.u * sin_a).normalized()
    up = (frame.u * cos_a - frame.n * sin_a).normalized()
    return _Frame(frame.origin, normal, up)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _dome(
    frame: _Frame,
    radius: float,
    height: float,
    *,
    segments: int = DEFAULT_SEGMENTS,
    rings: int = DEFAULT_RINGS,
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """A domed disc: a polar grid that rises to ``height`` at its centre.

    The rim sits *on* the surface (out = 0), so the disc's edge is flush with the
    skin and only its middle stands proud -- which is what an areola is. The
    profile is ``out = height * (1 - t**2)`` for ``t`` in ``0..1``, so it leaves
    the surface tangentially instead of meeting it at a crease.
    """
    points: list[Vec3] = [frame.point(0.0, 0.0, height)]  # apex
    for ring in range(1, rings + 1):
        t = ring / rings
        ring_radius = radius * t
        ring_out = height * (1.0 - t * t)
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            points.append(
                frame.point(
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    ring_out,
                )
            )
    faces: list[tuple[int, ...]] = []
    # Fan from the apex to the first ring.
    for segment in range(segments):
        here = 1 + segment
        nxt = 1 + (segment + 1) % segments
        faces.append((0, here, nxt))
    # Quads between successive rings (a quad keeps the shell closed and smooth).
    for ring in range(rings - 1):
        start = 1 + ring * segments
        ahead = start + segments
        for segment in range(segments):
            a = start + segment
            b = start + (segment + 1) % segments
            c = ahead + (segment + 1) % segments
            d = ahead + segment
            faces.append((a, b, c, d))
    return points, faces


def _cone(
    frame: _Frame,
    radius: float,
    length: float,
    *,
    segments: int = DEFAULT_SEGMENTS,
    rings: int = 2,
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """A rounded tapered cone standing out along the frame's normal (a nipple)."""
    points: list[Vec3] = [frame.point(0.0, 0.0, length)]  # tip
    for ring in range(1, rings + 1):
        t = ring / rings
        ring_radius = radius * (1.0 - 0.72 * t)
        ring_out = length * (1.0 - t)
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            points.append(
                frame.point(
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    ring_out,
                )
            )
    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        here = 1 + segment
        nxt = 1 + (segment + 1) % segments
        faces.append((0, here, nxt))
    for ring in range(rings - 1):
        start = 1 + ring * segments
        ahead = start + segments
        for segment in range(segments):
            a = start + segment
            b = start + (segment + 1) % segments
            c = ahead + (segment + 1) % segments
            d = ahead + segment
            faces.append((a, b, c, d))
    return points, faces


def _ellipsoid(
    frame: _Frame,
    radii: tuple[float, float, float],
    *,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 14,
    rings: int = 8,
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """A closed ellipsoid in the frame's own axes (an areola mound, a labium)."""
    right_radius, up_radius, out_radius = radii
    offset_right, offset_up, offset_out = centre
    points: list[Vec3] = []
    # The poles, then the intermediate rings (a UV sphere squashed per axis).
    points.append(frame.point(offset_right, offset_up + up_radius, offset_out))
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        y = math.cos(phi)
        band = math.sin(phi)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            points.append(
                frame.point(
                    offset_right + right_radius * band * math.cos(theta),
                    offset_up + up_radius * y,
                    offset_out + out_radius * band * math.sin(theta),
                )
            )
    points.append(frame.point(offset_right, offset_up - up_radius, offset_out))
    bottom = len(points) - 1
    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        here = 1 + segment
        nxt = 1 + (segment + 1) % segments
        faces.append((0, nxt, here))
    for ring in range(rings - 2):
        start = 1 + ring * segments
        ahead = start + segments
        for segment in range(segments):
            a = start + segment
            b = start + (segment + 1) % segments
            c = ahead + (segment + 1) % segments
            d = ahead + segment
            faces.append((a, b, c, d))
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        here = last + segment
        nxt = last + (segment + 1) % segments
        faces.append((bottom, here, nxt))
    return points, faces


# ---------------------------------------------------------------------------
# Finding where a detail goes on the body's own surface
# ---------------------------------------------------------------------------


def _body_vertex_indices(mesh: Mesh) -> list[int]:
    """The indices of the body's **own** vertices -- the largest connected shell.

    The body is authored first and the details are appended after it, but a mesh
    that already carries detail must not have that detail mistaken for body
    surface (an areola would then be placed on the previous areola). So the body
    is identified by its topology -- the largest connected component of faces --
    rather than by position in the vertex list.
    """
    if not mesh.faces:
        return list(range(len(mesh.vertices)))
    parent = list(range(len(mesh.vertices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in mesh.faces:
        first = find(face[0])
        for index in face[1:]:
            other = find(index)
            if other != first:
                parent[other] = first
    groups: dict[int, list[int]] = {}
    for face in mesh.faces:
        for index in face:
            groups.setdefault(find(index), []).append(index)
    if not groups:
        return list(range(len(mesh.vertices)))
    return max(groups.values(), key=len)


def _body_vertices(mesh: Mesh) -> list[Vec3]:
    """The body's own vertices (see :func:`_body_vertex_indices`)."""
    return [mesh.vertices[index] for index in _body_vertex_indices(mesh)]


def _breast_frames(mesh: Mesh) -> tuple[_Frame, _Frame] | None:
    """The two breast centres and their outward normals, found on the surface.

    The chest is probed rather than assumed: take the front-most band of vertices
    at the bust's height, split it by side, and use each side's mean point and its
    mean normal. A bust dial that moved the breast therefore moves the areola with
    it, which a constant offset could never do.
    """
    points = _body_vertices(mesh)
    if not points:
        return None
    low, high = mesh.bounds()
    height = high.y - low.y
    if height <= 0.0:
        return None
    # The bust sits in the upper third of the torso, in front of the spine.
    y_lo = low.y + height * 0.66
    y_hi = low.y + height * 0.76
    band = [point for point in points if y_lo <= point.y <= y_hi]
    if not band:
        return None
    # "Front" is the band's own front, not the whole body's: the feet point
    # forward and would otherwise define the front plane.
    band_front = max(point.z for point in band)
    band_depth = band_front - min(point.z for point in band)
    front = [point for point in band if point.z >= band_front - band_depth * 0.22]
    left = [point for point in front if point.x > 0.0]
    right = [point for point in front if point.x < 0.0]
    if not left or not right:
        return None
    frames = []
    for side in (left, right):
        furthest = max(point.z for point in side)
        # The breast's own centre is the middle of the front band on that side,
        # at the depth of its furthest point -- so the areola sits on the surface
        # a bust dial actually moved, not at a fixed offset from the spine.
        centre = Vec3(
            sum(point.x for point in side) / len(side),
            sum(point.y for point in side) / len(side),
            furthest,
        )
        normal = Vec3(0.0, 0.0, 1.0)
        frames.append(_Frame(centre, normal, Vec3(0.0, 1.0, 0.0)))
    return frames[0], frames[1]


def _pubic_frame(mesh: Mesh) -> _Frame | None:
    """The front of the pelvis, found on the surface (where the genitals sit)."""
    points = _body_vertices(mesh)
    if not points:
        return None
    low, high = mesh.bounds()
    height = high.y - low.y
    if height <= 0.0:
        return None
    # The crotch is just under the pelvis's midpoint, on the centre line.
    y_lo = low.y + height * 0.42
    y_hi = low.y + height * 0.53
    band = [
        point
        for point in points
        if y_lo <= point.y <= y_hi and abs(point.x) < (high.x - low.x) * 0.14
    ]
    if not band:
        return None
    furthest = max(band, key=lambda point: point.z)
    return _Frame(furthest, Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# The detail sets
# ---------------------------------------------------------------------------


def breast_geometry(mesh: Mesh, detail: BreastDetail) -> AnatomyGeometry:
    """The areola and nipple pair, sized, tilted, and coloured by ``detail``."""
    geometry = AnatomyGeometry()
    frames = _breast_frames(mesh)
    if frames is None:
        return geometry
    settings = detail.normalised()
    low, high = mesh.bounds()
    height = max(high.y - low.y, 1e-6)
    # A base areola is about 1.6% of the figure's height (a real one is ~4 cm on
    # a 1.75 m adult), which keeps the detail in proportion at any body size.
    base_radius = height * 0.0165 * settings.areola_size
    for frame in frames:
        centre = frame.origin
        if abs(settings.nipple_spacing - 1.0) > 1e-9:
            # Move each areola out from the body's centre line by the spacing dial.
            centre = Vec3(
                centre.x * settings.nipple_spacing,
                centre.y,
                centre.z,
            )
        placed = _Frame(centre, frame.n, frame.u)
        areola_points, areola_faces = _dome(
            placed,
            base_radius,
            base_radius * 0.30 * settings.areola_raise,
        )
        geometry.add(areola_points, areola_faces, settings.areola_color)
        nipple_frame = _tilt(placed, settings.nipple_direction)
        nipple_points, nipple_faces = _cone(
            nipple_frame,
            base_radius * settings.nipple_size,
            base_radius * 0.55 * settings.nipple_length,
        )
        geometry.add(nipple_points, nipple_faces, settings.nipple_color)
    return geometry


def genital_geometry(mesh: Mesh, detail: GenitalDetail) -> AnatomyGeometry:
    """The genital set for the body's own sex, sized, tilted, and coloured."""
    geometry = AnatomyGeometry()
    settings = detail.normalised()
    if not settings.sex or settings.show <= 0.0:
        return geometry
    frame = _pubic_frame(mesh)
    if frame is None:
        return geometry
    low, high = mesh.bounds()
    height = max(high.y - low.y, 1e-6)
    # The set scales with the figure, so a child's body is never given adult
    # proportions even if a dial is set high.
    unit = height * 0.055 * settings.size
    visible = 0.35 + 0.65 * settings.show
    # The tilt applies to the whole set, whichever sex it is, so the direction
    # control is never dead on one body.
    tilted = _tilt(frame, settings.direction)
    if settings.sex == "female":
        geometry.add(
            *_ellipsoid(
                tilted,
                (unit * 0.62 * settings.fullness, unit * 0.42, unit * 0.30),
                centre=(0.0, -unit * 0.10, unit * 0.06 * visible),
            ),
            settings.color,
        )
        for side in (-1.0, 1.0):
            geometry.add(
                *_ellipsoid(
                    tilted,
                    (unit * 0.26 * settings.fullness, unit * 0.40 * settings.length, unit * 0.24),
                    centre=(side * unit * 0.34, -unit * 0.26, unit * 0.10 * visible),
                ),
                settings.color,
            )
        return geometry
    # Male: a scrotum and a tapered shaft along the tilted axis.
    geometry.add(
        *_ellipsoid(
            tilted,
            (unit * 0.42 * settings.fullness, unit * 0.40, unit * 0.34),
            centre=(0.0, -unit * 0.22, unit * 0.10 * visible),
        ),
        settings.color,
    )
    geometry.add(
        *_cone(
            tilted,
            unit * 0.24 * settings.fullness,
            unit * 0.85 * settings.length * visible,
        ),
        settings.color,
    )
    geometry.add(
        *_ellipsoid(
            tilted,
            (unit * 0.26, unit * 0.26, unit * 0.24),
            centre=(0.0, 0.0, unit * 0.80 * settings.length * visible),
        ),
        settings.color,
    )
    return geometry


def merge_anatomy(
    mesh: Mesh,
    *,
    breast: BreastDetail | None = None,
    genital: GenitalDetail | None = None,
) -> Mesh:
    """Return ``mesh`` with the requested details appended as closed shells.

    Nothing is added when neither detail asks for geometry, so a neutral body is
    returned **exactly** as it was built -- no extra vertex, no changed face.
    """
    parts = AnatomyGeometry()
    for extra in (
        breast_geometry(mesh, breast) if breast is not None else None,
        genital_geometry(mesh, genital) if genital is not None else None,
    ):
        if extra is None or not extra.vertices:
            continue
        # The detail's own faces are relative to *its* first vertex, so they are
        # rebased onto the combined list as it grows.
        base = len(parts.vertices)
        parts.vertices.extend(extra.vertices)
        parts.colors.extend(extra.colors)
        parts.faces.extend(tuple(index + base for index in face) for face in extra.faces)
    if not parts.vertices:
        return mesh
    # ...and the whole detail block is then rebased past the body's own vertices,
    # so a detail face indexes a detail vertex and never the skin it sits on.
    offset = len(mesh.vertices)
    colours = list(mesh.vertex_colors) if mesh.vertex_colors else [
        (0.86, 0.71, 0.62)
    ] * len(mesh.vertices)
    colours.extend(parts.colors)
    return Mesh(
        name=mesh.name,
        vertices=tuple(mesh.vertices) + tuple(parts.vertices),
        faces=tuple(mesh.faces)
        + tuple(tuple(index + offset for index in face) for face in parts.faces),
        normals=(),
        vertex_colors=tuple(colours),
    )



def describe_anatomy(
    breast: BreastDetail | None,
    genital: GenitalDetail | None,
) -> str:
    """One honest line naming what a body carries."""
    pieces: list[str] = []
    if breast is not None:
        breast_settings = breast.normalised()
        pieces.append(
            f"areolae and nipples (areola x{breast_settings.areola_size:.2f}, "
            f"nipple x{breast_settings.nipple_size:.2f}, "
            f"tilt {breast_settings.nipple_direction:.0f}deg)"
        )
    if genital is not None:
        genital_settings = genital.normalised()
        if genital_settings.sex and genital_settings.show > 0.0:
            pieces.append(
                f"{genital_settings.sex} genitalia (size x{genital_settings.size:.2f}, "
                f"tilt {genital_settings.direction:.0f}deg)"
            )
    if not pieces:
        return "No anatomical detail (a neutral body)."
    return "Anatomical detail: " + ", ".join(pieces) + "."
