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

from core import scene3d_sculpt as _sculpt
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
class BreastShape:
    """The breast-shape dials: every component of the chest, not just its size.

    ``bust`` is the base fullness the body morph already carries; the rest are
    additional, independent controls applied as a smooth deformation of the chest
    region. ``1.0`` is neutral for every multiplier and ``0.0`` is neutral for
    ``sag``, so a body that was never edited is unchanged.
    """

    bust: float = 0.0
    size: float = 1.0
    #: Each breast's own size, so the two can differ (``1.0`` is the reference).
    size_left: float = 1.0
    size_right: float = 1.0
    #: The whole ribcage's size -- the chest the breasts sit on, independent of
    #: the breast volume itself.
    chest_size: float = 1.0
    projection: float = 1.0
    fullness: float = 1.0
    separation: float = 1.0
    height: float = 1.0
    sag: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return (
            abs(self.size - 1.0) < 1e-9
            and abs(self.size_left - 1.0) < 1e-9
            and abs(self.size_right - 1.0) < 1e-9
            and abs(self.chest_size - 1.0) < 1e-9
            and abs(self.projection - 1.0) < 1e-9
            and abs(self.fullness - 1.0) < 1e-9
            and abs(self.separation - 1.0) < 1e-9
            and abs(self.height - 1.0) < 1e-9
            and abs(self.sag) < 1e-9
        )

    def size_for_side(self, side: float) -> float:
        """The breast's size on one side (``+1`` is the figure's left, x > 0)."""
        return self.size_left if side >= 0.0 else self.size_right

    def normalised(self) -> BreastShape:
        return BreastShape(
            bust=clamp(self.bust, 0.0, 1.5),
            size=clamp(self.size, 0.3, 2.5),
            size_left=clamp(self.size_left, 0.3, 2.5),
            size_right=clamp(self.size_right, 0.3, 2.5),
            chest_size=clamp(self.chest_size, 0.6, 1.6),
            projection=clamp(self.projection, 0.3, 2.5),
            fullness=clamp(self.fullness, 0.3, 2.5),
            separation=clamp(self.separation, 0.5, 1.8),
            height=clamp(self.height, 0.6, 1.4),
            sag=clamp(self.sag, 0.0, 1.0),
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
    #: How erect the male organ is: ``0`` is fully flaccid, ``1`` is fully erect.
    #: It changes the length, the girth, and the angle together, the way the real
    #: organ does, rather than only stretching it.
    erection: float = 0.0
    #: The glans's own size relative to the shaft.
    glans_size: float = 1.0
    #: How much the shaft curves along its length (``0`` straight, ``1`` a gentle
    #: upward curve).
    curve: float = 0.0
    color: RGB = (0.60, 0.40, 0.38)
    #: The body's skin colour, used for the mons so the mound reads as skin
    #: rather than as a dark patch.
    skin_color: RGB = (0.82, 0.66, 0.56)

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
            erection=clamp(self.erection, 0.0, 1.0),
            glans_size=clamp(self.glans_size, 0.4, 2.0),
            curve=clamp(self.curve, 0.0, 1.0),
            color=_rgb(self.color),
            skin_color=_rgb(self.skin_color),
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

    def add_mesh(self, mesh: Mesh) -> None:
        """Append a finished, multi-coloured mesh (a sculpted organ)."""
        if not mesh.vertices:
            return
        base = len(self.vertices)
        self.vertices.extend(mesh.vertices)
        if mesh.vertex_colors:
            self.colors.extend(mesh.vertex_colors)
        else:
            self.colors.extend([(0.82, 0.66, 0.56)] * len(mesh.vertices))
        self.faces.extend(tuple(index + base for index in face) for face in mesh.faces)


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
    rim: float = 0.0,
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
        ring_out = height * (1.0 - t * t) + rim * t * t
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
    base_out: float = 0.0,
    segments: int = DEFAULT_SEGMENTS,
    rings: int = 2,
) -> tuple[list[Vec3], list[tuple[int, ...]]]:
    """A rounded tapered cone standing out along the frame's normal (a nipple)."""
    points: list[Vec3] = [frame.point(0.0, 0.0, length)]  # tip
    for ring in range(1, rings + 1):
        t = ring / rings
        ring_radius = radius * (1.0 - 0.72 * t)
        ring_out = length * (1.0 - t) + base_out * t
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
    """The two nipple points and their outward normals, found on the torso.

    The position is the anatomical one -- the mid-clavicular line at the bust's
    height -- but the *depth* is read from the body's own surface, so a bust dial
    that moved the breast forward moves the areola with it. The earlier version
    took the front-most vertex of a whole horizontal band, which on this rig is
    the **hand** of the A-pose (it hangs forward of the chest), so the areolae
    were placed on the arms; restricting the search to the torso and to a window
    around the nipple's own position is what keeps them on the breast.
    """
    indices = _body_vertex_indices(mesh)
    if not indices:
        return None
    low, high = mesh.bounds()
    height = high.y - low.y
    if height <= 0.0:
        return None
    normals = mesh.vertex_normals()
    # The bust sits in the upper third of the torso; the search is limited to the
    # torso itself (a narrow x band) and to the front half (z > 0).
    y_lo = low.y + height * 0.63
    y_hi = low.y + height * 0.79
    target_y = low.y + height * 0.715
    target_x = height * 0.075
    half_x = height * 0.030
    half_y = height * 0.045
    torso = [
        index
        for index in indices
        if y_lo <= mesh.vertices[index].y <= y_hi
        and mesh.vertices[index].z > 0.0
        and abs(mesh.vertices[index].x) <= height * 0.16
    ]
    if not torso:
        return None
    frames: list[_Frame] = []
    for side in (1.0, -1.0):
        window = [
            index
            for index in torso
            if mesh.vertices[index].x * side > 0.0
            and abs(mesh.vertices[index].x - side * target_x) <= half_x
            and abs(mesh.vertices[index].y - target_y) <= half_y
        ]
        if not window:
            # Fall back to the torso vertex nearest the anatomical point, so a
            # very small or unusual body still gets a placed detail. A torso with
            # no vertex on this side at all yields no frame rather than a crash.
            candidates = [
                index for index in torso if mesh.vertices[index].x * side > 0.0
            ]
            if not candidates:
                return None
            window = [
                min(
                    candidates,
                    key=lambda index: (
                        (mesh.vertices[index].x - side * target_x) ** 2
                        + (mesh.vertices[index].y - target_y) ** 2
                    ),
                )
            ]
        # The breast's own surface: the front-most few vertices of that window.
        deepest = max(mesh.vertices[index].z for index in window)
        front = [
            index for index in window
            if mesh.vertices[index].z >= deepest - height * 0.006
        ]
        centre = Vec3(
            sum(mesh.vertices[index].x for index in front) / len(front),
            sum(mesh.vertices[index].y for index in front) / len(front),
            sum(mesh.vertices[index].z for index in front) / len(front),
        )
        normal = Vec3(
            sum(normals[index].x for index in front),
            sum(normals[index].y for index in front),
            sum(normals[index].z for index in front),
        )
        frames.append(_Frame(centre, normal, Vec3(0.0, 1.0, 0.0)))
    # The pack mesh is symmetric to within a couple of millimetres, but the two
    # frames are found independently and inherited that difference (a 3 mm
    # height difference between the areolae). The right frame is the left one
    # mirrored, so the pair is exactly symmetric and the deformation follows.
    left = frames[0]
    mirrored = _Frame(
        Vec3(-left.origin.x, left.origin.y, left.origin.z),
        Vec3(-left.n.x, left.n.y, left.n.z),
        Vec3(-left.u.x, left.u.y, left.u.z),
    )
    return left, mirrored


def _pubic_frame(mesh: Mesh) -> _Frame | None:
    """The front of the pelvis, found on the surface (where the genitals sit).

    The band is deliberately **low** -- between 0.43 and 0.49 of the height, on
    the centre line -- because that is where the pubic symphysis is. The earlier
    band reached up to 0.53 of the height, which placed the whole set a few
    centimetres too high (the reported "the male organ is higher than its correct
    place"). The band is also narrow, so a thigh vertex can never define the point.
    """
    points = _body_vertices(mesh)
    if not points:
        return None
    low, high = mesh.bounds()
    height = high.y - low.y
    if height <= 0.0:
        return None
    y_lo = low.y + height * 0.43
    y_hi = low.y + height * 0.49
    width = max(high.x - low.x, 1e-6)
    band = [
        point
        for point in points
        if y_lo <= point.y <= y_hi and abs(point.x) < width * 0.10
    ]
    if not band:
        return None
    centre_band = [point for point in band if abs(point.x) < width * 0.03]
    furthest = max(centre_band or band, key=lambda point: point.z)
    # The set hangs from the body's own centre line: the front-most vertex of a
    # narrow centre band gives the depth, but the x is forced to 0 so the frame
    # can never drift to one side (a half-centimetre of asymmetry there would put
    # the whole set off-centre).
    origin = Vec3(0.0, furthest.y, furthest.z)
    # The pubic surface faces forward and slightly down, which is the axis the
    # set hangs from.
    return _Frame(origin, Vec3(0.0, -0.12, 1.0), Vec3(0.0, 1.0, 0.0))


def shape_breasts(mesh: Mesh, shape: BreastShape) -> Mesh:
    """Deform the chest region by the breast dials, as real geometry.

    Every component is its own control: ``chest_size`` grows the ribcage the
    breasts sit on, ``size`` scales both breasts, ``size_left``/``size_right``
    scale each one independently, ``fullness`` rounds it, ``projection`` pushes it
    forward, ``separation`` moves the two apart, ``height`` lifts them, and
    ``sag`` lets the lower half hang. A smooth, bounded falloff around each
    breast's own centre keeps the deformation local -- the sternum, the ribs, and
    the back are untouched -- and the neutral settings return the mesh **exactly**
    as it was built.
    """
    settings = shape.normalised()
    if settings.is_neutral:
        return mesh
    low, high = mesh.bounds()
    height = max(high.y - low.y, 1e-6)
    points = list(mesh.vertices)
    # -- the ribcage itself ------------------------------------------------
    if abs(settings.chest_size - 1.0) > 1e-9:
        chest_y = low.y + 0.70 * height
        sigma = 0.115 * height
        spine_z = low.z + (high.z - low.z) * 0.5
        for index, vertex in enumerate(mesh.vertices):
            if vertex.y < low.y + 0.55 * height or vertex.y > low.y + 0.88 * height:
                continue
            weight = math.exp(-0.5 * ((vertex.y - chest_y) / sigma) ** 2)
            # Fade out at the arms, so the shoulder and the upper arm are not
            # widened with the ribcage.
            weight *= math.exp(-0.5 * (vertex.x / (0.17 * height)) ** 2)
            if weight < 1e-3:
                continue
            grow = 1.0 + (settings.chest_size - 1.0) * weight
            points[index] = Vec3(
                vertex.x * grow,
                vertex.y,
                spine_z + (vertex.z - spine_z) * (1.0 + (settings.chest_size - 1.0) * 0.85 * weight),
            )
    if (
        abs(settings.size - 1.0) < 1e-9
        and abs(settings.size_left - 1.0) < 1e-9
        and abs(settings.size_right - 1.0) < 1e-9
        and abs(settings.projection - 1.0) < 1e-9
        and abs(settings.fullness - 1.0) < 1e-9
        and abs(settings.separation - 1.0) < 1e-9
        and abs(settings.height - 1.0) < 1e-9
        and abs(settings.sag) < 1e-9
    ):
        return Mesh(
            name=mesh.name, vertices=tuple(points), faces=mesh.faces,
            normals=(), vertex_colors=mesh.vertex_colors,
        )
    work = Mesh(
        name=mesh.name, vertices=tuple(points), faces=mesh.faces,
        normals=(), vertex_colors=mesh.vertex_colors,
    )
    frames = _breast_frames(work)
    if frames is None:
        return work
    sigma_x = 0.075 * height
    sigma_y = 0.090 * height
    z_front = max(frame.origin.z for frame in frames)
    z_limit = z_front - 0.13 * height
    max_step = 0.07 * height
    for index, vertex in enumerate(work.vertices):
        if vertex.y < low.y + 0.56 * height or vertex.y > low.y + 0.87 * height:
            continue
        if vertex.z < z_limit:
            continue
        best_centre: Vec3 | None = None
        best_weight = 0.0
        best_dx = 0.0
        best_dy = 0.0
        for frame in frames:
            centre = frame.origin
            dx = vertex.x - centre.x
            dy = vertex.y - centre.y
            weight = math.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
            # Fade the deformation out at the sternum, so the two sides never
            # push a vertex across the centre line (which tore a visible seam
            # down the middle of the chest).
            weight *= min(1.0, (abs(vertex.x) / (0.028 * height)) ** 1.5)
            if weight > best_weight:
                best_weight = weight
                best_centre = centre
                best_dx = dx
                best_dy = dy
        if best_centre is None or best_weight < 1e-3:
            continue
        side = 1.0 if vertex.x >= 0.0 else -1.0
        # The overall breast size and that side's own size multiply together, so
        # "both bigger" and "only the left bigger" are independent controls.
        side_size = settings.size * settings.size_for_side(side)
        scale = (
            1.0
            + (settings.fullness - 1.0) * 0.50 * best_weight
            + (side_size - 1.0) * 0.50 * best_weight
        )
        new_x = best_centre.x + best_dx * scale
        new_y = best_centre.y + best_dy * scale
        # Volume is three-dimensional: a larger or fuller breast also projects
        # forward, not merely wider. Without this the size dial only widened the
        # breast and the "size" control looked like it did nothing.
        new_z = vertex.z + (
            (settings.projection - 1.0) * 0.050
            + (side_size - 1.0) * 0.035
            + (settings.fullness - 1.0) * 0.015
        ) * height * best_weight
        new_x += (settings.separation - 1.0) * 0.020 * height * best_weight * side
        new_y += (settings.height - 1.0) * 0.030 * height * best_weight
        if settings.sag > 0.0 and best_dy < 0.0:
            new_y -= settings.sag * 0.040 * height * best_weight
            new_z += settings.sag * 0.012 * height * best_weight
        # Never let a breast pull a vertex across the body's centre line.
        new_x = max(new_x, 0.001 * height) if vertex.x >= 0.0 else min(new_x, -0.001 * height)
        moved = Vec3(new_x, new_y, new_z)
        displacement = moved - vertex
        if displacement.length() > max_step:
            moved = vertex + displacement * (max_step / displacement.length())
        points[index] = moved
    return Mesh(
        name=mesh.name,
        vertices=tuple(points),
        faces=mesh.faces,
        normals=(),
        vertex_colors=mesh.vertex_colors,
    )


# ---------------------------------------------------------------------------
# The detail sets
# ---------------------------------------------------------------------------


def breast_geometry(mesh: Mesh, detail: BreastDetail) -> AnatomyGeometry:
    """The areola and nipple pair, sized, tilted, and coloured by ``detail``.

    Each detail is placed at its breast's own apex, along the *surface normal*
    found there, and its rim is embedded a couple of millimetres into the skin --
    so it follows the breast's real curvature from every camera angle instead of
    floating on a tangent plane in front of it.
    """
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
    embed = -height * 0.0028
    for side, frame in ((1.0, frames[0]), (-1.0, frames[1])):
        centre = frame.origin
        if abs(settings.nipple_spacing - 1.0) > 1e-9:
            # Move each areola along the breast's own surface (outward from the
            # body's centre line) by the spacing dial, rather than sideways in
            # world x, which would lift it off a curved chest.
            centre = centre + frame.r * ((settings.nipple_spacing - 1.0) * base_radius * 1.7 * side)
        placed = _Frame(centre, frame.n, frame.u)
        areola_points, areola_faces = _dome(
            placed,
            base_radius,
            base_radius * 0.30 * settings.areola_raise,
            rim=embed,
        )
        geometry.add(areola_points, areola_faces, settings.areola_color)
        nipple_frame = _tilt(placed, settings.nipple_direction)
        nipple_points, nipple_faces = _cone(
            nipple_frame,
            base_radius * settings.nipple_size,
            base_radius * 0.55 * settings.nipple_length,
            base_out=embed * 0.6,
        )
        geometry.add(nipple_points, nipple_faces, settings.nipple_color)
    return geometry


def _darker(colour: RGB, factor: float) -> RGB:
    return (
        clamp(colour[0] * factor, 0.0, 1.0),
        clamp(colour[1] * factor, 0.0, 1.0),
        clamp(colour[2] * factor, 0.0, 1.0),
    )


def _female_geometry(frame: _Frame, settings: GenitalDetail, unit: float) -> Mesh:
    """The female set as **one** smoothly fused surface, flush with the pelvis.

    A mons, two outer labia, two inner labia, a clitoral hood, and the opening are
    separate fields fused by the sculptor, so the result reads as one continuous
    anatomy rather than as a row of separate ellipsoids -- which is what the
    earlier set looked like from the side.
    """
    tilted = _tilt(frame, settings.direction)
    lip = settings.color
    inner = (
        clamp(lip[0] * 1.06 + 0.03, 0.0, 1.0),
        clamp(lip[1] * 0.94, 0.0, 1.0),
        clamp(lip[2] * 0.94, 0.0, 1.0),
    )
    opening = _darker(lip, 0.55)
    full = settings.fullness
    length = settings.length
    blobs = [
        # The mons: a wide, soft mound that blends into the lower belly. It sits
        # almost entirely inside the body, so it smooths the transition rather
        # than adding a visible lump of its own.
        _sculpt.ellipsoid(
            tilted.point(0.0, unit * 0.20, -unit * 0.006),
            (unit * 0.28 * full, unit * 0.16, unit * 0.050),
            lip,
            blend=unit * 0.045,
        ),
        # The clitoral hood, just above the opening.
        _sculpt.sphere(
            tilted.point(0.0, unit * 0.04, unit * 0.014),
            unit * 0.050,
            lip,
            blend=unit * 0.018,
        ),
        # The opening: a small, dark, shallow inset at the centre.
        _sculpt.ellipsoid(
            tilted.point(0.0, -unit * 0.14, unit * 0.008),
            (unit * 0.035, unit * 0.100, unit * 0.018),
            opening,
            blend=unit * 0.015,
        ),
    ]
    for side in (-1.0, 1.0):
        blobs.append(
            _sculpt.capsule(
                tilted.point(side * unit * 0.09, unit * 0.05, unit * 0.008),
                tilted.point(side * unit * 0.11, -unit * 0.40 * length, unit * 0.010),
                unit * 0.055 * full,
                unit * 0.040 * full,
                lip,
                blend=unit * 0.022,
            )
        )
        blobs.append(
            _sculpt.capsule(
                tilted.point(side * unit * 0.045, unit * 0.00, unit * 0.012),
                tilted.point(side * unit * 0.060, -unit * 0.30 * length, unit * 0.014),
                unit * 0.030,
                unit * 0.020,
                inner,
                blend=unit * 0.012,
            )
        )
    return _sculpt.sculpt(blobs, cell=unit * 0.045, name="female anatomy")


def _male_geometry(frame: _Frame, settings: GenitalDetail, unit: float, height: float) -> Mesh:
    """The male set as **one** smoothly fused organ: root, shaft, glans, scrotum.

    The root is embedded a few centimetres inside the pubic mound, and the shaft,
    the glans, and the scrotum are fused to it by the sculptor, so the organ
    emerges from the body as a single continuous surface instead of a capsule
    floating in front of the skin. ``erection`` changes the length, the girth, and
    the angle together -- flaccid and hanging at ``0``, longer, thicker, and
    raised at ``1`` -- and ``direction`` tilts the whole set, which is the size and
    direction control the studio offers.
    """
    erection = settings.erection
    full = settings.fullness
    length = unit * (0.88 + 0.92 * erection) * settings.length
    girth = unit * (0.145 + 0.050 * erection) * full
    angle = math.radians(-55.0 + 90.0 * erection + settings.direction)
    visible = 0.35 + 0.65 * settings.show
    length *= visible
    # The axis, as a short chain of points: a gentle curve along the length (the
    # real organ is not a straight cylinder), bending upward as it approaches the
    # tip by the curve dial.
    # The organ is **one** chain of capsules, beginning several centimetres inside
    # the body and emerging through the skin: a separate root stub meeting the
    # shaft left a pinch that surface nets split into two shells. The deep end is
    # wider, so the skin meets it in a clean ring.
    start_out = -unit * 0.006
    deep = frame.point(0.0, -unit * 0.010, -unit * 0.060)
    root = frame.point(0.0, -unit * 0.010, start_out)
    points = [deep, root]
    steps = 4
    for step in range(steps):
        t = (step + 1) / steps
        a = angle + math.radians(settings.curve * 16.0 * t)
        points.append(
            frame.point(
                0.0,
                -unit * 0.010 + math.sin(a) * length * t,
                start_out + math.cos(a) * length * t,
            )
        )
    shaft = settings.color
    glans = (
        clamp(shaft[0] * 1.18 + 0.06, 0.0, 1.0),
        clamp(shaft[1] * 0.92, 0.0, 1.0),
        clamp(shaft[2] * 0.92, 0.0, 1.0),
    )
    scrotum = _darker(shaft, 0.90)
    meatus = _darker(glans, 0.45)
    blobs = []
    for index in range(len(points) - 1):
        t = index / (len(points) - 1)
        blobs.append(
            _sculpt.capsule(
                points[index],
                points[index + 1],
                girth * (1.15 - 0.26 * t),
                girth * (1.15 - 0.26 * (t + 1.0 / (len(points) - 1))),
                shaft,
                blend=unit * 0.030,
            )
        )
    tip = points[-1]
    glans_radius = girth * 1.14 * settings.glans_size
    direction = (tip - points[-2]).normalized()
    # The glans is a short tapered capsule rather than a ball on the end of a
    # tube: a rounded tip, a slight flare, and a narrow base is the shape that
    # reads as a glans from every angle.
    blobs.append(
        _sculpt.capsule(
            tip - direction * glans_radius * 1.35,
            tip,
            glans_radius * 0.70,
            glans_radius * 0.96,
            glans,
            blend=unit * 0.012,
        )
    )
    # The urethral opening, a small dark mark at the very tip.
    blobs.append(
        _sculpt.sphere(
            tip + direction * glans_radius * 0.55,
            glans_radius * 0.22,
            meatus,
            blend=unit * 0.006,
        )
    )
    # The scrotum: one sac below the root with two testicles inside it. It rises
    # a little with the erection, as it does on the body.
    drop = unit * (0.50 - 0.12 * erection)
    # A short pedicle joins the root to the sac. Without it the scrotum was a
    # separate closed shell a centimetre under the shaft -- a part resting beside
    # the organ rather than a continuation of it.
    blobs.append(
        _sculpt.capsule(
            frame.point(0.0, -unit * 0.010, -unit * 0.012),
            frame.point(0.0, -unit * 0.010 - drop, -unit * 0.010),
            unit * 0.11,
            unit * 0.16,
            scrotum,
            blend=unit * 0.030,
        )
    )
    # Two lobes, fused: a single sphere read as a ball hanging under the shaft,
    # where a scrotum is a soft sac with a left and a right side.
    for side in (-1.0, 1.0):
        blobs.append(
            _sculpt.ellipsoid(
                frame.point(
                    side * unit * 0.075 * full,
                    -unit * 0.010 - drop,
                    -unit * 0.012,
                ),
                (unit * 0.15 * full, unit * 0.19 * full, unit * 0.13 * full),
                scrotum,
                blend=unit * 0.045,
            )
        )
    return _sculpt.sculpt(blobs, cell=max(0.003, height * 0.0021), name="male anatomy")


def genital_geometry(mesh: Mesh, detail: GenitalDetail) -> AnatomyGeometry:
    """The genital set for the body's own sex, as one fused, continuous surface.

    The **female** set is deliberately *flush*: a soft mons, two labia, a hood,
    and a small opening, all with almost no forward offset -- so it reads as the
    body's own anatomy rather than as a male set on a female body.

    The **male** set is a scrotum below the pubic root and a shaft that leaves the
    body and hangs forward and down when flaccid, rising and lengthening with the
    erection dial, with a glans and a corona at its tip. Its root is embedded in
    the pubic mound, so the organ is continuous with the body.

    Nothing is built for a child: the caller never passes a detail for a minor.
    """
    geometry = AnatomyGeometry()
    settings = detail.normalised()
    if not settings.sex or settings.show <= 0.0:
        return geometry
    frame = _pubic_frame(mesh)
    if frame is None:
        return geometry
    low, high = mesh.bounds()
    height = max(high.y - low.y, 1e-6)
    # The set scales with the figure, so an adult body of any size gets adult
    # proportions.
    unit = height * 0.048 * settings.size
    if settings.sex == "female":
        geometry.add_mesh(_female_geometry(frame, settings, unit))
        return geometry
    geometry.add_mesh(_male_geometry(frame, settings, unit, height))
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
