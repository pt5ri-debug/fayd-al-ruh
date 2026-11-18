"""A small implicit-surface (SDF) sculptor that fuses parts into one smooth skin.

The figure builder used to assemble a body from separate primitives -- a torso
cylinder, two arm tubes, two leg tubes, a head, a neck, hands, feet -- and stitch
them together afterwards with subdivision and smoothing. The result was honest but
it read as a **doll**: the parts were distinct shells with visible gaps between
them (a shoulder ring that only *touched* an arm ring), so the figure looked like
pieces rather than one human body.

This module is the fix. Instead of building surfaces, it builds a **scalar field**:
every part is a smooth distance function (an ellipsoid, a tapered capsule, a
rounded box), and the parts are combined with a **smooth minimum** so they *fuse*
where they meet -- the shoulder blends into the arm, the neck into the chest, the
thigh into the hip, with no seam and no gap. The zero level set of that field is
then polygonized once, by **surface nets**, into a single closed mesh. The result
is one connected skin, which is what a body is.

It is deliberately small and dependency-light: the heavy lifting is a handful of
``numpy`` array operations (numpy is a base dependency of the project), and the
field is evaluated only inside each primitive's own bounding box, so the whole
build stays well under a second at the resolutions the studio uses.

The field is the honest part: what you see *is* the union of the parts you asked
for, smoothly fused. A part that is far from another simply never reaches it, so
there is no way for the sculptor to invent a connection the parameters did not
describe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.scene3d import Mesh, Vec3

#: The default grid spacing (metres) the sculptor samples the field at. Smaller is
#: finer and slower; this value gives a smooth silhouette at the studio's render
#: sizes while keeping a full body under a second to build.
DEFAULT_CELL = 0.010

#: A floor on the spacing, so a caller cannot ask for a grid so fine it hangs.
MIN_CELL = 0.003

#: How many lattice samples the sculptor will take before it coarsens the grid. The
#: rasteriser is fill-bound, so a mesh beyond this is slower to draw for no visible
#: gain; the cap keeps a very fine request honest (and bounded) rather than fatal.
MAX_SAMPLES = 6_000_000


@dataclass(frozen=True)
class _Blob:
    """One smooth part of the field: an ellipsoid, a tapered capsule, or a box.

    ``blend`` is the smooth-union radius: two parts closer than roughly their blend
    radii fuse with a rounded joint instead of a hard crease. ``colour`` is the RGB
    the surface carries where this part is the nearest one, so one fused mesh still
    shows skin, lips, irises, nails, and the like.
    """

    kind: str
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    r0: float
    r1: float
    colour: tuple[float, float, float]
    blend: float
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    subtract: bool = False


def _xyz(point: object) -> tuple[float, float, float]:
    """Coerce a point to a plain ``(x, y, z)`` -- accepting a ``Vec3`` or a sequence."""
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return (float(point.x), float(point.y), float(point.z))  # type: ignore[attr-defined]
    return (float(point[0]), float(point[1]), float(point[2]))  # type: ignore[index]


def _bounds(
    a: object, b: object, radius: float, blend: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = _xyz(a)
    b = _xyz(b)
    pad = max(radius, 0.0) + max(blend, 0.005) + 0.01
    lo = (min(a[0], b[0]) - pad, min(a[1], b[1]) - pad, min(a[2], b[2]) - pad)
    hi = (max(a[0], b[0]) + pad, max(a[1], b[1]) + pad, max(a[2], b[2]) + pad)
    return lo, hi


def sphere(
    centre: object,
    radius: float,
    colour: tuple[float, float, float],
    *,
    blend: float = 0.02,
) -> _Blob:
    return ellipsoid(centre, (radius, radius, radius), colour, blend=blend)


def ellipsoid(
    centre: object,
    radii: Sequence[float],
    colour: tuple[float, float, float],
    *,
    blend: float = 0.02,
) -> _Blob:
    c = _xyz(centre)
    r = (float(radii[0]), float(radii[1]), float(radii[2]))
    lo, hi = _bounds(c, c, max(r), blend)
    return _Blob("ell", c, r, 0.0, 0.0, colour, blend, lo, hi)


def capsule(
    a: object,
    b: object,
    radius_a: float,
    radius_b: float,
    colour: tuple[float, float, float],
    *,
    blend: float = 0.03,
) -> _Blob:
    p0 = _xyz(a)
    p1 = _xyz(b)
    lo, hi = _bounds(p0, p1, max(radius_a, radius_b), blend)
    return _Blob("cap", p0, p1, float(radius_a), float(radius_b), colour, blend, lo, hi)


def box(
    centre: object,
    half: Sequence[float],
    colour: tuple[float, float, float],
    *,
    blend: float = 0.02,
) -> _Blob:
    c = _xyz(centre)
    h = (float(half[0]), float(half[1]), float(half[2]))
    lo, hi = _bounds(c, c, max(h), blend)
    return _Blob("box", c, h, 0.0, 0.0, colour, blend, lo, hi)


def subtraction(blob: _Blob) -> _Blob:
    """A copy of ``blob`` marked to be **carved out** of the field."""
    return _Blob(
        blob.kind, blob.a, blob.b, blob.r0, blob.r1, blob.colour, blob.blend,
        blob.lo, blob.hi, True,
    )


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float | np.ndarray) -> np.ndarray:
    """The polynomial smooth minimum -- the operation that fuses two parts.

    Where the two fields are far apart it is the ordinary minimum (the nearer
    surface wins); where they are within ``k`` it curves the joint, so the union
    of the two parts reads as one smooth surface rather than two surfaces meeting
    at a crease.
    """
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smooth_max(a: np.ndarray, b: np.ndarray, k: float | np.ndarray) -> np.ndarray:
    """The smooth maximum -- used to **carve** one field out of another."""
    return -_smooth_min(-a, -b, k)


def _distance(blob: _Blob, points: np.ndarray) -> np.ndarray:
    """The signed distance from ``points`` (n, 3) to one blob's surface."""
    if blob.kind == "ell":
        centre = np.asarray(blob.a, dtype=float)
        radii = np.asarray(blob.b, dtype=float)
        scaled = (points - centre) / radii
        return (np.linalg.norm(scaled, axis=1) - 1.0) * float(radii.min())
    if blob.kind == "cap":
        start = np.asarray(blob.a, dtype=float)
        end = np.asarray(blob.b, dtype=float)
        axis = end - start
        length_sq = float(axis @ axis)
        if length_sq < 1e-12:
            return np.linalg.norm(points - start, axis=1) - blob.r0
        t = np.clip(((points - start) @ axis) / length_sq, 0.0, 1.0)
        projection = start + t[:, None] * axis
        radius = blob.r0 + (blob.r1 - blob.r0) * t
        return np.linalg.norm(points - projection, axis=1) - radius
    half = np.asarray(blob.b, dtype=float)
    centre = np.asarray(blob.a, dtype=float)
    q = np.abs(points - centre) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def _field_on_grid(
    blobs: Sequence[_Blob],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the fused field on a lattice, returning ``(distance, colour_index)``.

    Each blob is evaluated only over the sub-box its own bounding box covers, which
    is what keeps a full-body build fast: a fingertip's blob never touches the grid
    points near the feet.
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    distance = np.full((nx, ny, nz), 1e9)
    colour_index = np.zeros((nx, ny, nz), dtype=np.int16)
    for index, blob in enumerate(blobs):
        ix0 = int(np.searchsorted(xs, blob.lo[0], "left"))
        ix1 = int(np.searchsorted(xs, blob.hi[0], "right"))
        iy0 = int(np.searchsorted(ys, blob.lo[1], "left"))
        iy1 = int(np.searchsorted(ys, blob.hi[1], "right"))
        iz0 = int(np.searchsorted(zs, blob.lo[2], "left"))
        iz1 = int(np.searchsorted(zs, blob.hi[2], "right"))
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        gx, gy, gz = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], indexing="ij")
        points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        d = _distance(blob, points).reshape(ix1 - ix0, iy1 - iy0, iz1 - iz0)
        window = (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1))
        # ``current`` is a *view* into ``distance``: reading it into a fresh array
        # first is what keeps the "nearer blob" comparison correct, because writing
        # the new distance below would otherwise overwrite it in place.
        current = distance[window].copy()
        if blob.subtract:
            distance[window] = _smooth_max(current, -d, max(blob.blend, 1e-6))
            continue
        # Blend only where the two surfaces are close (a small, bounded k), so a
        # distant part never smears the field.
        k = np.maximum(np.minimum(blob.blend, np.abs(current)), 1e-6)
        distance[window] = _smooth_min(current, d, k)
        colour_index[window] = np.where(d < current, index, colour_index[window])
    return distance, colour_index


# Corner order for a cell, and the 12 edges between the corners.
_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)
_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (4, 5), (4, 6), (3, 7), (5, 7), (6, 7),
)


def _surface_nets(
    field: np.ndarray, origin: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Polygonize the zero level set of ``field`` into one quad mesh.

    This is **surface nets**: one vertex per cell the surface passes through, placed
    at the average of the cell's edge crossings, and one quad per lattice edge whose
    endpoints straddle the surface. It is the dual of marching cubes, but it yields a
    clean, connected quad mesh with no duplicated vertices at cell boundaries -- so
    the surface is a single shell wherever the field is a single body.
    """
    nx, ny, nz = field.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    sign = field < 0.0

    corner_sign = [sign[dx:dx + cx, dy:dy + cy, dz:dz + cz] for dx, dy, dz in _CORNERS]
    corner_field = [field[dx:dx + cx, dy:dy + cy, dz:dz + cz] for dx, dy, dz in _CORNERS]

    inside = np.zeros((cx, cy, cz), dtype=bool)
    outside = np.ones((cx, cy, cz), dtype=bool)
    for s in corner_sign:
        inside |= s
        outside &= s
    active = inside & ~outside

    offset_sum = np.zeros((cx, cy, cz, 3))
    crossing_count = np.zeros((cx, cy, cz))
    for a, b in _EDGES:
        straddles = corner_sign[a] != corner_sign[b]
        if not straddles.any():
            continue
        ca = np.array(_CORNERS[a], dtype=float)
        cb = np.array(_CORNERS[b], dtype=float)
        fa = corner_field[a]
        fb = corner_field[b]
        denom = fa - fb
        safe = np.abs(denom) > 1e-12
        t = np.where(safe, fa / np.where(safe, denom, 1.0), 0.5)
        offset_sum += np.where(straddles[..., None], ca + t[..., None] * (cb - ca), 0.0)
        crossing_count += straddles

    has_vertex = active & (crossing_count > 0)
    mean_offset = offset_sum / np.maximum(crossing_count, 1.0)[..., None]
    i = np.broadcast_to(np.arange(cx)[:, None, None], (cx, cy, cz))
    j = np.broadcast_to(np.arange(cy)[None, :, None], (cx, cy, cz))
    k = np.broadcast_to(np.arange(cz)[None, None, :], (cx, cy, cz))
    origin_grid = np.stack(
        [origin[0] + cell * i, origin[1] + cell * j, origin[2] + cell * k], axis=-1
    )
    positions = origin_grid + cell * mean_offset

    cell_vertex = np.full((cx, cy, cz), -1, dtype=np.int64)
    cell_vertex[has_vertex] = np.arange(int(has_vertex.sum()))
    vertices = positions[has_vertex]
    padded = np.full((cx + 2, cy + 2, cz + 2), -1, dtype=np.int64)
    padded[1:-1, 1:-1, 1:-1] = cell_vertex

    quads: list[np.ndarray] = []

    straddle_x = sign[:-1, :, :] != sign[1:, :, :]
    ii, jj, kk = np.nonzero(straddle_x)
    if len(ii):
        v0 = padded[ii + 1, jj, kk]
        v1 = padded[ii + 1, jj + 1, kk]
        v2 = padded[ii + 1, jj + 1, kk + 1]
        v3 = padded[ii + 1, jj, kk + 1]
        good = (v0 >= 0) & (v1 >= 0) & (v2 >= 0) & (v3 >= 0)
        quads.append(np.stack([v0[good], v1[good], v2[good], v3[good]], axis=1))

    straddle_y = sign[:, :-1, :] != sign[:, 1:, :]
    ii, jj, kk = np.nonzero(straddle_y)
    if len(ii):
        v0 = padded[ii, jj + 1, kk]
        v1 = padded[ii + 1, jj + 1, kk]
        v2 = padded[ii + 1, jj + 1, kk + 1]
        v3 = padded[ii, jj + 1, kk + 1]
        good = (v0 >= 0) & (v1 >= 0) & (v2 >= 0) & (v3 >= 0)
        quads.append(np.stack([v0[good], v1[good], v2[good], v3[good]], axis=1))

    straddle_z = sign[:, :, :-1] != sign[:, :, 1:]
    ii, jj, kk = np.nonzero(straddle_z)
    if len(ii):
        v0 = padded[ii, jj, kk + 1]
        v1 = padded[ii + 1, jj, kk + 1]
        v2 = padded[ii + 1, jj + 1, kk + 1]
        v3 = padded[ii, jj + 1, kk + 1]
        good = (v0 >= 0) & (v1 >= 0) & (v2 >= 0) & (v3 >= 0)
        quads.append(np.stack([v0[good], v1[good], v2[good], v3[good]], axis=1))

    faces = np.concatenate(quads, axis=0) if quads else np.zeros((0, 4), dtype=np.int64)
    return vertices, faces


def _bounds_of(blobs: Sequence[_Blob]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([math.inf, math.inf, math.inf])
    hi = np.array([-math.inf, -math.inf, -math.inf])
    for blob in blobs:
        lo = np.minimum(lo, np.asarray(blob.lo))
        hi = np.maximum(hi, np.asarray(blob.hi))
    return lo, hi


def sculpt(
    blobs: Sequence[_Blob],
    *,
    cell: float = DEFAULT_CELL,
    name: str = "figure",
) -> Mesh:
    """Fuse ``blobs`` into one connected mesh and return it.

    The grid spacing is coarsened automatically if the request would exceed
    :data:`MAX_SAMPLES`, so a very fine setting stays honest rather than hanging. The
    mesh carries a per-vertex colour taken from the nearest blob, so the fused skin
    still shows the lips, irises, nails, and the rest.
    """
    if not blobs:
        return Mesh(name, (), ())
    lo, hi = _bounds_of(blobs)
    spacing = max(MIN_CELL, float(cell))
    nx = ny = nz = 0
    for _ in range(8):
        nx = int((hi[0] - lo[0]) / spacing) + 2
        ny = int((hi[1] - lo[1]) / spacing) + 2
        nz = int((hi[2] - lo[2]) / spacing) + 2
        if nx * ny * nz <= MAX_SAMPLES:
            break
        spacing *= 1.3
    xs = lo[0] + spacing * np.arange(nx)
    ys = lo[1] + spacing * np.arange(ny)
    zs = lo[2] + spacing * np.arange(nz)
    field, colour_index = _field_on_grid(blobs, xs, ys, zs)
    vertices, faces = _surface_nets(field, lo, spacing)
    colours = _vertex_colours(blobs, vertices, colour_index, xs, ys, zs)
    return Mesh(
        name,
        tuple(Vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices),
        tuple(tuple(int(i) for i in face) for face in faces),
        (),
        tuple(colours),
    )


def _vertex_colours(
    blobs: Sequence[_Blob],
    vertices: np.ndarray,
    colour_index: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> list[tuple[float, float, float]]:
    """The colour each surface vertex carries: its nearest blob's own colour.

    The index grid is sampled at the vertex's cell, which is cheap and, at the
    sculpting resolutions used here, agrees with the true nearest blob on every
    vertex that matters (a vertex sits on the surface of the blob it belongs to).
    The lookup is vectorized -- a vertex at a time in Python would dominate the
    whole build on a fine body.
    """
    if len(vertices) == 0:
        return []
    nx, ny, nz = colour_index.shape
    i = np.clip(np.searchsorted(xs, vertices[:, 0]) - 1, 0, nx - 1)
    j = np.clip(np.searchsorted(ys, vertices[:, 1]) - 1, 0, ny - 1)
    k = np.clip(np.searchsorted(zs, vertices[:, 2]) - 1, 0, nz - 1)
    palette = np.array([blob.colour for blob in blobs], dtype=float)
    picked = palette[colour_index[i, j, k]]
    return [(float(r), float(g), float(b)) for r, g, b in picked]


__all__ = [
    "DEFAULT_CELL",
    "MAX_SAMPLES",
    "MIN_CELL",
    "box",
    "capsule",
    "ellipsoid",
    "sculpt",
    "sphere",
    "subtraction",
]
