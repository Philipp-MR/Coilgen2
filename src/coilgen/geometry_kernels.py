"""Float64 distance kernels; no fastmath or parallel reductions.

The scalar reference, batched NumPy path and optional JIT use the same clamped
geometry predicates. Importing this module does not import Numba.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.util import find_spec
from threading import Lock
import warnings

import numpy as np

_ACTIVE_BACKEND = ContextVar("coilgen_geometry_backend", default=None)


def active_geometry_backend():
    return _ACTIVE_BACKEND.get()


@contextmanager
def geometry_acceleration(backend="auto"):
    """Opt sweeps into batched distances without changing core defaults."""
    if backend not in {"auto", "reference", "numpy", "numba"}:
        raise ValueError("Distance backend must be auto, reference, numpy, or numba.")
    token = _ACTIVE_BACKEND.set(backend)
    try:
        yield
    finally:
        _ACTIVE_BACKEND.reset(token)


def _segment_sq(p, q, a, b):
    u, v, r = q - p, b - a, p - a
    uu, vv = np.dot(u, u), np.dot(v, v)
    if uu == 0.0 and vv == 0.0:
        return np.dot(r, r)
    if uu == 0.0:
        s, t = 0.0, min(max(np.dot(v, r) / vv, 0.0), 1.0)
    else:
        ur = np.dot(u, r)
        if vv == 0.0:
            s, t = min(max(-ur / uu, 0.0), 1.0), 0.0
        else:
            uv, vr = np.dot(u, v), np.dot(v, r)
            normal = np.cross(u, v)
            denominator = np.dot(normal, normal)
            numerator = np.dot(np.cross(-r, v), normal)
            s = 0.0 if denominator == 0.0 else min(max(numerator / denominator, 0.0), 1.0)
            t = (uv * s + vr) / vv
            if t < 0.0:
                t, s = 0.0, min(max(-ur / uu, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((uv - ur) / uu, 0.0), 1.0)
    delta = r + s * u - t * v
    return max(np.dot(delta, delta), 0.0)


def segment_distance(p, q, a, b) -> float:
    """Closest distance between two segments, including exact point segments."""
    points = (np.asarray(value, dtype=np.float64) for value in (p, q, a, b))
    return float(np.sqrt(_segment_sq(*points)))


def segment_distances(first_start, first_direction, second_start, second_direction):
    """Vectorized clamped distance for broadcastable ``(..., 3)`` arrays.

    Degeneracy means zero length, not an absolute length cutoff. Cross products
    avoid subtracting almost equal squared dot products for parallel lines and
    keep the interior-crossing decision independent of the geometry's units.
    Directions are accepted directly to avoid reconstructing rounded endpoints.
    """
    p, u, a, v = (
        np.asarray(value, dtype=np.float64) for value in (first_start, first_direction, second_start, second_direction)
    )
    s, t = segment_closest_parameters(p, u, a, v)
    return np.linalg.norm(p - a + s[..., None] * u - t[..., None] * v, axis=-1)


def segment_closest_parameters(first_start, first_direction, second_start, second_direction):
    """Clamped closest-point parameters using the same stable distance kernel."""
    p, u, a, v = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (first_start, first_direction, second_start, second_direction)
        )
    )
    r = p - a
    uu, vv = np.einsum("...i,...i->...", u, u), np.einsum("...i,...i->...", v, v)
    uv = np.einsum("...i,...i->...", u, v)
    ur, vr = np.einsum("...i,...i->...", u, r), np.einsum("...i,...i->...", v, r)
    normal = np.cross(u, v)
    denominator = np.einsum("...i,...i->...", normal, normal)
    numerator = np.einsum("...i,...i->...", np.cross(-r, v), normal)
    s = np.clip(np.divide(numerator, denominator, out=np.zeros_like(uu), where=denominator != 0.0), 0.0, 1.0)
    t = np.divide(uv * s + vr, vv, out=np.zeros_like(vv), where=vv != 0.0)
    before = t < 0.0
    after = t > 1.0
    first_projection = np.clip(np.divide(-ur, uu, out=np.zeros_like(uu), where=uu != 0.0), 0.0, 1.0)
    last_projection = np.clip(np.divide(uv - ur, uu, out=np.zeros_like(uu), where=uu != 0.0), 0.0, 1.0)
    s = np.where(before | (vv == 0.0), first_projection, s)
    s = np.where(after, last_projection, s)
    t = np.clip(t, 0.0, 1.0)
    return s, t


def _point_triangle_sq(p, triangle):
    a, b, c = triangle[0], triangle[1], triangle[2]
    ab, ac, ap = b - a, c - a, p - a
    normal = np.cross(ab, ac)
    if np.dot(normal, normal) == 0.0:
        return min(_segment_sq(p, p, a, b), _segment_sq(p, p, b, c), _segment_sq(p, p, c, a))
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.dot(ap, ap)
    bp = p - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        delta = p - (a + d1 / (d1 - d3) * ab)
        return np.dot(delta, delta)
    cp = p - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        delta = p - (a + d2 / (d2 - d6) * ac)
        return np.dot(delta, delta)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        delta = p - (b + fraction * (c - b))
        return np.dot(delta, delta)
    denominator = va + vb + vc
    if denominator == 0.0:
        return min(_segment_sq(p, p, a, b), _segment_sq(p, p, b, c), _segment_sq(p, p, c, a))
    delta = p - (a + (vb * ab + vc * ac) / denominator)
    return max(np.dot(delta, delta), 0.0)


def _segment_hits_triangle(start, end, triangle):
    direction, e1, e2 = end - start, triangle[1] - triangle[0], triangle[2] - triangle[0]
    cross = np.cross(direction, e2)
    determinant = np.dot(e1, cross)
    scale = np.sqrt(np.dot(direction, direction) * np.dot(e1, e1) * np.dot(e2, e2))
    if abs(determinant) <= 32.0 * np.finfo(np.float64).eps * scale:
        return False
    offset = start - triangle[0]
    first = np.dot(offset, cross) / determinant
    other = np.cross(offset, e1)
    second = np.dot(direction, other) / determinant
    fraction = np.dot(e2, other) / determinant
    return 0.0 <= first <= 1.0 and second >= 0.0 and first + second <= 1.0 and 0.0 <= fraction <= 1.0


def _triangle_sq(first, second):
    best = np.inf
    for i in range(3):
        j = (i + 1) % 3
        if _segment_hits_triangle(first[i], first[j], second) or _segment_hits_triangle(second[i], second[j], first):
            return 0.0
        best = min(best, _point_triangle_sq(first[i], second), _point_triangle_sq(second[i], first))
        for k in range(3):
            best = min(best, _segment_sq(first[i], first[j], second[k], second[(k + 1) % 3]))
    return max(best, 0.0)


def _segments_reference(first, second):
    values = np.empty(len(first), dtype=np.float64)
    for index in range(len(first)):
        values[index] = np.sqrt(_segment_sq(first[index, 0], first[index, 1], second[index, 0], second[index, 1]))
    return values


def _triangles_reference(first, second):
    values = np.empty(len(first), dtype=np.float64)
    for index in range(len(first)):
        values[index] = np.sqrt(_triangle_sq(first[index], second[index]))
    return values


def _dot_rows(first, second):
    return np.einsum("ij,ij->i", first, second)


def _point_segment_sq_numpy(points, start, end):
    direction, offset = end - start, points - start
    length_sq = _dot_rows(direction, direction)
    fraction = np.clip(
        np.divide(_dot_rows(direction, offset), length_sq, out=np.zeros_like(length_sq), where=length_sq != 0.0),
        0.0,
        1.0,
    )
    delta = offset - fraction[:, None] * direction
    return _dot_rows(delta, delta)


def _point_triangle_sq_numpy(points, triangles):
    """The scalar Voronoi-region tests, with their original branch ordering."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    active = _dot_rows(normal, normal) != 0.0
    values = np.empty(len(points), dtype=np.float64)
    fallback = ~active
    d1, d2 = _dot_rows(ab, ap), _dot_rows(ac, ap)

    mask = active & (d1 <= 0.0) & (d2 <= 0.0)
    values[mask] = _dot_rows(ap[mask], ap[mask])
    active &= ~mask
    bp = points - b
    d3, d4 = _dot_rows(ab, bp), _dot_rows(ac, bp)
    mask = active & (d3 >= 0.0) & (d4 <= d3)
    values[mask] = _dot_rows(bp[mask], bp[mask])
    active &= ~mask

    vc = d1 * d4 - d3 * d2
    mask = active & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(mask):
        delta = points[mask] - (a[mask] + (d1[mask] / (d1[mask] - d3[mask]))[:, None] * ab[mask])
        values[mask] = _dot_rows(delta, delta)
    active &= ~mask

    cp = points - c
    d5, d6 = _dot_rows(ab, cp), _dot_rows(ac, cp)
    mask = active & (d6 >= 0.0) & (d5 <= d6)
    values[mask] = _dot_rows(cp[mask], cp[mask])
    active &= ~mask

    vb = d5 * d2 - d1 * d6
    mask = active & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(mask):
        delta = points[mask] - (a[mask] + (d2[mask] / (d2[mask] - d6[mask]))[:, None] * ac[mask])
        values[mask] = _dot_rows(delta, delta)
    active &= ~mask

    va = d3 * d6 - d5 * d4
    mask = active & (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    if np.any(mask):
        fraction = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
        delta = points[mask] - (b[mask] + fraction[:, None] * (c[mask] - b[mask]))
        values[mask] = _dot_rows(delta, delta)
    active &= ~mask

    denominator = va + vb + vc
    fallback |= active & (denominator == 0.0)
    mask = active & ~fallback
    if np.any(mask):
        delta = points[mask] - (
            a[mask] + (vb[mask, None] * ab[mask] + vc[mask, None] * ac[mask]) / denominator[mask, None]
        )
        values[mask] = np.maximum(_dot_rows(delta, delta), 0.0)
    if np.any(fallback):
        p, aa, bb, cc = points[fallback], a[fallback], b[fallback], c[fallback]
        values[fallback] = np.minimum.reduce(
            (
                _point_segment_sq_numpy(p, aa, bb),
                _point_segment_sq_numpy(p, bb, cc),
                _point_segment_sq_numpy(p, cc, aa),
            )
        )
    return values


def _segment_hits_triangle_numpy(start, end, triangles):
    direction = end - start
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction, e2)
    determinant = _dot_rows(e1, cross)
    scale = np.sqrt(_dot_rows(direction, direction) * _dot_rows(e1, e1) * _dot_rows(e2, e2))
    active = np.abs(determinant) > 32.0 * np.finfo(np.float64).eps * scale
    hits = np.zeros(len(start), dtype=bool)
    if np.any(active):
        offset = start[active] - triangles[active, 0]
        other = np.cross(offset, e1[active])
        first = _dot_rows(offset, cross[active]) / determinant[active]
        second = _dot_rows(direction[active], other) / determinant[active]
        fraction = _dot_rows(e2[active], other) / determinant[active]
        hits[active] = (
            (first >= 0.0)
            & (first <= 1.0)
            & (second >= 0.0)
            & (first + second <= 1.0)
            & (fraction >= 0.0)
            & (fraction <= 1.0)
        )
    return hits


def _segments_numpy(first, second):
    values = np.empty(len(first), dtype=np.float64)
    for start in range(0, len(first), 4096):
        a, b = first[start : start + 4096], second[start : start + 4096]
        values[start : start + len(a)] = segment_distances(
            a[:, 0], a[:, 1] - a[:, 0], b[:, 0], b[:, 1] - b[:, 0]
        )
    return values


def _triangles_numpy(first, second):
    """All scalar primitive tests, in batches with bounded temporary storage."""
    values = np.empty(len(first), dtype=np.float64)
    for start in range(0, len(first), 4096):
        a, b = first[start : start + 4096], second[start : start + 4096]
        best = np.full(len(a), np.inf)
        for i in range(3):
            j = (i + 1) % 3
            hits = _segment_hits_triangle_numpy(a[:, i], a[:, j], b) | _segment_hits_triangle_numpy(
                b[:, i], b[:, j], a
            )
            best[hits] = 0.0
            best = np.minimum(best, _point_triangle_sq_numpy(a[:, i], b))
            best = np.minimum(best, _point_triangle_sq_numpy(b[:, i], a))
            for k in range(3):
                distances = segment_distances(
                    a[:, i], a[:, j] - a[:, i], b[:, k], b[:, (k + 1) % 3] - b[:, k]
                )
                best = np.minimum(best, distances * distances)
        values[start : start + len(a)] = np.sqrt(np.maximum(best, 0.0))
    return values


NUMBA_AVAILABLE = find_spec("numba") is not None
_segments_compiled = _triangles_compiled = None
_NUMBA_INITIALIZED = False
_NUMBA_FAILURE = None
_NUMBA_WARNING_EMITTED = False
_COMPILE_LOCK = Lock()


def _compiled_kernel(vertices):
    global _NUMBA_INITIALIZED, _segments_compiled, _triangles_compiled, _NUMBA_FAILURE
    if not NUMBA_AVAILABLE:
        return None
    with _COMPILE_LOCK:
        if _NUMBA_FAILURE is not None:
            raise RuntimeError(str(_NUMBA_FAILURE))
        if not _NUMBA_INITIALIZED:
            try:
                from numba import njit
                from numba.extending import register_jitable

                for helper in (_segment_sq, _point_triangle_sq, _segment_hits_triangle, _triangle_sq):
                    register_jitable(helper)
                _segments_compiled = njit(cache=True, fastmath=False)(_segments_reference)
                _triangles_compiled = njit(cache=True, fastmath=False)(_triangles_reference)
                _NUMBA_INITIALIZED = True
            except Exception as exc:
                # Keep the message without retaining traceback frames and their
                # distance-batch arrays for the lifetime of a design session.
                _NUMBA_FAILURE = str(exc)
                raise
    return _segments_compiled if vertices == 2 else _triangles_compiled


@dataclass(frozen=True)
class DistanceResult:
    values_m: np.ndarray
    backend: str


def _run(first, second, *, vertices, backend, reference, numpy_kernel):
    global _NUMBA_FAILURE, _NUMBA_WARNING_EMITTED
    if backend not in {"auto", "reference", "numpy", "numba"}:
        raise ValueError("Distance backend must be auto, reference, numpy, or numba.")
    first, second = np.ascontiguousarray(first, dtype=np.float64), np.ascontiguousarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (vertices, 3):
        raise ValueError(f"Paired distance inputs require matching (n, {vertices}, 3) arrays.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Distance inputs must be finite.")
    if backend not in {"reference", "numpy"}:
        try:
            if _NUMBA_FAILURE is not None:
                raise RuntimeError(str(_NUMBA_FAILURE))
            compiled = _compiled_kernel(vertices)
            if compiled is not None:
                return DistanceResult(compiled(first, second), "numba_float64")
            if backend == "numba":
                raise RuntimeError("Numba was requested but is not installed; select the reference backend.")
        except Exception as exc:
            if backend == "numba":
                raise
            with _COMPILE_LOCK:
                _NUMBA_FAILURE = str(exc)
                warn = not _NUMBA_WARNING_EMITTED
                _NUMBA_WARNING_EMITTED = True
            if warn:
                warnings.warn(
                    f"Numba distance kernel unavailable; using the vectorized NumPy implementation: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    if backend == "reference":
        return DistanceResult(reference(first, second), "reference_float64")
    return DistanceResult(numpy_kernel(first, second), "numpy_float64")


def segment_pair_distances(first, second, *, backend="auto") -> DistanceResult:
    return _run(first, second, vertices=2, backend=backend, reference=_segments_reference, numpy_kernel=_segments_numpy)


def triangle_pair_distances(first, second, *, backend="auto") -> DistanceResult:
    return _run(first, second, vertices=3, backend=backend, reference=_triangles_reference, numpy_kernel=_triangles_numpy)
