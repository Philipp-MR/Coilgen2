"""Optional float64 distance kernels; no fastmath or parallel reductions.

The NumPy/Python implementation is the reference. Numba is imported optionally
and compiled lazily, so geometry remains available without the accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import warnings

import numpy as np

_ACTIVE_BACKEND = ContextVar("coilgen_research_geometry_backend", default=None)


def active_geometry_backend():
    return _ACTIVE_BACKEND.get()


@contextmanager
def geometry_acceleration(backend="auto"):
    """Opt research sweeps into batched distances without changing core defaults."""
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
    delta = p + s * u - a - t * v
    return max(np.dot(delta, delta), 0.0)


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


try:
    from numba import njit
    from numba.extending import register_jitable

    for helper in (_segment_sq, _point_triangle_sq, _segment_hits_triangle, _triangle_sq):
        register_jitable(helper)
    _segments_compiled = njit(cache=True, fastmath=False)(_segments_reference)
    _triangles_compiled = njit(cache=True, fastmath=False)(_triangles_reference)
    NUMBA_AVAILABLE = True
except ImportError:
    _segments_compiled = _triangles_compiled = None
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class DistanceResult:
    values_m: np.ndarray
    backend: str


def _run(first, second, *, vertices, backend, reference, compiled):
    if backend not in {"auto", "reference", "numpy", "numba"}:
        raise ValueError("Distance backend must be auto, reference, numpy, or numba.")
    if backend == "numpy":
        backend = "reference"
    first, second = np.ascontiguousarray(first, dtype=np.float64), np.ascontiguousarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (vertices, 3):
        raise ValueError(f"Paired distance inputs require matching (n, {vertices}, 3) arrays.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Distance inputs must be finite.")
    if backend == "numba" and compiled is None:
        raise RuntimeError("Numba was requested but is not installed; select the reference backend.")
    if backend != "reference" and compiled is not None:
        try:
            return DistanceResult(compiled(first, second), "numba_float64")
        except Exception as exc:
            if backend == "numba":
                raise
            warnings.warn(f"Numba distance kernel failed; using the reference implementation: {exc}", RuntimeWarning)
    return DistanceResult(reference(first, second), "reference_float64")


def segment_pair_distances(first, second, *, backend="auto") -> DistanceResult:
    return _run(first, second, vertices=2, backend=backend, reference=_segments_reference, compiled=_segments_compiled)


def triangle_pair_distances(first, second, *, backend="auto") -> DistanceResult:
    return _run(
        first, second, vertices=3, backend=backend, reference=_triangles_reference, compiled=_triangles_compiled
    )
