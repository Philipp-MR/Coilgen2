"""Explicit overpass proposals. These are geometry, never acceptance receipts."""

import numpy as np
from .models import WirePathGeometry


def surface_crossing_stations(projector, first, second):
    """Intersect projected facet polylines in 3D, without inventing a flat chart."""
    first, second = np.asarray(first), np.asarray(second)
    a, _ = projector.project(first)
    b, _ = projector.project(second)
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    if len(da) * len(db) > 4_000_000:
        raise ValueError("Crossing search exceeds its bounded work budget.")
    la, lb = np.linalg.norm(np.diff(first, axis=0), axis=1), np.linalg.norm(np.diff(second, axis=0), axis=1)
    sa, sb = np.r_[0.0, np.cumsum(la)], np.r_[0.0, np.cumsum(lb)]
    tolerance = max(np.ptp(np.vstack((a, b)), axis=0).max() * 1e-8, 1e-10)
    hits = []
    bb = np.einsum("ij,ij->i", db, db)
    for i, direction in enumerate(da):
        aa, ab = direction @ direction, db @ direction
        q = b[:-1] - a[i]
        aq, bq = q @ direction, np.einsum("ij,ij->i", q, db)
        denominator = aa * bb - ab**2
        valid = denominator > 1e-14 * aa * bb
        t = np.divide(bb * aq - ab * bq, denominator, out=np.zeros(len(db)), where=valid)
        u = np.divide(ab * aq - aa * bq, denominator, out=np.zeros(len(db)), where=valid)
        distance = np.linalg.norm(a[i] + t[:, None] * direction - b[:-1] - u[:, None] * db, axis=1)
        for j in np.flatnonzero(valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (distance <= tolerance)):
            station = sa[i] + t[j] * la[i]
            if (
                station <= tolerance
                or station >= sa[-1] - tolerance
                or any(abs(station - h[0]) <= tolerance for h in hits)
            ):
                continue
            hits.append(
                (
                    station,
                    sb[j] + u[j] * lb[j],
                    first[i] + t[j] * (first[i + 1] - first[i]),
                    second[j] + u[j] * (second[j + 1] - second[j]),
                )
            )
    return hits


def crossing_stations(chart, first, second):
    a, b = chart.project(first), chart.project(second)
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    la, lb = np.linalg.norm(np.diff(first, axis=0), axis=1), np.linalg.norm(np.diff(second, axis=0), axis=1)
    if len(da) * len(db) > 4_000_000:
        raise ValueError("Crossing search exceeds its bounded work budget.")
    sa, sb = np.r_[0.0, np.cumsum(la)], np.r_[0.0, np.cumsum(lb)]

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    hits = []
    for i, direction in enumerate(da):
        if np.linalg.norm(direction) > 3 * la[i] + 1e-6:
            continue
        denominator = cross(direction, db)
        valid = (np.abs(denominator) > 1e-16) & (np.linalg.norm(db, axis=1) <= 3 * lb + 1e-6)
        t = np.divide(cross(b[:-1] - a[i], db), denominator, out=np.zeros(len(db)), where=valid)
        u = np.divide(cross(b[:-1] - a[i], direction), denominator, out=np.zeros(len(db)), where=valid)
        for j in np.flatnonzero(valid & (t >= -1e-9) & (t <= 1 + 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)):
            ta, tb = np.clip(t[j], 0, 1), np.clip(u[j], 0, 1)
            station = sa[i] + ta * la[i]
            if station <= 1e-9 or station >= sa[-1] - 1e-9 or any(abs(station - h[0]) < 1e-9 for h in hits):
                continue
            hits.append(
                (
                    station,
                    sb[j] + tb * lb[j],
                    first[i] + ta * (first[i + 1] - first[i]),
                    second[j] + tb * (second[j + 1] - second[j]),
                )
            )
    return hits


def realize_bridge(projector, spec):
    base = np.asarray(spec["base_points_m"], float)
    geometry = tuple(WirePathGeometry(g) for g in spec["base_geometry"])
    if len(geometry) != len(base) - 1:
        raise ValueError("Bridge support provenance is incomplete.")
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))]
    center, height, span = (float(spec[k]) for k in ("center_m", "height_m", "span_m"))
    if (
        not np.isfinite([center, height, span]).all()
        or height < 0
        or span <= 0
        or center - span / 2 < 0
        or center + span / 2 > station[-1]
    ):
        raise ValueError("The bridge needs room for both ramps before its attachment endpoints.")
    if not np.isfinite(spec["separation_m"]) or spec["separation_m"] <= 0:
        raise ValueError("Bridge separation must be finite and positive.")
    spacing = min(0.001, span / 32)
    count = max(33, int(np.ceil(span / spacing)) + 1)
    if count + len(station) + 1 > 50000:
        raise ValueError("Bridge exceeds the routing point budget.")
    samples = np.unique(np.r_[station, center, np.linspace(center - span / 2, center + span / 2, count)])
    samples = samples[np.r_[True, np.diff(samples) > 1e-12]]
    if len(samples) > 50000:
        raise ValueError("Bridge exceeds the routing point budget.")
    points = np.column_stack([np.interp(samples, station, base[:, i]) for i in range(3)])
    _, normals = projector.project(points)
    u = np.clip((samples - center) / span + 0.5, 0, 1)
    # Zero height, first derivative and second derivative at both ramp ends.
    lift = height * 64 * u**3 * (1 - u) ** 3
    points += lift[:, None] * normals
    points[0], points[-1] = base[0], base[-1]
    index = np.clip(np.searchsorted(station, (samples[:-1] + samples[1:]) / 2, side="right") - 1, 0, len(geometry) - 1)
    labels = tuple(
        WirePathGeometry.OFFSET_SURFACE if max(lift[i : i + 2]) > 1e-14 else geometry[j] for i, j in enumerate(index)
    )
    return points, labels
