"""Select the physical centroid trace shared by field, electrical and export APIs."""

from __future__ import annotations

import numpy as np


def conductor_trace(part, *, source: str = "auto") -> tuple[np.ndarray | None, str]:
    """Return a validated trace in metres; explicit sources never silently fall back.

    ``auto`` prefers the realized profile centroid, then the connected route.
    ``wire_path`` retains the earlier routed geometry for comparison scripts.
    """
    if source not in {"auto", "wire_path", "fabricated_centroid"}:
        raise ValueError(f"Unknown conductor trace source: {source!r}.")
    physical = getattr(part, "fabrication_path", None)
    if source != "wire_path" and physical is not None:
        points = np.asarray(physical.wire_points_m, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or not np.all(np.isfinite(points)):
            raise ValueError("The fabricated field trace must contain finite (N, 3) points.")
        return points, "fabricated_centroid"
    if source == "fabricated_centroid":
        raise ValueError("A fabricated centroid was requested but has not been realized.")
    wire = getattr(part, "wire_path", None)
    if wire is None:
        return None, "unconnected_contours"
    points = np.asarray(wire.v, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("The connected wire trace must contain finite (N, 3) points.")
    return (points, "connected_wire") if len(points) >= 2 else (None, "unconnected_contours")


def trace_segment_geometry(part, kind: str):
    """Use provenance from the selected representation, never index the earlier route."""
    trace = part.fabrication_path if kind == "fabricated_centroid" else part.wire_path
    return getattr(trace, "segment_geometry", ())
