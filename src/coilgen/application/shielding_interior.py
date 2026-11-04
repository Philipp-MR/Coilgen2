"""Opt-in physical field checks at symmetric degree-two triangular quadrature points."""

from __future__ import annotations

from dataclasses import replace
from numbers import Real
from types import SimpleNamespace

import numpy as np

from ..config import CoilGenConfig
from .field_sampling import MAX_PHYSICAL_INTERACTIONS, _cancel, _physical, _sheet_source, _wire_source


def interior_sources(record, drives):
    """Budget work before assembling any source operators or quadrature points."""
    names = set(record.array_names)
    if "shielding_surface_faces" not in names:
        return ()
    faces = record.array("shielding_surface_faces", copy=False)
    if faces.ndim != 2 or faces.shape[1] != 3 or not 0 < len(faces) <= 20_000:
        return ()
    metadata = record.metadata_view if hasattr(record, "metadata_view") else record.metadata
    count = 3 * len(faces) + 24
    sources = []
    if "sheet" in drives and {"vertices_m", "faces", "stream_function_A"} <= names:
        order = metadata.get("config", {}).get("gauss_order")
        vertices = record.array("vertices_m", copy=False)
        triangles = record.array("faces", copy=False)
        if (
            type(order) is int
            and 1 <= order <= 16
            and vertices.ndim == triangles.ndim == 2
            and vertices.shape[1] == triangles.shape[1] == 3
            and record.array("stream_function_A", copy=False).shape == (len(vertices),)
            and 0 < len(vertices) * len(triangles) * 24 <= 128 * 1024**2
            and 3 * len(triangles) * order**2 * count <= MAX_PHYSICAL_INTERACTIONS
        ):
            sources.append("sheet_interior")
    if "routed" in drives:
        evidence = (metadata.get("wire_fields") or {}).get("routed") or {}
        current = evidence.get("current_A")
        paths = [
            record.array(name, copy=False)
            for name in names
            if name.startswith("evaluated_winding_") and name.endswith("_points_m")
        ]
        geometry = evidence.get("geometry_sources") or ()
        if (
            evidence.get("available") is True
            and isinstance(current, Real)
            and not isinstance(current, bool)
            and np.isfinite(current)
            and isinstance(evidence.get("current_source"), str)
            and paths
            and len(paths) == len(geometry)
            and all(role in ("fabricated_centroid", "connected_wire") for role in geometry)
            and all(path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2 for path in paths)
            and sum(len(path) - 1 for path in paths) * count <= MAX_PHYSICAL_INTERACTIONS
        ):
            sources.append("routed_interior")
    return tuple(sources)


def evaluate_interiors(record, native, source, cancel_check):
    """Reproduce original samples, then apply the same field operator at new coordinates."""
    _cancel(cancel_check)
    triangles = native.points_m[native.faces]
    a, b, c = np.moveaxis(triangles, 1, 0)
    oriented = np.cross(b - a, c - a)
    double_area = np.linalg.norm(oriented, axis=1)
    if np.any(double_area <= 0) or not np.isfinite(double_area).all():
        raise ValueError("Direct surface quadrature requires nondegenerate finite triangles.")
    points = np.vstack(((4 * a + b + c) / 6, (a + 4 * b + c) / 6, (a + b + 4 * c) / 6))
    normals = np.tile(oriented / double_area[:, None], (3, 1))
    weights = np.tile(double_area / 6, 3)
    metadata = record.metadata_view if hasattr(record, "metadata_view") else record.metadata
    config = CoilGenConfig(
        gauss_order=metadata.get("config", {}).get("gauss_order", 2), parallel_worker_count=1, progress_flag=False
    )
    if source == "sheet":
        kind, physical = "sheet", _sheet_source(record, config, None)
    else:
        kind, physical = "wire", _wire_source(record, SimpleNamespace(key="routed"), None, None)

    check = np.unique(np.linspace(0, len(native.points_m) - 1, min(24, len(native.points_m)), dtype=int))
    reproduced, valid = _physical(native.points_m[check], kind, physical, config, cancel_check)
    if not valid.all():
        raise ValueError("Suppression samples intersect the saved physical source or have an undefined field.")
    expected = native.total_T[check]
    if not np.allclose(reproduced, expected, rtol=1e-8, atol=1e-15 + 1e-8 * np.max(np.abs(expected))):
        raise ValueError("Saved geometry and excitation do not reproduce the recorded suppression field.")
    actual, valid = _physical(points, kind, physical, config, cancel_check)
    if not valid.all():
        raise ValueError("Interior samples intersect the saved physical source or have an undefined field.")
    return replace(
        native,
        points_m=points,
        total_T=actual,
        primary_T=None,
        shield_T=None,
        normals=normals,
        area_weights_m2=weights,
        faces=None,
        sampling="triangle_interior",
        target_T=np.zeros_like(points) if native.target_T is not None and not np.any(native.target_T) else None,
    )
