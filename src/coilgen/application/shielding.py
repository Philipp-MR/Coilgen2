"""Applied fields on the executed run's native suppression samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .service import RunRecord


@dataclass(frozen=True, slots=True)
class ShieldingSurfaceFields:
    points_m: np.ndarray
    total_T: np.ndarray
    primary_T: np.ndarray | None
    shield_T: np.ndarray | None
    normals: np.ndarray | None
    area_weights_m2: np.ndarray | None
    faces: np.ndarray | None
    current_A: float | None
    current_source: str
    target_T: np.ndarray | None
    sampling: str = "native"


def _suppression_mask(record):
    names = set(getattr(record, "array_names", ()))
    if not {"target_points_m", "target_suppressed"} <= names or not callable(getattr(record, "array", None)):
        return None
    points = record.array("target_points_m", copy=False)
    mask = record.array("target_suppressed", copy=False)
    if points.ndim != 2 or points.shape[1] != 3 or mask.shape != (len(points),):
        return None
    if mask.dtype.kind not in "biu" or (mask.dtype.kind != "b" and not np.isin(mask, (0, 1)).all()):
        return None
    return mask.astype(bool, copy=False) if mask.any() else None


def _fabricated_currents(record, metadata):
    """Require a closed, identified electrical path for every recorded drive."""
    from .gradient_studies import _saved_winding

    evidence = metadata.get("fabricated_physics") or {}
    currents = evidence.get("currents_A")
    if evidence.get("current_source") != "original_contour_current_no_refit" or not isinstance(currents, Mapping):
        return None
    if not np.array_equal(record.array("fabricated_points_m", copy=False), record.array("target_points_m", copy=False)):
        return None
    try:
        _, drives, _, complete = _saved_winding(record, "original")
    except (KeyError, TypeError, ValueError):
        return None
    if not complete or not drives or not all(type(value) in (int, float) for value in currents.values()):
        return None
    return drives if drives == currents else None


def _source_drives(record, mask):
    names = set(record.array_names)
    metadata = record.metadata_view if hasattr(record, "metadata_view") else record.metadata
    wire = metadata.get("wire_fields") or {}
    drives = {
        source: (None, "solved_stream_function")
        if source == "sheet"
        else (wire.get(source, {}).get("current_A"), wire.get(source, {}).get("current_source", "unrecorded"))
        for source in ("sheet", "contour", "routed")
        if f"{source}_field_T" in names and record.array(f"{source}_field_T", copy=False).shape == (len(mask), 3)
    }
    if {"fabricated_field_T", "fabricated_points_m"} <= names and record.array(
        "fabricated_field_T", copy=False
    ).shape == (len(mask), 3):
        currents = _fabricated_currents(record, metadata)
        if currents:
            distinct = set(currents.values())
            drives["fabricated"] = (
                (next(iter(distinct)), "original_contour_current_no_refit")
                if len(distinct) == 1
                else (None, "original per-body currents (no refit)")
            )
    return drives


def shielding_sources(record: RunRecord) -> tuple[str, ...]:
    """Return only recorded vector fields covering the same native target rows."""
    from .shielding_interior import interior_sources

    mask = _suppression_mask(record)
    if mask is None:
        return ()
    drives = _source_drives(record, mask)
    return (*drives, *interior_sources(record, drives))


def shielding_surface_fields(record: RunRecord, source="sheet", *, cancel_check=None) -> ShieldingSurfaceFields:
    """Read colocated vectors in world coordinates at the recorded excitation.

    The total is the applied coil field. The primary baseline is the primary
    contribution of this coupled design with the shield drive removed. Neither
    field is normalized or refitted to the ROI.
    """
    mask = _suppression_mask(record)
    drives = _source_drives(record, mask) if mask is not None else {}
    interior = source.endswith("_interior")
    if interior:
        from .shielding_interior import interior_sources

        if source not in interior_sources(record, drives):
            raise ValueError(
                "Direct surface quadrature is unavailable for this saved source or exceeds its work budget."
            )
        source = source.removesuffix("_interior")
    if source not in drives:
        raise ValueError(f"No saved {source} vector field on suppression samples.")
    names = set(record.array_names)
    count = int(np.count_nonzero(mask))

    def samples(key, *, subset=False):
        if key not in names:
            return None
        value = record.array(key, copy=False)
        if value.shape != (count if subset else len(mask), 3) or value.dtype.kind not in "fiu":
            raise ValueError(f"Saved {key} must match the native sample coordinates.")
        value = value if subset else value[mask]
        if not np.isfinite(value).all():
            raise ValueError(f"Saved {key} contains nonfinite suppression samples.")
        return value

    points = samples("target_points_m")
    total = samples(f"{source}_field_T")
    primary = samples(f"shielding_{source}_primary_T", subset=True)
    shield = samples(f"shielding_{source}_shield_T", subset=True)
    if (primary is None) != (shield is None):
        raise ValueError("Incomplete primary and active-shield field decomposition.")
    if primary is not None:
        scale = max(float(np.max(np.abs(primary))), float(np.max(np.abs(shield))))
        if not np.allclose(primary + shield, total, rtol=1e-10, atol=1e-11 * scale):
            raise ValueError("Recorded primary and shield fields disagree with the total at the saved excitation.")
    normals = samples("shielding_surface_normals", subset=True)
    if normals is not None:
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / lengths[:, None] if np.all(lengths > 0) else None
    areas = None
    if "shielding_surface_area_weights_m2" in names:
        areas = record.array("shielding_surface_area_weights_m2", copy=False)
        if areas.shape != (count,) or not np.isfinite(areas).all() or np.any(areas < 0):
            raise ValueError("Suppression vertex areas must be finite, nonnegative and match the saved sample order.")
        if np.any(areas == 0):
            areas = None  # Some imported samples have no assigned area.
    faces = None
    if "shielding_surface_faces" in names:
        faces = record.array("shielding_surface_faces", copy=False)
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or not faces.size
            or faces.min() < 0
            or faces.max() >= count
        ):
            raise ValueError("Suppression surface triangles must index the saved suppression samples.")
    native = ShieldingSurfaceFields(
        points,
        total,
        primary,
        shield,
        normals,
        areas,
        faces,
        *drives[source],
        samples("target_field_T"),
    )
    if interior:
        from .shielding_interior import evaluate_interiors

        return evaluate_interiors(record, native, source, cancel_check)
    return native
