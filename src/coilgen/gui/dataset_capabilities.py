"""Describe displayable saved evidence without calculating or altering physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FIELD_KEYS = {
    "preview_target": "target_field_T",
    "target": "target_field_T",
    "sheet": "sheet_field_T",
    "validation": "validation_field_T",
    "contour_field": "contour_field_T",
    "contour_residual": "contour_field_T",
    "routed_field": "routed_field_T",
    "routed_residual": "routed_field_T",
    "fabricated": "fabricated_field_T",
}
WIRE_FIELDS = ("contour_field", "contour_residual", "routed_field", "routed_residual")
PATH_PREFIXES = {
    "contours": "contour_",
    "routes": "route_",
    "evaluated_winding": "evaluated_winding_",
    "fabrication": "fabrication_",
    "review_conductor": "review_conductor_",
    "review_electrical": "review_electrical_",
}
REQUIRED_STAGES = {
    "support": "Define support",
    "preview_target": "Define target and ROI",
    "target": "Prepare",
    "stream_function": "Optimize sheet",
    "sheet": "Optimize sheet",
    "contours": "Generate contours",
    "routes": "Connect / sweep",
    "fabrication": "Connect / sweep",
    "swept": "Connect / sweep",
    "validation": "Evaluate",
    "contour_field": "Evaluate",
    "contour_residual": "Evaluate",
    "routed_field": "Evaluate",
    "routed_residual": "Evaluate",
    "evaluated_winding": "Evaluate",
    "review_conductor": "Fabrication review",
    "review_electrical": "Fabrication review",
    "fabricated": "Fabrication review",
}


@dataclass(frozen=True, slots=True)
class DatasetCapability:
    available: bool
    reason: str = ""
    required_stage: str = ""
    invalid: bool = False

    @property
    def explanation(self):
        return self.reason + (f" Required step: {self.required_stage}." if self.required_stage else "")


def _vectors(arrays, name, minimum=1):
    value = arrays.get(name)
    if value is None:
        return f"Missing {name}.", False
    try:
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != 3:
            return f"Invalid {name}: expected Nx3 values.", True
        if len(value) < minimum:
            return f"Empty {name}." if not len(value) else f"Insufficient points in {name}.", False
        if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            return f"Invalid {name}: coordinates/values must be finite real numbers.", True
    except (TypeError, ValueError):
        return f"Invalid {name}: expected finite Nx3 values.", True
    return "", False


def _surface(arrays, vertices="vertices_m", faces="faces"):
    reason, invalid = _vectors(arrays, vertices)
    if reason:
        return reason, invalid
    if faces not in arrays:
        return f"Missing {faces}.", False
    try:
        triangles = np.asarray(arrays[faces])
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            return f"Invalid {faces}: expected Nx3 triangular indices.", True
        if not len(triangles):
            return f"Empty {faces}.", False
        if not np.issubdtype(triangles.dtype, np.integer):
            return f"Invalid {faces}: indices must be integers.", True
        if triangles.min() < 0 or triangles.max() >= len(arrays[vertices]):
            return f"Invalid {faces}: vertex index outside mesh.", True
    except (TypeError, ValueError):
        return f"Invalid {faces}: expected triangular indices.", True
    return "", False


def resolve_dataset(arrays, key, *, original=False):
    """Validate the exact geometry/map/current pair, including legitimate zeros.

    This checks display integrity only. Availability never implies independent
    field acceptance, feasible topology, or manufacturing acceptance.
    """
    required = REQUIRED_STAGES.get(key, "")
    reason, invalid = "", False
    if key == "preview_target":
        reason, invalid = _surface(arrays)
        if reason:
            return DatasetCapability(False, reason, required, invalid)
    if key in FIELD_KEYS:
        points = "validation_points_m" if key == "validation" else "target_points_m"
        if key == "fabricated" and "fabricated_points_m" in arrays:
            points = "fabricated_points_m"
        field = FIELD_KEYS[key]
        if key in WIRE_FIELDS and original:
            field = field.replace("_field_T", "_field_at_stream_current_T")
        for name in (points, field, *(("target_field_T",) if key.endswith("_residual") else ())):
            reason, invalid = _vectors(arrays, name)
            if reason:
                break
        if not reason:
            count = len(arrays[points])
            if len(arrays[field]) != count or key.endswith("_residual") and len(arrays["target_field_T"]) != count:
                reason, invalid = "Field coordinates, values and residual target have incompatible dimensions.", True
    elif key in {"support", "stream_function"}:
        reason, invalid = _surface(arrays)
        if not reason and key == "stream_function":
            values = arrays.get("stream_function_A")
            if values is None:
                reason = "Missing stream_function_A."
            else:
                try:
                    values = np.asarray(values)
                    count = len(arrays["vertices_m"])
                    if (
                        values.shape not in {(count,), (count, 1)}
                        or not np.issubdtype(values.dtype, np.number)
                        or np.iscomplexobj(values)
                        or not np.all(np.isfinite(values))
                    ):
                        reason, invalid = (
                            "Invalid stream function: expected one finite real coefficient per mesh vertex.",
                            True,
                        )
                except (TypeError, ValueError):
                    reason, invalid = "Invalid stream function coefficients.", True
    elif key in PATH_PREFIXES:
        names = sorted(name for name in arrays if name.startswith(PATH_PREFIXES[key]) and name.endswith("_points_m"))
        if not names:
            reason = "No saved paths for this dataset."
        for name in names:
            reason, invalid = _vectors(arrays, name, minimum=2)
            if reason:
                break
    elif key == "swept":
        names = sorted(name for name in arrays if name.startswith("swept_") and name.endswith("_vertices_m"))
        if not names:
            reason = "No saved conductor solids."
        for name in names:
            reason, invalid = _surface(arrays, name, name.removesuffix("_vertices_m") + "_faces")
            if reason:
                break
    else:
        reason = "Unknown dataset."
    return DatasetCapability(not reason, reason, required if reason else "", invalid)


def points_available(arrays, name="target_points_m"):
    return not _vectors(arrays, name)[0]


def dataset_capabilities(arrays, *, original=False):
    """Build a catalog when a source changes, rather than rechecking during camera edits."""
    return {key: resolve_dataset(arrays, key, original=original) for key in REQUIRED_STAGES}
