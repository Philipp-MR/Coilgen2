"""Labels and matched color limits for saved-field views."""

from collections.abc import Mapping
from numbers import Real
import math

import numpy as np


def plot_field_source(dataset):
    for source in ("contour", "routed", "winding_validation"):
        if dataset in {f"{source}_field", f"{source}_residual"}:
            return source
    return dataset


def field_provenance(source_key, metadata, sample_count, *, evidence=None):
    """Describe the saved samples and current that produced this field."""
    key = plot_field_source(source_key)
    original = key.endswith("_at_stream_current")
    source = key.removesuffix("_at_stream_current")
    names = {
        "preview_target": "Model target preview",
        "target": "Saved target",
        "sheet": "Stream function field",
        "validation": "Independent stream function field",
        "contour": "Unconnected contours",
        "routed": "Evaluated routed winding",
        "fabricated": "Reviewed conductor field",
        "winding_validation": "Independent realized winding field",
    }
    saved = {}
    if source in {"contour", "routed"}:
        saved = (metadata.get("wire_fields") or {}).get(source) or {}
        if original:
            saved = saved.get("at_stream_current") or {}
    elif source == "fabricated":
        saved = metadata.get("fabricated_physics") or {}
    elif source == "winding_validation":
        saved = metadata.get(source) or {}
    saved = {**saved, **(evidence or {})}
    row = saved.get("requirements_row") or {}
    grid = (
        str(row.get("sampling", "Saved")).capitalize() + " grid"
        if row
        else "Independent grid"
        if source in {"validation", "winding_validation"}
        else "Preview grid"
        if source == "preview_target"
        else "Design grid"
    )
    parts = [names.get(source, str(row.get("region") or "Saved field")), f"{grid} · {sample_count:,} samples"]
    if row.get("scenario"):
        parts.insert(1, str(row["scenario"]))
    if source not in {"target", "preview_target", "sheet", "validation"} and (
        source in names or saved.get("physical_representation") == "wire"
    ):
        circuit_currents = saved.get("currents_A")
        values = circuit_currents.values() if isinstance(circuit_currents, Mapping) else (saved.get("current_A"),)
        currents = sorted(
            {
                float(value)
                for value in values
                if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            }
        )
        if not currents:
            parts.insert(1, "Current unrecorded")
        else:
            value = f"{currents[0]:.6g} A" if len(currents) == 1 else f"{currents[0]:.6g} to {currents[-1]:.6g} A"
            current_source = saved.get("current_source", "")
            if saved.get("current_refitted") is True or "refitted" in str(current_source):
                policy = "Refitted current"
            elif original or saved.get("current_refitted") is False or str(current_source).startswith("original"):
                policy = "Original current"
            else:
                policy = "Recorded current"
            parts.insert(1, f"{policy} {value}")
    return " · ".join(parts)


def paired_color_limits(actual, target, *, signed=True, minimum=1e-12):
    """Give two comparable maps one zero-referenced domain in display units."""
    bound = minimum
    for values in (actual, target):
        if values is None:
            continue
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size:
            bound = max(bound, float(np.max(np.abs(finite))))
    return (-bound if signed else 0.0), bound


def field_comparison_values(field_T, target_T, component, *, b0_direction=(0, 0, 1), valid=None):
    """Target, actual and actual-minus-target on identical supported samples.

    For magnitude, the residual is |Bactual| - |Btarget|, not |ΔB|.
    Native samples and rectangular grids keep their original leading shape.
    """
    from ..field_components import field_component_values

    actual = np.asarray(field_T, dtype=float)
    target = np.full_like(actual, np.nan) if target_T is None else np.asarray(target_T, dtype=float)
    if actual.ndim < 2 or actual.shape[-1] != 3 or actual.shape != target.shape:
        raise ValueError("Field comparison requires matching vector arrays with a final dimension of 3.")
    mask = np.all(np.isfinite(actual), axis=-1) & np.all(np.isfinite(target), axis=-1)
    if valid is not None:
        if np.shape(valid) != mask.shape:
            raise ValueError("Comparison mask must match the field grid.")
        mask &= np.asarray(valid, dtype=bool)

    def project(vectors):
        if component in {"magnitude", "|B|", "Bmag"}:
            scalar = np.hypot.reduce(vectors, axis=-1)
        else:
            scalar = field_component_values(
                vectors.reshape(-1, 3),
                "BdotB0" if component == "Bparallel" else component,
                b0_direction=b0_direction,
            ).reshape(mask.shape)
        return np.where(mask, scalar, np.nan)

    actual, target = project(actual), project(target)
    residual = actual - target
    for values in (target, actual, residual):
        values.setflags(write=False)
    return target, actual, residual
