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
