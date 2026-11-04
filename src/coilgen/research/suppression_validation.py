"""Physical suppression residuals on the original design surface."""

import numpy as np

from ..field_components import field_component_values
from ..optimization import (
    _estimate_suppression_normals,
    _normalise_suppression_surface_weighting,
    _suppression_component,
)


def sampled_suppression(result, fabricated_field):
    """Use the design's selected component and physical T limits, without fitting a current."""
    config, target = result.input_data, result.target_field
    rms_limit = config.suppression_field_rms_limit_T if config.suppression_field_rms_limit_T > 0 else None
    peak_limit = config.suppression_field_peak_limit_T if config.suppression_field_peak_limit_T > 0 else None
    if rms_limit is None and peak_limit is None:
        return None
    component = _suppression_component(config)
    weighting = _normalise_suppression_surface_weighting(config)
    positive = np.isfinite(target.weights) & (target.weights > 0)
    mask = (
        target.suppressed & np.any(positive.reshape(len(target.coords), -1), axis=1)
        if len(target.coords)
        else target.suppressed
    )
    count = int(mask.sum())
    area = target.suppression_area_weights
    weights = np.ones(count) if weighting == "uniform" else None if area is None else np.asarray(area)[mask]
    valid_weights = bool(weights is not None and np.isfinite(weights).all() and np.all(weights > 0))

    def measure(field):
        review = dict(available=False, rms_T=None, peak_T=None, rms_passed=None, peak_passed=None, reason="")
        if not count:
            return {**review, "reason": "No active suppression design samples."}
        if field is None or np.shape(field) != target.b.shape:
            return {**review, "reason": "The field does not cover the recorded target coordinates."}
        residual = np.asarray(field)[mask] - target.b[mask]
        if not np.all(np.isfinite(residual)):
            return {**review, "reason": "The suppression field or target contains nonfinite values."}
        if component in {"vector", "Bmag", "components"}:
            absolute = np.linalg.norm(residual, axis=1)
            peak = np.abs(residual).max() if component == "components" else absolute.max()
        else:
            if component == "Bnormal":
                source = config.suppression_normal_source.strip().lower()
                if source in {"mesh", "target", "metadata"}:
                    if target.suppression_normals is None:
                        return {**review, "reason": "No recorded suppression surface normals."}
                    normals = np.asarray(target.suppression_normals)[mask]
                else:
                    model = "radial" if source in {"radial", "cylindrical"} else "box"
                    normals = _estimate_suppression_normals(target.coords[mask], model=model)
                lengths = np.linalg.norm(normals, axis=1)
                if not np.isfinite(lengths).all() or np.any(lengths <= 1e-15):
                    return {**review, "reason": "Suppression surface normals are invalid."}
                absolute = np.abs(np.einsum("ij,ij->i", residual, normals / lengths[:, None]))
            else:
                absolute = np.abs(field_component_values(residual, component, b0_direction=config.b_0_direction))
            peak = absolute.max()
        rms = float(np.sqrt(np.average(absolute**2, weights=weights))) if valid_weights else None
        return {
            "available": valid_weights or rms_limit is None,
            "rms_T": rms,
            "peak_T": float(peak),
            "rms_passed": bool(rms <= rms_limit) if rms is not None and rms_limit is not None else None,
            "peak_passed": bool(peak <= peak_limit) if peak_limit is not None else None,
            "reason": "" if valid_weights else "Physical RMS unavailable: missing or invalid suppression vertex areas.",
        }

    return {
        "scope": "suppression_design_samples",
        "independent": False,
        "component": component,
        "weighting": weighting,
        "sample_count": count,
        "rms_limit_T": rms_limit,
        "peak_limit_T": peak_limit,
        "sheet": measure(getattr(result, "b_field_opt_sf", None)),
        "fabricated": measure(fabricated_field),
    }
