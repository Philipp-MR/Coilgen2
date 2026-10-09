"""Analysis of immutable saved vector fields; never re-solve or fit wire current.

Statistics describe the selected samples. Saved objective weights are not
assumed to be volume quadrature weights. Scalar solid harmonics describe a
source-free domain only when that physical assumption is independently valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Mapping

import numpy as np

from ..spherical_harmonics import fit_spherical_harmonics_many, spherical_harmonic_coefficient_rows
from .project import _freeze


def _vector_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
        raise ValueError(f"{name} must contain nonempty real Nx3 SI values.")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite SI values.")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class FieldDataset:
    key: str
    label: str
    points_m: np.ndarray
    field_T: np.ndarray
    target_T: np.ndarray | None = None
    weights: np.ndarray | None = None
    scope: str = ""
    provenance: Mapping = field(default_factory=dict)
    target_geometry: Mapping | None = None

    def __post_init__(self):
        from ..persistence import json_value

        if not isinstance(self.key, str) or not self.key or not isinstance(self.label, str):
            raise ValueError("Field source key and label must be strings; key cannot be empty.")
        points, actual = _vector_array(self.points_m, "points_m"), _vector_array(self.field_T, "field_T")
        if actual.shape != points.shape:
            raise ValueError("Field and coordinates must share exactly the same saved sampling.")
        target = None if self.target_T is None else _vector_array(self.target_T, "target_T")
        if target is not None and target.shape != points.shape:
            raise ValueError("Desired field must share exactly the same saved sampling.")
        weights = None
        if self.weights is not None:
            raw = np.asarray(self.weights)
            if raw.dtype.kind not in "fiu" or raw.shape != (len(points),):
                raise ValueError("weights must contain one real scalar per sample.")
            weights = np.array(raw, dtype=float, copy=True)
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must be finite and nonnegative.")
            weights.flags.writeable = False
        for name, value in (("points_m", points), ("field_T", actual), ("target_T", target), ("weights", weights)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "provenance", _freeze(json_value(dict(self.provenance), nonfinite="raise")))
        if self.target_geometry is not None:
            object.__setattr__(
                self, "target_geometry", _freeze(json_value(dict(self.target_geometry), nonfinite="raise"))
            )

    def to_dict(self):
        from ..persistence import json_value

        return json_value({item.name: getattr(self, item.name) for item in fields(self)})


def _saved_sources(record):
    """Descriptors only: keep normal GUI availability updates independent of N."""
    names, metadata = set(record.array_names), record.metadata
    definitions = [
        (
            "target",
            "Saved desired field",
            "target_points_m",
            "target_field_T",
            "target_field_T",
            "desired_target",
            "samples",
            {},
        ),
        (
            "sheet",
            "Continuous sheet · design samples",
            "target_points_m",
            "sheet_field_T",
            "target_field_T",
            "continuous_sheet",
            "sheet",
            {},
        ),
        (
            "validation",
            "Continuous sheet · independent samples",
            "validation_points_m",
            "validation_field_T",
            "validation_target_T",
            "independent_sheet",
            "sheet",
            metadata.get("validation", {}),
        ),
        (
            "fabricated",
            "Realized conductor field · original current",
            "fabricated_points_m" if "fabricated_points_m" in names else "target_points_m",
            "fabricated_field_T",
            "target_field_T",
            "saved_fabricated_field",
            "wire",
            metadata.get("fabricated_physics") or {},
        ),
        (
            "winding_validation",
            "Realized winding · independent samples",
            "winding_validation_points_m",
            "winding_validation_field_T",
            "winding_validation_target_T",
            "independent_realized_winding",
            "wire",
            metadata.get("winding_validation") or {},
        ),
    ]
    for source, title in (("contour", "Unconnected contours"), ("routed", "Evaluated routed winding")):
        evidence = metadata.get("wire_fields", {}).get(source, {})
        definitions.append(
            (
                source,
                title + " · recorded current",
                "target_points_m",
                source + "_field_T",
                "target_field_T",
                "design_grid_wire_field",
                "wire",
                evidence,
            )
        )
        definitions.append(
            (
                source + "_at_stream_current",
                title + " · original contour current",
                "target_points_m",
                source + "_field_at_stream_current_T",
                "target_field_T",
                "design_grid_wire_field",
                "wire",
                evidence.get("at_stream_current", {}),
            )
        )
    for row in metadata.get("requirements_review", {}).get("rows", ()):
        prefix = row.get("array_prefix")
        if row.get("available") is not True or not prefix:
            continue
        label = f"{row['region']} · {row['scenario']} · {row['sampling']} · {row['scope']}"
        representation = "sheet" if row["scope"] == "continuous_sheet" else "wire"
        definitions.append(
            (
                prefix,
                label,
                prefix + "_points_m",
                prefix + "_field_T",
                prefix + "_target_T",
                row["scope"],
                representation,
                {"requirements_row": row},
            )
        )
    for definition in definitions:
        if definition[2] in names and definition[3] in names and definition[7].get("available") is not False:
            yield definition


def field_analysis_available(record) -> bool:
    """Cheap UI gate for a recorded computed vector field, excluding target-only runs."""
    if record is None or not hasattr(record, "array_names") or not callable(getattr(record, "array", None)):
        return False
    for item in _saved_sources(record):
        if item[0] == "target":
            continue
        points, actual = record.array(item[2], copy=False), record.array(item[3], copy=False)
        if (
            points.ndim == 2
            and len(points)
            and points.shape[1] == 3
            and actual.shape == points.shape
            and actual.dtype.kind in "fiu"
        ):
            return True
    return False


def _source_geometry(record, spec, row):
    from .gradient_studies import RigidTransform

    identity = RigidTransform().to_dict()
    if row is not None:
        if spec is None:
            return None
        region = next((item for item in spec.regions if item.name == row["region"]), None)
        if region is None:
            return None
        dataset = region.validation if row["sampling"] == "independent_validation" else region.design
        if dataset is None or dataset.recipe is None:
            return None
        scenario = next((item for item in spec.scenarios if item.name == row["scenario"]), None)
        if scenario is None and row["scenario"] != "Nominal":
            return None
        if row.get("dataset_sha256") is not None and row["dataset_sha256"] != dataset.digest:
            return None
        return {
            "region": dict(dataset.recipe["region"]),
            "registration": dataset.registration.to_dict(),
            "sample_transform": scenario.sample_transform.to_dict() if scenario else identity,
        }
    if record.metadata.get("requirements_review"):
        return None  # The combined named grid has no single authored target volume.
    config = record.metadata.get("config", {})
    if str(config.get("target_field_definition_file", "none")).lower() not in {"none", ""}:
        return None  # Imported desired samples do not inherit unused analytic ROI settings.
    geometry = config.get("target_region_geometry")
    if geometry is None:
        if str(config.get("target_field_definition_file", "none")).lower() not in {"none", ""} or str(
            config.get("target_mesh_file", "none")
        ).lower() not in {"none", ""}:
            return None
        radius = config.get("target_region_radius")
        if not isinstance(radius, Real) or not np.isfinite(radius) or radius <= 0:
            return None
        center = np.zeros(3)
        if config.get("set_roi_into_mesh_center"):
            if "vertices_m" not in record.array_names:
                return None
            center = np.mean(record.array("vertices_m", copy=False), axis=0)
        geometry = {"shape": "sphere", "radius_m": float(radius), "center_m": center.tolist()}
    return {"region": geometry, "registration": identity, "sample_transform": identity}


def source_catalog(record, *, request=None) -> tuple[FieldDataset, ...]:
    """Resolve fields and their own saved desired vectors from a verified RunRecord.

    This function consumes the snapshot only. Optional ``request`` must come
    from ``evidence.load_saved_request``; it supplies authored named ROI shapes.
    Missing desired vectors remain unavailable rather than being reconstructed.
    """
    names = set(record.array_names)
    b0 = record.metadata.get("config", {}).get("b_0_direction", (0, 0, 1))
    spec = None
    if request is not None and request.get("requirements") is not None:
        from .research_requirements import ResearchRequirements

        spec = ResearchRequirements.from_dict(request["requirements"])
    suppression_count = None
    if {"target_points_m", "target_suppressed"} <= names:
        suppressed = record.array("target_suppressed", copy=False)
        if (
            suppressed.shape != (len(record.array("target_points_m", copy=False)),)
            or suppressed.dtype.kind not in "biu"
            or not np.all((suppressed == 0) | (suppressed == 1))
        ):
            raise ValueError("Saved suppression mask must match the exact native target sample order.")
        suppression_count = int(np.count_nonzero(suppressed))
    result = []
    for key, label, points, actual, target, scope, representation, evidence in _saved_sources(record):
        row = evidence.get("requirements_row")
        target_key = target if target in names else None
        if (
            target_key == "target_field_T"
            and points != "target_points_m"
            and (
                "target_points_m" not in names
                or not np.array_equal(record.array(points, copy=False), record.array("target_points_m", copy=False))
            )
        ):
            target_key = None
        provenance = {
            **evidence,
            "points_key": points,
            "field_key": actual,
            "target_key": target_key,
            "physical_representation": representation,
            "current_policy": "original"
            if key.endswith("_at_stream_current") or key in {"fabricated", "winding_validation"} or row
            else "recorded",
            "b0_direction": b0,
            "acceptance_claim": False,
            "sample_scope": "All selected saved samples; no implicit target/suppression filtering",
            "native_sample_count": len(record.array(points, copy=False)),
        }
        selected_suppression = None
        if (points == "target_points_m" or target_key == "target_field_T") and suppression_count is not None:
            selected_suppression = suppression_count
        elif row is not None and isinstance(row.get("protected"), bool):
            selected_suppression = provenance["native_sample_count"] if row["protected"] else 0
        if selected_suppression is not None:
            provenance.update(
                primary_sample_count=provenance["native_sample_count"] - selected_suppression,
                suppression_sample_count=selected_suppression,
                includes_suppression=selected_suppression > 0,
                sample_scope="All saved primary and suppression samples"
                if 0 < selected_suppression < provenance["native_sample_count"]
                else "Saved protected-region samples"
                if selected_suppression
                else "Saved primary target samples",
            )
        weights = (
            record.array("target_weights", copy=False)
            if points == "target_points_m" and "target_weights" in names
            else None
        )
        result.append(
            FieldDataset(
                key,
                label,
                record.array(points, copy=False),
                record.array(actual, copy=False),
                record.array(target_key, copy=False) if target_key is not None else None,
                weights,
                scope,
                provenance,
                _source_geometry(record, spec, row),
            )
        )
    return tuple(result)


def _weights(dataset, sampling):
    if sampling == "equal":
        return np.ones(len(dataset.points_m)) / len(dataset.points_m), "Equal sample statistics; not a volume integral"
    if sampling != "weighted":
        raise ValueError("sampling must be equal or weighted.")
    if dataset.weights is None or not np.any(dataset.weights > 0):
        raise ValueError("Positive saved sample weights are unavailable for this source.")
    weights = dataset.weights / np.max(dataset.weights)
    return weights / weights.sum(), "Positive saved sample weights; not assumed to be volume quadrature"


def _scalar(field, component, direction):
    if component in {"Bx", "By", "Bz"}:
        return field[:, {"Bx": 0, "By": 1, "Bz": 2}[component]]
    if component in {"magnitude", "|B|"}:
        return np.hypot.reduce(field, axis=1)
    if component != "Bparallel":
        raise ValueError("component must be Bx, By, Bz, Bparallel or magnitude.")
    raw = np.asarray(direction)
    if raw.dtype.kind not in "fiu" or raw.shape != (3,) or not np.all(np.isfinite(raw)) or not np.any(raw):
        raise ValueError("Bparallel requires a finite nonzero real B0 direction.")
    direction = np.asarray(raw, dtype=float) / np.max(np.abs(raw))
    direction /= np.linalg.norm(direction)
    return field @ direction


def _statistics(values, weights):
    active = weights > 0
    values, weights = np.asarray(values)[active], weights[active]
    if not np.all(np.isfinite(values)):
        raise ValueError("Derived field metric exceeds finite SI reporting range.")
    scale = float(np.max(np.abs(values)))
    scaled = values / scale if scale else values
    mean = float(weights @ scaled)
    absolute = np.abs(values)
    if np.allclose(weights, weights[0], rtol=1e-14, atol=0):
        p95 = float(np.percentile(absolute, 95))
        quantile = "linear equal-sample absolute-value percentile"
    else:
        order = np.argsort(absolute, kind="stable")
        positions = np.cumsum(weights[order]) - weights[order] / 2
        p95 = float(np.interp(0.95, positions, absolute[order]))
        quantile = "linear interpolation at weighted CDF midpoints of absolute values"
    return {
        "rms": scale * float(np.sqrt(weights @ (scaled**2))),
        "peak": scale,
        "std": scale * float(np.sqrt(weights @ ((scaled - mean) ** 2))),
        "mean": scale * mean,
        "p95": p95,
        "p95_convention": quantile,
        "sample_count": len(values),
    }


@dataclass(frozen=True, slots=True)
class MetricMap:
    values: np.ndarray
    unit: str
    label: str
    summary: Mapping
    normalization_scale_T: float | None
    sampling_convention: str
    component: str
    metric: str

    def to_dict(self):
        from ..persistence import json_value

        return json_value({item.name: getattr(self, item.name) for item in fields(self)})


def metric_map(
    dataset: FieldDataset, component="Bz", metric="field", *, b0_direction=None, sampling="equal", normalization="none"
) -> MetricMap:
    """Global normalization is optional; individual desired zeros never divide.

    ``vector_error`` is |B - Bdesired|; a magnitude component instead reports
    the scalar magnitude difference |B| - |Bdesired|. Percent maps divide by one
    selected desired RMS/peak over the entire positively weighted source.
    """
    weights, convention = _weights(dataset, sampling)
    direction = dataset.provenance.get("b0_direction", (0, 0, 1)) if b0_direction is None else b0_direction
    actual = _scalar(dataset.field_T, component, direction)
    target = None if dataset.target_T is None else _scalar(dataset.target_T, component, direction)
    if metric not in {"field", "target", "signed_residual", "absolute_error", "vector_error"}:
        raise ValueError("Unknown field metric.")
    if (metric != "field" or normalization != "none") and target is None:
        raise ValueError("This saved source has no paired desired field.")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = None if target is None else actual - target
        values = (
            actual
            if metric == "field"
            else target
            if metric == "target"
            else residual
            if metric == "signed_residual"
            else np.abs(residual)
            if metric == "absolute_error"
            else np.hypot.reduce(dataset.field_T - dataset.target_T, axis=1)
        )
    desired = np.hypot.reduce(dataset.target_T, axis=1) if metric == "vector_error" else target
    summary = _statistics(values, weights)
    if residual is not None:
        errors = values if metric == "vector_error" else residual
        error_stats, target_stats = _statistics(errors, weights), _statistics(desired, weights)
        summary.update(
            error_rms_T=error_stats["rms"],
            error_peak_T=error_stats["peak"],
            error_std_T=error_stats["std"],
            error_p95_T=error_stats["p95"],
            target_rms_T=target_stats["rms"],
            target_peak_T=target_stats["peak"],
            relative_error_rms=error_stats["rms"] / target_stats["rms"] if target_stats["rms"] > 0 else None,
        )
    scale, unit = None, "T"
    if normalization != "none":
        if normalization not in {"target_rms", "target_peak"}:
            raise ValueError("normalization must be none, target_rms or target_peak.")
        scale = summary["target_rms_T" if normalization == "target_rms" else "target_peak_T"]
        if scale == 0:
            raise ValueError("A zero desired field cannot define global normalization; use SI error metrics.")
        with np.errstate(over="ignore", invalid="ignore"):
            values = (values / scale) * 100
        summary.update(_statistics(values, weights))
        unit = "%"
    values = np.array(values, copy=True)
    values.flags.writeable = False
    summary.update(unit=unit, sampling_convention=convention, normalization=normalization, acceptance_claim=False)
    for key in ("sample_scope", "primary_sample_count", "suppression_sample_count", "includes_suppression"):
        if key in dataset.provenance:
            summary[key] = dataset.provenance[key]
    label = {
        "field": component,
        "target": "Desired " + component,
        "signed_residual": "Signed " + component + " residual",
        "absolute_error": "Absolute " + component + " error",
        "vector_error": "Vector residual magnitude",
    }[metric]
    return MetricMap(values, unit, label, _freeze(summary), scale, convention, component, metric)


def _checkpoint(progress, cancel_check, percent, message):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Field analysis cancelled.")
    if progress is not None:
        progress(percent, message)


def analyze_harmonics(
    dataset: FieldDataset,
    *,
    component="Bz",
    max_order=4,
    center_m=None,
    radius_m=None,
    sampling="equal",
    b0_direction=None,
    rcond=None,
    progress=None,
    cancel_check=None,
) -> dict:
    """Fit scalar Cartesian/B0 projections on one common sample support.

    Magnitude and vector-error norms are nonlinear scalars and are not accepted
    as harmonic magnetic field components. Rank/condition do not establish a
    source-free fitting volume, and no fit residual becomes engineering acceptance.
    """
    if component in {"magnitude", "|B|"}:
        raise ValueError("Magnetic field magnitude is not a scalar harmonic field component.")
    if (
        isinstance(max_order, (bool, np.bool_))
        or not isinstance(max_order, (int, np.integer))
        or not 0 <= max_order <= 12
    ):
        raise ValueError("max_order must be an integer from 0 to 12.")
    if rcond is not None and (
        isinstance(rcond, bool) or not isinstance(rcond, Real) or not np.isfinite(rcond) or rcond < 0
    ):
        raise ValueError("rcond must be a finite nonnegative scalar or None.")
    if radius_m is not None and (
        isinstance(radius_m, bool) or not isinstance(radius_m, Real) or not np.isfinite(radius_m) or radius_m <= 0
    ):
        raise ValueError("radius_m must be finite and positive.")
    if center_m is not None:
        center = np.asarray(center_m)
        if center.dtype.kind not in "fiu" or center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_m must contain three finite real SI coordinates.")
    weights, convention = _weights(dataset, sampling)
    active = weights > 0
    count, terms = int(np.count_nonzero(active)), (max_order + 1) ** 2
    if count * terms * 8 > 128 * 1024**2:
        raise ValueError("The harmonic basis exceeds 128 MiB; reduce order or use a smaller explicitly sampled source.")
    direction = dataset.provenance.get("b0_direction", (0, 0, 1)) if b0_direction is None else b0_direction
    values = {"field": _scalar(dataset.field_T, component, direction)}
    if dataset.target_T is not None:
        values["target"] = _scalar(dataset.target_T, component, direction)
        with np.errstate(over="ignore", invalid="ignore"):
            values["residual"] = values["field"] - values["target"]
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("Derived scalar fields exceed finite SI reporting range.")
    _checkpoint(progress, cancel_check, 10, "Fit shared solid harmonic basis")
    fits = fit_spherical_harmonics_many(
        dataset.points_m,
        values,
        max_order=max_order,
        weights=weights,
        center=center_m,
        radius=radius_m,
        rcond=rcond,
        unit="T",
    )
    _checkpoint(progress, cancel_check, 90, "Summarize harmonic rank and coefficients")
    reference = fits["field"]
    singular = reference.singular_values
    condition = (
        float(singular[0] / singular[-1])
        if reference.rank == terms and len(singular) == terms and singular[-1] > 0
        else float("inf")
    )
    if not all(np.all(np.isfinite(fit.coefficients)) for fit in fits.values()):
        raise ValueError("Harmonic coefficients exceed finite SI reporting range.")
    rows = {
        key: spherical_harmonic_coefficient_rows(fit, min_contribution_fraction=0, max_terms=None)
        for key, fit in fits.items()
    }
    result = {
        "fits": fits,
        "coefficient_rows": rows,
        "rank": reference.rank,
        "condition": condition,
        "term_count": terms,
        "sample_count": count,
        "sample_indices": np.flatnonzero(active),
        "center_m": reference.center.copy(),
        "radius_m": reference.radius,
        "component": component,
        "max_order": max_order,
        "sampling_convention": convention,
        "full_rank": reference.rank == terms,
        "source_free_verified": False,
        "sourcefree_caveat": "Cartesian magnetic field components satisfy Laplace's equation only in a source-free homogeneous domain. This sample fit does not verify that domain, Maxwell consistency, unsampled errors or engineering acceptance.",
        "acceptance_claim": False,
        "source_key": dataset.key,
        "sample_scope": dataset.provenance.get("sample_scope", "All selected saved samples"),
        "primary_sample_count": dataset.provenance.get("primary_sample_count"),
        "suppression_sample_count": dataset.provenance.get("suppression_sample_count"),
        "diagnostics": []
        if reference.rank == terms
        else [
            "The selected sampling does not identify every requested harmonic coefficient; the fit is rank deficient."
        ],
    }
    if dataset.provenance.get("primary_sample_count", 0) and dataset.provenance.get("suppression_sample_count", 0):
        result["diagnostics"].append(
            "Primary and suppression samples are fitted together. They may span current-carrying material; one sample fit does not establish a source-free expansion ball."
        )
    _checkpoint(progress, cancel_check, 100, "Harmonic analysis complete")
    return result
