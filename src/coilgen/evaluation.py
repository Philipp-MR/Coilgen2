from __future__ import annotations

import numpy as np

from .config import CoilGenConfig
from .conductor_geometry import conductor_trace
from .contours import contour_evaluation_points
from .engineering_metrics import fit_bz_field_quality
from .field_components import component_sample_values, field_component_values, normalize_field_component
from .models import CoilPart, TargetField
from .metrics import dsv_target_mask, fit_target_mode_current
from .parallel import parallel_map, resolve_worker_count, should_parallelize
from .physics import biot_savart_wire_path
from .progress import ProgressReporter
from .routing_field_objective import longitudinal_field_distortion


def layout_field_geometry(part: CoilPart, config: CoilGenConfig | None = None):
    """Use the finished conductor centroid whenever postprocessing produced it."""
    if not bool(getattr(config, "skip_postprocessing", False)):
        return conductor_trace(part)
    return None, "unconnected_contours"


def evaluate_field_errors(
    parts: list[CoilPart],
    config: CoilGenConfig,
    target_field: TargetField,
    sf_b_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], float, np.ndarray, TargetField]:
    worker_count = resolve_worker_count(config)
    reporter = ProgressReporter.from_config(config, label="Field evaluation")
    combined_loops_per_amp = calculate_unconnected_contour_field(
        parts,
        config,
        target_field,
        worker_count=worker_count,
        reporter=reporter,
    )
    for part_ind, part in enumerate(parts, start=1):
        points, geometry_source = layout_field_geometry(part, config)
        if points is not None:
            part.field_by_layout = biot_savart_wire_path(
                points,
                target_field.coords,
                chunk_size=int(config.biot_savart_chunk_size),
                worker_count=worker_count,
                min_work_size=int(config.parallel_min_work_size),
                max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
                progress=reporter,
                progress_label=f"part {part_ind}/{len(parts)} {geometry_source} chunks",
            )
        else:
            part.field_by_layout = part.field_by_loops.copy()

    combined_layout_per_amp = sum(
        (part.field_by_layout for part in parts), start=np.zeros_like(parts[0].field_by_layout)
    )

    contour_step = parts[0].contour_step or 1.0
    sf_b_field_1a = sf_b_field / contour_step
    target_field_1a = TargetField(
        coords=target_field.coords.copy(),
        b=target_field.b / contour_step,
        weights=target_field.weights.copy(),
        suppressed=target_field.suppressed.copy(),
        target_field_group_inds=target_field.target_field_group_inds.copy(),
        target_gradient_dbdxyz=None
        if target_field.target_gradient_dbdxyz is None
        else target_field.target_gradient_dbdxyz.copy(),
        suppression_normals=None
        if target_field.suppression_normals is None
        else target_field.suppression_normals.copy(),
        suppression_area_weights=None
        if target_field.suppression_area_weights is None
        else target_field.suppression_area_weights.copy(),
        target_diagnostics=dict(target_field.target_diagnostics),
    )

    component = getattr(config, "target_field_component", "Bz")
    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    target_values = field_component_values(target_field.b, component, b0_direction=b0_direction)
    dsv_mask = dsv_target_mask(target_field)
    component_dsv_mask = component_sample_values(dsv_mask, component).astype(bool)
    component_weights = component_sample_values(np.asarray(target_field.weights, dtype=float), component)
    opt_current_layout = _field_evaluation_current(
        combined_layout_per_amp,
        target_values,
        float(config.target_gradient_strength),
        component_dsv_mask,
        contour_step=float(contour_step),
        source=field_evaluation_current_source(config),
        component=component,
        b0_direction=b0_direction,
        sample_weights=component_weights,
    )
    combined_layout = combined_layout_per_amp * opt_current_layout
    combined_loops = combined_loops_per_amp * opt_current_layout
    combined_layout_at_sf_current = combined_layout_per_amp * contour_step
    combined_loops_at_sf_current = combined_loops_per_amp * contour_step

    errors = _field_error_values(
        target_values,
        field_component_values(sf_b_field, component, b0_direction=b0_direction),
        field_component_values(combined_layout, component, b0_direction=b0_direction),
        field_component_values(combined_loops, component, b0_direction=b0_direction),
        layout_stream_z=field_component_values(combined_layout_at_sf_current, component, b0_direction=b0_direction),
        loop_stream_z=field_component_values(combined_loops_at_sf_current, component, b0_direction=b0_direction),
        sample_mask=component_dsv_mask,
    )
    return (
        combined_layout,
        combined_loops,
        combined_layout_per_amp,
        combined_loops_per_amp,
        errors,
        opt_current_layout,
        sf_b_field_1a,
        target_field_1a,
    )


def final_field_diagnostics(parts, config, target_field, layout_per_amp, loops_per_amp, reported_current_a):
    """Identify final field geometry/current and report routing error without a refit.

    Historical target-current reporting remains configured independently. The
    routing comparison always uses the common original contour-step current;
    unavailable or inconsistent current context is reported as unchecked.
    """
    sources = tuple(layout_field_geometry(part, config)[1] for part in parts)
    source = sources[0] if sources and len(set(sources)) == 1 else "mixed" if sources else "unavailable"
    current_source = field_evaluation_current_source(config)
    stream_aliases = {"stream_function", "stream", "contour_step", "sf"}
    diagnostics = {
        "field_evaluation_geometry_source": source,
        "field_evaluation_part_geometry_sources": ";".join(sources),
        "field_evaluation_requested_current_source": str(config.field_evaluation_current_source),
        "field_evaluation_current_source": current_source,
        "field_evaluation_current_a": float(reported_current_a),
        "field_evaluation_current_refitted": current_source.strip().lower().replace("-", "_") not in stream_aliases,
        "layout_target_source": source,
        "routing_distortion_checked": False,
        "routing_distortion_reference": "original_oriented_contours",
        "routing_distortion_geometry_source": source,
        "routing_distortion_current_refitted": False,
    }
    try:
        current = float(parts[0].contour_step)
        if (
            not np.isfinite(current)
            or current == 0
            or any(
                part.contour_step is None
                or not np.isfinite(part.contour_step)
                or not np.isclose(part.contour_step, current, rtol=1e-12, atol=0.0)
                for part in parts
            )
        ):
            raise ValueError("A common finite nonzero contour-step current is required.")
        values = longitudinal_field_distortion(
            loops_per_amp, layout_per_amp, target_field, config.b_0_direction, current
        )
    except (IndexError, TypeError, ValueError) as error:
        diagnostics["routing_distortion_reason"] = str(error)
        return diagnostics
    diagnostics.update(
        routing_distortion_checked=True,
        routing_distortion_reason="",
        routing_distortion_current_a=values["current_a"],
        routing_distortion_longitudinal_rms_T=values["rms_T"],
        routing_distortion_longitudinal_peak_T=values["peak_T"],
        routing_distortion_active_sample_count=values["active_sample_count"],
    )
    return diagnostics


def layout_target_residual_diagnostics(
    field_by_layout: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
) -> dict[str, float | int | bool | str]:
    """Validate the realized layout against its own absolute field specification."""

    component = getattr(config, "target_field_component", "Bz")
    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    target_values = field_component_values(target_field.b, component, b0_direction=b0_direction).reshape(-1)
    layout_values = field_component_values(field_by_layout, component, b0_direction=b0_direction).reshape(-1)
    mask = component_sample_values(dsv_target_mask(target_field), component).astype(bool).reshape(-1)
    weights = component_sample_values(np.asarray(target_field.weights, dtype=float), component).reshape(-1)
    active = mask & np.isfinite(weights) & (weights > 0.0)
    finite = active & np.isfinite(target_values) & np.isfinite(layout_values)
    invalid_sample_count = int(np.count_nonzero(active & ~finite))
    metric_key = str(getattr(config, "target_error_constraint_metric", "std"))
    metric_key = metric_key.strip().lower().replace("-", "_")
    std_limit = max(float(getattr(config, "layout_target_error_std_limit_T", 0.0)), 0.0)
    rms_limit = max(float(getattr(config, "layout_target_error_rms_limit_T", 0.0)), 0.0)
    if metric_key in {"std", "standard_deviation", "homogeneity"} or (std_limit > 0.0 and rms_limit <= 0.0):
        metric = "std"
        limit = std_limit
    elif metric_key in {"rms", "root_mean_square"} or rms_limit > 0.0:
        metric = "rms"
        limit = rms_limit
    else:
        metric = ""
        limit = 0.0
    acceptance = max(float(getattr(config, "layout_target_error_acceptance_absolute_T", 0.0)), 0.0)
    checked = bool(limit > 0.0)
    if not np.any(finite):
        return {
            "layout_target_checked": checked,
            "layout_target_met": not checked,
            "layout_target_metric": metric,
            "layout_target_residual_std_T": float("nan"),
            "layout_target_residual_rms_T": float("nan"),
            "layout_target_metric_T": float("nan"),
            "layout_target_limit_T": float(limit) if checked else float("nan"),
            "layout_target_effective_limit_T": float(limit + acceptance) if checked else float("nan"),
            "layout_target_acceptance_absolute_T": float(acceptance),
            "layout_target_metric_ratio": float("nan"),
            "layout_target_sample_count": 0,
            "layout_target_invalid_sample_count": invalid_sample_count,
            "layout_target_source": "connected_wire" if not config.skip_postprocessing else "unconnected_contours",
        }
    residual = layout_values[finite] - target_values[finite]
    active_weights = weights[finite]
    weight_sum = float(np.sum(active_weights))
    residual_mean = float(np.dot(active_weights, residual) / weight_sum)
    residual_std = float(np.sqrt(np.dot(active_weights, (residual - residual_mean) ** 2) / weight_sum))
    residual_rms = float(np.sqrt(np.dot(active_weights, residual**2) / weight_sum))
    metric_value = residual_std if metric == "std" else residual_rms
    effective_limit = limit + acceptance
    met = bool(
        not checked or (invalid_sample_count == 0 and np.isfinite(metric_value) and metric_value <= effective_limit)
    )
    return {
        "layout_target_checked": checked,
        "layout_target_met": met,
        "layout_target_metric": metric,
        "layout_target_residual_std_T": residual_std,
        "layout_target_residual_rms_T": residual_rms,
        "layout_target_metric_T": float(metric_value),
        "layout_target_limit_T": float(limit) if checked else float("nan"),
        "layout_target_effective_limit_T": float(effective_limit) if checked else float("nan"),
        "layout_target_acceptance_absolute_T": float(acceptance),
        "layout_target_metric_ratio": float(metric_value / max(effective_limit, 1.0e-30)) if checked else float("nan"),
        "layout_target_sample_count": int(np.count_nonzero(finite)),
        "layout_target_invalid_sample_count": invalid_sample_count,
        "layout_target_source": "connected_wire" if not config.skip_postprocessing else "unconnected_contours",
    }


def calculate_unconnected_contour_field(
    parts: list[CoilPart],
    config: CoilGenConfig,
    target_field: TargetField,
    *,
    worker_count: int | None = None,
    reporter: ProgressReporter | None = None,
    progress_label: str = "Field evaluation",
) -> np.ndarray:
    if not parts:
        return np.zeros((len(target_field.coords), 3), dtype=float)
    resolved_workers = resolve_worker_count(config) if worker_count is None else int(worker_count)
    active_reporter = ProgressReporter.from_config(config, label=progress_label) if reporter is None else reporter
    for part_ind, part in enumerate(parts, start=1):
        part.field_by_loops = _field_by_contours(
            part,
            target_field,
            config,
            resolved_workers,
            active_reporter,
            part_ind,
            len(parts),
        )
    return sum((part.field_by_loops for part in parts), start=np.zeros_like(parts[0].field_by_loops))


def _field_by_contours(
    part: CoilPart,
    target_field: TargetField,
    config: CoilGenConfig,
    worker_count: int,
    reporter: ProgressReporter,
    part_ind: int,
    part_count: int,
) -> np.ndarray:
    target_coords = target_field.coords
    if not part.contour_lines:
        return np.zeros((len(target_coords), 3), dtype=float)
    contour_paths = [(contour, contour_evaluation_points(part.mesh, contour, config)) for contour in part.contour_lines]
    work_size = sum(max(len(points) - 1, 0) for _contour, points in contour_paths) * len(target_coords)
    task = reporter.task(f"part {part_ind}/{part_count} contour fields", len(part.contour_lines))
    if should_parallelize(work_size, config, worker_count=worker_count) and len(part.contour_lines) > 1:
        fields = parallel_map(
            lambda item: (
                biot_savart_wire_path(
                    item[1],
                    target_coords,
                    chunk_size=int(config.biot_savart_chunk_size),
                    worker_count=1,
                    min_work_size=int(config.parallel_min_work_size),
                    max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
                )
                * float(item[0].turn_multiplicity)
            ),
            contour_paths,
            worker_count=worker_count,
            progress_callback=lambda done, total: task.update(done),
        )
        task.done()
        return np.sum(fields, axis=0)
    loops_field = np.zeros((len(target_coords), 3), dtype=float)
    for contour_ind, (contour, points) in enumerate(contour_paths, start=1):
        loops_field += float(contour.turn_multiplicity) * biot_savart_wire_path(
            points,
            target_coords,
            chunk_size=int(config.biot_savart_chunk_size),
            worker_count=worker_count,
            min_work_size=int(config.parallel_min_work_size),
            max_chunk_memory_mb=float(config.parallel_max_chunk_memory_mb),
        )
        task.update(contour_ind)
    task.done()
    return loops_field


def calculate_layout_gradient(field: np.ndarray, target_field: TargetField) -> dict[str, float]:
    """Fit connected-layout Bz over the DSV with MR-engineering diagnostics."""

    fit = fit_bz_field_quality(field, target_field, sample_mask=dsv_target_mask(target_field))
    data = fit.as_dict()
    return {
        "offset": float(data["offset_T"]),
        "dbz_dx": float(data["dbz_dx_T_per_m"]),
        "dbz_dy": float(data["dbz_dy_T_per_m"]),
        "dbz_dz": float(data["dbz_dz_T_per_m"]),
        "rms_residual": float(data["linear_rms_residual_T"]),
        "sample_count": float(data["sample_count"]),
        "desired_dbz_dx": float(data["desired_dbz_dx_T_per_m"]),
        "desired_dbz_dy": float(data["desired_dbz_dy_T_per_m"]),
        "desired_dbz_dz": float(data["desired_dbz_dz_T_per_m"]),
        "desired_gradient": float(data["desired_gradient_T_per_m"]),
        "achieved_gradient": float(data["achieved_gradient_T_per_m"]),
        "cross_gradient": float(data["cross_gradient_T_per_m"]),
        "cross_gradient_percent_of_desired": float(data["cross_gradient_percent_of_desired"]),
        "gradient_alignment_cosine": float(data["gradient_alignment_cosine"]),
        "linear_rms_residual_percent_of_peak": float(data["linear_rms_residual_percent_of_peak"]),
        "quadratic_rms_residual": float(data["quadratic_rms_residual_T"]),
        "quadratic_rms_residual_percent_of_peak": float(data["quadratic_rms_residual_percent_of_peak"]),
        "qxx": float(data["qxx_T_per_m2"]),
        "qyy": float(data["qyy_T_per_m2"]),
        "qzz": float(data["qzz_T_per_m2"]),
        "qxy": float(data["qxy_T_per_m2"]),
        "qxz": float(data["qxz_T_per_m2"]),
        "qyz": float(data["qyz_T_per_m2"]),
        "second_order_rms": float(data["second_order_rms_T_per_m2"]),
    }


def field_evaluation_current_source(config: CoilGenConfig) -> str:
    """Keep physical optimization's certified current throughout discretization.

    Refitting a coil's drive after solving would change its power, energy,
    torque and scenario bounds. Legacy designs retain the requested policy.
    """
    if config.research_physics is not None:
        return "stream_function"
    return str(getattr(config, "field_evaluation_current_source", "target_fit"))


def _field_evaluation_current(
    field_per_amp: np.ndarray,
    target_values: np.ndarray,
    target_scale: float,
    sample_mask: np.ndarray,
    *,
    contour_step: float,
    source: str = "target_fit",
    component: str = "Bz",
    b0_direction: object = (0.0, 0.0, 1.0),
    sample_weights: np.ndarray | None = None,
) -> float:
    key = str(source).strip().lower().replace("-", "_")
    if key in {"stream_function", "stream", "contour_step", "sf"}:
        return float(contour_step)
    resolved_component = normalize_field_component(component)
    field_values = (
        np.asarray(field_per_amp, dtype=float)[:, 2]
        if resolved_component == "Bz"
        else field_component_values(field_per_amp, resolved_component, b0_direction=b0_direction)
    )
    if key in {"target_projection", "projection", "least_squares", "least_squares_projection"}:
        return _target_projection_current(field_values, target_values, sample_mask)
    if key in {
        "target_std_projection",
        "std_projection",
        "homogeneity_projection",
        "mean_free_projection",
    }:
        return _target_std_projection_current(
            field_values,
            target_values,
            sample_mask,
            sample_weights=sample_weights,
        )
    return _target_mode_current_or_projection_fallback(
        field_per_amp,
        field_values,
        target_values,
        target_scale,
        sample_mask,
        component=resolved_component,
    )


def _target_mode_current_or_projection_fallback(
    field_per_amp: np.ndarray,
    field_values: np.ndarray,
    target_values: np.ndarray,
    target_scale: float,
    sample_mask: np.ndarray,
    *,
    component: str = "Bz",
) -> float:
    if normalize_field_component(component) == "Bz":
        current = fit_target_mode_current(field_per_amp, target_values, target_scale, sample_mask=sample_mask)
        if np.isfinite(current) and abs(current) > 1e-30:
            return float(current)
    return _target_projection_current(field_values, target_values, sample_mask)


def _target_projection_current(
    field_values: np.ndarray,
    target_values: np.ndarray,
    sample_mask: np.ndarray,
) -> float:
    coil_values = np.asarray(field_values, dtype=float)
    if coil_values.ndim == 2:
        if coil_values.shape[1] != 3:
            raise ValueError("field_values must be a vector or an (n, 3) field array.")
        coil_values = coil_values[:, 2]
    coil_values = coil_values.reshape(-1)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    finite = np.isfinite(coil_values) & np.isfinite(target) & np.asarray(sample_mask, dtype=bool).reshape(-1)
    if not np.any(finite):
        return 0.0
    coil_vec = coil_values[finite]
    target_vec = target[finite]
    denom = float(coil_vec @ coil_vec)
    return float((coil_vec @ target_vec) / denom) if denom > 1e-30 else 0.0


def _target_std_projection_current(
    field_values: np.ndarray,
    target_values: np.ndarray,
    sample_mask: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
) -> float:
    """Return the current minimizing weighted mean-free target residual variance.

    A shim-current fit should not spend current reproducing a constant B0 offset
    when homogeneity is the requested metric.  For residual r = b - I g, this
    solves the one-dimensional weighted least-squares problem after projecting
    both b and g orthogonally to the constant field.
    """

    coil_values = np.asarray(field_values, dtype=float)
    if coil_values.ndim == 2:
        if coil_values.shape[1] != 3:
            raise ValueError("field_values must be a vector or an (n, 3) field array.")
        coil_values = coil_values[:, 2]
    coil_values = coil_values.reshape(-1)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
    if len(coil_values) != len(target) or len(mask) != len(target):
        raise ValueError("field, target, and sample mask must have matching lengths.")

    if sample_weights is None:
        weights = np.ones(len(target), dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        if len(weights) != len(target):
            raise ValueError("sample_weights must match the target length.")
    finite = mask & np.isfinite(coil_values) & np.isfinite(target) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return 0.0

    coil_vec = coil_values[finite]
    target_vec = target[finite]
    weight_vec = weights[finite]
    weight_sum = float(np.sum(weight_vec))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return 0.0
    coil_centered = coil_vec - float(np.dot(weight_vec, coil_vec) / weight_sum)
    target_centered = target_vec - float(np.dot(weight_vec, target_vec) / weight_sum)
    denominator = float(np.dot(weight_vec, coil_centered * coil_centered))
    if denominator <= 1e-30 or not np.isfinite(denominator):
        return 0.0
    numerator = float(np.dot(weight_vec, coil_centered * target_centered))
    return float(numerator / denominator)


def _field_error_values(
    target_z: np.ndarray,
    sf_z: np.ndarray,
    layout_z: np.ndarray,
    loop_z: np.ndarray,
    *,
    layout_stream_z: np.ndarray | None = None,
    loop_stream_z: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
    suffix: str = "",
) -> dict[str, float]:
    """Return target and stream-function field mismatch metrics.

    ``layout_z`` and ``loop_z`` are the routed/unconnected fields scaled to the
    target-current fit and are therefore only used against ``target_z``.
    Stream-function comparisons must use the contour-step current convention;
    otherwise active-shielded designs can compare a target-scaled layout against
    the raw optimized stream function and report meaningless errors.
    """

    target_z = np.asarray(target_z, dtype=float).reshape(-1)
    sf_z = np.asarray(sf_z, dtype=float).reshape(-1)
    layout_z = np.asarray(layout_z, dtype=float).reshape(-1)
    loop_z = np.asarray(loop_z, dtype=float).reshape(-1)
    if layout_stream_z is None:
        layout_stream_z = layout_z
    if loop_stream_z is None:
        loop_stream_z = loop_z
    layout_stream_z = np.asarray(layout_stream_z, dtype=float).reshape(-1)
    loop_stream_z = np.asarray(loop_stream_z, dtype=float).reshape(-1)
    finite = (
        np.isfinite(target_z)
        & np.isfinite(sf_z)
        & np.isfinite(layout_z)
        & np.isfinite(loop_z)
        & np.isfinite(layout_stream_z)
        & np.isfinite(loop_stream_z)
    )
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if len(mask) != len(finite):
            return _nan_field_error_values(suffix=suffix)
        finite &= mask
    if not np.any(finite):
        return _nan_field_error_values(suffix=suffix)
    target_z = target_z[finite]
    sf_z = sf_z[finite]
    layout_z = layout_z[finite]
    loop_z = loop_z[finite]
    layout_stream_z = layout_stream_z[finite]
    loop_stream_z = loop_stream_z[finite]
    target_scale = max(float(np.max(np.abs(target_z))), 1e-30)
    sf_scale = max(float(np.max(np.abs(sf_z))), 1e-30)
    return {
        f"max_rel_error_layout_vs_target{suffix}": float(np.max(np.abs((layout_z - target_z) / target_scale)) * 100.0),
        f"mean_rel_error_layout_vs_target{suffix}": float(
            np.mean(np.abs((layout_z - target_z) / target_scale)) * 100.0
        ),
        f"max_rel_error_unconnected_contours_vs_target{suffix}": float(
            np.max(np.abs((loop_z - target_z) / target_scale)) * 100.0
        ),
        f"mean_rel_error_unconnected_contours_vs_target{suffix}": float(
            np.mean(np.abs((loop_z - target_z) / target_scale)) * 100.0
        ),
        f"max_rel_error_layout_vs_stream_function_field{suffix}": float(
            np.max(np.abs((layout_stream_z - sf_z) / sf_scale)) * 100.0
        ),
        f"mean_rel_error_layout_vs_stream_function_field{suffix}": float(
            np.mean(np.abs((layout_stream_z - sf_z) / sf_scale)) * 100.0
        ),
        f"max_rel_error_unconnected_contours_vs_stream_function_field{suffix}": float(
            np.max(np.abs((loop_stream_z - sf_z) / sf_scale)) * 100.0
        ),
        f"mean_rel_error_unconnected_contours_vs_stream_function_field{suffix}": float(
            np.mean(np.abs((loop_stream_z - sf_z) / sf_scale)) * 100.0
        ),
    }


def _nan_field_error_values(*, suffix: str = "") -> dict[str, float]:
    return {
        f"max_rel_error_layout_vs_target{suffix}": float("nan"),
        f"mean_rel_error_layout_vs_target{suffix}": float("nan"),
        f"max_rel_error_unconnected_contours_vs_target{suffix}": float("nan"),
        f"mean_rel_error_unconnected_contours_vs_target{suffix}": float("nan"),
        f"max_rel_error_layout_vs_stream_function_field{suffix}": float("nan"),
        f"mean_rel_error_layout_vs_stream_function_field{suffix}": float("nan"),
        f"max_rel_error_unconnected_contours_vs_stream_function_field{suffix}": float("nan"),
        f"mean_rel_error_unconnected_contours_vs_stream_function_field{suffix}": float("nan"),
    }
