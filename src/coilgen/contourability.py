from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CoilGenConfig
from .contours import (
    calc_potential_levels,
    contour_current_alignment,
    extract_contours,
    filter_loop_significance,
    regularize_contour_paths,
)
from .evaluation import _field_evaluation_current, calculate_unconnected_contour_field
from .field_components import component_sample_values, field_component_values
from .metrics import dsv_target_mask
from .models import CoilPart, Mesh, TargetField
from .symmetry import normalise_symmetry_mode


@dataclass(frozen=True)
class _ContourRepairCandidate:
    levels: int
    phase_offset: float
    range_percentile: float
    zero_crossing_strategy: str


def validate_and_repair_contourability(
    parts: list[CoilPart],
    combined_mesh: Mesh,
    target_field: TargetField,
    sf_b_field: np.ndarray,
    config: CoilGenConfig,
) -> tuple[list[CoilPart], dict[str, Any]]:
    """Validate that finite contours still represent the optimized sheet current.

    The stream-function optimizer solves for a continuous surface-current sheet.
    The later contouring step is a discretization of that sheet into equal-current
    loops.  This check catches cases where closed loops exist but are too sparse,
    collapsed, badly oriented, or electromagnetically far from the continuous
    stream-function field.
    """

    mode = normalise_symmetry_mode(getattr(config, "contourability_validation", "auto"))
    if mode == "off":
        return parts, {"contourability_checked": False, "contourability_validation_mode": "off"}

    requested_levels = int(getattr(config, "levels", 0))
    requested_candidate = _current_contour_candidate(config, requested_levels)
    baseline = _assess_contourability(parts, target_field, sf_b_field, config, candidate_levels=requested_levels)
    _annotate_candidate_report(baseline, requested_candidate, requested_candidate)
    baseline["contourability_validation_mode"] = mode
    baseline["contourability_requested_levels"] = requested_levels
    baseline["contourability_repaired"] = False
    baseline["contourability_candidate_count"] = 1
    _write_part_contourability_diagnostics(parts, baseline)
    if baseline["contourability_ok"]:
        if bool(getattr(config, "contour_field_constrained_smoothing_enabled", False)):
            parts, smoothing_report = _select_field_constrained_contour_smoothing(
                parts,
                target_field,
                sf_b_field,
                config,
                base_report=baseline,
                candidate_levels=requested_levels,
            )
            baseline.update(smoothing_report)
            baseline["contourability_reason"] = _contourability_reason(baseline)
            _write_part_contourability_diagnostics(parts, baseline)
        return parts, baseline

    best = baseline
    best_snapshot = _snapshot_parts(parts)
    original_snapshot = _snapshot_parts(parts)
    selected_candidate = requested_candidate
    repair_candidates: list[_ContourRepairCandidate] = []
    pruned_candidate_count = 0
    if bool(getattr(config, "contourability_repair_enabled", True)):
        repair_candidates, pruned_candidate_count = _repair_candidates_for_report(
            requested_candidate,
            baseline,
            config,
        )
    baseline["contourability_repair_pruned_candidate_count"] = int(pruned_candidate_count)
    candidate_reports = [dict(baseline)]

    if repair_candidates:
        for repair_candidate in repair_candidates:
            if repair_candidate == requested_candidate:
                continue
            _restore_parts(parts, original_snapshot)
            _apply_contour_candidate(config, repair_candidate)
            parts, _primary_surface_ind = calc_potential_levels(parts, combined_mesh, config)
            parts = extract_contours(parts, config)
            parts = filter_loop_significance(parts, target_field, config)
            parts = regularize_contour_paths(parts, config)
            candidate = _assess_contourability(
                parts,
                target_field,
                sf_b_field,
                config,
                candidate_levels=int(repair_candidate.levels),
            )
            _annotate_candidate_report(candidate, requested_candidate, repair_candidate)
            candidate["contourability_validation_mode"] = mode
            candidate["contourability_requested_levels"] = requested_levels
            candidate["contourability_repaired"] = repair_candidate != requested_candidate
            candidate_reports.append(dict(candidate))
            if _contourability_score(candidate) < _contourability_score(best):
                best = candidate
                best_snapshot = _snapshot_parts(parts)
                selected_candidate = repair_candidate

    _restore_parts(parts, best_snapshot)
    _apply_contour_candidate(config, selected_candidate)
    best["contourability_validation_mode"] = mode
    best["contourability_requested_levels"] = requested_levels
    best["contourability_selected_levels"] = int(selected_candidate.levels)
    best["contourability_repaired"] = selected_candidate != requested_candidate
    best["contourability_candidate_count"] = int(len(candidate_reports))
    best["contourability_repair_pruned_candidate_count"] = int(pruned_candidate_count)
    best["contourability_candidate_reports"] = candidate_reports
    if bool(getattr(config, "contour_field_constrained_smoothing_enabled", False)):
        parts, smoothing_report = _select_field_constrained_contour_smoothing(
            parts,
            target_field,
            sf_b_field,
            config,
            base_report=best,
            candidate_levels=int(selected_candidate.levels),
        )
        merged_report = dict(best)
        merged_report.update(smoothing_report)
        best = merged_report
    best["contourability_reason"] = _contourability_reason(best)
    _write_part_contourability_diagnostics(parts, best)

    if mode == "strict" and not bool(best["contourability_ok"]):
        raise ValueError("Contourability validation failed: " + str(best["contourability_reason"]))
    return parts, best


def _assess_contourability(
    parts: list[CoilPart],
    target_field: TargetField,
    sf_b_field: np.ndarray,
    config: CoilGenConfig,
    *,
    candidate_levels: int,
) -> dict[str, Any]:
    geometry = _contour_geometry_report(parts, config)
    roughness = _contour_roughness_report(parts)
    field = _contour_field_report(parts, target_field, sf_b_field, config)
    limits = _contourability_limits(config)
    field_ok = (
        np.isfinite(field["contourability_field_rms_error_percent"])
        and field["contourability_field_rms_error_percent"] <= limits["max_rms_error_percent"]
        and field["contourability_field_mean_abs_error_percent"] <= limits["max_mean_error_percent"]
        and field["contourability_field_peak_error_percent"] <= limits["max_peak_error_percent"]
    )
    geometry_ok = (
        geometry["contourability_contour_count"] > 0
        and (
            limits["max_contour_count"] <= 0 or geometry["contourability_contour_count"] <= limits["max_contour_count"]
        )
        and geometry["contourability_open_loop_count"] == 0
        and geometry["contourability_bad_orientation_count"] == 0
        and geometry["contourability_tiny_loop_fraction"] <= limits["max_tiny_loop_fraction"]
        and geometry["contourability_max_closure_gap_m"] <= geometry["contourability_closure_tolerance_m"]
        and (
            not np.isfinite(geometry["contourability_max_surface_error_m"])
            or geometry["contourability_max_surface_error_m"] <= geometry["contourability_surface_tolerance_m"]
        )
    )
    target_limit_enforced = bool(field.get("contourability_target_limit_enforced", False))
    target_metric = float(field.get("contourability_target_metric_T", np.nan))
    target_limit = float(field.get("contourability_target_effective_allowed_metric_T", np.nan))
    target_ok = bool(
        not target_limit_enforced
        or (np.isfinite(target_metric) and np.isfinite(target_limit) and target_metric <= target_limit)
    )
    report: dict[str, Any] = {
        "contourability_checked": True,
        "contourability_candidate_levels": int(candidate_levels),
        "contourability_selected_levels": int(candidate_levels),
        **limits,
        **geometry,
        **roughness,
        **field,
        "contourability_geometry_ok": bool(geometry_ok),
        "contourability_field_ok": bool(field_ok),
        "contourability_target_ok": bool(target_ok),
    }
    report["contourability_ok"] = bool(geometry_ok and field_ok and target_ok)
    report["contourability_reason"] = _contourability_reason(report)
    return report


def _contour_geometry_report(parts: list[CoilPart], config: CoilGenConfig) -> dict[str, Any]:
    scale = _parts_geometry_scale(parts)
    closure_tolerance = _positive_or_default(getattr(config, "contourability_max_closure_gap_m", 0.0), 1.0e-8 * scale)
    surface_tolerance = _positive_or_default(getattr(config, "contourability_max_surface_error_m", 0.0), 1.0e-8 * scale)
    min_loop_length = _positive_or_default(getattr(config, "contourability_min_loop_length_m", 0.0), 1.0e-5 * scale)
    contour_count = 0
    point_count = 0
    open_count = 0
    tiny_count = 0
    bad_orientation_count = 0
    orientation_checked_count = 0
    orientation_indeterminate_count = 0
    lengths: list[float] = []
    closure_gaps: list[float] = []
    surface_errors: list[float] = []

    for part in parts:
        mesh = part.mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        for contour in part.contour_lines:
            contour_count += 1
            points = np.asarray(contour.v, dtype=float)
            point_count += int(len(points))
            length = float(contour.length)
            lengths.append(length)
            if length <= min_loop_length:
                tiny_count += 1
            closure_gap = float(np.linalg.norm(points[0] - points[-1])) if len(points) >= 2 else float("inf")
            closure_gaps.append(closure_gap)
            if not np.isfinite(closure_gap) or closure_gap > closure_tolerance:
                open_count += 1

            alignment, alignment_scale = contour_current_alignment(part, contour)
            if alignment_scale > 0.0:
                orientation_checked_count += 1
                if alignment < -1.0e-10 * alignment_scale:
                    bad_orientation_count += 1
            else:
                orientation_indeterminate_count += 1

            if (
                contour.surface_faces is None
                or contour.surface_barycentric is None
                or len(vertices) == 0
                or len(faces) == 0
            ):
                continue
            face_indices = np.asarray(contour.surface_faces, dtype=np.int64).reshape(-1)
            barycentric = np.asarray(contour.surface_barycentric, dtype=float)
            n_points = min(len(points), len(face_indices), len(barycentric))
            if n_points == 0 or barycentric.ndim != 2 or barycentric.shape[1] != 3:
                continue
            valid = (face_indices[:n_points] >= 0) & (face_indices[:n_points] < len(faces))
            if not np.any(valid):
                continue
            triangles = vertices[faces[face_indices[:n_points][valid]]]
            reconstructed = np.einsum("ij,ijk->ik", barycentric[:n_points][valid], triangles)
            errors = np.linalg.norm(reconstructed - points[:n_points][valid], axis=1)
            surface_errors.extend(float(value) for value in errors if np.isfinite(value))

    return {
        "contourability_geometry_scale_m": float(scale),
        "contourability_closure_tolerance_m": float(closure_tolerance),
        "contourability_surface_tolerance_m": float(surface_tolerance),
        "contourability_min_loop_length_m": float(min_loop_length),
        "contourability_contour_count": int(contour_count),
        "contourability_point_count": int(point_count),
        "contourability_mean_points_per_contour": float(point_count / contour_count) if contour_count else float("nan"),
        "contourability_total_length_m": float(np.sum(lengths)) if lengths else 0.0,
        "contourability_median_loop_length_m": float(np.median(lengths)) if lengths else float("nan"),
        "contourability_open_loop_count": int(open_count),
        "contourability_tiny_loop_count": int(tiny_count),
        "contourability_tiny_loop_fraction": float(tiny_count / contour_count) if contour_count else 1.0,
        "contourability_max_closure_gap_m": float(np.max(closure_gaps)) if closure_gaps else float("inf"),
        "contourability_max_surface_error_m": float(np.max(surface_errors)) if surface_errors else float("nan"),
        "contourability_bad_orientation_count": int(bad_orientation_count),
        "contourability_orientation_checked_count": int(orientation_checked_count),
        "contourability_orientation_indeterminate_count": int(orientation_indeterminate_count),
    }


def _contour_roughness_report(parts: list[CoilPart]) -> dict[str, Any]:
    """Return a sampling-aware discrete bending metric for contour paths."""

    bending_energy = 0.0
    total_length = 0.0
    maximum_turning_angle = 0.0
    evaluated_count = 0
    for part in parts:
        for contour in part.contour_lines:
            points = np.asarray(contour.v, dtype=float)
            if len(points) < 3 or not np.all(np.isfinite(points)):
                continue
            scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
            closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1.0e-10 * scale)
            if closed:
                points = points[:-1]
            if len(points) < 3:
                continue
            segments = np.roll(points, -1, axis=0) - points if closed else np.diff(points, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            valid = np.isfinite(lengths) & (lengths > 1.0e-15 * scale)
            if np.count_nonzero(valid) < (3 if closed else 2):
                continue
            lengths = lengths[valid]
            tangents = segments[valid] / lengths[:, None]
            if closed:
                previous = np.roll(tangents, 1, axis=0)
                local_ds = 0.5 * (lengths + np.roll(lengths, 1))
            else:
                previous = tangents[:-1]
                tangents = tangents[1:]
                local_ds = 0.5 * (lengths[:-1] + lengths[1:])
            tangent_delta = tangents - previous
            bending_energy += float(np.sum(np.einsum("ij,ij->i", tangent_delta, tangent_delta) / local_ds))
            dot_products = np.clip(np.einsum("ij,ij->i", tangents, previous), -1.0, 1.0)
            maximum_turning_angle = max(maximum_turning_angle, float(np.max(np.arccos(dot_products))))
            total_length += float(np.sum(lengths))
            evaluated_count += 1
    curvature_rms = (
        float(np.sqrt(bending_energy / total_length))
        if total_length > 0.0 and np.isfinite(bending_energy)
        else float("nan")
    )
    return {
        "contourability_bending_energy_per_m": float(bending_energy),
        "contourability_curvature_rms_per_m": curvature_rms,
        "contourability_max_turning_angle_deg": float(np.rad2deg(maximum_turning_angle)),
        "contourability_roughness_path_count": int(evaluated_count),
    }


def _contour_field_report(
    parts: list[CoilPart],
    target_field: TargetField,
    sf_b_field: np.ndarray,
    config: CoilGenConfig,
) -> dict[str, Any]:
    calculate_unconnected_contour_field(parts, config, target_field, progress_label="Contourability")
    component = str(getattr(config, "target_field_component", "Bz"))
    b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    sf_values = field_component_values(sf_b_field, component, b0_direction=b0_direction).reshape(-1)
    contour_stream_values = np.zeros_like(sf_values)
    contour_field_per_amp = np.zeros_like(np.asarray(sf_b_field, dtype=float))
    for part in parts:
        if part.field_by_loops is None:
            continue
        part_field = np.asarray(part.field_by_loops, dtype=float)
        contour_field_per_amp += part_field
        contour_stream_values += field_component_values(
            part_field * float(part.contour_step or 0.0),
            component,
            b0_direction=b0_direction,
        ).reshape(-1)
    mask = component_sample_values(dsv_target_mask(target_field), component).astype(bool).reshape(-1)
    finite = np.asarray(mask, dtype=bool) & np.isfinite(sf_values) & np.isfinite(contour_stream_values)
    if not np.any(finite):
        return {
            "contourability_field_sample_count": 0,
            "contourability_field_rms_error_T": float("nan"),
            "contourability_field_rms_reference_T": float("nan"),
            "contourability_field_peak_reference_T": float("nan"),
            "contourability_field_rms_error_percent": float("nan"),
            "contourability_field_mean_abs_error_percent": float("nan"),
            "contourability_field_peak_error_percent": float("nan"),
            "contourability_target_residual_std_T": float("nan"),
            "contourability_target_residual_rms_T": float("nan"),
            "contourability_target_metric_T": float("nan"),
            "contourability_target_metric_ratio": float("nan"),
        }
    diff = contour_stream_values[finite] - sf_values[finite]
    sf = sf_values[finite]
    rms_reference = _rms(sf)
    mean_abs_reference = max(float(np.mean(np.abs(sf))), 1.0e-30)
    peak_reference = max(float(np.max(np.abs(sf))), 1.0e-30)
    rms_error = _rms(diff)
    report = {
        "contourability_field_sample_count": int(np.count_nonzero(finite)),
        "contourability_field_rms_error_T": float(rms_error),
        "contourability_field_rms_reference_T": float(rms_reference),
        "contourability_field_peak_reference_T": float(peak_reference),
        "contourability_field_rms_error_percent": float(100.0 * rms_error / max(rms_reference, 1.0e-30)),
        "contourability_field_mean_abs_error_percent": float(100.0 * float(np.mean(np.abs(diff))) / mean_abs_reference),
        "contourability_field_peak_error_percent": float(
            100.0 * float(np.max(np.abs(diff))) / max(peak_reference, 1.0e-30)
        ),
    }
    report.update(
        _target_contour_field_report(
            parts,
            contour_field_per_amp,
            target_field,
            config,
            component=component,
            b0_direction=b0_direction,
            mask=mask,
        )
    )
    return report


def _target_contour_field_report(
    parts: list[CoilPart],
    contour_field_per_amp: np.ndarray,
    target_field: TargetField,
    config: CoilGenConfig,
    *,
    component: str,
    b0_direction: Any,
    mask: np.ndarray,
) -> dict[str, Any]:
    target_values = field_component_values(
        target_field.b,
        component,
        b0_direction=b0_direction,
    ).reshape(-1)
    contour_values_per_amp = field_component_values(
        contour_field_per_amp,
        component,
        b0_direction=b0_direction,
    ).reshape(-1)
    weights = component_sample_values(np.asarray(target_field.weights, dtype=float), component).reshape(-1)
    contour_step = float(parts[0].contour_step or 1.0) if parts else 1.0
    fitted_current = _field_evaluation_current(
        contour_field_per_amp,
        target_values,
        float(getattr(config, "target_gradient_strength", 1.0)),
        mask,
        contour_step=contour_step,
        source=str(getattr(config, "field_evaluation_current_source", "target_fit")),
        component=component,
        b0_direction=b0_direction,
        sample_weights=weights,
    )
    residual = target_values - fitted_current * contour_values_per_amp
    finite = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(residual)
        & np.isfinite(target_values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(finite):
        return {
            "contourability_target_fitted_current_A": float(fitted_current),
            "contourability_target_residual_std_T": float("nan"),
            "contourability_target_residual_rms_T": float("nan"),
            "contourability_target_metric": "",
            "contourability_target_metric_T": float("nan"),
            "contourability_target_reference_metric_T": float("nan"),
            "contourability_target_allowed_metric_T": float("nan"),
            "contourability_target_effective_allowed_metric_T": float("nan"),
            "contourability_target_acceptance_absolute_T": float("nan"),
            "contourability_target_limit_source": "",
            "contourability_target_limit_enforced": False,
            "contourability_target_metric_ratio": float("nan"),
        }
    residual_values = residual[finite]
    target_finite = target_values[finite]
    weight_finite = weights[finite]
    residual_std = _weighted_std(residual_values, weight_finite)
    residual_rms = _weighted_rms(residual_values, weight_finite)
    target_std = _weighted_std(target_finite, weight_finite)
    target_rms = _weighted_rms(target_finite, weight_finite)
    metric = str(getattr(config, "target_error_constraint_metric", "norm")).strip().lower().replace("-", "_")
    if metric in {"std", "standard_deviation", "homogeneity"}:
        metric = "std"
        metric_value = residual_std
        reference = target_std
        layout_allowed = max(float(getattr(config, "layout_target_error_std_limit_T", 0.0)), 0.0)
        continuous_allowed = max(float(getattr(config, "target_error_std_limit_T", 0.0)), 0.0)
    else:
        metric = "rms"
        metric_value = residual_rms
        reference = target_rms
        layout_allowed = max(float(getattr(config, "layout_target_error_rms_limit_T", 0.0)), 0.0)
        continuous_allowed = max(float(getattr(config, "target_error_rms_limit_T", 0.0)), 0.0)
    limit_enforced = layout_allowed > 0.0
    allowed = layout_allowed if limit_enforced else continuous_allowed
    acceptance = (
        max(float(getattr(config, "layout_target_error_acceptance_absolute_T", 0.0)), 0.0) if limit_enforced else 0.0
    )
    effective_allowed = allowed + acceptance if allowed > 0.0 else 0.0
    denominator = effective_allowed if effective_allowed > 0.0 else reference
    return {
        "contourability_target_fitted_current_A": float(fitted_current),
        "contourability_target_residual_std_T": float(residual_std),
        "contourability_target_residual_rms_T": float(residual_rms),
        "contourability_target_metric": metric,
        "contourability_target_metric_T": float(metric_value),
        "contourability_target_reference_metric_T": float(reference),
        "contourability_target_allowed_metric_T": float(allowed) if allowed > 0.0 else float("nan"),
        "contourability_target_effective_allowed_metric_T": (
            float(effective_allowed) if effective_allowed > 0.0 else float("nan")
        ),
        "contourability_target_acceptance_absolute_T": float(acceptance),
        "contourability_target_limit_source": "layout" if limit_enforced else "continuous_stream_function",
        "contourability_target_limit_enforced": bool(limit_enforced),
        "contourability_target_metric_ratio": float(metric_value / max(denominator, 1.0e-30)),
    }


def _contourability_limits(config: CoilGenConfig) -> dict[str, float]:
    return {
        "max_rms_error_percent": float(getattr(config, "contourability_max_rms_error_percent", 30.0)),
        "max_mean_error_percent": float(getattr(config, "contourability_max_mean_error_percent", 25.0)),
        "max_peak_error_percent": float(getattr(config, "contourability_max_peak_error_percent", 100.0)),
        "max_tiny_loop_fraction": float(getattr(config, "contourability_max_tiny_loop_fraction", 0.5)),
        "max_contour_count": float(max(int(getattr(config, "contourability_max_contour_count", 0)), 0)),
    }


def _select_field_constrained_contour_smoothing(
    parts: list[CoilPart],
    target_field: TargetField,
    sf_b_field: np.ndarray,
    config: CoilGenConfig,
    *,
    base_report: dict[str, Any],
    candidate_levels: int,
) -> tuple[list[CoilPart], dict[str, Any]]:
    """Select the smoothest on-surface contours that remain field-feasible."""

    original_snapshot = _snapshot_parts(parts)
    baseline = dict(base_report)
    baseline.update(_contour_roughness_report(parts))
    baseline["contour_field_smoothing_iterations"] = 0
    baseline["contour_field_smoothing_factor"] = 0.0
    candidate_reports = [dict(baseline)]
    best_report = baseline
    best_snapshot = original_snapshot
    best_iterations = 0
    best_factor = 0.0

    for iterations, factor in _field_smoothing_candidates(config):
        _restore_parts(parts, original_snapshot)
        candidate_config = copy(config)
        candidate_config.contour_path_regularization_mode = "smooth"
        candidate_config.contour_smoothing_iterations = int(iterations)
        candidate_config.contour_smoothing_factor = float(factor)
        parts = regularize_contour_paths(parts, candidate_config)
        candidate_report = _assess_contourability(
            parts,
            target_field,
            sf_b_field,
            config,
            candidate_levels=int(candidate_levels),
        )
        candidate_report["contour_field_smoothing_iterations"] = int(iterations)
        candidate_report["contour_field_smoothing_factor"] = float(factor)
        candidate_reports.append(dict(candidate_report))
        if _smoothing_report_is_better(candidate_report, best_report):
            best_report = candidate_report
            best_snapshot = _snapshot_parts(parts)
            best_iterations = int(iterations)
            best_factor = float(factor)

    baseline_roughness = float(baseline.get("contourability_curvature_rms_per_m", np.nan))
    selected_roughness = float(best_report.get("contourability_curvature_rms_per_m", np.nan))
    improvement = (
        100.0 * (baseline_roughness - selected_roughness) / baseline_roughness
        if np.isfinite(baseline_roughness) and baseline_roughness > 0.0 and np.isfinite(selected_roughness)
        else 0.0
    )
    required_improvement = max(
        float(getattr(config, "contour_field_smoothing_min_improvement_percent", 0.0)),
        0.0,
    )
    if best_iterations > 0 and bool(baseline.get("contourability_ok", False)) and improvement < required_improvement:
        best_report = baseline
        best_snapshot = original_snapshot
        best_iterations = 0
        best_factor = 0.0
        selected_roughness = baseline_roughness
        improvement = 0.0

    _restore_parts(parts, best_snapshot)
    selected = dict(best_report)
    selected.update(
        {
            "contour_field_constrained_smoothing_checked": True,
            "contour_field_constrained_smoothing_applied": bool(best_iterations > 0),
            "contour_field_smoothing_candidate_count": int(len(candidate_reports)),
            "contour_field_smoothing_selected_iterations": int(best_iterations),
            "contour_field_smoothing_selected_factor": float(best_factor),
            "contour_field_smoothing_roughness_before_per_m": float(baseline_roughness),
            "contour_field_smoothing_roughness_after_per_m": float(selected_roughness),
            "contour_field_smoothing_improvement_percent": float(improvement),
            "contour_field_smoothing_required_improvement_percent": float(required_improvement),
            "contour_field_smoothing_candidate_reports": candidate_reports,
        }
    )
    return parts, selected


def _field_smoothing_candidates(config: CoilGenConfig) -> list[tuple[int, float]]:
    candidates: list[tuple[int, float]] = []
    seen: set[tuple[int, float]] = set()
    for raw_candidate in tuple(getattr(config, "contour_field_smoothing_candidates", ())):
        try:
            raw_iterations, raw_factor = raw_candidate
            iterations = int(raw_iterations)
            factor = float(raw_factor)
        except (TypeError, ValueError):
            continue
        key = (iterations, round(factor, 12))
        if iterations < 1 or not np.isfinite(factor) or factor <= 0.0 or factor > 1.0 or key in seen:
            continue
        seen.add(key)
        candidates.append((iterations, factor))
    return candidates


def _smoothing_report_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_ok = bool(candidate.get("contourability_ok", False))
    incumbent_ok = bool(incumbent.get("contourability_ok", False))
    if candidate_ok != incumbent_ok:
        return candidate_ok
    if not candidate_ok:
        return _contourability_score(candidate) < _contourability_score(incumbent)
    candidate_roughness = float(candidate.get("contourability_curvature_rms_per_m", np.inf))
    incumbent_roughness = float(incumbent.get("contourability_curvature_rms_per_m", np.inf))
    if np.isfinite(candidate_roughness) and candidate_roughness < incumbent_roughness * (1.0 - 1.0e-12):
        return True
    if np.isclose(candidate_roughness, incumbent_roughness, rtol=1.0e-12, atol=0.0):
        candidate_ratio = float(candidate.get("contourability_target_metric_ratio", np.inf))
        incumbent_ratio = float(incumbent.get("contourability_target_metric_ratio", np.inf))
        return candidate_ratio < incumbent_ratio
    return False


def _repair_level_candidates(requested_levels: int, config: CoilGenConfig) -> list[int]:
    min_levels = max(int(getattr(config, "contourability_repair_min_levels", 4)), 1)
    max_levels = max(int(getattr(config, "contourability_repair_max_levels", 160)), min_levels)
    factors = tuple(float(value) for value in getattr(config, "contourability_repair_level_factors", (1.0,)))
    candidates = {int(requested_levels)}
    for factor in factors:
        if np.isfinite(factor) and factor > 0.0:
            candidates.add(int(round(float(requested_levels) * factor)))
    return sorted({max(min(levels, max_levels), min_levels) for levels in candidates})


def _repair_contour_candidates(
    requested_candidate: _ContourRepairCandidate,
    config: CoilGenConfig,
) -> list[_ContourRepairCandidate]:
    levels = _repair_level_candidates(requested_candidate.levels, config)
    phases = _repair_phase_candidates(config, requested_candidate.phase_offset)
    percentiles = _repair_range_candidates(config, requested_candidate.range_percentile)
    strategies = _repair_zero_crossing_candidates(config, requested_candidate.zero_crossing_strategy)

    candidates = [requested_candidate]
    seen = {_candidate_key(requested_candidate)}
    for level_count in levels:
        for phase_offset in phases:
            for range_percentile in percentiles:
                for strategy in strategies:
                    candidate = _ContourRepairCandidate(
                        levels=int(level_count),
                        phase_offset=float(phase_offset),
                        range_percentile=float(range_percentile),
                        zero_crossing_strategy=str(strategy),
                    )
                    key = _candidate_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)
    return candidates


def _repair_candidates_for_report(
    requested_candidate: _ContourRepairCandidate,
    report: dict[str, Any],
    config: CoilGenConfig,
) -> tuple[list[_ContourRepairCandidate], int]:
    """Prune denser candidates when the requested layout already violates a hard loop cap."""

    candidates = _repair_contour_candidates(requested_candidate, config)
    max_contour_count = int(report.get("max_contour_count", 0) or 0)
    contour_count = int(report.get("contourability_contour_count", 0) or 0)
    if max_contour_count <= 0 or contour_count <= max_contour_count:
        return candidates, 0
    kept = [candidate for candidate in candidates if candidate.levels <= requested_candidate.levels]
    return kept, int(len(candidates) - len(kept))


def _current_contour_candidate(config: CoilGenConfig, requested_levels: int) -> _ContourRepairCandidate:
    return _ContourRepairCandidate(
        levels=int(requested_levels),
        phase_offset=_safe_phase_offset(getattr(config, "pot_offset_factor", 0.5), default=0.5),
        range_percentile=_safe_range_percentile(
            getattr(config, "contour_level_range_percentile", 100.0),
            default=100.0,
        ),
        zero_crossing_strategy=_normalise_repair_zero_crossing_strategy(
            getattr(config, "zero_crossing_level_strategy", "paired")
        ),
    )


def _apply_contour_candidate(config: CoilGenConfig, candidate: _ContourRepairCandidate) -> None:
    config.levels = int(candidate.levels)
    config.pot_offset_factor = float(candidate.phase_offset)
    config.contour_level_range_percentile = float(candidate.range_percentile)
    config.zero_crossing_level_strategy = str(candidate.zero_crossing_strategy)


def _annotate_candidate_report(
    report: dict[str, Any],
    requested: _ContourRepairCandidate,
    candidate: _ContourRepairCandidate,
) -> None:
    report["contourability_requested_phase_offset"] = float(requested.phase_offset)
    report["contourability_requested_range_percentile"] = float(requested.range_percentile)
    report["contourability_requested_zero_crossing_strategy"] = str(requested.zero_crossing_strategy)
    report["contourability_candidate_phase_offset"] = float(candidate.phase_offset)
    report["contourability_candidate_range_percentile"] = float(candidate.range_percentile)
    report["contourability_candidate_zero_crossing_strategy"] = str(candidate.zero_crossing_strategy)
    report["contourability_selected_phase_offset"] = float(candidate.phase_offset)
    report["contourability_selected_range_percentile"] = float(candidate.range_percentile)
    report["contourability_selected_zero_crossing_strategy"] = str(candidate.zero_crossing_strategy)


def _repair_phase_candidates(config: CoilGenConfig, original: float) -> list[float]:
    values = [original, *tuple(getattr(config, "contourability_repair_phase_offsets", ()))]
    candidates: list[float] = []
    seen: set[float] = set()
    for value in values:
        phase = _safe_phase_offset(value, default=float("nan"))
        key = round(phase, 12)
        if not np.isfinite(phase) or key in seen:
            continue
        seen.add(key)
        candidates.append(float(phase))
    return candidates or [float(original)]


def _repair_range_candidates(config: CoilGenConfig, original: float) -> list[float]:
    values = [original, *tuple(getattr(config, "contourability_repair_range_percentiles", ()))]
    candidates: list[float] = []
    seen: set[float] = set()
    for value in values:
        percentile = _safe_range_percentile(value, default=float("nan"))
        key = round(percentile, 12)
        if not np.isfinite(percentile) or key in seen:
            continue
        seen.add(key)
        candidates.append(float(percentile))
    return candidates or [float(original)]


def _repair_zero_crossing_candidates(config: CoilGenConfig, original: str) -> list[str]:
    values = [original, *tuple(getattr(config, "contourability_repair_zero_crossing_strategies", ()))]
    candidates: list[str] = []
    seen: set[str] = set()
    for value in values:
        strategy = _normalise_repair_zero_crossing_strategy(value)
        if strategy in seen:
            continue
        seen.add(strategy)
        candidates.append(strategy)
    return candidates or [str(original)]


def _candidate_key(candidate: _ContourRepairCandidate) -> tuple[int, float, float, str]:
    return (
        int(candidate.levels),
        round(float(candidate.phase_offset), 12),
        round(float(candidate.range_percentile), 12),
        str(candidate.zero_crossing_strategy),
    )


def _safe_phase_offset(value: Any, *, default: float) -> float:
    try:
        phase = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(phase) or phase < 0.0 or phase >= 1.0:
        return float(default)
    return float(phase)


def _safe_range_percentile(value: Any, *, default: float) -> float:
    try:
        percentile = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(percentile) or percentile <= 0.0 or percentile > 100.0:
        return float(default)
    return float(percentile)


def _normalise_repair_zero_crossing_strategy(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"paired", "symmetric"}:
        return "paired"
    if key in {"full_range", "centered"}:
        return "full_range"
    return "paired"


def _contourability_score(report: dict[str, Any]) -> float:
    if int(report.get("contourability_contour_count", 0)) <= 0:
        return float("inf")

    score = 0.0
    score += 1.0e9 * int(report.get("contourability_open_loop_count", 0))
    score += 1.0e9 * int(report.get("contourability_bad_orientation_count", 0))

    max_contour_count = int(report.get("max_contour_count", 0) or 0)
    contour_count = int(report.get("contourability_contour_count", 0) or 0)
    if max_contour_count > 0 and contour_count > max_contour_count:
        score += 1.0e8 * (contour_count - max_contour_count)

    target_ratio = float(report.get("contourability_target_metric_ratio", np.nan))
    if np.isfinite(target_ratio):
        if bool(report.get("contourability_target_limit_enforced", False)):
            score += 1.0e7 * max(target_ratio - 1.0, 0.0)
        score += 10.0 * target_ratio

    closure_gap = float(report.get("contourability_max_closure_gap_m", 0.0))
    closure_limit = float(report.get("contourability_closure_tolerance_m", np.inf))
    if np.isfinite(closure_gap) and np.isfinite(closure_limit) and closure_limit > 0.0:
        score += 1.0e6 * max(closure_gap / closure_limit - 1.0, 0.0)

    rms_error = float(report.get("contourability_field_rms_error_percent", np.inf))
    mean_error = float(report.get("contourability_field_mean_abs_error_percent", np.inf))
    peak_error = float(report.get("contourability_field_peak_error_percent", np.inf))
    max_rms = max(float(report.get("max_rms_error_percent", 30.0)), 1.0e-30)
    max_mean = max(float(report.get("max_mean_error_percent", 25.0)), 1.0e-30)
    max_peak = max(float(report.get("max_peak_error_percent", 100.0)), 1.0e-30)
    score += rms_error / max_rms
    score += 0.5 * mean_error / max_mean
    score += 0.1 * peak_error / max_peak

    tiny_fraction = float(report.get("contourability_tiny_loop_fraction", 0.0))
    max_tiny = max(float(report.get("max_tiny_loop_fraction", 0.5)), 1.0e-30)
    score += 0.05 * max(tiny_fraction / max_tiny - 1.0, 0.0)
    if not np.isfinite(score):
        return float("inf")
    return score


def _contourability_reason(report: dict[str, Any]) -> str:
    if bool(report.get("contourability_ok", False)):
        if bool(report.get("contourability_repaired", False)):
            return (
                "passed after contour-grid repair "
                f"({_format_report_candidate(report, 'requested')} -> {_format_report_candidate(report, 'selected')})"
            )
        return "passed"
    reasons: list[str] = []
    if int(report.get("contourability_contour_count", 0)) <= 0:
        reasons.append("no contours extracted")
    max_contour_count = int(report.get("max_contour_count", 0) or 0)
    contour_count = int(report.get("contourability_contour_count", 0) or 0)
    if max_contour_count > 0 and contour_count > max_contour_count:
        reasons.append(f"{contour_count} contours > configured maximum {max_contour_count}")
    if int(report.get("contourability_open_loop_count", 0)):
        reasons.append(f"{report.get('contourability_open_loop_count')} open contour(s)")
    if int(report.get("contourability_bad_orientation_count", 0)):
        reasons.append(f"{report.get('contourability_bad_orientation_count')} reversed contour(s)")
    if float(report.get("contourability_tiny_loop_fraction", 0.0)) > float(report.get("max_tiny_loop_fraction", 0.5)):
        reasons.append("too many tiny loops")
    target_metric = float(report.get("contourability_target_metric_T", np.nan))
    target_limit = float(report.get("contourability_target_effective_allowed_metric_T", np.nan))
    if (
        bool(report.get("contourability_target_limit_enforced", False))
        and np.isfinite(target_metric)
        and np.isfinite(target_limit)
        and target_metric > target_limit
    ):
        metric_name = str(report.get("contourability_target_metric", "field metric"))
        reasons.append(f"contour target residual {metric_name} {target_metric:.4g} T > {target_limit:.4g} T")
    rms_error = float(report.get("contourability_field_rms_error_percent", np.nan))
    max_rms = float(report.get("max_rms_error_percent", np.nan))
    if np.isfinite(rms_error) and np.isfinite(max_rms) and rms_error > max_rms:
        reasons.append(f"contour field RMS error {rms_error:.3g}% > {max_rms:.3g}%")
    mean_error = float(report.get("contourability_field_mean_abs_error_percent", np.nan))
    max_mean = float(report.get("max_mean_error_percent", np.nan))
    if np.isfinite(mean_error) and np.isfinite(max_mean) and mean_error > max_mean:
        reasons.append(f"contour field mean error {mean_error:.3g}% > {max_mean:.3g}%")
    peak_error = float(report.get("contourability_field_peak_error_percent", np.nan))
    max_peak = float(report.get("max_peak_error_percent", np.nan))
    if np.isfinite(peak_error) and np.isfinite(max_peak) and peak_error > max_peak:
        reasons.append(f"contour field peak error {peak_error:.3g}% > {max_peak:.3g}%")
    return "; ".join(reasons) if reasons else "failed unspecified contourability checks"


def _format_report_candidate(report: dict[str, Any], prefix: str) -> str:
    levels = report.get(f"contourability_{prefix}_levels", report.get("contourability_candidate_levels", "?"))
    phase = report.get(f"contourability_{prefix}_phase_offset", "?")
    percentile = report.get(f"contourability_{prefix}_range_percentile", "?")
    strategy = report.get(f"contourability_{prefix}_zero_crossing_strategy", "?")
    return f"{levels} levels, phase {phase}, range {percentile}%, {strategy}"


def _write_part_contourability_diagnostics(parts: list[CoilPart], report: dict[str, Any]) -> None:
    for part in parts:
        part.contour_level_diagnostics.update(report)


def _snapshot_parts(parts: list[CoilPart]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for part in parts:
        snapshots.append(
            {
                "potential_level_list": None
                if part.potential_level_list is None
                else np.asarray(part.potential_level_list, dtype=float).copy(),
                "contour_step": part.contour_step,
                "contour_lines": deepcopy(part.contour_lines),
                "combined_loop_length": part.combined_loop_length,
                "contour_level_diagnostics": dict(part.contour_level_diagnostics),
                "field_by_loops": None
                if part.field_by_loops is None
                else np.asarray(part.field_by_loops, dtype=float).copy(),
            }
        )
    return snapshots


def _restore_parts(parts: list[CoilPart], snapshots: list[dict[str, Any]]) -> None:
    for part, snapshot in zip(parts, snapshots):
        part.potential_level_list = snapshot["potential_level_list"]
        part.contour_step = snapshot["contour_step"]
        part.contour_lines = deepcopy(snapshot["contour_lines"])
        part.combined_loop_length = snapshot["combined_loop_length"]
        part.contour_level_diagnostics = dict(snapshot["contour_level_diagnostics"])
        part.field_by_loops = (
            None if snapshot["field_by_loops"] is None else np.asarray(snapshot["field_by_loops"]).copy()
        )


def _parts_geometry_scale(parts: list[CoilPart]) -> float:
    vertices = [np.asarray(part.mesh.vertices, dtype=float) for part in parts if part.mesh.n_vertices]
    if not vertices:
        return 1.0
    points = np.vstack(vertices)
    extent = np.ptp(points, axis=0)
    scale = float(np.linalg.norm(extent))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _positive_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if np.isfinite(number) and number > 0.0 else float(default)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if values.size == 0 or weight_sum <= 0.0:
        return float("nan")
    return float(np.sqrt(np.dot(weights, values * values) / weight_sum))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if values.size == 0 or weight_sum <= 0.0:
        return float("nan")
    mean = float(np.dot(weights, values) / weight_sum)
    centered = values - mean
    return float(np.sqrt(np.dot(weights, centered * centered) / weight_sum))
