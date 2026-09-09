from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config_validation import ConfigValidationReport, validate_config
from .design_context import (
    CoilDesignContext,
    EvaluationFeature,
    normalize_design_context,
)
from .field_units import (
    PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
    config_field_display_units,
    config_target_field_units,
    field_values_from_tesla,
)
from .field_maps import shim_homogeneity_stats
from .gradient_linearity import gradient_linearity_enabled, gradient_linearity_summary
from .gradient_slice_maps import gradient_slice_map_summary_dict
from .engineering_metrics import (
    ConcomitantFieldReport,
    CurrentDensityReport,
    ReadinessReport,
    ThermalScreeningReport,
)
from .metrics import dsv_target_mask, suppression_target_mask, target_mode_fit_performance
from .models import CoilGenResult
from .regularization_diagnostics import regularization_selection_report
from .report_schema import REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ResultQuality:
    status: str
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def result_design_context(result: CoilGenResult) -> CoilDesignContext:
    return normalize_design_context(getattr(getattr(result, "input_data", None), "design_context", "mr_gradient"))


def result_quality_status(
    result: CoilGenResult,
    *,
    max_mean_field_error_percent: float = 10.0,
    max_peak_field_error_percent: float = 30.0,
) -> ResultQuality:
    """Classify a CoilGen result as PASS, WARNING, or FAIL for lab inspection.

    FAIL includes explicitly diagnostic constrained solves and failed solver
    contracts, so study ranking excludes them by default. WARNING requires
    inspection; PASS means no reported failures or warnings, not certification
    of checks that were never run. Legacy results need no solver-contract flag.
    """

    failures: list[str] = []
    warnings: list[str] = []
    route_statuses = [str(part.route_diagnostics.get("route_quality_status", "unknown")) for part in result.coil_parts]
    if any(status == "fail" for status in route_statuses):
        failures.append("Route-quality diagnostics report at least one failed coil part.")
    if any(status == "warn" for status in route_statuses):
        warnings.append("Route-quality diagnostics report at least one warning.")
    if any(
        part.manufacturing_diagnostics and not part.manufacturing_diagnostics.get("manufacturing_ok", True)
        for part in result.coil_parts
    ):
        failures.append("Manufacturing diagnostics report invalid swept-body geometry.")

    mean_error = float(result.error_vals.get("mean_rel_error_layout_vs_target", np.nan))
    peak_error = float(result.error_vals.get("max_rel_error_layout_vs_target", np.nan))
    if np.isfinite(mean_error) and mean_error > max_mean_field_error_percent:
        warnings.append(
            f"Mean DSV layout-vs-target Bz error is {mean_error:.3g}%, above {max_mean_field_error_percent:.3g}%."
        )
    if np.isfinite(peak_error) and peak_error > max_peak_field_error_percent:
        warnings.append(
            f"Peak DSV layout-vs-target Bz error is {peak_error:.3g}%, above {max_peak_field_error_percent:.3g}%."
        )

    diagnostics = result.regularization_diagnostics
    contract_met = diagnostics.get("constrained_solver_contract_met")
    physical_constraints_met = diagnostics.get("constrained_solver_physical_constraints_met")
    solver_failures: list[str] = []
    if diagnostics.get("constrained_solver_diagnostic_result", False):
        solver_failures.append("diagnostic-only result")
    if contract_met is not None and not bool(contract_met):
        solver_failures.append("full solver contract not met")
    if physical_constraints_met is not None and not bool(physical_constraints_met):
        solver_failures.append("physical constraints not met")
    if solver_failures:
        reason = str(diagnostics.get("constrained_solver_diagnostic_reason", "")).strip()
        if not reason:
            reason = ", ".join(str(name) for name in diagnostics.get("constrained_solver_failed_constraints", ()))
        detail = f" {reason}" if reason else ""
        failures.append(f"Constrained optimization result is not accepted: {', '.join(solver_failures)}.{detail}")
    config = getattr(result, "input_data", None)
    if config is not None and gradient_linearity_enabled(config):
        linearity = gradient_linearity_summary(
            _gradient_linearity_field_for_result(result), result.target_field, config
        )
        if not bool(linearity.get("linearity_met", True)):
            detail = str(linearity.get("linearity_failure_reasons", "")).strip()
            suffix = f" {detail}" if detail else ""
            failures.append(f"Gradient linearity specification failed.{suffix}")
    if diagnostics.get("fallback"):
        warnings.append(f"Regularization used fallback: {diagnostics.get('fallback_reason', 'unspecified')}.")
    constrained_backend = str(diagnostics.get("constrained_solver_backend", "")).strip().lower()
    if constrained_backend and "fallback" in constrained_backend:
        reason = str(diagnostics.get("constrained_solver_fallback_reason", "")).strip()
        detail = f" Reason: {reason}" if reason else ""
        warnings.append(
            "Constrained quadratic optimization did not use a direct conic solver; "
            f"backend is {constrained_backend}.{detail}"
        )
    if diagnostics.get("target_error_contour_step_guard_relaxed_for_constraint"):
        warnings.append(
            "The stream-function target was satisfied only by relaxing the contour-step guard; "
            "the SF target and layout discretization target are likely jointly infeasible for this fallback solve."
        )
    clearance_status = str(diagnostics.get("suppression_clearance_status", "")).upper()
    if clearance_status == "FAIL":
        failures.append("Generated suppression target violates the configured source-mesh clearance.")
    elif clearance_status == "WARNING":
        warnings.append("Generated suppression target has clearance warnings.")
    if diagnostics.get("turn_spacing_measured_threshold_met") is False:
        warnings.append("Measured contour spacing is below the requested min_turn_distance.")
    if diagnostics.get("surface_current_density_limit_met") is False:
        warnings.append("Solved stream-function surface-current density exceeds the requested engineering limit.")
    if diagnostics.get("contourability_checked") and not diagnostics.get("contourability_ok", True):
        warnings.append(f"Contourability validation failed: {diagnostics.get('contourability_reason', 'unspecified')}.")
    elif diagnostics.get("contourability_repaired"):
        warnings.append(
            f"Contourability repair changed contour levels: {diagnostics.get('contourability_reason', '')}."
        )
    if diagnostics.get("layout_target_checked") and not diagnostics.get("layout_target_met", False):
        metric = str(diagnostics.get("layout_target_metric", "field metric"))
        measured = float(diagnostics.get("layout_target_metric_T", np.nan))
        limit = float(diagnostics.get("layout_target_effective_limit_T", np.nan))
        source = str(diagnostics.get("layout_target_source", "layout"))
        invalid_samples = int(diagnostics.get("layout_target_invalid_sample_count", 0))
        if invalid_samples:
            failures.append(
                f"Realized {source} has {invalid_samples} non-finite active field samples; "
                "the layout target could not be validated."
            )
        else:
            failures.append(f"Realized {source} {metric} residual {measured:.6g} T exceeds the layout limit {limit:.6g} T.")
    try:
        regularization_report = regularization_selection_report(result)
    except ValueError:
        regularization_report = None
    if regularization_report is not None:
        warnings.extend(regularization_report.warnings)
    warnings.extend(contour_stability_warnings(result))

    status = "FAIL" if failures else "WARNING" if warnings else "PASS"
    return ResultQuality(status=status, failures=tuple(failures), warnings=tuple(warnings))


def _gradient_linearity_field_for_result(result: CoilGenResult) -> np.ndarray:
    """Return connected-layout field at the fitted gradient operating current."""

    field_per_amp = getattr(result, "field_layout_per1Amp", None)
    current = _gradient_linearity_layout_current_A(result)
    if field_per_amp is not None and np.isfinite(current):
        field = np.asarray(field_per_amp, dtype=float)
        if field.ndim == 2 and field.shape[1] == 3:
            return field * current
    return np.asarray(result.field_by_layout, dtype=float)


def _gradient_linearity_layout_current_A(result: CoilGenResult) -> float:
    layout_gradient = getattr(result, "layout_gradient", None)
    config = getattr(result, "input_data", None)
    if isinstance(layout_gradient, dict):
        try:
            achieved_per_amp = float(layout_gradient.get("achieved_gradient", np.nan))
        except (TypeError, ValueError):
            achieved_per_amp = float("nan")
        try:
            requested = abs(float(getattr(config, "target_gradient_strength", np.nan)))
        except (TypeError, ValueError):
            requested = float("nan")
        if not np.isfinite(requested) or requested <= 0.0:
            try:
                requested = abs(float(layout_gradient.get("desired_gradient", np.nan)))
            except (TypeError, ValueError):
                requested = float("nan")
        if np.isfinite(achieved_per_amp) and abs(achieved_per_amp) > 1.0e-30 and np.isfinite(requested):
            return float(requested / achieved_per_amp)
    try:
        return float(getattr(result, "needed_current_layout", np.nan))
    except (TypeError, ValueError):
        return float("nan")


def result_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return a compact SI-unit summary suitable for Markdown/CSV reports."""

    config = getattr(result, "input_data", None)
    design_context = result_design_context(result)
    sparsified_parts = [part for part in result.coil_parts if part.contour_sparsification is not None]
    sparsification_diagnostics = sparsified_parts[0].contour_sparsification.diagnostics if sparsified_parts else {}
    route_statuses = [str(part.route_diagnostics.get("route_quality_status", "unknown")) for part in result.coil_parts]
    min_wire_distances = [
        float(part.route_diagnostics.get("min_wire_wire_distance", np.nan))
        for part in result.coil_parts
        if part.route_diagnostics
    ]
    unconnected_contour_wire_distances = [
        float(
            (part.unconnected_contour_wire_distance_diagnostics or {}).get(
                "unconnected_contours_min_wire_distance",
                getattr(part, "unconnected_contour_min_wire_distance", np.nan),
            )
        )
        for part in result.coil_parts
        if int(
            (part.unconnected_contour_wire_distance_diagnostics or {}).get(
                "unconnected_contours_wire_distance_pair_count",
                0,
            )
        )
        > 0
    ]
    wire_clearance_failures = [
        int(part.route_diagnostics.get("wire_clearance_failure_count", 0))
        for part in result.coil_parts
        if part.route_diagnostics
    ]
    wire_clearance_failure_thresholds = [
        float(part.route_diagnostics.get("wire_clearance_failure_threshold", np.nan))
        for part in result.coil_parts
        if part.route_diagnostics
    ]
    wire_clearance_warnings = [
        int(part.route_diagnostics.get("wire_clearance_warning_count", 0))
        for part in result.coil_parts
        if part.route_diagnostics
    ]
    wire_clearance_warning_thresholds = [
        float(part.route_diagnostics.get("wire_clearance_warning_threshold", np.nan))
        for part in result.coil_parts
        if part.route_diagnostics
    ]
    contour_distances = [
        float(part.minimal_contour_distance)
        for part in result.coil_parts
        if np.isfinite(part.minimal_contour_distance) and part.minimal_contour_distance > 0.0
    ]
    symmetry = symmetry_summary_dict(result)
    inductance_summary = inductance_matrix_summary_dict(result.inductance_matrix)
    layout_gradient_summary = (
        layout_gradient_summary_dict(result.layout_gradient)
        if design_context.enabled(EvaluationFeature.GRADIENT_FIT)
        else {}
    )
    linearity_summary = (
        gradient_linearity_summary(_gradient_linearity_field_for_result(result), result.target_field, config)
        if config is not None
        else {}
    )
    shim_summary = (
        shim_homogeneity_summary_dict(result) if design_context.enabled(EvaluationFeature.SHIM_HOMOGENEITY) else {}
    )
    target_mode_summary = (
        target_mode_summary_dict(result)
        if design_context.enabled(EvaluationFeature.TARGET_MODE_FIT)
        else target_mode_not_applicable_summary_dict(result)
    )
    dsv_mask = dsv_target_mask(result.target_field)
    suppressed_mask = suppression_target_mask(result.target_field)
    try:
        regularization_report = regularization_selection_report(result)
        selected_scan = _selected_regularization_scan_values(result)
        regularization_summary = {
            "regularization_knee_type": regularization_report.knee_type,
            "regularization_residual_log_span": regularization_report.residual_log_span,
            "regularization_resistance_log_span": regularization_report.regularization_log_span,
            "regularization_selected_residual_position": regularization_report.selected_residual_position,
            "regularization_selected_resistance_position": regularization_report.selected_regularization_position,
            **selected_scan,
        }
    except ValueError:
        diagnostics = result.regularization_diagnostics
        regularization_summary = {
            "selected_target_residual_norm_T_per_A": float(
                diagnostics.get("target_error_constraint_residual_norm_T", np.nan)
            ),
            "selected_resistance_seminorm_ohm_A": float(
                diagnostics.get("target_error_constraint_resistance_norm", np.nan)
            ),
        }
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "problem_schema_version": str(getattr(config, "problem_schema_version", "")),
        "design_context": design_context.key,
        "design_context_is_mr": bool(design_context.is_mr),
        "design_context_evaluation_features": ", ".join(design_context.feature_keys()),
        "target_field_component": str(getattr(config, "target_field_component", "Bz")),
        "field_evaluation_current_source": str(getattr(config, "field_evaluation_current_source", "target_fit")),
        "field_source_label": (
            f"{getattr(config, 'target_field_component', 'Bz')} target, "
            f"{getattr(config, 'field_evaluation_current_source', 'target_fit')} current"
        ),
        "layout_terminal_strategy": str(getattr(config, "layout_terminal_strategy", "undefined")),
        "layout_terminal_pad_count": int(getattr(config, "layout_terminal_pad_count", 0)),
        "layout_terminal_exit_count": int(getattr(config, "layout_terminal_exit_count", 0)),
        "layout_keepout_region_count": int(getattr(config, "layout_keepout_region_count", 0)),
        "layout_manufacturing_intent": str(getattr(config, "layout_manufacturing_intent", "")),
        "part_count": len(result.coil_parts),
        "target_point_count": int(len(result.target_field.coords)),
        "dsv_target_point_count": int(np.count_nonzero(dsv_mask)),
        "suppression_point_count": int(np.count_nonzero(suppressed_mask)),
        **field_unit_summary_dict(result),
        **shim_summary,
        "contour_count": int(sum(len(part.contour_lines) for part in result.coil_parts)),
        "nominal_contour_geometry_count": int(sum(part.nominal_contour_count for part in result.coil_parts)),
        "physical_turn_count": int(sum(part.physical_turn_count for part in result.coil_parts)),
        "contour_sparsification_mode": str(getattr(config, "contour_sparsification_mode", "off")),
        "contour_sparsification_accepted_merge_count": int(
            sum(
                int(part.contour_sparsification.diagnostics.get("contour_sparsification_accepted_merge_count", 0))
                for part in sparsified_parts
            )
        ),
        "contour_sparsification_completed_pass_count": int(
            sparsification_diagnostics.get("contour_sparsification_completed_pass_count", 0)
        ),
        "contour_sparsification_achieved_max_turn_multiplicity": int(
            max(
                (
                    int(
                        part.contour_sparsification.diagnostics.get(
                            "contour_sparsification_achieved_max_turn_multiplicity",
                            0,
                        )
                    )
                    for part in sparsified_parts
                ),
                default=0,
            )
        ),
        "contour_sparsification_global_rms_field_error_percent": float(
            sparsification_diagnostics.get("contour_sparsification_global_rms_field_error_percent", 0.0)
        ),
        "contour_sparsification_global_peak_field_error_percent": float(
            sparsification_diagnostics.get("contour_sparsification_global_peak_field_error_percent", 0.0)
        ),
        "contour_sparsification_merge_rms_field_error_percent": float(
            sparsification_diagnostics.get(
                "contour_sparsification_merge_rms_field_error_percent",
                sparsification_diagnostics.get("contour_sparsification_global_rms_field_error_percent", 0.0),
            )
        ),
        "contour_sparsification_merge_peak_field_error_percent": float(
            sparsification_diagnostics.get(
                "contour_sparsification_merge_peak_field_error_percent",
                sparsification_diagnostics.get("contour_sparsification_global_peak_field_error_percent", 0.0),
            )
        ),
        "contour_sparsification_residual_budget_enabled": bool(
            sparsification_diagnostics.get("contour_sparsification_residual_budget_enabled", False)
        ),
        "contour_sparsification_residual_baseline_metric_T": float(
            sparsification_diagnostics.get("contour_sparsification_residual_baseline_metric_T", np.nan)
        ),
        "contour_sparsification_residual_allowed_metric_T": float(
            sparsification_diagnostics.get("contour_sparsification_residual_allowed_metric_T", np.nan)
        ),
        "contour_sparsification_merge_selected_residual_metric_T": float(
            sparsification_diagnostics.get("contour_sparsification_merge_selected_residual_metric_T", np.nan)
        ),
        "contour_sparsification_merge_residual_metric_change_T": float(
            sparsification_diagnostics.get("contour_sparsification_merge_residual_metric_change_T", np.nan)
        ),
        "contour_sparsification_pruning_mode": str(
            sparsification_diagnostics.get(
                "contour_sparsification_pruning_mode",
                getattr(config, "contour_sparsification_pruning_mode", "off"),
            )
        ),
        "contour_sparsification_pruning_removed_geometry_count": int(
            sparsification_diagnostics.get("contour_sparsification_pruning_removed_geometry_count", 0)
        ),
        "contour_sparsification_pruning_removed_turn_count": int(
            sparsification_diagnostics.get("contour_sparsification_pruning_removed_turn_count", 0)
        ),
        "contour_sparsification_pruning_baseline_metric_T": float(
            sparsification_diagnostics.get("contour_sparsification_pruning_baseline_metric_T", np.nan)
        ),
        "contour_sparsification_pruning_selected_metric_T": float(
            sparsification_diagnostics.get("contour_sparsification_pruning_selected_metric_T", np.nan)
        ),
        "contour_sparsification_pruning_metric_change_T": float(
            sparsification_diagnostics.get("contour_sparsification_pruning_metric_change_T", np.nan)
        ),
        "contour_sparsification_pruning_incremental_rms_field_error_percent": float(
            sparsification_diagnostics.get(
                "contour_sparsification_pruning_incremental_rms_field_error_percent",
                0.0,
            )
        ),
        "contour_sparsification_pruning_incremental_peak_field_error_percent": float(
            sparsification_diagnostics.get(
                "contour_sparsification_pruning_incremental_peak_field_error_percent",
                0.0,
            )
        ),
        "topological_group_count": int(sum(len(part.groups) for part in result.coil_parts)),
        "effective_contour_levels": int(result.num_levels),
        "contour_discretization_method": str(
            result.regularization_diagnostics.get(
                "contour_discretization_method",
                getattr(config, "contour_discretization_method", "level_count"),
            )
        ),
        "max_wire_current_A": float(
            result.regularization_diagnostics.get(
                "max_wire_current_A",
                max(float(getattr(config, "max_wire_current_A", 0.0)), float(getattr(config, "max_current_A", 0.0))),
            )
        ),
        "min_wire_distance_m": float(
            result.regularization_diagnostics.get(
                "min_wire_distance_m",
                max(
                    float(getattr(config, "min_wire_distance_m", 0.0)), float(getattr(config, "min_turn_distance", 0.0))
                ),
            )
        ),
        "engineering_max_contour_step_A": float(
            result.regularization_diagnostics.get("engineering_max_contour_step_A", np.nan)
        ),
        "engineering_required_contour_step_A": float(
            result.regularization_diagnostics.get("engineering_required_contour_step_A", np.nan)
        ),
        "engineering_current_limit_feasible": bool(
            result.regularization_diagnostics.get("engineering_current_limit_feasible", True)
        ),
        "surface_current_density_limit_A_per_m": float(
            result.regularization_diagnostics.get("surface_current_density_limit_A_per_m_effective", np.nan)
        ),
        "surface_current_density_limit_met": bool(
            result.regularization_diagnostics.get("surface_current_density_limit_met", True)
        ),
        "surface_current_density_max_A_per_m": float(
            result.regularization_diagnostics.get("surface_current_density_max_A_per_m", np.nan)
        ),
        "surface_current_density_p99_A_per_m": float(
            result.regularization_diagnostics.get("surface_current_density_p99_A_per_m", np.nan)
        ),
        "target_gradient_normalization_method": str(
            result.regularization_diagnostics.get("target_gradient_normalization_method", "")
        ),
        "target_gradient_axis": str(result.regularization_diagnostics.get("target_gradient_axis", "")),
        "target_gradient_requested_T_per_m": float(
            result.regularization_diagnostics.get("target_gradient_requested_T_per_m", np.nan)
        ),
        "target_gradient_hardware_effective_T_per_m": float(
            result.regularization_diagnostics.get("target_gradient_hardware_effective_T_per_m", np.nan)
        ),
        "gradient_hardware_feasible": bool(result.regularization_diagnostics.get("gradient_hardware_feasible", True)),
        "gradient_hardware_scale_factor": float(
            result.regularization_diagnostics.get("gradient_hardware_scale_factor", np.nan)
        ),
        **linearity_summary,
        "contourability_checked": bool(result.regularization_diagnostics.get("contourability_checked", False)),
        "contourability_ok": bool(result.regularization_diagnostics.get("contourability_ok", True)),
        "contourability_repaired": bool(result.regularization_diagnostics.get("contourability_repaired", False)),
        "contourability_reason": str(result.regularization_diagnostics.get("contourability_reason", "")),
        "contourability_requested_levels": int(
            result.regularization_diagnostics.get("contourability_requested_levels", result.num_levels)
        ),
        "contourability_selected_levels": int(
            result.regularization_diagnostics.get("contourability_selected_levels", result.num_levels)
        ),
        "contourability_field_rms_error_percent": float(
            result.regularization_diagnostics.get("contourability_field_rms_error_percent", np.nan)
        ),
        "contourability_field_mean_abs_error_percent": float(
            result.regularization_diagnostics.get("contourability_field_mean_abs_error_percent", np.nan)
        ),
        "contourability_field_peak_error_percent": float(
            result.regularization_diagnostics.get("contourability_field_peak_error_percent", np.nan)
        ),
        "contourability_total_length_m": float(
            result.regularization_diagnostics.get("contourability_total_length_m", np.nan)
        ),
        "layout_target_checked": bool(result.regularization_diagnostics.get("layout_target_checked", False)),
        "layout_target_met": bool(result.regularization_diagnostics.get("layout_target_met", True)),
        "layout_target_metric": str(result.regularization_diagnostics.get("layout_target_metric", "")),
        "layout_target_metric_T": float(result.regularization_diagnostics.get("layout_target_metric_T", np.nan)),
        "layout_target_limit_T": float(result.regularization_diagnostics.get("layout_target_limit_T", np.nan)),
        "layout_target_effective_limit_T": float(
            result.regularization_diagnostics.get("layout_target_effective_limit_T", np.nan)
        ),
        "layout_target_metric_ratio": float(
            result.regularization_diagnostics.get("layout_target_metric_ratio", np.nan)
        ),
        "layout_target_source": str(result.regularization_diagnostics.get("layout_target_source", "")),
        "contourability_target_metric_T": float(
            result.regularization_diagnostics.get("contourability_target_metric_T", np.nan)
        ),
        "contourability_target_allowed_metric_T": float(
            result.regularization_diagnostics.get("contourability_target_allowed_metric_T", np.nan)
        ),
        "contour_field_smoothing_applied": bool(
            result.regularization_diagnostics.get("contour_field_constrained_smoothing_applied", False)
        ),
        "contour_field_smoothing_improvement_percent": float(
            result.regularization_diagnostics.get("contour_field_smoothing_improvement_percent", np.nan)
        ),
        "contour_path_smoothing_scale_mode": str(
            result.regularization_diagnostics.get("contour_path_smoothing_scale_mode", "legacy")
        ),
        "contour_path_smoothing_mesh_scale_m": float(
            result.regularization_diagnostics.get("contour_path_smoothing_mesh_scale_m", np.nan)
        ),
        "contour_path_smoothing_mesh_gain_requested": float(
            result.regularization_diagnostics.get("contour_path_smoothing_mesh_gain_requested", np.nan)
        ),
        "contour_path_smoothing_mesh_gain_applied": float(
            result.regularization_diagnostics.get("contour_path_smoothing_mesh_gain_applied", np.nan)
        ),
        "contour_path_smoothing_displacement_cap_enabled": bool(
            result.regularization_diagnostics.get("contour_path_smoothing_displacement_cap_enabled", False)
        ),
        "contour_path_smoothing_max_applied_displacement_m": float(
            result.regularization_diagnostics.get("contour_path_smoothing_max_applied_displacement_m", np.nan)
        ),
        "contour_path_smoothing_contour_count": int(
            result.regularization_diagnostics.get("contour_path_smoothing_contour_count", 0)
        ),
        "contour_path_smoothing_spacing_rejected_count": int(
            result.regularization_diagnostics.get("contour_path_smoothing_spacing_rejected_count", 0)
        ),
        "contour_loop_pruning_mode": str(result.regularization_diagnostics.get("contour_loop_pruning_mode", "off")),
        "contour_loop_pruning_raw_loop_count": int(
            result.regularization_diagnostics.get("contour_loop_pruning_raw_loop_count", 0)
        ),
        "contour_loop_pruning_significance_removed_loop_count": int(
            result.regularization_diagnostics.get("contour_loop_pruning_significance_removed_loop_count", 0)
        ),
        "contour_loop_pruning_removed_loop_count": int(
            result.regularization_diagnostics.get("contour_loop_pruning_removed_loop_count", 0)
        ),
        "contour_loop_pruning_baseline_metric_T": float(
            result.regularization_diagnostics.get("contour_loop_pruning_baseline_metric_T", np.nan)
        ),
        "contour_loop_pruning_allowed_metric_T": float(
            result.regularization_diagnostics.get("contour_loop_pruning_allowed_metric_T", np.nan)
        ),
        "contour_loop_pruning_selected_metric_T": float(
            result.regularization_diagnostics.get("contour_loop_pruning_selected_metric_T", np.nan)
        ),
        "contour_loop_pruning_metric_change_T": float(
            result.regularization_diagnostics.get("contour_loop_pruning_metric_change_T", np.nan)
        ),
        "contour_loop_pruning_symmetry_grouping_complete": bool(
            result.regularization_diagnostics.get("contour_loop_pruning_symmetry_grouping_complete", False)
        ),
        "potential_step_A": float(result.potential_step),
        "needed_current_A": float(result.needed_current_layout),
        "coil_length_m": _first_or_nan(result.coil_lengths),
        "coil_length_total_m": _sum_or_nan(result.coil_lengths),
        "coil_resistance_ohm": _first_or_nan(result.resistance),
        "coil_resistance_total_ohm": _sum_or_nan(result.resistance),
        "coil_inductance_H": _matrix_first_or_nan(result.inductance_matrix),
        "coil_inductance_same_current_H": _same_current_inductance(result.inductance_matrix),
        "min_contour_distance_m": min(contour_distances) if contour_distances else float("nan"),
        "unconnected_contours_min_wire_distance_m": (
            min(unconnected_contour_wire_distances) if unconnected_contour_wire_distances else float("nan")
        ),
        "min_wire_wire_distance_m": min(min_wire_distances) if min_wire_distances else float("nan"),
        "wire_clearance_failure_count": int(sum(wire_clearance_failures)),
        "wire_clearance_failure_threshold_m": (
            max(wire_clearance_failure_thresholds) if wire_clearance_failure_thresholds else float("nan")
        ),
        "wire_clearance_warning_count": int(sum(wire_clearance_warnings)),
        "wire_clearance_warning_threshold_m": (
            max(wire_clearance_warning_thresholds) if wire_clearance_warning_thresholds else float("nan")
        ),
        "route_statuses": ", ".join(route_statuses),
        "stream_function_scheme": str(result.regularization_diagnostics.get("stream_function_scheme", "unknown")),
        "regularization_method": str(result.regularization_diagnostics.get("selection_method", "unknown")),
        "regularization_factor": float(result.regularization_factor),
        "regularization_lambda": float(result.regularization_lambda),
        "regularization_auto_range": bool(result.regularization_diagnostics.get("regularization_auto_range", False)),
        "estimated_regularization_contour_step_A": float(
            result.regularization_diagnostics.get("estimated_selected_contour_step_A", np.nan)
        ),
        "regularization_auto_range_max_contour_step_A": float(
            result.regularization_diagnostics.get("regularization_auto_range_max_contour_step_A", np.nan)
        ),
        "target_error_contour_step_guard_relaxed_for_constraint": bool(
            result.regularization_diagnostics.get("target_error_contour_step_guard_relaxed_for_constraint", False)
        ),
        "contour_step_constraint_enabled": bool(
            result.regularization_diagnostics.get("contour_step_constraint_enabled", False)
        ),
        "contour_step_constraint_source": str(
            result.regularization_diagnostics.get("contour_step_constraint_source", "")
        ),
        "contour_step_constraint_max_step_A": float(
            result.regularization_diagnostics.get("contour_step_constraint_max_step_A", np.nan)
        ),
        "contour_step_constraint_level_count": int(
            result.regularization_diagnostics.get("contour_step_constraint_level_count", 0)
        ),
        "contour_step_constraint_max_range_A": float(
            result.regularization_diagnostics.get("contour_step_constraint_max_range_A", np.nan)
        ),
        "surface_current_density_constraint_enabled": bool(
            result.regularization_diagnostics.get("surface_current_density_constraint_enabled", False)
        ),
        "surface_current_density_constraint_source": str(
            result.regularization_diagnostics.get("surface_current_density_constraint_source", "")
        ),
        "surface_current_density_constraint_limit_A_per_m": float(
            result.regularization_diagnostics.get("surface_current_density_constraint_limit_A_per_m", np.nan)
        ),
        "surface_current_density_constraint_derived_limit_A_per_m": float(
            result.regularization_diagnostics.get(
                "surface_current_density_constraint_derived_limit_A_per_m",
                np.nan,
            )
        ),
        "shield_gradient_opposition_requested": bool(
            result.regularization_diagnostics.get("shield_gradient_opposition_requested", False)
        ),
        "shield_gradient_opposition_enabled": bool(
            result.regularization_diagnostics.get("shield_gradient_opposition_enabled", False)
        ),
        "shield_gradient_opposition_constraint_met": bool(
            result.regularization_diagnostics.get("shield_gradient_opposition_constraint_met", True)
        ),
        "shield_gradient_opposition_value_T_per_m": float(
            result.regularization_diagnostics.get("shield_gradient_opposition_value_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_limit_T_per_m": float(
            result.regularization_diagnostics.get("shield_gradient_opposition_limit_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_margin_T_per_m": float(
            result.regularization_diagnostics.get("shield_gradient_opposition_margin_T_per_m", np.nan)
        ),
        "shield_gradient_opposition_reason": str(
            result.regularization_diagnostics.get("shield_gradient_opposition_reason", "")
        ),
        "shield_gradient_opposition_part_indices": str(
            result.regularization_diagnostics.get("shield_gradient_opposition_part_indices", "")
        ),
        "estimated_guarded_contour_step_A": float(
            result.regularization_diagnostics.get("estimated_guarded_contour_step_A", np.nan)
        ),
        "target_error_constraint_mode": str(result.regularization_diagnostics.get("target_error_constraint_mode", "")),
        "target_error_constraint_metric": str(
            result.regularization_diagnostics.get("target_error_constraint_metric", "")
        ),
        "target_error_constraint_met": bool(
            result.regularization_diagnostics.get("target_error_constraint_met", False)
        ),
        "target_error_tolerance_percent": float(
            result.regularization_diagnostics.get("target_error_tolerance_percent", np.nan)
        ),
        "target_error_constraint_relative_residual_percent": float(
            result.regularization_diagnostics.get("target_error_constraint_relative_residual_percent", np.nan)
        ),
        "target_error_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("target_error_constraint_field_metric_T", np.nan)
        ),
        "target_error_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("target_error_allowed_field_metric_T", np.nan)
        ),
        "linearity_rms_constraint_met": bool(
            result.regularization_diagnostics.get("linearity_rms_constraint_met", False)
        ),
        "linearity_rms_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("linearity_rms_constraint_field_metric_T", np.nan)
        ),
        "linearity_rms_constraint_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("linearity_rms_constraint_allowed_field_metric_T", np.nan)
        ),
        "linearity_peak_constraint_met": bool(
            result.regularization_diagnostics.get("linearity_peak_constraint_met", False)
        ),
        "linearity_peak_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("linearity_peak_constraint_field_metric_T", np.nan)
        ),
        "linearity_peak_constraint_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("linearity_peak_constraint_allowed_field_metric_T", np.nan)
        ),
        "gradient_strength_constraint_met": bool(
            result.regularization_diagnostics.get("gradient_strength_constraint_met", False)
        ),
        "gradient_strength_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("gradient_strength_constraint_field_metric_T", np.nan)
        ),
        "gradient_strength_constraint_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("gradient_strength_constraint_allowed_field_metric_T", np.nan)
        ),
        "cross_gradient_constraint_met": bool(
            result.regularization_diagnostics.get("cross_gradient_constraint_met", False)
        ),
        "cross_gradient_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("cross_gradient_constraint_field_metric_T", np.nan)
        ),
        "cross_gradient_constraint_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("cross_gradient_constraint_allowed_field_metric_T", np.nan)
        ),
        "target_error_auto_retry_count": int(result.regularization_diagnostics.get("target_error_auto_retry_count", 0)),
        "target_error_auto_retry_selected_rerun": int(
            result.regularization_diagnostics.get("target_error_auto_retry_selected_rerun", 0)
        ),
        "target_error_auto_retry_selected_violation_ratio": float(
            result.regularization_diagnostics.get("target_error_auto_retry_selected_violation_ratio", np.nan)
        ),
        "dsv_constraint_met": bool(result.regularization_diagnostics.get("dsv_constraint_met", False)),
        "dsv_constraint_metric": str(result.regularization_diagnostics.get("dsv_constraint_metric", "")),
        "dsv_constraint_relative_residual_percent": float(
            result.regularization_diagnostics.get("dsv_constraint_relative_residual_percent", np.nan)
        ),
        "dsv_constraint_allowed_residual_norm_T": float(
            result.regularization_diagnostics.get("dsv_constraint_allowed_residual_norm_T", np.nan)
        ),
        "dsv_constraint_field_metric_T": float(
            result.regularization_diagnostics.get("dsv_constraint_field_metric_T", np.nan)
        ),
        "dsv_constraint_allowed_field_metric_T": float(
            result.regularization_diagnostics.get("dsv_constraint_allowed_field_metric_T", np.nan)
        ),
        "dsv_constraint_std_T": float(result.regularization_diagnostics.get("dsv_constraint_std_T", np.nan)),
        "dsv_constraint_allowed_std_T": float(
            result.regularization_diagnostics.get("dsv_constraint_allowed_std_T", np.nan)
        ),
        "dsv_constraint_rms_residual_T": float(
            result.regularization_diagnostics.get("dsv_constraint_rms_residual_T", np.nan)
        ),
        "suppression_constraint_enabled": bool(
            result.regularization_diagnostics.get("suppression_constraint_enabled", False)
        ),
        "suppression_constraint_met": bool(result.regularization_diagnostics.get("suppression_constraint_met", False)),
        "suppression_constraint_relative_residual_percent": float(
            result.regularization_diagnostics.get("suppression_constraint_relative_residual_percent", np.nan)
        ),
        "suppression_constraint_allowed_residual_norm_T": float(
            result.regularization_diagnostics.get("suppression_constraint_allowed_residual_norm_T", np.nan)
        ),
        "suppression_constraint_rms_residual_T": float(
            result.regularization_diagnostics.get("suppression_constraint_rms_residual_T", np.nan)
        ),
        "suppression_target_enabled": bool(result.regularization_diagnostics.get("suppression_target_enabled", False)),
        "suppression_target_component": str(result.regularization_diagnostics.get("suppression_target_component", "")),
        "suppression_surface_weighting": str(
            result.regularization_diagnostics.get("suppression_surface_weighting", "")
        ),
        "suppression_target_sample_count": int(
            result.regularization_diagnostics.get("suppression_target_sample_count", 0)
        ),
        "suppression_target_weight_mean": float(
            result.regularization_diagnostics.get("suppression_target_weight_mean", np.nan)
        ),
        "suppression_target_rms_residual_T": float(
            result.regularization_diagnostics.get("suppression_target_rms_residual_T", np.nan)
        ),
        "suppression_target_peak_abs_residual_T": float(
            result.regularization_diagnostics.get("suppression_target_peak_abs_residual_T", np.nan)
        ),
        "suppression_target_weighted_rms_residual_T": float(
            result.regularization_diagnostics.get("suppression_target_weighted_rms_residual_T", np.nan)
        ),
        "suppression_field_rms_T": float(result.regularization_diagnostics.get("suppression_field_rms_T", np.nan)),
        "suppression_field_rms_limit_T": float(
            result.regularization_diagnostics.get("suppression_field_rms_limit_T", np.nan)
        ),
        "suppression_field_peak_T": float(result.regularization_diagnostics.get("suppression_field_peak_T", np.nan)),
        "suppression_field_peak_limit_T": float(
            result.regularization_diagnostics.get(
                "suppression_field_peak_limit_T",
                result.regularization_diagnostics.get("suppression_peak_constraint_allowed_field_metric_T", np.nan),
            )
        ),
        **suppression_clearance_summary_dict(result),
        "combined_weighted_residual_norm_T": float(
            result.regularization_diagnostics.get("combined_weighted_residual_norm_T", np.nan)
        ),
        "constrained_solver_backend": str(result.regularization_diagnostics.get("constrained_solver_backend", "")),
        "constrained_solver_name": str(result.regularization_diagnostics.get("constrained_solver_name", "")),
        "constrained_solver_status": str(result.regularization_diagnostics.get("constrained_solver_status", "")),
        "constrained_solver_contract_met": result.regularization_diagnostics.get("constrained_solver_contract_met"),
        "constrained_solver_physical_constraints_met": result.regularization_diagnostics.get(
            "constrained_solver_physical_constraints_met"
        ),
        "constrained_solver_diagnostic_result": bool(
            result.regularization_diagnostics.get("constrained_solver_diagnostic_result", False)
        ),
        "constrained_solver_diagnostic_reason": str(
            result.regularization_diagnostics.get("constrained_solver_diagnostic_reason", "")
        ),
        "constrained_solver_failed_constraints": ", ".join(
            str(name) for name in result.regularization_diagnostics.get("constrained_solver_failed_constraints", ())
        ),
        "constrained_solver_fallback_reason": str(
            result.regularization_diagnostics.get("constrained_solver_fallback_reason", "")
        ),
        "stream_function_smoothness_enabled": bool(
            result.regularization_diagnostics.get("stream_function_smoothness_enabled", False)
        ),
        "stream_function_smoothness_factor": float(
            result.regularization_diagnostics.get("stream_function_smoothness_factor", np.nan)
        ),
        "stream_function_basis": str(result.regularization_diagnostics.get("stream_function_basis", "")),
        "stream_function_spectral_active": bool(
            result.regularization_diagnostics.get("stream_function_spectral_active", False)
        ),
        "stream_function_spectral_unknown_count_before": int(
            result.regularization_diagnostics.get("stream_function_spectral_unknown_count_before", 0)
        ),
        "stream_function_spectral_unknown_count_after": int(
            result.regularization_diagnostics.get("stream_function_spectral_unknown_count_after", 0)
        ),
        "stream_function_spectral_reduction_factor": float(
            result.regularization_diagnostics.get("stream_function_spectral_reduction_factor", np.nan)
        ),
        "stream_function_spectral_cutoff_length_m": float(
            result.regularization_diagnostics.get("stream_function_spectral_cutoff_length_m", np.nan)
        ),
        "stream_function_spectral_min_retained_wavelength_m": float(
            result.regularization_diagnostics.get("stream_function_spectral_min_retained_wavelength_m", np.nan)
        ),
        "sensitivity_target_point_count": int(
            result.regularization_diagnostics.get("sensitivity_target_point_count", 0)
        ),
        "sensitivity_mesh_vertex_count": int(result.regularization_diagnostics.get("sensitivity_mesh_vertex_count", 0)),
        "sensitivity_unknown_count": int(result.regularization_diagnostics.get("sensitivity_unknown_count", 0)),
        "sensitivity_matrix_part_count": int(result.regularization_diagnostics.get("sensitivity_matrix_part_count", 0)),
        "sensitivity_matrix_bytes": int(result.regularization_diagnostics.get("sensitivity_matrix_bytes", 0)),
        "sensitivity_matrix_MiB": float(result.regularization_diagnostics.get("sensitivity_matrix_MiB", np.nan)),
        "sensitivity_matrix_shapes": str(result.regularization_diagnostics.get("sensitivity_matrix_shapes", "")),
        "execution_time_s": _execution_time(result),
        **regularization_summary,
        **gradient_slice_map_summary_dict(result),
        **concomitant_field_summary_dict(result),
        **current_density_summary_dict(result),
        **thermal_screening_summary_dict(result),
        **readiness_summary_dict(result),
        **target_mode_summary,
        **layout_gradient_summary,
        **symmetry,
        **inductance_summary,
        **{key + "_percent": float(value) for key, value in result.error_vals.items()},
    }


def result_performance_table_rows(result: CoilGenResult) -> list[list[str]]:
    """Return the formatted rows used by the plot-book and terminal result summaries."""

    summary = result_summary_dict(result)
    quality = result_quality_status(result)
    design_context = result_design_context(result)
    rows: list[list[str]] = []

    def add(group: str, quantity: str, key: str | None = None, unit: str = "-", *, value: Any = None) -> None:
        raw_value = summary.get(key, value) if key is not None else value
        formatted = format_result_summary_value(raw_value)
        if formatted:
            rows.append([group, quantity, formatted, unit])

    part_count = int(summary.get("part_count", 0))
    add("Quality", "overall status", value=quality.status)
    add("Quality", "warning count", value=len(quality.warnings))
    add("Quality", "failure count", value=len(quality.failures))
    add("Conventions", "problem schema", "problem_schema_version")
    add("Conventions", "design context", "design_context")
    add("Conventions", "target field component", "target_field_component")
    add("Conventions", "field-current source", "field_evaluation_current_source")
    add("Conventions", "layout terminal strategy", "layout_terminal_strategy")
    add("Conventions", "layout manufacturing intent", "layout_manufacturing_intent")
    add("Design size", "coil parts", "part_count")
    add("Design size", "all field-sample points", "target_point_count")
    add("Design size", "DSV target points", "dsv_target_point_count")
    add("Design size", "suppression points", "suppression_point_count")
    add("Design size", "stream-function contours", "contour_count")
    add("Design size", "topological loop groups", "topological_group_count")
    add("Design size", "effective contour levels", "effective_contour_levels")
    if bool(summary.get("gradient_slice_maps_enabled", False)):
        add("Gradient slice maps", "status", "gradient_slice_maps_status")
        add("Gradient slice maps", "source", "gradient_slice_map_source")
        add("Gradient slice maps", "current convention", "gradient_slice_map_current")
        add("Gradient slice maps", "slice axis", "gradient_slice_map_axis")
        add("Gradient slice maps", "slice count", "gradient_slice_map_slice_count")
        add("Gradient slice maps", "pixel samples", "gradient_slice_map_sample_count")
        add("Gradient slice maps", "operating current", "gradient_slice_map_current_A", "A")
        add("Gradient slice maps", "pixel linearity RMS", "gradient_slice_map_linearity_rms_percent", "%")
        add("Gradient slice maps", "pixel linearity peak", "gradient_slice_map_linearity_peak_percent", "%")
        add("Gradient slice maps", "concomitant peak", "gradient_slice_map_concomitant_peak_abs_T", "T")

    if design_context.enabled(EvaluationFeature.TARGET_MODE_FIT):
        add(
            "DSV field fit",
            "connected-wire gradient efficiency",
            "layout_gradient_efficiency_mT_per_m_per_A",
            "mT m^-1 A^-1",
        )
        add("DSV field fit", "connected-wire fitted current", "layout_needed_current_A", "A")
        add("DSV field fit", "connected-wire mean relative error", "layout_mean_rel_error_percent", "%")
        add("DSV field fit", "connected-wire max relative error", "layout_max_rel_error_percent", "%")
        add("Gradient linearity", "status", "linearity_status")
        add("Gradient linearity", "gradient strength error", "linearity_gradient_strength_error_percent", "%")
        add("Gradient linearity", "cross-gradient error", "linearity_cross_gradient_percent", "%")
        add("Gradient linearity", "RMS nonlinear residual", "linearity_rms_percent", "%")
        add("Gradient linearity", "peak nonlinear residual", "linearity_peak_percent", "%")
        add(
            "DSV field fit",
            "unconnected-contour gradient efficiency",
            "unconnected_contours_gradient_efficiency_mT_per_m_per_A",
            "mT m^-1 A^-1",
        )
        add(
            "DSV field fit",
            "stream-function gradient efficiency",
            "stream_function_gradient_efficiency_mT_per_m_per_A",
            "mT m^-1 A^-1",
        )
        add("DSV field fit", "stream-function mean relative error", "stream_function_mean_rel_error_percent", "%")
    elif design_context.enabled(EvaluationFeature.FIELD_ERROR):
        add("Target field fit", "connected-wire fitted current", "layout_needed_current_A", "A")
        add("Target field fit", "connected-wire mean relative error", "layout_mean_rel_error_percent", "%")
        add("Target field fit", "connected-wire max relative error", "layout_max_rel_error_percent", "%")
    display_unit = str(summary.get("field_display_units", "T"))
    if design_context.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        add("Shimming", "wire-layout remaining std", f"shim_layout_residual_std_{display_unit}", display_unit)
        add("Shimming", "wire-layout remaining RMS", f"shim_layout_residual_rms_{display_unit}", display_unit)
        add("Shimming", "wire-layout p95 |remaining|", f"shim_layout_residual_p95_abs_{display_unit}", display_unit)
        add("Shimming", "wire-layout max |remaining|", f"shim_layout_residual_max_abs_{display_unit}", display_unit)
        add(
            "Shimming",
            "wire-layout remaining peak-to-peak",
            f"shim_layout_residual_peak_to_peak_{display_unit}",
            display_unit,
        )
        add(
            "Shimming",
            "stream-function diagnostic std",
            f"shim_stream_function_residual_std_{display_unit}",
            display_unit,
        )
        add(
            "Shimming",
            "stream-function diagnostic RMS",
            f"shim_stream_function_residual_rms_{display_unit}",
            display_unit,
        )
    if design_context.enabled(EvaluationFeature.TARGET_MODE_FIT):
        add(
            "Active shielding",
            "connected-wire suppression RMS Bz per ampere",
            "layout_suppressed_rms_Bz_T_per_A",
            "T A^-1",
        )
        add(
            "Active shielding",
            "connected-wire suppression peak |Bz| per ampere",
            "layout_suppressed_peak_abs_Bz_T_per_A",
            "T A^-1",
        )
        add(
            "Active shielding",
            "connected-wire suppression RMS |B| per ampere",
            "layout_suppressed_rms_Bmag_T_per_A",
            "T A^-1",
        )
        add(
            "Active shielding",
            "connected-wire suppression peak |B| per ampere",
            "layout_suppressed_peak_Bmag_T_per_A",
            "T A^-1",
        )
        add(
            "Active shielding",
            "connected-wire suppression RMS Bz at fitted current",
            "layout_suppressed_rms_Bz_T_at_fitted_current",
            "T",
        )
        add(
            "Active shielding",
            "connected-wire suppression peak |Bz| at fitted current",
            "layout_suppressed_peak_abs_Bz_T_at_fitted_current",
            "T",
        )
        add(
            "Active shielding",
            "connected-wire suppression RMS |B| at fitted current",
            "layout_suppressed_rms_Bmag_T_at_fitted_current",
            "T",
        )
        add(
            "Active shielding",
            "connected-wire suppression peak |B| at fitted current",
            "layout_suppressed_peak_Bmag_T_at_fitted_current",
            "T",
        )
        add(
            "Active shielding",
            "stream-function suppression RMS Bz at fitted current",
            "stream_function_suppressed_rms_Bz_T_at_fitted_current",
            "T",
        )

    if part_count > 1:
        add("Electrical", "total conductor length", "coil_length_total_m", "m")
        add("Electrical", "sum of part resistances", "coil_resistance_total_ohm", "Ohm")
        add("Electrical", "same-current total inductance", "coil_inductance_same_current_H", "H")
        add("Electrical", "first-part self inductance", "coil_inductance_H", "H")
    else:
        add("Electrical", "conductor length", "coil_length_m", "m")
        add("Electrical", "resistance", "coil_resistance_ohm", "Ohm")
        add("Electrical", "self inductance", "coil_inductance_H", "H")

    add("Routing", "route-quality statuses", "route_statuses")
    add("Routing", "minimum contour spacing", "min_contour_distance_m", "m")
    add("Routing", "minimum unconnected-contour wire clearance", "unconnected_contours_min_wire_distance_m", "m")
    add("Routing", "minimum wire-wire clearance", "min_wire_wire_distance_m", "m")
    add("Routing", "wire-clearance failure threshold", "wire_clearance_failure_threshold_m", "m")
    add("Routing", "wire-clearance failure count", "wire_clearance_failure_count")
    add("Routing", "potential step per turn", "potential_step_A", "A")

    add("Optimization", "stream-function solver", "stream_function_scheme")
    add("Optimization", "regularization method", "regularization_method")
    add("Optimization", "regularization factor", "regularization_factor")
    add("Optimization", "regularization lambda", "regularization_lambda")
    add("Optimization", "regularization auto range", "regularization_auto_range")
    add("Optimization", "estimated regularization contour step", "estimated_regularization_contour_step_A", "A")
    add("Optimization", "regularization contour-step guard", "regularization_auto_range_max_contour_step_A", "A")
    add(
        "Optimization",
        "contour-step guard relaxed for target",
        "target_error_contour_step_guard_relaxed_for_constraint",
    )
    add("Optimization", "contour-step constraint active", "contour_step_constraint_enabled")
    add("Optimization", "contour-step constraint source", "contour_step_constraint_source")
    add("Optimization", "contour-step constraint max step", "contour_step_constraint_max_step_A", "A")
    add("Optimization", "contour-step constraint levels", "contour_step_constraint_level_count")
    add("Optimization", "contour-step constraint max range", "contour_step_constraint_max_range_A", "A")
    add("Optimization", "surface-current-density constraint active", "surface_current_density_constraint_enabled")
    add("Optimization", "surface-current-density constraint source", "surface_current_density_constraint_source")
    add(
        "Optimization",
        "surface-current-density constraint limit",
        "surface_current_density_constraint_limit_A_per_m",
        "A/m",
    )
    add(
        "Optimization",
        "surface-current-density derived limit",
        "surface_current_density_constraint_derived_limit_A_per_m",
        "A/m",
    )
    add("Optimization", "estimated guarded contour step", "estimated_guarded_contour_step_A", "A")
    add("Optimization", "target-error constraint mode", "target_error_constraint_mode")
    add("Optimization", "target-error constraint metric", "target_error_constraint_metric")
    add("Optimization", "target-error constraint met", "target_error_constraint_met")
    add("Optimization", "target-error tolerance", "target_error_tolerance_percent", "%")
    add("Optimization", "achieved target-error residual", "target_error_constraint_relative_residual_percent", "%")
    add("Optimization", "target-error constrained field metric", "target_error_constraint_field_metric_T", "T")
    add("Optimization", "target-error allowed field metric", "target_error_allowed_field_metric_T", "T")
    add("Optimization", "linearity RMS constraint met", "linearity_rms_constraint_met")
    add("Optimization", "linearity peak constraint met", "linearity_peak_constraint_met")
    add("Optimization", "gradient-strength constraint met", "gradient_strength_constraint_met")
    add("Optimization", "cross-gradient constraint met", "cross_gradient_constraint_met")
    add("Optimization", "target-error auto retry count", "target_error_auto_retry_count")
    add("Optimization", "target-error auto retry selected rerun", "target_error_auto_retry_selected_rerun")
    add("Optimization", "target-error auto retry violation ratio", "target_error_auto_retry_selected_violation_ratio")
    add("Optimization", "DSV constraint met", "dsv_constraint_met")
    add("Optimization", "DSV constraint metric", "dsv_constraint_metric")
    add("Optimization", "DSV residual", "dsv_constraint_relative_residual_percent", "%")
    add("Optimization", "DSV residual std", "dsv_constraint_std_T", "T")
    add("Optimization", "DSV allowed residual std", "dsv_constraint_allowed_std_T", "T")
    add("Optimization", "suppression constraint enabled", "suppression_constraint_enabled")
    add("Optimization", "suppression constraint met", "suppression_constraint_met")
    add("Optimization", "suppression residual", "suppression_constraint_relative_residual_percent", "%")
    add("Optimization", "suppression target enabled", "suppression_target_enabled")
    add("Optimization", "suppression target component", "suppression_target_component")
    add("Optimization", "suppression surface weighting", "suppression_surface_weighting")
    add("Optimization", "suppression target RMS", "suppression_target_rms_residual_T", "T")
    add("Optimization", "suppression target weight", "suppression_target_weight_mean")
    add("Optimization", "suppression RMS field", "suppression_field_rms_T", "T")
    add("Optimization", "suppression RMS limit", "suppression_field_rms_limit_T", "T")
    add("Optimization", "suppression peak field", "suppression_field_peak_T", "T")
    add("Optimization", "suppression peak limit", "suppression_field_peak_limit_T", "T")
    add("Optimization", "constrained solver backend", "constrained_solver_backend")
    add("Optimization", "constrained solver contract met", "constrained_solver_contract_met")
    add("Optimization", "constrained physical constraints met", "constrained_solver_physical_constraints_met")
    add("Optimization", "constrained diagnostic-only result", "constrained_solver_diagnostic_result")
    add("Optimization", "constrained diagnostic reason", "constrained_solver_diagnostic_reason")
    add("Optimization", "constrained failed constraints", "constrained_solver_failed_constraints")
    add("Optimization", "constrained solver fallback reason", "constrained_solver_fallback_reason")
    add("Optimization", "stream-function smoothness factor", "stream_function_smoothness_factor")
    add("Optimization", "stream-function basis", "stream_function_basis")
    add("Optimization", "spectral basis active", "stream_function_spectral_active")
    add("Optimization", "spectral unknowns before", "stream_function_spectral_unknown_count_before")
    add("Optimization", "spectral unknowns after", "stream_function_spectral_unknown_count_after")
    add("Optimization", "spectral reduction factor", "stream_function_spectral_reduction_factor")
    add("Optimization", "spectral cutoff length", "stream_function_spectral_cutoff_length_m", "m")
    add("Optimization", "spectral retained min wavelength", "stream_function_spectral_min_retained_wavelength_m", "m")
    add("Optimization", "execution time", "execution_time_s", "s")

    add("Symmetry", "symmetry reduction active", "symmetry_reduction_active")
    add("Symmetry", "symmetry reduction factor", "symmetry_reduction_factor")
    add("Symmetry", "contour symmetry pairs", "contour_symmetry_pair_count")
    add("Symmetry", "contour symmetry failures", "contour_symmetry_failed_pair_count")
    return rows


def format_result_summary_value(value: Any) -> str:
    """Format a scalar result value for human-readable terminal/table output."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (np.bool_,)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return ""
        magnitude = abs(number)
        if number == 0.0:
            return "0"
        if magnitude < 1e-3 or magnitude >= 1e4:
            return f"{number:.4e}"
        return f"{number:.6g}"
    if isinstance(value, str):
        text = value.strip()
        return text if text and text.lower() != "nan" else ""
    return str(value)


def field_unit_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return user-facing field-unit metadata and optional display-unit stats."""

    config = result.input_data
    target_units = config_target_field_units(config)
    display_units = config_field_display_units(config)
    summary: dict[str, Any] = {
        "target_field_units": target_units,
        "field_display_units": display_units,
        "proton_frequency_Hz_per_T": PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
    }
    if display_units == "T":
        return summary

    dsv_mask = dsv_target_mask(result.target_field)
    if not np.any(dsv_mask):
        return summary
    suffix = display_units
    target = field_values_from_tesla(np.asarray(result.target_field.b, dtype=float)[:, 2], display_units)
    layout = field_values_from_tesla(np.asarray(result.field_by_layout, dtype=float)[:, 2], display_units)
    stream = field_values_from_tesla(np.asarray(result.b_field_opt_sf, dtype=float)[:, 2], display_units)
    target_1a = field_values_from_tesla(np.asarray(result.target_field_1A.b, dtype=float)[:, 2], display_units)
    layout_1a = field_values_from_tesla(np.asarray(result.field_layout_per1Amp, dtype=float)[:, 2], display_units)
    summary.update(
        {
            f"target_peak_{suffix}": _finite_max_abs(target[dsv_mask]),
            f"layout_peak_{suffix}": _finite_max_abs(layout[dsv_mask]),
            f"stream_function_peak_{suffix}": _finite_max_abs(stream[dsv_mask]),
            f"layout_rms_error_{suffix}": _finite_rms(layout[dsv_mask] - target[dsv_mask]),
            f"stream_function_rms_error_{suffix}": _finite_rms(stream[dsv_mask] - target[dsv_mask]),
            f"target_peak_{suffix}_per_A": _finite_max_abs(target_1a[dsv_mask]),
            f"layout_peak_{suffix}_per_A": _finite_max_abs(layout_1a[dsv_mask]),
        }
    )
    return summary


def shim_homogeneity_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return final-current shimming residual statistics in display units."""

    summary: dict[str, Any] = {}
    for source, label in (("after", "layout"), ("after_stream", "stream_function")):
        stats = shim_homogeneity_stats(result, source=source, domain="dsv")
        unit = str(stats.get("unit", config_field_display_units(result.input_data)))
        prefix = f"shim_{label}_residual"
        summary[f"{prefix}_sample_count"] = int(stats.get("sample_count", 0.0))
        for key in ("mean", "median", "std", "rms", "p95_abs", "max_abs", "peak_to_peak"):
            summary[f"{prefix}_{key}_{unit}"] = float(stats.get(key, np.nan))
    return summary


def _selected_regularization_scan_values(result: CoilGenResult) -> dict[str, float]:
    diagnostics = getattr(result, "regularization_diagnostics", {}) or {}
    selected_index = int(diagnostics.get("selected_index", -1))
    residuals = np.asarray(diagnostics.get("residual_norms", ()), dtype=float)
    resistance = np.asarray(diagnostics.get("regularization_norms", ()), dtype=float)
    selected_residual = float(diagnostics.get("target_error_constraint_residual_norm_T", np.nan))
    selected_resistance = float(diagnostics.get("target_error_constraint_resistance_norm", np.nan))
    if 0 <= selected_index < len(residuals) and np.isfinite(residuals[selected_index]):
        selected_residual = float(residuals[selected_index])
    if 0 <= selected_index < len(resistance) and np.isfinite(resistance[selected_index]):
        selected_resistance = float(resistance[selected_index])
    return {
        "selected_target_residual_norm_T_per_A": selected_residual,
        "selected_resistance_seminorm_ohm_A": selected_resistance,
    }


def inductance_matrix_summary_dict(values: Any) -> dict[str, float]:
    """Return per-part self and mutual inductance entries in SI units.

    The matrix signs include the actual wire/path orientation. For a coil whose
    independent parts are driven with the same current amplitude and direction,
    the effective total inductance is ``1.T @ L @ 1`` and is reported separately
    as ``coil_inductance_same_current_H``.
    """

    if values is None or np.size(values) == 0:
        return {}
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return {}
    summary: dict[str, float] = {
        "coil_inductance_matrix_part_count": float(matrix.shape[0]),
    }
    for index in range(matrix.shape[0]):
        summary[f"coil_inductance_self_part{index + 1}_H"] = float(matrix[index, index])
    for row in range(matrix.shape[0]):
        for col in range(row + 1, matrix.shape[1]):
            summary[f"coil_mutual_inductance_part{row + 1}_part{col + 1}_H"] = float(matrix[row, col])
    return summary


def layout_gradient_summary_dict(values: Any) -> dict[str, float]:
    """Flatten the optional DSV gradient-tensor fit into report-safe keys."""

    if not isinstance(values, dict):
        return {}
    key_map = {
        "offset": "layout_gradient_offset_T",
        "dbz_dx": "layout_gradient_dbz_dx_T_per_m",
        "dbz_dy": "layout_gradient_dbz_dy_T_per_m",
        "dbz_dz": "layout_gradient_dbz_dz_T_per_m",
        "desired_dbz_dx": "layout_gradient_desired_dbz_dx_T_per_m",
        "desired_dbz_dy": "layout_gradient_desired_dbz_dy_T_per_m",
        "desired_dbz_dz": "layout_gradient_desired_dbz_dz_T_per_m",
        "desired_gradient": "layout_gradient_desired_T_per_m",
        "achieved_gradient": "layout_gradient_achieved_T_per_m",
        "cross_gradient": "layout_gradient_cross_T_per_m",
        "cross_gradient_percent_of_desired": "layout_gradient_cross_percent_of_desired",
        "gradient_alignment_cosine": "layout_gradient_alignment_cosine",
        "rms_residual": "layout_gradient_linear_rms_residual_T",
        "linear_rms_residual_percent_of_peak": "layout_gradient_linear_rms_residual_percent_of_peak",
        "quadratic_rms_residual": "layout_gradient_quadratic_rms_residual_T",
        "quadratic_rms_residual_percent_of_peak": "layout_gradient_quadratic_rms_residual_percent_of_peak",
        "qxx": "layout_gradient_qxx_T_per_m2",
        "qyy": "layout_gradient_qyy_T_per_m2",
        "qzz": "layout_gradient_qzz_T_per_m2",
        "qxy": "layout_gradient_qxy_T_per_m2",
        "qxz": "layout_gradient_qxz_T_per_m2",
        "qyz": "layout_gradient_qyz_T_per_m2",
        "second_order_rms": "layout_gradient_second_order_rms_T_per_m2",
        "sample_count": "layout_gradient_fit_sample_count",
    }
    summary: dict[str, float] = {}
    for source, dest in key_map.items():
        try:
            summary[dest] = float(values.get(source, np.nan))
        except (TypeError, ValueError):
            summary[dest] = float("nan")
    return summary


def concomitant_field_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return Maxwell concomitant-field metrics for the fitted layout gradient."""

    if not result_design_context(result).enabled(EvaluationFeature.CONCOMITANT_FIELD):
        data = ConcomitantFieldReport.nan().as_dict()
        data["concomitant_gradient_source"] = "not_applicable_for_design_context"
        return data
    if not isinstance(result.layout_gradient, dict):
        return ConcomitantFieldReport.nan().as_dict()
    gradient_per_amp = np.asarray(
        [
            result.layout_gradient.get("dbz_dx", np.nan),
            result.layout_gradient.get("dbz_dy", np.nan),
            result.layout_gradient.get("dbz_dz", np.nan),
        ],
        dtype=float,
    )
    current = float(getattr(result, "needed_current_layout", np.nan))
    gradient_at_current = gradient_per_amp * current if np.isfinite(current) else np.full(3, np.nan, dtype=float)
    dsv_mask = dsv_target_mask(result.target_field)
    target_bz = np.asarray(result.target_field.b, dtype=float)[:, 2]
    target_peak = _finite_max_abs(target_bz[dsv_mask]) if np.any(dsv_mask) else float("nan")
    config = result.input_data
    report = ConcomitantFieldReport.from_linear_gradient(
        result.target_field.coords,
        gradient_at_current,
        main_field_T=float(getattr(config, "b_0_strength_T", np.nan)),
        b0_direction=getattr(config, "b_0_direction", (0.0, 0.0, 1.0)),
        sample_mask=dsv_mask,
        target_peak_T=target_peak,
    )
    data = report.as_dict()
    data["concomitant_gradient_source"] = "connected_layout_fit_at_needed_current"
    return data


def current_density_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return configured-conductor current-density screening metrics."""

    return CurrentDensityReport.from_result(result).as_dict()


def suppression_clearance_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return geometric clearance checks for generated suppression targets."""

    diagnostics = getattr(result, "regularization_diagnostics", {}) or {}
    return {
        "suppression_clearance_status": str(diagnostics.get("suppression_clearance_status", "not_checked")),
        "suppression_clearance_generator": str(diagnostics.get("suppression_clearance_generator", "")),
        "suppression_clearance_action": str(diagnostics.get("suppression_clearance_action", "")),
        "suppression_clearance_requested_m": float(diagnostics.get("suppression_clearance_requested_m", np.nan)),
        "suppression_clearance_min_allowed_m": float(diagnostics.get("suppression_clearance_min_allowed_m", np.nan)),
        "suppression_clearance_min_m": float(diagnostics.get("suppression_clearance_min_m", np.nan)),
        "suppression_clearance_p05_m": float(diagnostics.get("suppression_clearance_p05_m", np.nan)),
        "suppression_clearance_median_m": float(diagnostics.get("suppression_clearance_median_m", np.nan)),
        "suppression_clearance_max_m": float(diagnostics.get("suppression_clearance_max_m", np.nan)),
        "suppression_clearance_input_vertex_count": int(diagnostics.get("suppression_clearance_input_vertex_count", 0)),
        "suppression_clearance_output_vertex_count": int(
            diagnostics.get("suppression_clearance_output_vertex_count", 0)
        ),
        "suppression_clearance_output_face_count": int(diagnostics.get("suppression_clearance_output_face_count", 0)),
        "suppression_clearance_dropped_face_count": int(diagnostics.get("suppression_clearance_dropped_face_count", 0)),
        "suppression_clearance_violation_count": int(
            diagnostics.get(
                "suppression_clearance_violation_count",
                diagnostics.get("suppression_clearance_violating_output_vertex_count", 0),
            )
        ),
        "suppression_clearance_warning": str(diagnostics.get("suppression_clearance_warning", "")),
    }


def suppression_clearance_report_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return compact table values for the suppression-clearance report."""

    summary = suppression_clearance_summary_dict(result)
    keys = (
        "suppression_clearance_status",
        "suppression_clearance_generator",
        "suppression_clearance_action",
        "suppression_clearance_requested_m",
        "suppression_clearance_min_allowed_m",
        "suppression_clearance_min_m",
        "suppression_clearance_p05_m",
        "suppression_clearance_median_m",
        "suppression_clearance_max_m",
        "suppression_clearance_input_vertex_count",
        "suppression_clearance_output_vertex_count",
        "suppression_clearance_output_face_count",
        "suppression_clearance_dropped_face_count",
        "suppression_clearance_violation_count",
        "suppression_clearance_warning",
    )
    return {key: summary.get(key, "") for key in keys}


def thermal_screening_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return thermal-screening status for reports.

    CoilGen does not yet evaluate thermal duty cycle, cooling, or temperature
    rise. This explicit report prevents a magnetostatic PASS from being read as
    full engineering readiness.
    """

    return ThermalScreeningReport.not_evaluated().as_dict()


def readiness_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return design-readiness status separated from magnetostatic feasibility."""

    quality = result_quality_status(result)
    route_statuses = [str(part.route_diagnostics.get("route_quality_status", "unknown")) for part in result.coil_parts]
    if any(status == "fail" for status in route_statuses):
        layout_status = "FAIL"
    elif any(status == "warn" for status in route_statuses):
        layout_status = "WARNING"
    elif route_statuses and all(status == "pass" for status in route_statuses):
        layout_status = "PASS"
    else:
        layout_status = "not_checked"
    return ReadinessReport.from_status(
        magnetostatic_status=quality.status,
        layout_status=layout_status,
    ).as_dict()


def target_mode_not_applicable_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return target-mode keys without running gradient-oriented fit metrics."""

    nan = float("nan")
    summary: dict[str, Any] = {}
    error_map = {
        "layout": "layout",
        "unconnected_contours": "unconnected_contours",
    }
    for source in ("layout", "unconnected_contours", "stream_function"):
        summary[f"{source}_gradient_efficiency_mT_per_m_per_A"] = nan
        summary[f"{source}_needed_current_A"] = float(result.needed_current_layout) if source == "layout" else nan
        if source in error_map:
            stem = error_map[source]
            summary[f"{source}_mean_rel_error_percent"] = float(
                result.error_vals.get(f"mean_rel_error_{stem}_vs_target", nan)
            )
            summary[f"{source}_max_rel_error_percent"] = float(
                result.error_vals.get(f"max_rel_error_{stem}_vs_target", nan)
            )
        else:
            summary[f"{source}_mean_rel_error_percent"] = nan
            summary[f"{source}_max_rel_error_percent"] = nan
        summary[f"{source}_fit_sample_count"] = 0
    return summary


def target_mode_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return target-mode fit metrics for stream, contours, and routed layout."""

    target_bz = np.asarray(result.target_field.b, dtype=float)[:, 2]
    dsv_mask = dsv_target_mask(result.target_field)
    suppressed_mask = suppression_target_mask(result.target_field)
    target_scale = float(result.input_data.target_gradient_strength)
    fits = {
        "layout": target_mode_fit_performance(
            result.field_layout_per1Amp,
            target_bz,
            target_scale,
            sample_mask=dsv_mask,
        ),
        "unconnected_contours": target_mode_fit_performance(
            result.field_loops_per1Amp,
            target_bz,
            target_scale,
            sample_mask=dsv_mask,
        ),
        "stream_function": target_mode_fit_performance(
            result.b_field_opt_sf_1A,
            target_bz,
            target_scale,
            sample_mask=dsv_mask,
        ),
    }
    summary: dict[str, Any] = {}
    for source, fit in fits.items():
        summary[f"{source}_gradient_efficiency_mT_per_m_per_A"] = float(fit["gradient_efficiency_mT_per_m_per_A"])
        summary[f"{source}_needed_current_A"] = float(fit["needed_current_A"])
        summary[f"{source}_mean_rel_error_percent"] = float(fit["mean_rel_error_percent"])
        summary[f"{source}_max_rel_error_percent"] = float(fit["max_rel_error_percent"])
        summary[f"{source}_fit_sample_count"] = int(fit["sample_count"])
        summary.update(
            _suppression_field_summary(
                source,
                {
                    "layout": result.field_layout_per1Amp,
                    "unconnected_contours": result.field_loops_per1Amp,
                    "stream_function": result.b_field_opt_sf_1A,
                }[source],
                suppressed_mask=suppressed_mask,
                current=float(fit["needed_current_A"]),
            )
        )
    return summary


def _suppression_field_summary(
    source: str,
    field_per_amp: Any,
    *,
    suppressed_mask: np.ndarray,
    current: float,
) -> dict[str, float]:
    if field_per_amp is None or not np.any(suppressed_mask):
        return {
            f"{source}_suppressed_rms_Bz_T_per_A": float("nan"),
            f"{source}_suppressed_peak_abs_Bz_T_per_A": float("nan"),
            f"{source}_suppressed_rms_Bmag_T_per_A": float("nan"),
            f"{source}_suppressed_peak_Bmag_T_per_A": float("nan"),
            f"{source}_suppressed_rms_Bz_T_at_fitted_current": float("nan"),
            f"{source}_suppressed_peak_abs_Bz_T_at_fitted_current": float("nan"),
            f"{source}_suppressed_rms_Bmag_T_at_fitted_current": float("nan"),
            f"{source}_suppressed_peak_Bmag_T_at_fitted_current": float("nan"),
        }
    field = np.asarray(field_per_amp, dtype=float)
    if field.ndim == 2:
        if field.shape[1] < 3:
            bz = np.zeros(0, dtype=float)
            bmag = np.zeros(0, dtype=float)
        else:
            bz = field[:, 2]
            bmag = np.linalg.norm(field[:, :3], axis=1)
    else:
        bz = field.reshape(-1)
        bmag = np.abs(bz)
    mask = np.asarray(suppressed_mask, dtype=bool).reshape(-1)
    if len(bz) != len(mask):
        selected = np.zeros(0, dtype=float)
        selected_bmag = np.zeros(0, dtype=float)
    else:
        selected = bz[mask]
        selected = selected[np.isfinite(selected)]
        selected_bmag = bmag[mask]
        selected_bmag = selected_bmag[np.isfinite(selected_bmag)]
    if selected.size == 0:
        rms_per_amp = float("nan")
        peak_per_amp = float("nan")
    else:
        rms_per_amp = float(np.sqrt(np.mean(selected**2)))
        peak_per_amp = float(np.max(np.abs(selected)))
    if selected_bmag.size == 0:
        rms_bmag_per_amp = float("nan")
        peak_bmag_per_amp = float("nan")
    else:
        rms_bmag_per_amp = float(np.sqrt(np.mean(selected_bmag**2)))
        peak_bmag_per_amp = float(np.max(selected_bmag))
    if not np.isfinite(current):
        current = float("nan")
    scale = abs(float(current))
    return {
        f"{source}_suppressed_rms_Bz_T_per_A": rms_per_amp,
        f"{source}_suppressed_peak_abs_Bz_T_per_A": peak_per_amp,
        f"{source}_suppressed_rms_Bmag_T_per_A": rms_bmag_per_amp,
        f"{source}_suppressed_peak_Bmag_T_per_A": peak_bmag_per_amp,
        f"{source}_suppressed_rms_Bz_T_at_fitted_current": rms_per_amp * scale,
        f"{source}_suppressed_peak_abs_Bz_T_at_fitted_current": peak_per_amp * scale,
        f"{source}_suppressed_rms_Bmag_T_at_fitted_current": rms_bmag_per_amp * scale,
        f"{source}_suppressed_peak_Bmag_T_at_fitted_current": peak_bmag_per_amp * scale,
    }


def symmetry_summary_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return compact symmetry diagnostics for reports and CSV summaries."""

    rows = symmetry_diagnostics_rows(result)
    contour_rows = [row for row in rows if row["source"] == "contours"]
    failed_rows = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARNING"]
    optimization = result.regularization_diagnostics
    active = bool(optimization.get("symmetry_reduction_active", False))
    before = _safe_int(optimization.get("symmetry_reduction_unknown_count_before"))
    after = _safe_int(optimization.get("symmetry_reduction_unknown_count_after"))
    return {
        "symmetry_diagnostics_available": bool(rows),
        "symmetry_reduction_active": active,
        "symmetry_reduction_factor": float(optimization.get("symmetry_reduction_factor", np.nan)),
        "symmetry_reduction_unknown_count_before": before,
        "symmetry_reduction_unknown_count_after": after,
        "symmetry_reduction_reason": str(optimization.get("symmetry_reduction_reason", "")),
        "contour_symmetry_pair_count": len(contour_rows),
        "contour_symmetry_failed_pair_count": len([row for row in contour_rows if row["status"] == "FAIL"]),
        "symmetry_warning_count": len(warnings),
        "symmetry_failure_count": len(failed_rows),
    }


def symmetry_diagnostics_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    """Flatten optimization and contour symmetry diagnostics into table rows."""

    rows: list[dict[str, Any]] = []
    diagnostics = result.regularization_diagnostics
    if any(str(key).startswith("symmetry_reduction") for key in diagnostics):
        active = bool(diagnostics.get("symmetry_reduction_active", False))
        reason = str(diagnostics.get("symmetry_reduction_reason", ""))
        mode = str(diagnostics.get("symmetry_reduction_mode", "unknown"))
        status = "PASS" if active else "WARNING" if mode != "off" else "INFO"
        if mode == "strict" and not active:
            status = "FAIL"
        rows.append(
            {
                "source": "optimization",
                "part_or_pair": "all",
                "check": "symmetry_reduction",
                "status": status,
                "mode": mode,
                "geometry": _status_from_bool(None),
                "stream_function": _status_from_bool(None),
                "levels": _status_from_bool(None),
                "contours": _status_from_bool(None),
                "factor": diagnostics.get("symmetry_reduction_factor", ""),
                "unknowns": _unknown_count_text(diagnostics),
                "reason": reason,
            }
        )

    seen_pairs: set[str] = set()
    for part_index, part in enumerate(result.coil_parts, start=1):
        part_diagnostics = getattr(part, "contour_level_diagnostics", {}) or {}
        pair = str(part_diagnostics.get("symmetry_pair", "")).strip()
        if not pair or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        status = _contour_symmetry_status(part_diagnostics)
        rows.append(
            {
                "source": "contours",
                "part_or_pair": pair,
                "check": "mirror_pair",
                "status": status,
                "mode": str(part_diagnostics.get("symmetry_validation_mode", "")),
                "geometry": _checked_status(part_diagnostics, "symmetry_geometry_checked", "symmetry_geometry_ok"),
                "stream_function": _checked_status(
                    part_diagnostics,
                    "symmetry_stream_function_checked",
                    "symmetry_stream_function_ok",
                ),
                "levels": _checked_status(part_diagnostics, "symmetry_level_checked", "symmetry_level_ok"),
                "contours": _checked_status(
                    part_diagnostics,
                    "symmetry_extracted_contour_checked",
                    "symmetry_extracted_contour_ok",
                ),
                "factor": "",
                "unknowns": "",
                "reason": _contour_symmetry_reason(part_diagnostics, part_index),
            }
        )
    return rows


def contour_stability_warnings(result: CoilGenResult) -> tuple[str, ...]:
    """Return concise warnings for contour sets likely to be unstable."""

    warnings: list[str] = []
    for part_index, part in enumerate(result.coil_parts, start=1):
        diagnostics = getattr(part, "contour_level_diagnostics", {}) or {}
        levels = part.potential_level_list
        contours = part.contour_lines
        if levels is not None and len(levels) > 0:
            levels_array = np.asarray(levels, dtype=float)
            crosses_zero = np.nanmin(levels_array) < 0.0 < np.nanmax(levels_array)
            zero_crossing_strategy = str(diagnostics.get("zero_crossing_level_strategy", "paired")).strip().lower()
            if crosses_zero and len(levels_array) % 2 and zero_crossing_strategy not in {"full_range", "centered"}:
                warnings.append(
                    f"Part {part_index} has an odd number of zero-crossing contour levels; "
                    "saddle-like center contours may be unstable."
                )
            if np.any(np.isclose(levels_array, 0.0, rtol=0.0, atol=1e-12)):
                warnings.append(f"Part {part_index} has a contour level at zero stream function.")
        if diagnostics.get("symmetry_extracted_contour_checked") and not diagnostics.get(
            "symmetry_extracted_contour_ok", True
        ):
            warnings.append(f"Part {part_index} failed extracted-contour symmetry validation.")
        if diagnostics.get("contourability_checked") and not diagnostics.get("contourability_ok", True):
            warnings.append(
                f"Part {part_index} failed contourability validation: {diagnostics.get('contourability_reason', '')}."
            )
        short_count = sum(1 for contour in contours if contour.length <= 1e-12)
        if short_count:
            warnings.append(f"Part {part_index} contains {short_count} degenerate or near-zero-length contour(s).")
    return tuple(dict.fromkeys(warnings))


def concomitant_field_report_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return a compact report table for the Markdown concomitant-field section."""

    summary = concomitant_field_summary_dict(result)
    keys = (
        "concomitant_model",
        "concomitant_assumption",
        "concomitant_gradient_source",
        "concomitant_main_field_T",
        "concomitant_b0_direction_x",
        "concomitant_b0_direction_y",
        "concomitant_b0_direction_z",
        "concomitant_gradient_x_T_per_m",
        "concomitant_gradient_y_T_per_m",
        "concomitant_gradient_z_T_per_m",
        "concomitant_sample_count",
        "concomitant_rms_T",
        "concomitant_p95_abs_T",
        "concomitant_max_abs_T",
        "concomitant_rms_ppm_of_B0",
        "concomitant_max_abs_ppm_of_B0",
        "concomitant_rms_percent_of_target_peak",
        "concomitant_max_abs_percent_of_target_peak",
    )
    return {key: summary.get(key, "") for key in keys}


def current_density_report_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return a compact report table for configured-conductor current density."""

    return current_density_summary_dict(result)


def readiness_report_dict(result: CoilGenResult) -> dict[str, Any]:
    """Return a compact table for engineering readiness and missing checks."""

    return readiness_summary_dict(result)


def write_design_report(
    result: CoilGenResult,
    path: str | Path,
    *,
    title: str = "CoilGen Design Report",
    design: Any | None = None,
    validation_report: ConfigValidationReport | None = None,
) -> Path:
    """Write a user-readable Markdown report with units and diagnostics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation_report = validation_report or validate_config(result.input_data)
    quality = result_quality_status(result)
    summary = result_summary_dict(result)
    design_context = result_design_context(result)
    lines: list[str] = [
        f"# {title}",
        "",
        f"Overall status: **{quality.status}**",
        f"Design context: **{design_context.key}**",
        "",
    ]
    if quality.failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {item}" for item in quality.failures)
        lines.append("")
    if quality.warnings or validation_report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {item}" for item in quality.warnings)
        lines.extend(f"- {item.format()}" for item in validation_report.warnings)
        lines.append("")
    if design is not None and hasattr(design, "as_user_parameters"):
        lines.extend(["## User Design Settings", ""])
        lines.extend(_markdown_table(_flatten_dict(design.as_user_parameters())))
        lines.append("")
    lines.extend(["## Magnetostatic Feasibility", ""])
    lines.extend(_markdown_table(_magnetostatic_feasibility_values(summary, quality)))
    lines.append("")
    lines.extend(["## Layout Feasibility", ""])
    lines.extend(_markdown_table(_layout_feasibility_values(summary)))
    lines.append("")
    lines.extend(["## Engineering Checks Not Included", ""])
    lines.extend(_markdown_table(readiness_report_dict(result)))
    lines.append("")
    lines.extend(["## Suppression-Clearance Report", ""])
    lines.append(
        "Checks generated zero-field suppression target points against the source mesh used for clearance validation."
    )
    lines.append("")
    lines.extend(_markdown_table(suppression_clearance_report_dict(result)))
    lines.append("")
    lines.extend(["## Numerical Summary", ""])
    lines.extend(_markdown_table(summary))
    lines.append("")
    pruning_summary = _sparsification_pruning_summary(result)
    pruning_rows = _sparsification_pruning_audit_rows(result)
    if pruning_summary:
        lines.extend(["## Post-Sparsification Pruning Audit", ""])
        lines.extend(_markdown_table(pruning_summary))
        lines.append("")
    if pruning_rows:
        lines.append("Each row is one selected multiplicity-weighted representative or complete symmetry-orbit removal.")
        lines.append("")
        lines.extend(
            _markdown_rows_table(
                pruning_rows,
                (
                    "step",
                    "representatives",
                    "source_contours",
                    "turns",
                    "nominal_length_m",
                    "significance_percent",
                    "residual_before_T",
                    "residual_after_T",
                    "residual_change_T",
                    "total_rms_field_error_percent",
                    "total_peak_field_error_percent",
                ),
            )
        )
        lines.append("")
    consolidation_summary = _post_pruning_merge_summary(result)
    consolidation_rows = _post_pruning_merge_audit_rows(result)
    if consolidation_summary:
        lines.extend(["## Post-Pruning Contour Consolidation", ""])
        lines.extend(_markdown_table(consolidation_summary))
        lines.append("")
    if consolidation_rows:
        lines.extend(
            _markdown_rows_table(
                consolidation_rows,
                (
                    "pass",
                    "source_representatives",
                    "source_contours",
                    "selected_potential_A",
                    "multiplicity",
                    "accepted",
                    "reason",
                    "residual_metric_T",
                    "total_rms_field_error_percent",
                    "minimum_spacing_m",
                    "manufacturing_ok",
                    "curvature_ratio",
                    "curvature_limit",
                ),
            )
        )
        lines.append("")
    try:
        regularization_report = regularization_selection_report(result)
    except ValueError:
        regularization_report = None
    if regularization_report is not None:
        lines.extend(["## Regularization Selection", ""])
        regularization_values = regularization_report.as_dict()
        regularization_values["warnings"] = "; ".join(regularization_report.warnings)
        lines.extend(_markdown_table(regularization_values))
        lines.append("")
    lines.extend(["## DSV Field Error", ""])
    lines.extend(_markdown_table({key + " [%]": value for key, value in result.error_vals.items()}))
    lines.append("")
    lines.extend(["## Concomitant Field Report", ""])
    if design_context.enabled(EvaluationFeature.CONCOMITANT_FIELD):
        lines.append(
            "Uses the fitted connected-layout gradient at the required target current. "
            "The model assumes the useful field component is parallel to the configured main field."
        )
    else:
        lines.append(f"Not applicable for design_context={design_context.key}.")
    lines.append("")
    lines.extend(_markdown_table(concomitant_field_report_dict(result)))
    lines.append("")
    lines.extend(["## Current-Density Report", ""])
    lines.append(
        "Uses the configured conductor cross section and fitted layout current. "
        "It is not a thermal or duty-cycle calculation."
    )
    lines.append("")
    lines.extend(_markdown_table(current_density_report_dict(result)))
    lines.append("")
    inductance_rows = _inductance_matrix_rows(result.inductance_matrix)
    if inductance_rows:
        lines.extend(["## Inductance Matrix", ""])
        lines.append(
            "Rows and columns are independent coil parts. Off-diagonal entries are mutual inductances; "
            "signs include the actual current/path orientation."
        )
        lines.append("")
        lines.extend(_markdown_rows_table(inductance_rows, ("row", "column", "type", "inductance_H")))
        lines.append("")
    symmetry_rows = symmetry_diagnostics_rows(result)
    if symmetry_rows:
        lines.extend(["## Symmetry Diagnostics", ""])
        lines.extend(
            _markdown_rows_table(
                symmetry_rows,
                (
                    "source",
                    "part_or_pair",
                    "check",
                    "status",
                    "mode",
                    "geometry",
                    "stream_function",
                    "levels",
                    "contours",
                    "factor",
                    "unknowns",
                    "reason",
                ),
            )
        )
        lines.append("")
    lines.extend(["## Timing", ""])
    if result.timing_summary:
        timing_rows = {
            str(row["stage"]): f"{float(row['stage_seconds']):.3f} s; elapsed {float(row['elapsed_seconds']):.3f} s"
            for row in result.timing_summary
        }
        lines.extend(_markdown_table(timing_rows))
    else:
        lines.append("No timing data available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_design_summary_csv(result: CoilGenResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = result_summary_dict(result)
    lines = ["quantity,value"]
    for key, value in summary.items():
        lines.append(f"{key},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sparsification_pruning_summary(result: CoilGenResult) -> dict[str, Any]:
    for part in result.coil_parts:
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        diagnostics = provenance.diagnostics
        if str(diagnostics.get("contour_sparsification_pruning_mode", "off")) == "off":
            continue
        return {
            "search mode": diagnostics.get("contour_sparsification_pruning_search_mode", ""),
            "optimality proven": diagnostics.get("contour_sparsification_pruning_optimality_proven", False),
            "evaluated subsets": diagnostics.get("contour_sparsification_pruning_evaluated_subset_count", 0),
            "feasible subsets": diagnostics.get("contour_sparsification_pruning_feasible_subset_count", 0),
            "removable symmetry orbits": diagnostics.get(
                "contour_sparsification_pruning_removable_orbit_count", 0
            ),
            "pre-pruning geometries": diagnostics.get("contour_sparsification_pruning_pre_geometry_count", 0),
            "retained geometries": diagnostics.get("contour_sparsification_pruning_retained_geometry_count", 0),
            "target geometry count": diagnostics.get("contour_sparsification_pruning_target_geometry_count", 0),
            "target count feasible": diagnostics.get(
                "contour_sparsification_pruning_target_count_feasible", False
            ),
            "final symmetry closure": diagnostics.get(
                "contour_sparsification_pruning_final_symmetry_ok", False
            ),
            "baseline residual metric [T]": diagnostics.get(
                "contour_sparsification_pruning_baseline_metric_T", np.nan
            ),
            "selected residual metric [T]": diagnostics.get(
                "contour_sparsification_pruning_selected_metric_T", np.nan
            ),
            "allowed residual metric [T]": diagnostics.get(
                "contour_sparsification_pruning_allowed_metric_T", np.nan
            ),
        }
    return {}


def _sparsification_pruning_audit_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    for part in result.coil_parts:
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        records = provenance.diagnostics.get("contour_sparsification_pruning_removal_records", ())
        if not records:
            continue
        rows: list[dict[str, Any]] = []
        for record in records:
            representatives = "; ".join(
                f"part {int(part_index) + 1}, geometry {int(contour_index)}"
                for part_index, contour_index in record.get("representative_indices", ())
            )
            source_contours = "; ".join(
                f"part {int(source.get('part_index', 0)) + 1}: "
                + ",".join(str(int(index)) for index in source.get("indices", ()))
                for source in record.get("source_contour_indices", ())
            )
            rows.append(
                {
                    "step": int(record.get("step", len(rows) + 1)),
                    "representatives": representatives,
                    "source_contours": source_contours,
                    "turns": int(record.get("removed_turn_count", 0)),
                    "nominal_length_m": float(record.get("nominal_length_m", np.nan)),
                    "significance_percent": float(record.get("significance_percent", np.nan)),
                    "residual_before_T": float(record.get("residual_metric_before_T", np.nan)),
                    "residual_after_T": float(record.get("residual_metric_after_T", np.nan)),
                    "residual_change_T": float(record.get("residual_metric_change_T", np.nan)),
                    "total_rms_field_error_percent": float(record.get("total_rms_field_error_percent", np.nan)),
                    "total_peak_field_error_percent": float(record.get("total_peak_field_error_percent", np.nan)),
                }
            )
        return rows
    return []


def _post_pruning_merge_summary(result: CoilGenResult) -> dict[str, Any]:
    for part in result.coil_parts:
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        diagnostics = provenance.diagnostics
        if str(diagnostics.get("contour_sparsification_post_pruning_merge_mode", "off")) == "off":
            continue
        return {
            "mode": diagnostics.get("contour_sparsification_post_pruning_merge_mode", "off"),
            "completed passes": diagnostics.get(
                "contour_sparsification_post_pruning_completed_pass_count", 0
            ),
            "proposed merges": diagnostics.get(
                "contour_sparsification_post_pruning_proposed_merge_count", 0
            ),
            "accepted merges": diagnostics.get(
                "contour_sparsification_post_pruning_accepted_merge_count", 0
            ),
            "pre-consolidation geometries": diagnostics.get(
                "contour_sparsification_post_pruning_pre_geometry_count", 0
            ),
            "final geometries": diagnostics.get(
                "contour_sparsification_post_pruning_final_geometry_count", 0
            ),
            "represented turns": diagnostics.get(
                "contour_sparsification_post_pruning_final_turn_count", 0
            ),
            "baseline residual metric [T]": diagnostics.get(
                "contour_sparsification_post_pruning_baseline_metric_T", np.nan
            ),
            "selected residual metric [T]": diagnostics.get(
                "contour_sparsification_post_pruning_selected_metric_T", np.nan
            ),
            "allowed residual metric [T]": diagnostics.get(
                "contour_sparsification_post_pruning_allowed_metric_T", np.nan
            ),
            "total RMS field error [%]": diagnostics.get(
                "contour_sparsification_post_pruning_total_rms_field_error_percent", np.nan
            ),
            "total peak field error [%]": diagnostics.get(
                "contour_sparsification_post_pruning_total_peak_field_error_percent", np.nan
            ),
        }
    return {}


def _post_pruning_merge_audit_rows(result: CoilGenResult) -> list[dict[str, Any]]:
    for part in result.coil_parts:
        provenance = part.contour_sparsification
        if provenance is None:
            continue
        records = provenance.diagnostics.get(
            "contour_sparsification_post_pruning_merge_records", ()
        )
        if not records:
            continue
        return [
            {
                "pass": int(record.get("pass_index", 0)),
                "source_representatives": ",".join(
                    str(int(index)) for index in record.get("source_representative_indices", ())
                ),
                "source_contours": ",".join(
                    str(int(index)) for index in record.get("source_contour_indices", ())
                ),
                "selected_potential_A": float(record.get("selected_potential_A", np.nan)),
                "multiplicity": int(record.get("turn_multiplicity", 0)),
                "accepted": bool(record.get("accepted", False)),
                "reason": str(record.get("rejection_reason", "")),
                "residual_metric_T": float(record.get("residual_metric_T", np.nan)),
                "total_rms_field_error_percent": float(
                    record.get("total_rms_field_error_percent", np.nan)
                ),
                "minimum_spacing_m": float(record.get("minimum_spacing_m", np.nan)),
                "manufacturing_ok": bool(record.get("manufacturing_ok", False)),
                "curvature_ratio": float(
                    record.get("max_curvature_envelope_ratio", np.nan)
                ),
                "curvature_limit": float(record.get("curvature_envelope_limit", np.nan)),
            }
            for record in records
        ]
    return []


def _inductance_matrix_rows(values: Any) -> list[dict[str, Any]]:
    if values is None or np.size(values) == 0:
        return []
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return []
    rows: list[dict[str, Any]] = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            rows.append(
                {
                    "row": row + 1,
                    "column": col + 1,
                    "type": "self" if row == col else "mutual",
                    "inductance_H": float(matrix[row, col]),
                }
            )
    return rows


def _magnetostatic_feasibility_values(summary: dict[str, Any], quality: ResultQuality) -> dict[str, Any]:
    display_unit = str(summary.get("field_display_units", "T"))
    return {
        "overall_status": quality.status,
        "target_point_count": summary.get("target_point_count", ""),
        "dsv_target_point_count": summary.get("dsv_target_point_count", ""),
        "layout_gradient_efficiency_mT_per_m_per_A": summary.get("layout_gradient_efficiency_mT_per_m_per_A", ""),
        "layout_needed_current_A": summary.get("layout_needed_current_A", ""),
        "layout_mean_rel_error_percent": summary.get("layout_mean_rel_error_percent", ""),
        "layout_max_rel_error_percent": summary.get("layout_max_rel_error_percent", ""),
        f"shim_layout_residual_rms_{display_unit}": summary.get(f"shim_layout_residual_rms_{display_unit}", ""),
        f"shim_layout_residual_max_abs_{display_unit}": summary.get(f"shim_layout_residual_max_abs_{display_unit}", ""),
        "target_error_constraint_met": summary.get("target_error_constraint_met", ""),
        "dsv_constraint_met": summary.get("dsv_constraint_met", ""),
        "constrained_solver_contract_met": summary.get("constrained_solver_contract_met"),
        "constrained_solver_physical_constraints_met": summary.get("constrained_solver_physical_constraints_met"),
        "constrained_solver_diagnostic_result": summary.get("constrained_solver_diagnostic_result", False),
        "constrained_solver_failed_constraints": summary.get("constrained_solver_failed_constraints", ""),
    }


def _layout_feasibility_values(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layout_status": summary.get("layout_status", ""),
        "route_statuses": summary.get("route_statuses", ""),
        "contour_count": summary.get("contour_count", ""),
        "topological_group_count": summary.get("topological_group_count", ""),
        "min_contour_distance_m": summary.get("min_contour_distance_m", ""),
        "unconnected_contours_min_wire_distance_m": summary.get("unconnected_contours_min_wire_distance_m", ""),
        "min_wire_wire_distance_m": summary.get("min_wire_wire_distance_m", ""),
        "wire_clearance_failure_threshold_m": summary.get("wire_clearance_failure_threshold_m", ""),
        "wire_clearance_failure_count": summary.get("wire_clearance_failure_count", ""),
        "current_density_class": summary.get("current_density_class", ""),
        "finished_engineering_design_ready": summary.get("finished_engineering_design_ready", ""),
    }


def _markdown_table(values: dict[str, Any]) -> list[str]:
    lines = ["| Quantity | Value |", "|---|---:|"]
    for key, value in values.items():
        lines.append(f"| `{key}` | {_format_value(value)} |")
    return lines


def _markdown_rows_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[str]:
    lines = [
        "| " + " | ".join(f"`{column}`" for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    return lines


def _flatten_dict(values: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=path))
        else:
            flat[path] = value
    return flat


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _first_or_nan(values: Any) -> float:
    if values is None or len(values) == 0:
        return float("nan")
    return float(values[0])


def _sum_or_nan(values: Any) -> float:
    if values is None or len(values) == 0:
        return float("nan")
    values_array = np.asarray(values, dtype=float)
    finite = values_array[np.isfinite(values_array)]
    return float(np.sum(finite)) if finite.size else float("nan")


def _matrix_first_or_nan(values: Any) -> float:
    if values is None or np.size(values) == 0:
        return float("nan")
    return float(np.asarray(values)[0, 0])


def _same_current_inductance(values: Any) -> float:
    if values is None or np.size(values) == 0:
        return float("nan")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return float("nan")
    currents = np.ones(matrix.shape[0], dtype=float)
    return float(currents @ matrix @ currents)


def _finite_max_abs(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = np.abs(array[np.isfinite(array)])
    return float(np.max(array)) if array.size else float("nan")


def _finite_rms(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(array * array))) if array.size else float("nan")


def _execution_time(result: CoilGenResult) -> float:
    if not result.timing_summary:
        return float("nan")
    return float(result.timing_summary[-1]["elapsed_seconds"])


def _safe_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number


def _unknown_count_text(diagnostics: dict[str, Any]) -> str:
    before = diagnostics.get("symmetry_reduction_unknown_count_before", "")
    after = diagnostics.get("symmetry_reduction_unknown_count_after", "")
    if before == "" and after == "":
        return ""
    return f"{before} -> {after}"


def _status_from_bool(value: bool | None) -> str:
    if value is None:
        return "not checked"
    return "PASS" if bool(value) else "FAIL"


def _checked_status(diagnostics: dict[str, Any], checked_key: str, ok_key: str) -> str:
    if not bool(diagnostics.get(checked_key, False)):
        return "not checked"
    return _status_from_bool(bool(diagnostics.get(ok_key, False)))


def _contour_symmetry_status(diagnostics: dict[str, Any]) -> str:
    statuses = [
        _checked_status(diagnostics, "symmetry_geometry_checked", "symmetry_geometry_ok"),
        _checked_status(diagnostics, "symmetry_stream_function_checked", "symmetry_stream_function_ok"),
        _checked_status(diagnostics, "symmetry_level_checked", "symmetry_level_ok"),
        _checked_status(diagnostics, "symmetry_extracted_contour_checked", "symmetry_extracted_contour_ok"),
    ]
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    warnings = diagnostics.get("symmetry_warnings", [])
    if warnings:
        return "WARNING"
    return "PASS"


def _contour_symmetry_reason(diagnostics: dict[str, Any], part_index: int) -> str:
    reasons: list[str] = []
    warnings = diagnostics.get("symmetry_warnings", [])
    if isinstance(warnings, (tuple, list)):
        reasons.extend(str(item) for item in warnings if str(item))
    checks = (
        ("symmetry_geometry_checked", "symmetry_geometry_ok", "geometry mismatch"),
        ("symmetry_stream_function_checked", "symmetry_stream_function_ok", "stream-function parity mismatch"),
        ("symmetry_level_checked", "symmetry_level_ok", "contour level mismatch"),
        ("symmetry_extracted_contour_checked", "symmetry_extracted_contour_ok", "extracted contour count mismatch"),
    )
    for checked_key, ok_key, message in checks:
        if bool(diagnostics.get(checked_key, False)) and not bool(diagnostics.get(ok_key, False)):
            reasons.append(message)
    if not reasons:
        return f"reported by part {part_index}"
    return "; ".join(dict.fromkeys(reasons))
