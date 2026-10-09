"""Independent field sampling and explicit post-fabrication physics checks.

Final field budgets are resolved from the executed CoilGenConfig, divided by
the requested design margin. They never come from stale frontend controls.
Unknown/imported/custom contracts are not promoted to verified acceptance.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..basis import calculate_basis_functions, calculate_one_ring_by_mesh
from ..field_components import field_component_values, normalize_field_component
from ..field_units import field_values_to_tesla
from ..geometry import compute_face_normals, compute_vertex_normals
from ..models import TargetField
from ..physics import biot_savart_wire_path, calculate_sensitivity_matrix
from ..target import evaluate_field_expression
from .convex import ResearchPhysicsSpec


def _spec(result: Any) -> ResearchPhysicsSpec:
    return ResearchPhysicsSpec.from_dict(result.input_data.research_physics or {})


def _unsupported_contract(result: Any) -> str:
    from ..optimization import _normalise_target_error_constraint_metric

    if not _spec(result).include_default_field_constraints:
        return "Custom-only field constraints require a separately supplied validation operator/dataset."
    metric = _normalise_target_error_constraint_metric(result.input_data)
    if metric not in {"norm", "rms"}:
        return f"Independent validation of {metric!r} field contracts is not implemented; no acceptance claim is made."
    return ""


def _selected_vectors(field: np.ndarray, component: str, direction: Any) -> np.ndarray:
    selected = field_component_values(field, component, b0_direction=direction)
    return np.asarray(selected).reshape(len(field), 3 if normalize_field_component(component) == "vector" else 1)


def field_budget_check(
    result: Any,
    actual_T: np.ndarray,
    target_T: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    design_margin: float = 1.0,
) -> dict[str, Any]:
    """Check resolved norm/RMS budgets using physical scalar/vector RMS [T].

    CoilGen's vector DSV ``rms`` currently averages Cartesian components.
    This function reports vector-magnitude RMS and converts that configured
    component-RMS limit by sqrt(3), preserving the executed solver contract.
    Relative norm tolerances are independent of this convention.
    """
    if not np.isfinite(design_margin) or not 0 < design_margin <= 1:
        raise ValueError("design_margin must be a finite fraction in (0,1].")
    reason = _unsupported_contract(result)
    if reason:
        return {"verified": False, "passed": None, "reason": reason}
    config, spec = result.input_data, _spec(result)
    actual = _selected_vectors(np.asarray(actual_T, dtype=float), config.target_field_component, config.b_0_direction)
    target = _selected_vectors(np.asarray(target_T, dtype=float), config.target_field_component, config.b_0_direction)
    weights = np.ones(len(target)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if actual.shape != target.shape or len(weights) != len(target):
        raise ValueError("Validation fields/weights must have matching point counts.")
    active = np.isfinite(weights) & (weights > 0)
    if not np.any(active):
        return {"verified": False, "passed": None, "reason": "No positive-weight field samples are available."}
    actual, target, weights = actual[active], target[active], weights[active]
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(target)):
        return {
            "verified": True,
            "passed": False,
            "value": None,
            "limit": None,
            "unit": "T",
            "rms_error_T": None,
            "peak_error_T": None,
            "relative_field_error": None,
            "reason": "Nonfinite active validation fields cannot pass physical acceptance.",
        }
    weights = weights / np.sum(weights)
    error_squared = np.sum((actual - target) ** 2, axis=1)
    reference_squared = np.sum(target**2, axis=1)
    rms = float(np.sqrt(weights @ error_squared))
    target_rms = float(np.sqrt(weights @ reference_squared))
    relative = rms / target_rms if target_rms > 0 else None
    component_factor = np.sqrt(target.shape[1])
    from ..optimization import _normalise_target_error_constraint_metric

    metric = _normalise_target_error_constraint_metric(config)
    relative_budget = float(config.target_error_tolerance_percent) / 100.0
    if metric == "rms" and config.target_error_rms_limit_T > 0:
        solve_limit = float(config.target_error_rms_limit_T) * component_factor
        kind = "absolute_rms_vector" if target.shape[1] == 3 else "absolute_rms_scalar"
    else:
        absolute = float(config.target_error_tolerance_absolute_T)
        if metric == "norm":
            design_weights = np.asarray(result.target_field.weights, dtype=float)
            design_active = (
                ~np.asarray(result.target_field.suppressed, bool) & np.isfinite(design_weights) & (design_weights > 0)
            )
            denominator = np.sqrt(np.sum(design_weights[design_active]))
            if denominator <= 0:
                return {
                    "verified": False,
                    "passed": None,
                    "reason": "No original design samples define the absolute norm budget.",
                }
            absolute /= denominator
        else:
            absolute *= component_factor
        solve_limit = target_rms * relative_budget + absolute
        kind = "relative_rms_plus_absolute" if absolute else "relative_rms"
    limit = solve_limit / design_margin
    allowance = spec.acceptance_absolute_by_unit.get("T", 0.0) + spec.acceptance_relative_tolerance * limit
    return {
        "verified": True,
        "passed": bool(rms <= limit + allowance),
        "contract_kind": kind,
        "value": rms,
        "limit": limit,
        "effective_limit": limit + allowance,
        "unit": "T",
        "rms_error_T": rms,
        "peak_error_T": float(np.sqrt(np.max(error_squared))),
        "relative_field_error": relative,
        "target_rms_T": target_rms,
        "relative_limit": relative_budget / design_margin if kind == "relative_rms" else None,
        "design_margin": design_margin,
        "configured_metric": metric,
        "rms_convention": "vector_magnitude" if target.shape[1] == 3 else "scalar",
    }


def sample_independent_target(result: Any, *, resolution: int = 12) -> TargetField:
    """Sample the executed analytic target without fitting a new normalization.

    Coordinates remain in the world frame. The saved expression scale and
    configured units are reused; imported targets need a separate dataset.
    """
    config = result.input_data
    imported_map = str(config.target_field_definition_file).strip().lower() not in {"", "none"}
    legacy_mesh = config.target_region_geometry is None and str(config.target_mesh_file).strip().lower() not in {
        "",
        "none",
    }
    if imported_map or legacy_mesh:
        raise ValueError("Imported targets require a separately supplied independent validation dataset.")
    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
        raise ValueError("Independent-grid resolution must be an integer >=2.")
    if config.target_region_geometry is not None:
        from ..target_regions import sample_target_region_from_config

        points = sample_target_region_from_config(config, resolution=resolution, independent=True)
    else:
        radius = float(config.target_region_radius)
        axis = (np.arange(resolution) + 0.5) * 2 * radius / resolution - radius
        points = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1).T
        points = points[np.linalg.norm(points, axis=1) <= radius]
        if config.set_roi_into_mesh_center:
            # Expressions remain in world coordinates, as define_target_field.
            points += result.combined_mesh.vertices.mean(axis=0)
    raw = np.zeros((len(points), 3))
    scale = float(getattr(result, "regularization_diagnostics", {}).get("target_gradient_normalization_scale", 1.0))
    raw[:, 2] = evaluate_field_expression(config.field_shape_function, points) * scale
    target = field_values_to_tesla(raw, config.target_field_units)
    return TargetField(points, target, np.ones(len(points)), np.zeros(len(points), bool), np.ones(len(points), int))


def validate_on_independent_grid(result: Any, *, resolution: int = 12, design_margin: float = 1.0) -> dict[str, Any]:
    """Evaluate unchanged sheet current on a distinct grid without refitting.

    Rebuild basis data on copied parts, so low-memory results remain usable.
    This changes sample locations, not the field-kernel quadrature accuracy.
    """
    config = result.input_data
    from ..progress import ProgressReporter, progress_scope

    reporter = ProgressReporter.from_config(config, label="Independent sheet validation")
    reason = _unsupported_contract(result)
    imported_map = str(config.target_field_definition_file).strip().lower() not in {"", "none"}
    legacy_mesh = config.target_region_geometry is None and str(config.target_mesh_file).strip().lower() not in {
        "",
        "none",
    }
    if not reason and (imported_map or legacy_mesh):
        reason = "Imported targets require a separately supplied independent validation dataset."
    if reason:
        return {"available": False, "verified": False, "passed": None, "reason": reason}
    with progress_scope(0, 10):
        reporter.report(None, "Sampling the independent desired-field grid")
        validation = sample_independent_target(result, resolution=resolution)
    points, target = validation.coords, validation.b
    parts = [
        replace(
            part,
            mesh=part.mesh.copy(),
            sensitivity_matrix=None,
            basis_elements=[],
            one_ring_list=[],
            node_triangles=[],
            current_density_mat=None,
        )
        for part in result.coil_parts
    ]
    for index, part in enumerate(parts):
        part.mesh.face_normals = compute_face_normals(part.mesh)
        part.mesh.vertex_normals = compute_vertex_normals(part.mesh)
        reporter.report(10 + 5 * (index + 1) / len(parts), "Preparing independent surface geometry")
    with progress_scope(15, 20):
        calculate_one_ring_by_mesh(parts)
    with progress_scope(20, 30):
        calculate_basis_functions(parts)
    with progress_scope(30, 95):
        reporter.report(0, "Evaluating independent current-sheet samples")
        calculate_sensitivity_matrix(parts, validation, config)
    field = sum(np.einsum("ctv,v->tc", part.sensitivity_matrix, part.stream_function) for part in parts)
    check = field_budget_check(result, field, target, design_margin=design_margin)
    reporter.finish("Independent sheet validation complete")
    return {
        "available": True,
        "point_count": len(points),
        **check,
        "source": "continuous_sheet_independent_grid_no_current_refit",
        "points_m": points,
        "field_T": field,
        "target_T": target,
        "field_quadrature_order": config.gauss_order,
        "independent_quadrature_check": False,
        "sampling_weights": "equal_point_weights",
        "target_region_geometry": config.target_region_geometry,
    }


def _winding_linearity(result, target, field):
    """Fit compatible Bz gradients, keeping descriptive evidence separate."""
    from ..gradient_linearity import gradient_linearity_enabled, gradient_linearity_summary

    config = result.input_data
    enabled = gradient_linearity_enabled(config)
    if normalize_field_component(config.target_field_component) != "Bz" or config.design_context != "mr_gradient":
        reason = "Independent winding linearity currently supports Cartesian Bz MR-gradient contracts only."
        return {}, [], reason if enabled else ""
    region = str(config.linearity_region).strip().lower()
    if region in {"suppression", "secondary", "shield"} or (
        region in {"all", "any"} and np.any(result.target_field.suppressed)
    ):
        reason = "Independent suppression-region winding linearity requires separate validation samples."
        return {}, [], reason if enabled else ""
    if np.linalg.matrix_rank(np.column_stack((np.ones(len(target.coords)), target.coords))) < 4:
        reason = "Independent winding samples cannot identify a complete Cartesian gradient fit."
        return {}, [], reason if enabled else ""
    if not enabled:
        from ..engineering_metrics import fit_bz_field_quality

        # A constant/nonlinear target has no unique desired gradient axis.
        # Roundoff in its first-order projection must not create an efficiency.
        target_fit = fit_bz_field_quality(
            target.b, target, fit_offset=bool(getattr(config, "linearity_fit_offset", True))
        )
        tolerance = 128 * np.finfo(float).eps * float(np.max(np.abs(target.b[:, 2])))
        gradient_scale = target_fit.desired_gradient_T_per_m * float(np.max(np.ptp(target.coords, axis=0)))
        if (
            not np.all(np.isfinite(field))
            or not np.all(np.isfinite([target_fit.linear_max_abs_residual_T, gradient_scale]))
            or target_fit.linear_max_abs_residual_T > tolerance
            or gradient_scale <= tolerance
        ):
            return {}, [], ""
    summary = gradient_linearity_summary(field, target, config)
    if not enabled:
        summary["linearity_status"] = "not_checked"
        for key in summary:
            if key.endswith("_met"):
                summary[key] = None
        return summary, [], ""
    checks = []
    for title, value, limit in (
        ("Gradient strength", "linearity_gradient_strength_error_percent", "gradient_strength_tolerance_percent"),
        ("Gradient RMS linearity", "linearity_rms_percent", "linearity_rms_tolerance_percent"),
        ("Gradient peak linearity", "linearity_peak_percent", "linearity_peak_tolerance_percent"),
        ("Cross-gradient", "linearity_cross_gradient_percent", "linearity_cross_gradient_tolerance_percent"),
    ):
        maximum = float(getattr(config, limit))
        if maximum > 0:
            observed = float(summary[value])
            checks.append(
                {
                    "name": "Independent winding " + title.lower(),
                    "value": observed if np.isfinite(observed) else None,
                    "limit": maximum,
                    "effective_limit": maximum * (1 + 1e-10),
                    "unit": "%",
                    "verified": True,
                    "passed": bool(np.isfinite(observed) and observed <= maximum * (1 + 1e-10)),
                }
            )
    return summary, checks, ""


def _samples_on_wire(path, samples):
    """Exclude singular thin-wire samples instead of accepting kernel zeros."""
    for start, vector in zip(path[:-1], np.diff(path, axis=0)):
        length_squared = float(vector @ vector)
        if length_squared <= 1e-30:
            continue
        offsets = samples - start
        fractions = np.clip(offsets @ vector / length_squared, 0, 1)
        distances_squared = np.sum((offsets - fractions[:, None] * vector) ** 2, axis=1)
        if np.any(distances_squared <= 1e-30):
            return True
    return False


def validate_winding_on_independent_grid(
    result: Any, fabrication: Any, *, resolution: int = 12, design_margin: float = 1.0
) -> dict[str, Any]:
    """Check recorded electrical paths at their original signed body currents.

    A collinear segment-bisection check exercises the existing analytically
    integrated thin-wire kernel without changing the path geometry. It is not
    curve-discretization convergence or finite-cross-section validation.
    Unsupported additional contracts remain unknown even when ROI fields are
    available. This field check cannot certify drive feeds or construction.
    """
    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
        raise ValueError("Independent-grid resolution must be an integer >=2.")
    if not np.isfinite(design_margin) or not 0 < design_margin <= 1:
        raise ValueError("design_margin must be a finite fraction in (0,1].")
    config, spec = result.input_data, _spec(result)
    from ..progress import ProgressReporter, progress_scope

    reporter = ProgressReporter.from_config(config, label="Independent winding validation")
    report = {
        "available": False,
        "verified": False,
        "passed": None,
        "source": "fabricated_electrical_paths_independent_grid_original_current",
        "geometry_source": "recorded_fabricated_electrical_paths",
        "field_model": "thin_wire_exact_segments",
        "current_source": "original_contour_current_no_refit",
        "current_refitted": False,
        "component": normalize_field_component(config.target_field_component),
        "coordinate_frame": "world",
        "sampling_weights": "equal_point_weights",
        "target_region_geometry": config.target_region_geometry,
        "resolution": int(resolution),
        "point_count": 0,
        "distinct_from_design_grid": False,
        "design_grid_overlap_point_count": 0,
        "new_point_count": 0,
        "overlap_tolerance_m": None,
        "currents_A": {},
        "linearity": {},
        "integration_check": {"available": False},
        "checks": [],
        "unknown_reasons": [],
    }

    def unavailable(reason):
        report.update(reason=reason, unknown_reasons=[reason])
        return report

    try:
        with progress_scope(0, 5):
            reporter.report(None, "Sampling the independent winding-validation grid")
            target = sample_independent_target(result, resolution=resolution)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return unavailable(str(exc))
    points = np.asarray(target.coords, dtype=float)
    if not len(points) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(target.b)):
        return unavailable("Independent analytic target samples must be nonempty and finite.")
    design_points = np.asarray(result.target_field.coords, dtype=float)
    if design_points.ndim != 2 or design_points.shape[1] != 3 or not np.all(np.isfinite(design_points)):
        return unavailable("Original design sample locations must be finite world-frame XYZ coordinates.")
    # Equivalent locations can differ by floating-point evaluation order.
    # Tree queries avoid a dense validation-by-design distance matrix.
    from scipy.spatial import cKDTree

    overlap_tolerance = (
        64
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(points))), float(np.max(np.abs(design_points))) if len(design_points) else 0.0)
    )
    overlap = int(np.count_nonzero(cKDTree(design_points).query(points)[0] <= overlap_tolerance))
    report.update(
        point_count=len(points),
        design_grid_overlap_point_count=overlap,
        new_point_count=len(points) - overlap,
        overlap_tolerance_m=overlap_tolerance,
    )
    if overlap == len(points):
        return unavailable("The requested winding validation grid repeats only design sample locations.")
    plans = list(getattr(fabrication, "winding_steps", ()))
    bodies = list(getattr(fabrication, "electrical_metrics", {}).get("bodies", ()))
    ids = [plan.get("body_id") for plan in plans]
    body_ids = [body.get("body_id") for body in bodies]
    if not ids or any(not isinstance(id_, str) or not id_ for id_ in ids) or len(set(ids)) != len(ids):
        return unavailable("Recorded winding paths require unique nonempty body identities.")
    if (
        any(not isinstance(id_, str) or not id_ for id_ in body_ids)
        or len(body_ids) != len(set(body_ids))
        or set(ids) != set(body_ids)
    ):
        return unavailable("Recorded electrical paths and original currents do not cover the same bodies.")
    bodies = {body["body_id"]: body for body in bodies}
    field, bisected_field = np.zeros_like(points), np.zeros_like(points)
    unknown = []
    for index, plan in enumerate(plans):
        id_ = plan["body_id"]
        try:
            path = np.asarray(plan["electrical_points_m"], dtype=float)
            current = float(bodies[id_]["current_A"])
            if (
                path.ndim != 2
                or path.shape[1] != 3
                or len(path) < 2
                or not np.all(np.isfinite(path))
                or not np.isfinite(current)
                or not np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-15)
            ):
                raise ValueError("finite Nx3 path and finite recorded current are required")
        except (ValueError, TypeError, KeyError) as exc:
            return unavailable(f"Recorded winding body {id_} is invalid: {exc}.")
        if _samples_on_wire(path, points):
            return unavailable(
                "Independent samples intersect a thin-wire centerline; a finite-cross-section field model is required."
            )
        if not plan.get("circuit_complete", False) or np.linalg.norm(path[-1] - path[0]) > 1e-10 * max(
            float(np.max(np.ptp(path, axis=0))), 0.01
        ):
            unknown.append(f"Recorded winding body {id_} has no complete modeled return circuit.")
        report["currents_A"][id_] = current
        start, span = 5 + 85 * index / len(plans), 85 / len(plans)
        with progress_scope(start, start + span / 2):
            reporter.report(0, f"Evaluating unchanged winding body {index + 1}/{len(plans)}")
            field += current * biot_savart_wire_path(path, points, worker_count=1)
        bisected = np.empty((2 * len(path) - 1, 3))
        bisected[::2], bisected[1::2] = path, (path[:-1] + path[1:]) / 2
        with progress_scope(start + span / 2, start + span):
            reporter.report(0, f"Checking exact-segment integration for body {index + 1}/{len(plans)}")
            bisected_field += current * biot_savart_wire_path(bisected, points, worker_count=1)
    reporter.report(90, "Comparing independent winding fields, linearity and error bounds")
    if spec.field_constraints or spec.scenarios:
        unknown.append("Custom field/scenario contracts require separate independent winding operators and datasets.")
    if (
        np.any(result.target_field.suppressed)
        or config.suppression_field_rms_limit_T > 0
        or config.suppression_field_peak_limit_T > 0
    ):
        unknown.append("Suppression contracts require separate independent winding validation samples.")
    if config.shield_gradient_opposition_enabled:
        unknown.append("Shield gradient-opposition requires an independent per-support winding operator.")
    check = field_budget_check(result, field, target.b, design_margin=design_margin)
    checks = [{"name": "Independent winding field error (original current)", **check}]
    if not check["verified"]:
        unknown.append(check["reason"])
    linearity, gradient_checks, reason = _winding_linearity(result, target, field)
    checks.extend(gradient_checks)
    if reason:
        unknown.append(reason)
    difference = float(np.max(np.linalg.norm(bisected_field - field, axis=1)))
    scale = float(np.max(np.linalg.norm(field, axis=1)))
    integration_limit = max(1e-15, 2e-10 * scale)
    integration = {
        "available": True,
        "method": "collinear_segment_bisection_invariance",
        "geometry_changed": False,
        "curve_discretization_convergence": False,
        "finite_cross_section_check": False,
        "max_deviation_T": difference,
        "effective_limit_T": integration_limit,
        "passed": bool(np.isfinite(difference) and difference <= integration_limit),
    }
    checks.append(
        {
            "name": "Independent winding exact-segment integration consistency",
            "value": difference if np.isfinite(difference) else None,
            "limit": integration_limit,
            "effective_limit": integration_limit,
            "unit": "T",
            "verified": True,
            "passed": integration["passed"],
        }
    )
    unknown = list(dict.fromkeys(unknown))
    checks.extend(
        {"name": reason, "value": None, "limit": None, "unit": "", "verified": False, "passed": None}
        for reason in unknown
    )
    for row in checks:
        row.update(
            scope="independent_realized_winding",
            geometry_source=report["geometry_source"],
            current_source=report["current_source"],
        )
    verified = not unknown and check["verified"]
    report.update(check)
    report.update(
        available=True,
        verified=verified,
        passed=bool(all(row["passed"] is True for row in checks)) if verified else None,
        point_count=len(points),
        distinct_from_design_grid=True,
        points_m=points,
        field_T=field,
        target_T=target.b,
        linearity=linearity,
        integration_check=integration,
        checks=checks,
        unknown_reasons=unknown,
    )
    reporter.finish("Independent winding validation complete")
    return report


def wire_force_torque(
    points_m: np.ndarray, current_A: float, b0_T: Any, origin_m: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Exact segment-integrated force and torque in a fixed uniform B0."""
    points, b0, origin = (
        np.asarray(points_m, dtype=float),
        np.asarray(b0_T, dtype=float),
        np.asarray(origin_m, dtype=float),
    )
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("Mechanical validation requires finite electrical path points in metres.")
    if b0.shape != (3,) or origin.shape != (3,) or not np.all(np.isfinite(np.r_[b0, origin, current_A])):
        raise ValueError("Mechanical validation requires finite uniform B0, origin and current.")
    force_segments = float(current_A) * np.cross(np.diff(points, axis=0), b0)
    torque_segments = np.cross((points[:-1] + points[1:]) / 2 - origin, force_segments)
    return np.sum(force_segments, axis=0), np.sum(torque_segments, axis=0)


def validate_fabricated_constraints(
    result: Any, fabrication: Any, *, design_margin: float = 1.0, check_default_field: bool = True
) -> dict[str, Any]:
    """Recheck final wire fields and uniform-B loads, flag unverifiable contracts.

    Arbitrary surface matrices cannot be reused for a routed wire. Their
    requested scenario/custom bounds need new wire couplings and validation
    coordinates, so their absence is an explicit unknown acceptance blocker.
    """
    config, spec = result.input_data, _spec(result)
    from ..progress import ProgressReporter, progress_scope

    reporter = ProgressReporter.from_config(config, label="Fabricated physics validation")
    unknown: list[str] = []
    checks: list[dict[str, Any]] = []
    if spec.sheet_power_limit_W is not None:
        unknown.append(
            "The optimized sheet-power limit is not a wire-power limit; a conductor/fill model and final circuit-power budget are required."
        )
    if spec.magnetic_energy_limit_J is not None:
        unknown.append("The sheet-energy bound needs complete-circuit self and mutual energy validation after routing.")
    if spec.field_constraints or spec.scenarios:
        unknown.append(
            "Custom field/scenario matrices act on the original sheet; post-route validation operators and target coordinates are required."
        )
    if (
        (check_default_field and np.any(result.target_field.suppressed))
        or config.suppression_field_rms_limit_T > 0
        or config.suppression_field_peak_limit_T > 0
    ):
        unknown.append("Post-route suppression constraints have not been independently verified.")
    if config.shield_gradient_opposition_enabled or any(
        getattr(config, key) > 0
        for key in (
            "gradient_strength_tolerance_percent",
            "linearity_rms_tolerance_percent",
            "linearity_peak_tolerance_percent",
            "linearity_cross_gradient_tolerance_percent",
        )
    ):
        unknown.append("Post-route gradient/linearity/shield constraints have not been independently verified.")
    mechanical_requested = any(
        value is not None
        for value in (
            spec.force_limit_N,
            spec.torque_limit_Nm,
            spec.force_equality_N,
            spec.torque_equality_Nm,
        )
    )
    if spec.background_B_per_face_T is not None and mechanical_requested:
        unknown.append("Nonuniform per-face B0 has no defined field interpolation along the routed electrical paths.")
    bodies = {row["body_id"]: row for row in fabrication.electrical_metrics.get("bodies", ())}
    plans = list(fabrication.winding_steps)
    paths_valid = bool(plans) and len(plans) == len(bodies) and {plan.get("body_id") for plan in plans} == set(bodies)
    if not paths_valid:
        unknown.append("Electrical path/current records do not completely cover fabricated bodies.")
    field = np.zeros_like(result.target_field.b, dtype=float)
    force, torque = np.zeros(3), np.zeros(3)
    complete_circuits = paths_valid
    for index, plan in enumerate(plans):
        body = bodies.get(plan.get("body_id"))
        if body is None:
            paths_valid = False
            complete_circuits = False
            continue
        if not plan.get("circuit_complete", False):
            complete_circuits = False
            unknown.append(
                f"Electrical path {plan.get('body_id')} omits a return; complete-circuit physics is unverified."
            )
        try:
            points = np.asarray(plan["electrical_points_m"], dtype=float)
            current = float(body["current_A"])
            f, t = wire_force_torque(points, current, spec.b0_T, spec.torque_origin_m)
            if plan.get("circuit_complete", False) and np.linalg.norm(points[-1] - points[0]) > 1e-10 * max(
                float(np.max(np.ptp(points, axis=0))), 0.01
            ):
                complete_circuits = False
                unknown.append(
                    f"Electrical path {plan.get('body_id')} claims a complete circuit but is geometrically open."
                )
            with progress_scope(95 * index / len(plans), 95 * (index + 1) / len(plans)):
                reporter.report(0, f"Rechecking original-current field and mechanical loads: {plan.get('body_id')}")
                field += current * biot_savart_wire_path(points, result.target_field.coords, worker_count=1)
        except (KeyError, TypeError, ValueError) as exc:
            unknown.append(f"Electrical path {plan.get('body_id')} is invalid: {exc}")
            paths_valid = False
            complete_circuits = False
            continue
        force += f
        torque += t
    if paths_valid:
        if check_default_field:
            mask = ~np.asarray(result.target_field.suppressed, dtype=bool)
            check = field_budget_check(
                result,
                field[mask],
                result.target_field.b[mask],
                weights=np.asarray(result.target_field.weights)[mask],
                design_margin=design_margin,
            )
            if check["verified"]:
                checks.append({"name": "Fabricated path field error (original current)", **check})
            else:
                unknown.append(check["reason"])
        if complete_circuits and spec.background_B_per_face_T is None:
            for name, vector, limit, equality, unit in (
                ("force", force, spec.force_limit_N, spec.force_equality_N, "N"),
                ("torque", torque, spec.torque_limit_Nm, spec.torque_equality_Nm, "N m"),
            ):
                for suffix, target, maximum in (
                    ("", np.zeros(3), limit),
                    (" equality", equality, 0.0 if equality is not None else None),
                ):
                    if maximum is None:
                        continue
                    value = float(np.max(np.abs(vector - target))) if suffix else float(np.linalg.norm(vector))
                    allowance = spec.acceptance_absolute_by_unit.get(
                        unit, 0
                    ) + spec.acceptance_relative_tolerance * max(float(maximum), float(np.max(np.abs(target))))
                    checks.append(
                        {
                            "name": "Fabricated wire " + name + suffix,
                            "value": value,
                            "limit": maximum,
                            "effective_limit": maximum + allowance,
                            "unit": unit,
                            "passed": bool(value <= maximum + allowance),
                        }
                    )
    unknown = list(dict.fromkeys(unknown))
    checks.extend({"name": reason, "value": None, "limit": None, "unit": "", "passed": None} for reason in unknown)
    uniform_paths_available = paths_valid and spec.background_B_per_face_T is None
    uniform_available = uniform_paths_available and complete_circuits
    reporter.finish("Fabricated physical limits reviewed")
    return {
        "checks": checks,
        "unknown_reasons": unknown,
        "verified": not unknown and paths_valid,
        "passed": bool(not unknown and paths_valid and all(check["passed"] is True for check in checks)),
        "force_N": force.tolist() if uniform_available else None,
        "torque_Nm": torque.tolist() if uniform_available else None,
        "force_model": "uniform_B0_complete_electrical_polyline"
        if uniform_available
        else "uniform_B0_partial_electrical_paths"
        if uniform_paths_available
        else "unverified",
        "partial_path_force_N": force.tolist() if uniform_paths_available and not complete_circuits else None,
        "partial_path_torque_Nm": torque.tolist() if uniform_paths_available and not complete_circuits else None,
        "field_T": field if paths_valid else None,
        "current_source": "original_contour_current_no_refit",
    }
