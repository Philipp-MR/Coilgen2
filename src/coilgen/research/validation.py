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


def validate_on_independent_grid(result: Any, *, resolution: int = 12, design_margin: float = 1.0) -> dict[str, Any]:
    """Evaluate unchanged sheet current on a distinct grid without refitting.

    Rebuild basis data on copied parts, so low-memory results remain usable.
    This changes sample locations, not the field-kernel quadrature accuracy.
    """
    config = result.input_data
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
    scale = float(result.regularization_diagnostics.get("target_gradient_normalization_scale", 1.0))
    raw[:, 2] = evaluate_field_expression(config.field_shape_function, points) * scale
    target = field_values_to_tesla(raw, config.target_field_units)
    validation = TargetField(
        points, target, np.ones(len(points)), np.zeros(len(points), bool), np.ones(len(points), int)
    )
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
    for part in parts:
        part.mesh.face_normals = compute_face_normals(part.mesh)
        part.mesh.vertex_normals = compute_vertex_normals(part.mesh)
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    calculate_sensitivity_matrix(parts, validation, config)
    field = sum(np.einsum("ctv,v->tc", part.sensitivity_matrix, part.stream_function) for part in parts)
    check = field_budget_check(result, field, target, design_margin=design_margin)
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


def validate_fabricated_constraints(result: Any, fabrication: Any, *, design_margin: float = 1.0) -> dict[str, Any]:
    """Recheck final wire fields and uniform-B loads, flag unverifiable contracts.

    Arbitrary surface matrices cannot be reused for a routed wire. Their
    requested scenario/custom bounds need new wire couplings and validation
    coordinates, so their absence is an explicit unknown acceptance blocker.
    """
    config, spec = result.input_data, _spec(result)
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
        np.any(result.target_field.suppressed)
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
    for plan in plans:
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
            field += current * biot_savart_wire_path(points, result.target_field.coords, worker_count=1)
        except (KeyError, TypeError, ValueError) as exc:
            unknown.append(f"Electrical path {plan.get('body_id')} is invalid: {exc}")
            paths_valid = False
            complete_circuits = False
            continue
        force += f
        torque += t
    if paths_valid:
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
