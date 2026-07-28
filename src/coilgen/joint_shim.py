"""Joint optimization of a stream-function surface and linear field actuators.

The auxiliary actuators are intentionally field-model agnostic. A channel may
represent an MRI scanner spherical-harmonic channel, a previously manufactured
shim coil, or any other calibrated field source whose field is linear in its
command value.

This module owns the coupled continuous-current solve. Contour extraction still
acts only on the returned surface stream function; callers must add the fixed
auxiliary field when evaluating the total contoured-layout residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import CoilGenConfig
from .engineering_limits import engineering_limit_diagnostics
from .field_components import sensitivity_component_matrix
from .field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from .models import CoilPart, TargetField
from .mri_shim_metric import (
    MRIShimMetricModel,
    MRIShimMetricOptions,
    build_cvxpy_mri_shim_objective,
)
from .optimization import (
    ResidualConstraintBlock,
    StreamFunctionOptimizationProblem,
    _constrained_contour_step_limit,
    _cvxpy_conic_solver_order,
    _cvxpy_solver_options,
    _cvxpy_surface_current_density_constraints,
    _solution_max_surface_current_density,
    _symmetric_quadratic_matrix,
    _target_error_constraint_blocks,
    _weighted_constraint_system,
    _mirror_vertex_mapping_for_single_mesh,
)


@dataclass(frozen=True, slots=True)
class LinearShimActuator:
    """One calibrated collection of linear field channels.

    ``field_vectors_T_per_unit`` has shape ``(samples, 3, channels)``. Channel
    coefficients therefore retain their physical command units: amperes for a
    measured coil, or tesla-valued solid-harmonic coefficients for an analytic
    scanner model. Bounds and reference values use the same units.
    """

    name: str
    field_vectors_T_per_unit: np.ndarray
    channel_labels: tuple[str, ...] = ()
    coefficient_unit: str = "unit"
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    reference_values: np.ndarray | None = None
    sample_coords_m: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Linear shim actuator name must not be empty.")
        fields = np.asarray(self.field_vectors_T_per_unit, dtype=float)
        if fields.ndim != 3 or fields.shape[1] != 3 or fields.shape[2] < 1:
            raise ValueError("field_vectors_T_per_unit must have shape (samples, 3, channels).")
        if not np.all(np.isfinite(fields)):
            raise ValueError("Auxiliary actuator fields must contain only finite values.")
        channel_count = int(fields.shape[2])
        labels = tuple(str(label).strip() for label in self.channel_labels)
        if not labels:
            labels = tuple(f"channel_{index + 1}" for index in range(channel_count))
        if len(labels) != channel_count or any(not label for label in labels):
            raise ValueError("channel_labels must provide one non-empty label per channel.")
        if len(set(labels)) != len(labels):
            raise ValueError("channel_labels must be unique within one actuator.")

        lower = _channel_vector(self.lower_bounds, channel_count, -np.inf, "lower_bounds")
        upper = _channel_vector(self.upper_bounds, channel_count, np.inf, "upper_bounds")
        if np.any(lower > upper):
            raise ValueError("Every auxiliary actuator lower bound must be <= its upper bound.")
        references = _reference_values(self.reference_values, lower, upper)

        coords = None
        if self.sample_coords_m is not None:
            coords = np.asarray(self.sample_coords_m, dtype=float)
            if coords.shape != (fields.shape[0], 3) or not np.all(np.isfinite(coords)):
                raise ValueError("sample_coords_m must have finite shape (samples, 3).")

        fields = fields.copy()
        lower = lower.copy()
        upper = upper.copy()
        references = references.copy()
        for values in (fields, lower, upper, references):
            values.setflags(write=False)
        if coords is not None:
            coords = coords.copy()
            coords.setflags(write=False)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field_vectors_T_per_unit", fields)
        object.__setattr__(self, "channel_labels", labels)
        object.__setattr__(self, "coefficient_unit", str(self.coefficient_unit).strip() or "unit")
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "reference_values", references)
        object.__setattr__(self, "sample_coords_m", coords)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def sample_count(self) -> int:
        return int(self.field_vectors_T_per_unit.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.field_vectors_T_per_unit.shape[2])

    def component_matrix(
        self,
        component: str,
        *,
        b0_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        sensitivity = np.moveaxis(self.field_vectors_T_per_unit, 1, 0)
        return np.asarray(
            sensitivity_component_matrix(
                sensitivity,
                component,
                b0_direction=b0_direction,
            ),
            dtype=float,
        )

    def evaluate_field(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=float).reshape(-1)
        if len(values) != self.channel_count:
            raise ValueError(f"Actuator {self.name!r} expects {self.channel_count} coefficients, got {len(values)}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Auxiliary actuator coefficients must be finite.")
        return np.einsum("nck,k->nc", self.field_vectors_T_per_unit, values)

    @classmethod
    def from_scalar_bz(
        cls,
        name: str,
        bz_matrix_T_per_unit: np.ndarray,
        **kwargs: Any,
    ) -> "LinearShimActuator":
        bz = np.asarray(bz_matrix_T_per_unit, dtype=float)
        if bz.ndim != 2:
            raise ValueError("bz_matrix_T_per_unit must have shape (samples, channels).")
        fields = np.zeros((bz.shape[0], 3, bz.shape[1]), dtype=float)
        fields[:, 2, :] = bz
        return cls(name=name, field_vectors_T_per_unit=fields, **kwargs)


@dataclass(frozen=True, slots=True)
class JointShimOptimizationProblem:
    """Coupled surface/auxiliary problem in one exact residual metric space."""

    surface_problem: StreamFunctionOptimizationProblem
    target_field: TargetField
    residual_block: ResidualConstraintBlock
    actuators: tuple[LinearShimActuator, ...]
    actuator_residual_matrices: tuple[np.ndarray, ...]
    raw_surface_matrix: np.ndarray
    raw_target: np.ndarray
    raw_actuator_matrices: tuple[np.ndarray, ...]
    dsv_coords_m: np.ndarray
    dsv_weights: np.ndarray
    dsv_mask: np.ndarray
    component: str
    b0_direction: tuple[float, float, float]

    @classmethod
    def from_stream_function_problem(
        cls,
        surface_problem: StreamFunctionOptimizationProblem,
        target_field: TargetField,
        config: CoilGenConfig,
        actuators: Sequence[LinearShimActuator],
        *,
        coordinate_tolerance_m: float = 1.0e-9,
    ) -> "JointShimOptimizationProblem":
        actuator_tuple = tuple(actuators)
        if not actuator_tuple:
            raise ValueError("Joint shim optimization requires at least one auxiliary actuator.")
        names = [actuator.name for actuator in actuator_tuple]
        if len(set(names)) != len(names):
            raise ValueError("Auxiliary actuator names must be unique.")

        blocks = _target_error_constraint_blocks(surface_problem, config)
        if len(blocks) != 1 or blocks[0].name != "dsv":
            names_text = ", ".join(block.name for block in blocks)
            raise NotImplementedError(
                "Joint auxiliary fields currently support one DSV residual block. "
                f"The configured solve produced: {names_text or 'no blocks'}. "
                "Additional suppression/linearity blocks must provide matching auxiliary matrices."
            )
        block = blocks[0]
        target_rows = np.asarray(surface_problem.unweighted_target_z, dtype=float).reshape(-1)
        target_coords = np.asarray(target_field.coords, dtype=float)
        if len(target_rows) != len(target_coords):
            raise ValueError("Surface target rows and target coordinates do not match.")
        dsv_mask = _active_dsv_mask(surface_problem, len(target_rows))
        weights = _active_dsv_weights(surface_problem, dsv_mask)
        component = str(getattr(config, "target_field_component", "Bz"))
        raw_direction = np.asarray(getattr(config, "b_0_direction", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
        direction_norm = float(np.linalg.norm(raw_direction))
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError("b_0_direction must be a finite non-zero vector.")
        direction = tuple(float(value) for value in raw_direction / direction_norm)

        raw_surface_all = surface_problem.unweighted_reduced_sensitivity
        if raw_surface_all is None:
            raise ValueError("Joint MRI metrics require the unweighted surface sensitivity matrix.")
        raw_surface = np.asarray(raw_surface_all, dtype=float)[dsv_mask]
        raw_target = target_rows[dsv_mask]
        dsv_coords = target_coords[dsv_mask]

        matrices: list[np.ndarray] = []
        raw_matrices: list[np.ndarray] = []
        for actuator in actuator_tuple:
            if actuator.sample_count != len(target_coords):
                raise ValueError(
                    f"Actuator {actuator.name!r} has {actuator.sample_count} samples; "
                    f"the target has {len(target_coords)}."
                )
            _validate_actuator_coordinates(
                actuator,
                target_coords,
                coordinate_tolerance_m=coordinate_tolerance_m,
            )
            raw_matrix = actuator.component_matrix(component, b0_direction=direction)
            active_raw_matrix = np.asarray(raw_matrix[dsv_mask], dtype=float)
            transformed, _zero_target, _weight_sum = _weighted_constraint_system(
                active_raw_matrix,
                np.zeros(int(np.count_nonzero(dsv_mask)), dtype=float),
                weights,
                center=str(block.metric).lower() == "std",
            )
            if transformed.shape[0] != block.matrix.shape[0]:
                raise ValueError(f"Actuator {actuator.name!r} does not match the DSV residual rows.")
            transformed_matrix = np.asarray(transformed, dtype=float).copy()
            raw_active_matrix = np.asarray(active_raw_matrix, dtype=float).copy()
            transformed_matrix.setflags(write=False)
            raw_active_matrix.setflags(write=False)
            matrices.append(transformed_matrix)
            raw_matrices.append(raw_active_matrix)

        mask = np.asarray(dsv_mask, dtype=bool).copy()
        mask.setflags(write=False)
        raw_surface = np.asarray(raw_surface, dtype=float).copy()
        raw_target = np.asarray(raw_target, dtype=float).copy()
        dsv_coords = np.asarray(dsv_coords, dtype=float).copy()
        weights = np.asarray(weights, dtype=float).copy()
        for values in (raw_surface, raw_target, dsv_coords, weights):
            values.setflags(write=False)
        return cls(
            surface_problem=surface_problem,
            target_field=target_field,
            residual_block=block,
            actuators=actuator_tuple,
            actuator_residual_matrices=tuple(matrices),
            raw_surface_matrix=raw_surface,
            raw_target=raw_target,
            raw_actuator_matrices=tuple(raw_matrices),
            dsv_coords_m=dsv_coords,
            dsv_weights=weights,
            dsv_mask=mask,
            component=component,
            b0_direction=direction,
        )


@dataclass(frozen=True, slots=True)
class JointShimSolveOptions:
    """Numerical and lexicographic settings for a joint solve."""

    lexicographic_secondary: bool = True
    residual_relative_slack: float = 1.0e-6
    residual_absolute_slack_field_T: float = 0.0
    surface_cost_weight: float = 1.0
    surface_smoothness_cost_weight: float = 0.0
    default_actuator_cost_weight: float = 1.0
    actuator_cost_weights: Mapping[str, float] = field(default_factory=dict)
    required_residual_field_T: float | None = None
    mri_metric_options: MRIShimMetricOptions | None = None
    project_stream_function_to_contour_symmetry: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.residual_relative_slack) or self.residual_relative_slack < 0.0:
            raise ValueError("residual_relative_slack must be finite and non-negative.")
        if not np.isfinite(self.residual_absolute_slack_field_T) or self.residual_absolute_slack_field_T < 0.0:
            raise ValueError("residual_absolute_slack_field_T must be finite and non-negative.")
        if not np.isfinite(self.surface_cost_weight) or self.surface_cost_weight < 0.0:
            raise ValueError("surface_cost_weight must be finite and non-negative.")
        if not np.isfinite(self.surface_smoothness_cost_weight) or self.surface_smoothness_cost_weight < 0.0:
            raise ValueError("surface_smoothness_cost_weight must be finite and non-negative.")
        if not np.isfinite(self.default_actuator_cost_weight) or self.default_actuator_cost_weight < 0.0:
            raise ValueError("default_actuator_cost_weight must be finite and non-negative.")
        for name, value in self.actuator_cost_weights.items():
            if not str(name).strip() or not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("Actuator cost weights must use non-empty names and non-negative finite values.")
        if self.required_residual_field_T is not None:
            value = float(self.required_residual_field_T)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("required_residual_field_T must be finite and positive when set.")


@dataclass(frozen=True, slots=True)
class JointShimSolution:
    reduced_stream_function: np.ndarray
    full_stream_function: np.ndarray
    actuator_coefficients: Mapping[str, np.ndarray]
    surface_field_T: np.ndarray
    auxiliary_field_T: np.ndarray
    combined_field_T: np.ndarray
    residual_field_T: np.ndarray
    diagnostics: Mapping[str, Any]


def solve_joint_shim_problem(
    problem: JointShimOptimizationProblem,
    config: CoilGenConfig,
    options: JointShimSolveOptions | None = None,
) -> JointShimSolution:
    """Solve the coupled actuator problem with an optional efficiency tie-break."""

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("CVXPY is required for joint shim optimization.") from exc

    solve_options = options or JointShimSolveOptions()
    mri_options = solve_options.mri_metric_options or MRIShimMetricOptions.from_config(config)
    mri_model = (
        MRIShimMetricModel.from_samples(
            problem.dsv_coords_m,
            problem.dsv_weights,
            mri_options,
        )
        if mri_options.report_enabled
        else None
    )
    surface_matrix = np.asarray(problem.residual_block.matrix, dtype=float)
    target = np.asarray(problem.residual_block.target, dtype=float).reshape(-1)
    surface_variable = cp.Variable(surface_matrix.shape[1], name="surface_stream_function")
    actuator_variables: list[Any] = []
    actuator_predictions: list[Any] = []
    raw_actuator_predictions: list[Any] = []
    constraints = _surface_constraints(cp, problem.surface_problem, surface_variable, config)

    for actuator, matrix, raw_matrix in zip(
        problem.actuators,
        problem.actuator_residual_matrices,
        problem.raw_actuator_matrices,
    ):
        normalized = cp.Variable(actuator.channel_count, name=f"{actuator.name}_normalized")
        actuator_variables.append(normalized)
        references = np.asarray(actuator.reference_values, dtype=float)
        actuator_predictions.append((np.asarray(matrix, dtype=float) * references[None, :]) @ normalized)
        raw_actuator_predictions.append((np.asarray(raw_matrix, dtype=float) * references[None, :]) @ normalized)
        lower = np.asarray(actuator.lower_bounds, dtype=float) / references
        upper = np.asarray(actuator.upper_bounds, dtype=float) / references
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if np.any(finite_lower):
            constraints.append(normalized[finite_lower] >= lower[finite_lower])
        if np.any(finite_upper):
            constraints.append(normalized[finite_upper] <= upper[finite_upper])

    prediction = surface_matrix @ surface_variable
    for actuator_prediction in actuator_predictions:
        prediction = prediction + actuator_prediction
    residual = prediction - target
    raw_prediction = np.asarray(problem.raw_surface_matrix, dtype=float) @ surface_variable
    for actuator_prediction in raw_actuator_predictions:
        raw_prediction = raw_prediction + actuator_prediction
    raw_residual = raw_prediction - np.asarray(problem.raw_target, dtype=float)
    numerical_scale = _joint_numerical_scale(
        surface_matrix,
        target,
        problem.actuator_residual_matrices,
        problem.actuators,
    )

    mri_score_hz = None
    primary_metric_name = str(problem.residual_block.metric)
    primary_constraints = list(constraints)
    if mri_options.optimize_enabled:
        if mri_model is None:
            raise RuntimeError("MRI shim optimize mode did not construct its metric model.")
        mri_score_hz, mri_constraints, _mri_expressions = build_cvxpy_mri_shim_objective(
            cp,
            mri_model,
            raw_residual,
        )
        primary_constraints.extend(mri_constraints)
        primary_objective = mri_score_hz / float(mri_options.reference_frequency_hz)
        primary_metric_name = "mri_acquisition_agnostic_risk"
    else:
        primary_objective = cp.norm(residual / numerical_scale, 2)

    primary_problem = cp.Problem(cp.Minimize(primary_objective), primary_constraints)
    primary_status, primary_solver = _solve_cvxpy_problem(primary_problem, cp, config)
    primary_surface = _variable_value(surface_variable, "surface stream function")
    primary_normalized = [
        _variable_value(variable, f"actuator {actuator.name}")
        for actuator, variable in zip(problem.actuators, actuator_variables)
    ]
    primary_coefficients = [
        values * np.asarray(actuator.reference_values, dtype=float)
        for actuator, values in zip(problem.actuators, primary_normalized)
    ]
    primary_residual = _residual_vector(
        surface_matrix,
        primary_surface,
        problem.actuator_residual_matrices,
        primary_coefficients,
        target,
    )
    primary_norm = float(np.linalg.norm(primary_residual))
    primary_raw_residual = _residual_vector(
        problem.raw_surface_matrix,
        primary_surface,
        problem.raw_actuator_matrices,
        primary_coefficients,
        problem.raw_target,
    )
    primary_mri_metrics = mri_model.evaluate(primary_raw_residual) if mri_model is not None else {}
    primary_objective_score_hz = (
        float(primary_mri_metrics["mri_shim_composite_score_Hz"]) if mri_options.optimize_enabled else float("nan")
    )

    selected_surface = primary_surface
    selected_coefficients = primary_coefficients
    selected_status = primary_status
    selected_solver = primary_solver
    secondary_used = False
    secondary_status = "not_requested"
    secondary_solver = ""
    if solve_options.lexicographic_secondary:
        if mri_options.optimize_enabled:
            if mri_score_hz is None:
                raise RuntimeError("MRI shim objective expression is unavailable for the secondary solve.")
            allowed_score_hz = primary_objective_score_hz * (
                1.0 + float(solve_options.residual_relative_slack)
            ) + PROTON_GYROMAGNETIC_RATIO_HZ_PER_T * float(solve_options.residual_absolute_slack_field_T)
            secondary_constraints = [*primary_constraints, mri_score_hz <= allowed_score_hz]
        else:
            denominator = float(problem.residual_block.field_metric_denominator or 1.0)
            allowed_norm = primary_norm * (1.0 + float(solve_options.residual_relative_slack)) + denominator * float(
                solve_options.residual_absolute_slack_field_T
            )
            secondary_constraints = [
                *primary_constraints,
                cp.norm(residual / numerical_scale, 2) <= allowed_norm / numerical_scale,
            ]
        secondary_objective = _secondary_objective(
            cp,
            problem,
            surface_variable,
            actuator_variables,
            solve_options,
        )
        secondary_problem = cp.Problem(cp.Minimize(secondary_objective), secondary_constraints)
        try:
            secondary_status, secondary_solver = _solve_cvxpy_problem(secondary_problem, cp, config)
            selected_surface = _variable_value(surface_variable, "surface stream function")
            selected_normalized = [
                _variable_value(variable, f"actuator {actuator.name}")
                for actuator, variable in zip(problem.actuators, actuator_variables)
            ]
            selected_coefficients = [
                values * np.asarray(actuator.reference_values, dtype=float)
                for actuator, values in zip(problem.actuators, selected_normalized)
            ]
            selected_status = secondary_status
            selected_solver = secondary_solver
            secondary_used = True
        except RuntimeError as exc:
            secondary_status = f"failed: {exc}"

    symmetry_projection_diagnostics: dict[str, Any] = {
        "joint_shim_stream_symmetry_projection_requested": bool(
            solve_options.project_stream_function_to_contour_symmetry
        ),
        "joint_shim_stream_symmetry_projection_applied": False,
    }
    if solve_options.project_stream_function_to_contour_symmetry:
        selected_surface, symmetry_projection_diagnostics = _project_stream_function_to_single_mesh_symmetry(
            problem,
            selected_surface,
            config,
        )

    selected_residual = _residual_vector(
        surface_matrix,
        selected_surface,
        problem.actuator_residual_matrices,
        selected_coefficients,
        target,
    )
    selected_raw_residual = _residual_vector(
        problem.raw_surface_matrix,
        selected_surface,
        problem.raw_actuator_matrices,
        selected_coefficients,
        problem.raw_target,
    )
    selected_mri_metrics = mri_model.evaluate(selected_raw_residual) if mri_model is not None else {}
    denominator = float(problem.residual_block.field_metric_denominator or 1.0)
    primary_field_metric = primary_norm / denominator
    selected_field_metric = float(np.linalg.norm(selected_residual)) / denominator
    selected_metric_gap = selected_field_metric - primary_field_metric
    metric_optimality_tolerance = max(
        1.0e-12,
        1.0e-7 * max(abs(primary_field_metric), 1.0e-30),
    )
    target_metric_was_primary = primary_metric_name == str(problem.residual_block.metric)
    selected_retains_primary_minimum = bool(
        target_metric_was_primary and selected_metric_gap <= metric_optimality_tolerance
    )
    required = solve_options.required_residual_field_T

    smoothness_metric = problem.surface_problem.reduced_smoothness_metric
    smoothness_available = smoothness_metric is not None and smoothness_metric.size > 0
    primary_smoothness_energy = _surface_quadratic_energy(
        smoothness_metric,
        primary_surface,
    )
    selected_smoothness_energy = _surface_quadratic_energy(
        smoothness_metric,
        selected_surface,
    )

    full_stream = np.asarray(problem.surface_problem.reducer.expansion_matrix, dtype=float) @ selected_surface
    surface_field = problem.surface_problem.evaluate_full_b_field(full_stream)
    coefficient_map = {
        actuator.name: np.asarray(coefficients, dtype=float)
        for actuator, coefficients in zip(problem.actuators, selected_coefficients)
    }
    auxiliary_field = np.zeros_like(surface_field)
    for actuator in problem.actuators:
        auxiliary_field += actuator.evaluate_field(coefficient_map[actuator.name])
    combined_field = surface_field + auxiliary_field
    target_field_values = np.asarray(problem.target_field.b, dtype=float)
    residual_field = combined_field - target_field_values

    contour = _constrained_contour_step_limit(config)
    density_limit = float(engineering_limit_diagnostics(config)["engineering_surface_current_density_limit_A_per_m"])
    max_density = _solution_max_surface_current_density(problem.surface_problem, full_stream)
    stream_range = float(np.ptp(full_stream))
    range_limit = float(contour["contour_step_constraint_max_range_A"])
    diagnostics: dict[str, Any] = {
        "joint_shim_enabled": True,
        "joint_shim_actuator_count": int(len(problem.actuators)),
        "joint_shim_auxiliary_channel_count": int(sum(a.channel_count for a in problem.actuators)),
        "joint_shim_component": problem.component,
        "joint_shim_metric": str(problem.residual_block.metric),
        "joint_shim_primary_metric": primary_metric_name,
        "joint_shim_primary_status": primary_status,
        "joint_shim_primary_solver": primary_solver,
        "joint_shim_primary_floor_residual_norm_T": primary_norm,
        "joint_shim_primary_floor_field_metric_T": primary_field_metric,
        "joint_shim_selected_status": selected_status,
        "joint_shim_selected_solver": selected_solver,
        "joint_shim_selected_residual_norm_T": float(np.linalg.norm(selected_residual)),
        "joint_shim_selected_field_metric_T": selected_field_metric,
        "joint_shim_target_metric_was_primary_objective": bool(target_metric_was_primary),
        "joint_shim_selected_minus_primary_field_metric_T": float(selected_metric_gap),
        "joint_shim_primary_metric_optimality_tolerance_T": float(metric_optimality_tolerance),
        "joint_shim_selected_retains_primary_metric_minimum": bool(selected_retains_primary_minimum),
        "joint_shim_lexicographic_secondary_requested": bool(solve_options.lexicographic_secondary),
        "joint_shim_lexicographic_secondary_used": bool(secondary_used),
        "joint_shim_secondary_status": secondary_status,
        "joint_shim_secondary_solver": secondary_solver,
        "joint_shim_surface_resistance_cost_weight": float(solve_options.surface_cost_weight),
        "joint_shim_surface_smoothness_cost_weight": float(solve_options.surface_smoothness_cost_weight),
        "joint_shim_surface_smoothness_metric_available": bool(smoothness_available),
        "joint_shim_primary_surface_biharmonic_energy_A2_per_m2": float(primary_smoothness_energy),
        "joint_shim_selected_surface_biharmonic_energy_A2_per_m2": float(selected_smoothness_energy),
        "joint_shim_surface_biharmonic_energy_change_fraction": (
            float(selected_smoothness_energy / primary_smoothness_energy - 1.0)
            if primary_smoothness_energy > 0.0
            else float("nan")
        ),
        "joint_shim_required_field_metric_T": float(required) if required is not None else float("nan"),
        "joint_shim_required_field_metric_feasible": (
            bool(primary_field_metric <= float(required) * (1.0 + 1.0e-7)) if required is not None else None
        ),
        "joint_shim_surface_current_density_limit_A_per_m": density_limit,
        "joint_shim_surface_current_density_max_A_per_m": float(max_density),
        "joint_shim_surface_current_density_utilization": (
            float(max_density / density_limit) if density_limit > 0.0 else float("nan")
        ),
        "joint_shim_stream_function_range_limit_A": range_limit,
        "joint_shim_stream_function_range_A": stream_range,
        "joint_shim_stream_function_range_utilization": (
            float(stream_range / range_limit) if range_limit > 0.0 else float("nan")
        ),
        "joint_shim_actuators": [
            _actuator_diagnostics(actuator, coefficient_map[actuator.name]) for actuator in problem.actuators
        ],
        **symmetry_projection_diagnostics,
    }
    if mri_model is not None:
        diagnostics.update(
            {
                "joint_shim_mri_metric_options": mri_options.as_dict(),
                "joint_shim_mri_metric_spatial_scale_m": float(mri_model.spatial_scale_m),
                "joint_shim_primary_mri_metrics": dict(primary_mri_metrics),
                "joint_shim_selected_mri_metrics": dict(selected_mri_metrics),
                "joint_shim_primary_mri_score_Hz": float(primary_mri_metrics["mri_shim_composite_score_Hz"]),
                "joint_shim_selected_mri_score_Hz": float(selected_mri_metrics["mri_shim_composite_score_Hz"]),
                "joint_shim_selected_mri_score_per_reference": float(
                    selected_mri_metrics["mri_shim_composite_score_per_reference"]
                ),
                "joint_shim_selected_mri_rms_Hz": float(selected_mri_metrics["mri_shim_residual_rms_Hz"]),
                "joint_shim_selected_mri_cvar_abs_Hz": float(selected_mri_metrics["mri_shim_residual_cvar_abs_Hz"]),
                "joint_shim_selected_mri_scaled_gradient_rms_Hz": float(
                    selected_mri_metrics["mri_shim_scaled_gradient_rms_Hz"]
                ),
            }
        )
    return JointShimSolution(
        reduced_stream_function=np.asarray(selected_surface, dtype=float),
        full_stream_function=np.asarray(full_stream, dtype=float),
        actuator_coefficients=coefficient_map,
        surface_field_T=np.asarray(surface_field, dtype=float),
        auxiliary_field_T=np.asarray(auxiliary_field, dtype=float),
        combined_field_T=np.asarray(combined_field, dtype=float),
        residual_field_T=np.asarray(residual_field, dtype=float),
        diagnostics=diagnostics,
    )


def save_joint_shim_solution_npz(
    path: str | Path,
    problem: JointShimOptimizationProblem,
    solution: JointShimSolution,
) -> Path:
    """Save a portable joint solution without Python object/pickle arrays.

    The ``stream_function`` key is deliberately compatible with CoilGen's
    preoptimized-stream-function NPZ loader. Auxiliary commands remain separate
    because only the surface stream function is contoured.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "stream_function": np.asarray(solution.full_stream_function, dtype=float),
        "reduced_stream_function": np.asarray(solution.reduced_stream_function, dtype=float),
        "target_coords_m": np.asarray(problem.target_field.coords, dtype=float),
        "surface_field_T": np.asarray(solution.surface_field_T, dtype=float),
        "auxiliary_field_T": np.asarray(solution.auxiliary_field_T, dtype=float),
        "combined_field_T": np.asarray(solution.combined_field_T, dtype=float),
        "residual_field_T": np.asarray(solution.residual_field_T, dtype=float),
        "actuator_names": np.asarray([actuator.name for actuator in problem.actuators], dtype=str),
    }
    for index, actuator in enumerate(problem.actuators):
        prefix = f"actuator_{index}"
        payload[f"{prefix}_name"] = np.asarray(actuator.name, dtype=str)
        payload[f"{prefix}_coefficient_unit"] = np.asarray(actuator.coefficient_unit, dtype=str)
        payload[f"{prefix}_channel_labels"] = np.asarray(actuator.channel_labels, dtype=str)
        payload[f"{prefix}_coefficients"] = np.asarray(
            solution.actuator_coefficients[actuator.name],
            dtype=float,
        )
        payload[f"{prefix}_lower_bounds"] = np.asarray(actuator.lower_bounds, dtype=float)
        payload[f"{prefix}_upper_bounds"] = np.asarray(actuator.upper_bounds, dtype=float)
        payload[f"{prefix}_reference_values"] = np.asarray(actuator.reference_values, dtype=float)
    np.savez_compressed(output, **payload)
    return output


def _surface_constraints(cp: Any, problem: StreamFunctionOptimizationProblem, variable: Any, config: Any) -> list[Any]:
    constraints: list[Any] = []
    contour = _constrained_contour_step_limit(config)
    if bool(contour["contour_step_constraint_enabled"]):
        expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
        full_stream = expansion @ variable
        stream_min = cp.Variable(name="surface_stream_min")
        stream_max = cp.Variable(name="surface_stream_max")
        constraints.extend(
            [
                full_stream >= stream_min,
                full_stream <= stream_max,
                stream_max - stream_min <= float(contour["contour_step_constraint_max_range_A"]),
            ]
        )
    density_limit = float(engineering_limit_diagnostics(config)["engineering_surface_current_density_limit_A_per_m"])
    if density_limit > 0.0:
        density_constraints, _count = _cvxpy_surface_current_density_constraints(
            cp,
            problem,
            variable,
            density_limit,
        )
        constraints.extend(density_constraints)
    return constraints


def _project_stream_function_to_single_mesh_symmetry(
    problem: JointShimOptimizationProblem,
    reduced_stream_function: np.ndarray,
    config: CoilGenConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a solved stream function onto its closest mirror-parity field.

    This is intentionally a post-solve projection, not a reduction of the
    optimization basis. It therefore preserves all spectral degrees of freedom
    during optimization while providing the exact mirror consistency required
    by strict contour extraction. The projected field must remain representable
    in the existing reduced/spectral basis; otherwise the function fails rather
    than silently changing the modeled solution space.
    """

    source = str(getattr(config, "symmetry_reduction_source", "")).strip().lower().replace("-", "_")
    if source not in {"single", "single_mesh", "mesh", "half", "half_mesh"}:
        raise ValueError(
            "Joint stream-function symmetry projection currently requires symmetry_reduction_source='single_mesh'."
        )
    expansion = np.asarray(problem.surface_problem.reducer.expansion_matrix, dtype=float)
    reduced = np.asarray(reduced_stream_function, dtype=float).reshape(-1)
    full = expansion @ reduced
    geometry = _mirror_vertex_mapping_for_single_mesh(
        CoilPart(mesh=problem.surface_problem.combined_mesh),
        config,
    )
    if not bool(geometry.get("ok", False)):
        raise ValueError(
            "Joint stream-function symmetry projection could not establish an exact "
            f"single-mesh mirror mapping: {geometry.get('reason', 'unknown reason')}."
        )
    match_indices = np.asarray(geometry["vertex_match_indices"], dtype=np.int64)
    before = _stream_mirror_residual(full, match_indices)
    sign = float(before["sign"])
    offset = float(before["offset_A"])
    projected_full = 0.5 * (full + sign * (full[match_indices] - offset))

    projected_reduced, *_ = np.linalg.lstsq(expansion, projected_full, rcond=None)
    reconstructed_full = expansion @ projected_reduced
    representation_error = reconstructed_full - projected_full
    stream_scale = max(float(np.ptp(projected_full)), float(np.max(np.abs(projected_full))), 1.0e-30)
    representation_rms = float(np.sqrt(np.mean(np.square(representation_error))))
    representation_max = float(np.max(np.abs(representation_error)))
    tolerance = float(getattr(config, "contour_symmetry_stream_function_tolerance", 1.0e-5))
    if representation_rms / stream_scale > max(0.1 * tolerance, 1.0e-12):
        raise ValueError(
            "Mirror-projected stream function is not accurately representable in the active "
            f"spectral basis (normalized RMS {representation_rms / stream_scale:.6g})."
        )
    after = _stream_mirror_residual(reconstructed_full, match_indices)
    if float(after["normalized_rms"]) > tolerance:
        raise ValueError(
            "Mirror projection did not meet the configured contour symmetry tolerance "
            f"({after['normalized_rms']:.6g} > {tolerance:.6g})."
        )
    change = reconstructed_full - full
    diagnostics: dict[str, Any] = {
        "joint_shim_stream_symmetry_projection_requested": True,
        "joint_shim_stream_symmetry_projection_applied": True,
        "joint_shim_stream_symmetry_projection_source": "single_mesh",
        "joint_shim_stream_symmetry_projection_sign": int(sign),
        "joint_shim_stream_symmetry_before_rms_A": float(before["rms_A"]),
        "joint_shim_stream_symmetry_before_normalized_rms": float(before["normalized_rms"]),
        "joint_shim_stream_symmetry_after_rms_A": float(after["rms_A"]),
        "joint_shim_stream_symmetry_after_normalized_rms": float(after["normalized_rms"]),
        "joint_shim_stream_symmetry_projection_change_rms_A": float(np.sqrt(np.mean(np.square(change)))),
        "joint_shim_stream_symmetry_projection_change_max_A": float(np.max(np.abs(change))),
        "joint_shim_stream_symmetry_spectral_reconstruction_rms_A": representation_rms,
        "joint_shim_stream_symmetry_spectral_reconstruction_max_A": representation_max,
    }
    return np.asarray(projected_reduced, dtype=float), diagnostics


def _stream_mirror_residual(stream: np.ndarray, match_indices: np.ndarray) -> dict[str, float]:
    values = np.asarray(stream, dtype=float).reshape(-1)
    matched = values[np.asarray(match_indices, dtype=np.int64)]
    candidates: list[dict[str, float]] = []
    for sign in (1.0, -1.0):
        offset = float(np.mean(matched - sign * values))
        residual = matched - (sign * values + offset)
        rms = float(np.sqrt(np.mean(np.square(residual))))
        candidates.append(
            {
                "sign": sign,
                "offset_A": offset,
                "rms_A": rms,
                "max_A": float(np.max(np.abs(residual))),
            }
        )
    selected = min(candidates, key=lambda row: row["rms_A"])
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0e-30)
    selected["normalized_rms"] = float(selected["rms_A"] / scale)
    selected["normalized_max"] = float(selected["max_A"] / scale)
    return selected


def _secondary_objective(
    cp: Any,
    problem: JointShimOptimizationProblem,
    surface_variable: Any,
    actuator_variables: Sequence[Any],
    options: JointShimSolveOptions,
) -> Any:
    terms: list[Any] = []
    surface_weight = float(options.surface_cost_weight)
    if surface_weight > 0.0:
        resistance = _symmetric_quadratic_matrix(problem.surface_problem.reduced_resistance)
        scale = _matrix_max_abs(resistance)
        if scale > 0.0:
            terms.append(surface_weight * cp.quad_form(surface_variable, cp.psd_wrap(resistance / scale)))
    smoothness_weight = float(options.surface_smoothness_cost_weight)
    if smoothness_weight > 0.0:
        raw_smoothness = problem.surface_problem.reduced_smoothness_metric
        if raw_smoothness is None or raw_smoothness.size == 0:
            raise ValueError(
                "Joint surface smoothing was requested, but the stream-function "
                "problem has no biharmonic smoothness metric. Set "
                "stream_function_smoothness_factor > 0 before building the problem."
            )
        smoothness = _symmetric_quadratic_matrix(raw_smoothness)
        scale = _matrix_max_abs(smoothness)
        if scale <= 0.0:
            raise ValueError("Joint surface smoothing was requested, but the biharmonic metric is zero.")
        terms.append(smoothness_weight * cp.quad_form(surface_variable, cp.psd_wrap(smoothness / scale)))
    for actuator, variable in zip(problem.actuators, actuator_variables):
        weight = float(
            options.actuator_cost_weights.get(
                actuator.name,
                options.default_actuator_cost_weight,
            )
        )
        if weight > 0.0:
            terms.append(weight * cp.sum_squares(variable))
    if not terms:
        return cp.sum_squares(surface_variable) * 0.0
    objective = terms[0]
    for term in terms[1:]:
        objective = objective + term
    return objective


def _solve_cvxpy_problem(problem: Any, cp: Any, config: Any) -> tuple[str, str]:
    errors: list[str] = []
    verbose = bool(getattr(config, "constrained_optimization_verbose", False))
    for solver_name in _cvxpy_conic_solver_order(cp, config):
        try:
            problem.solve(
                solver=solver_name,
                verbose=verbose,
                **_cvxpy_solver_options(solver_name, config),
            )
        except Exception as exc:
            errors.append(f"{solver_name}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return str(problem.status), str(solver_name)
        errors.append(f"{solver_name}: status {problem.status}")
    details = "; ".join(errors) if errors else "no installed conic solver"
    raise RuntimeError(f"Joint shim CVXPY solve failed ({details}).")


def _variable_value(variable: Any, label: str) -> np.ndarray:
    if variable.value is None:
        raise RuntimeError(f"Joint shim solver returned no value for {label}.")
    value = np.asarray(variable.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"Joint shim solver returned non-finite values for {label}.")
    return value


def _residual_vector(
    surface_matrix: np.ndarray,
    surface_values: np.ndarray,
    actuator_matrices: Sequence[np.ndarray],
    actuator_coefficients: Sequence[np.ndarray],
    target: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(surface_matrix, dtype=float) @ np.asarray(surface_values, dtype=float)
    for matrix, coefficients in zip(actuator_matrices, actuator_coefficients):
        prediction = prediction + np.asarray(matrix, dtype=float) @ np.asarray(coefficients, dtype=float)
    return np.asarray(prediction - target, dtype=float)


def _joint_numerical_scale(
    surface_matrix: np.ndarray,
    target: np.ndarray,
    actuator_matrices: Sequence[np.ndarray],
    actuators: Sequence[LinearShimActuator],
) -> float:
    values = [float(np.linalg.norm(target)), float(np.linalg.norm(surface_matrix, ord="fro"))]
    values.extend(
        float(np.linalg.norm(matrix * np.asarray(actuator.reference_values)[None, :], ord="fro"))
        for matrix, actuator in zip(actuator_matrices, actuators)
    )
    return max(*values, 1.0e-30)


def _active_dsv_mask(problem: StreamFunctionOptimizationProblem, row_count: int) -> np.ndarray:
    weights = problem.target_weights
    active = np.ones(row_count, dtype=bool) if weights is None else np.isfinite(weights) & (np.asarray(weights) > 0.0)
    suppressed = problem.target_suppressed
    suppressed_mask = (
        np.zeros(row_count, dtype=bool) if suppressed is None else np.asarray(suppressed, dtype=bool).reshape(-1)
    )
    if len(active) != row_count or len(suppressed_mask) != row_count:
        raise ValueError("Target weights/suppression rows do not match the target.")
    dsv = active & ~suppressed_mask
    if not np.any(dsv):
        dsv = active
    if not np.any(dsv):
        raise ValueError("No active DSV target rows are available.")
    return dsv


def _active_dsv_weights(problem: StreamFunctionOptimizationProblem, mask: np.ndarray) -> np.ndarray:
    if problem.target_weights is None:
        return np.ones(int(np.count_nonzero(mask)), dtype=float)
    return np.asarray(problem.target_weights, dtype=float).reshape(-1)[mask]


def _validate_actuator_coordinates(
    actuator: LinearShimActuator,
    target_coords: np.ndarray,
    *,
    coordinate_tolerance_m: float,
) -> None:
    if actuator.sample_coords_m is None:
        return
    tolerance = float(coordinate_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("coordinate_tolerance_m must be finite and non-negative.")
    distances = np.linalg.norm(np.asarray(actuator.sample_coords_m) - target_coords, axis=1)
    max_distance = float(np.max(distances)) if len(distances) else 0.0
    if max_distance > tolerance:
        raise ValueError(
            f"Actuator {actuator.name!r} coordinates differ from target rows by up to "
            f"{max_distance:.6g} m, exceeding {tolerance:.6g} m. Interpolate the calibrated "
            "field maps explicitly before joint optimization."
        )


def _channel_vector(values: np.ndarray | None, count: int, default: float, label: str) -> np.ndarray:
    if values is None:
        return np.full(count, default, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) != count or np.any(np.isnan(array)):
        raise ValueError(f"{label} must contain one non-NaN value per channel.")
    return array


def _reference_values(values: np.ndarray | None, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if values is not None:
        references = np.asarray(values, dtype=float).reshape(-1)
        if len(references) != len(lower) or not np.all(np.isfinite(references)) or np.any(references <= 0.0):
            raise ValueError("reference_values must contain one finite positive value per channel.")
        return references
    references = np.maximum(np.abs(lower), np.abs(upper))
    invalid = ~np.isfinite(references) | (references <= 0.0)
    references[invalid] = 1.0
    return references


def _matrix_max_abs(matrix: np.ndarray) -> float:
    values = np.abs(np.asarray(matrix, dtype=float))
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else 0.0


def _surface_quadratic_energy(
    matrix: np.ndarray | None,
    values: np.ndarray,
) -> float:
    """Evaluate a PSD surface metric while clipping round-off-scale negativity."""

    if matrix is None or matrix.size == 0:
        return float("nan")
    quadratic = _symmetric_quadratic_matrix(matrix)
    vector = np.asarray(values, dtype=float).reshape(-1)
    if quadratic.shape != (len(vector), len(vector)):
        raise ValueError("Surface quadratic metric and stream function dimensions differ.")
    energy = float(vector @ quadratic @ vector)
    roundoff_scale = max(
        float(np.linalg.norm(quadratic, ord=np.inf)) * float(vector @ vector),
        1.0,
    )
    if energy < 0.0 and abs(energy) <= 1.0e-10 * roundoff_scale:
        return 0.0
    if energy < 0.0:
        raise ValueError("Surface quadratic metric produced a negative energy.")
    return energy


def _actuator_diagnostics(actuator: LinearShimActuator, coefficients: np.ndarray) -> dict[str, Any]:
    values = np.asarray(coefficients, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(actuator.channel_labels):
        lower = float(actuator.lower_bounds[index])
        upper = float(actuator.upper_bounds[index])
        coefficient = float(values[index])
        finite_scale = max(
            abs(lower) if np.isfinite(lower) else 0.0,
            abs(upper) if np.isfinite(upper) else 0.0,
        )
        rows.append(
            {
                "label": label,
                "coefficient": coefficient,
                "unit": actuator.coefficient_unit,
                "lower_bound": lower,
                "upper_bound": upper,
                "bound_utilization": (abs(coefficient) / finite_scale if finite_scale > 0.0 else float("nan")),
            }
        )
    return {
        "name": actuator.name,
        "channel_count": actuator.channel_count,
        "coefficient_unit": actuator.coefficient_unit,
        "channels": rows,
        "metadata": dict(actuator.metadata),
    }
