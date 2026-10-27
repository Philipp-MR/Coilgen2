"""Population-optimized multichannel shim design.

This module keeps the cohort mathematics independent of dataset-specific inputs and
of any particular current-surface geometry.  A :class:`ShimScenario` contains
one physical disturbance field.  A :class:`SurfaceShimScenario` additionally
contains the CoilGen stream-function operator for the registered mask placement
used by that scenario.

The principal design route extends the physically constrained stream-function
and SVD construction described by Jia et al. (MRM 2020): solve one canonical
stream-function inverse problem per cohort scenario, perform a POD in the
resistance metric, calibrate the retained modes as fields per ampere, and then
evaluate/refine the channels with the real current limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import CoilGenConfig
from .cohort_sectorization import (
    AdaptiveSectorizationOptions,
    AdaptiveSectorizationResult,
    adaptively_sectorize_cohort_channels,
)
from .engineering_limits import engineering_surface_current_density_limit_A_per_m
from .field_components import normalized_b0_direction
from .joint_shim import LinearShimActuator
from .optimization import (
    StreamFunctionOptimizationProblem,
    _cvxpy_conic_solver_order,
    _cvxpy_solver_options,
    _cvxpy_surface_current_density_constraints,
    solve_stream_function_problem,
)
from .progress import DesignCancelled, ProgressReporter, progress_scope


ProgressCallback = Callable[[str], None]


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    ProgressReporter(label="Cohort design").report(None, message)
    if callback is not None:
        callback(message)


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """One active rigid transform using homogeneous column vectors.

    ``matrix_m`` maps coordinates expressed in ``source_frame`` to coordinates
    expressed in ``target_frame``.  Its translation is always stored in metres.
    """

    source_frame: str
    target_frame: str
    matrix_m: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = str(self.source_frame).strip()
        target = str(self.target_frame).strip()
        if not source or not target:
            raise ValueError("Rigid-transform frame names must be non-empty.")
        matrix = np.asarray(self.matrix_m, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("RigidTransform.matrix_m must be a finite 4x4 matrix.")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12, rtol=0.0):
            raise ValueError("Rigid-transform homogeneous bottom row must be [0, 0, 0, 1].")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("Rigid-transform rotation must be orthonormal.")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("Rigid-transform rotation must be right-handed with determinant +1.")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "source_frame", source)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "matrix_m", matrix)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix_m[:3, :3]

    @property
    def translation_m(self) -> np.ndarray:
        return self.matrix_m[:3, 3]

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("RigidTransform.apply_points expects finite shape (samples, 3).")
        return values @ self.rotation.T + self.translation_m

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("RigidTransform.apply_vectors expects finite vectors ending in dimension 3.")
        return values @ self.rotation.T

    def inverse(self) -> "RigidTransform":
        rotation = self.rotation.T
        inverse = np.eye(4, dtype=float)
        inverse[:3, :3] = rotation
        inverse[:3, 3] = -(rotation @ self.translation_m)
        return RigidTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            matrix_m=inverse,
            metadata={"inverse_of": dict(self.metadata)},
        )


@dataclass(frozen=True, slots=True)
class ShimScenario:
    """One cohort disturbance field sampled at physical scanner coordinates."""

    scenario_id: str
    subject_id: str
    coords_scanner_m: np.ndarray
    disturbance_T: np.ndarray
    scanner_from_mask: RigidTransform
    weights: np.ndarray | None = None
    b0_direction_scanner: Sequence[float] = (0.0, 0.0, 1.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        subject_id = str(self.subject_id).strip()
        if not scenario_id or not subject_id:
            raise ValueError("ShimScenario IDs must be non-empty.")
        coords = _readonly_array(self.coords_scanner_m)
        disturbance = _readonly_array(self.disturbance_T).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords_scanner_m must have shape (samples, 3).")
        if len(coords) != len(disturbance):
            raise ValueError("Scenario coordinates and disturbance must have equal sample counts.")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(disturbance)):
            raise ValueError("Scenario coordinates and disturbance must be finite.")
        if self.scanner_from_mask.target_frame != "scanner":
            raise ValueError("scanner_from_mask must target the canonical 'scanner' frame.")
        weights = None
        if self.weights is not None:
            weights = _readonly_array(self.weights).reshape(-1)
            if len(weights) != len(coords) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("Scenario weights must be finite, non-negative, and match the samples.")
            if float(np.sum(weights)) <= 0.0:
                raise ValueError("Scenario weights must contain positive total weight.")
        direction = tuple(float(value) for value in normalized_b0_direction(self.b0_direction_scanner))
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "coords_scanner_m", coords)
        object.__setattr__(self, "disturbance_T", disturbance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "b0_direction_scanner", direction)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def sample_count(self) -> int:
        return len(self.disturbance_T)

    @property
    def cancellation_target_T(self) -> np.ndarray:
        return -np.asarray(self.disturbance_T)


@dataclass(frozen=True, slots=True)
class PopulationShimDataset:
    scenarios: tuple[ShimScenario, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("A population shim dataset requires at least one scenario.")
        identifiers = [scenario.scenario_id for scenario in scenarios]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Population scenario IDs must be unique.")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(scenario.subject_id for scenario in self.scenarios))


@dataclass(frozen=True, slots=True)
class ChannelCurrentLimits:
    """Electrical bounds for the independently driven mask channels."""

    lower_A: np.ndarray
    upper_A: np.ndarray
    resistance_ohm: np.ndarray | None = None
    total_power_limit_W: float | None = None

    def __post_init__(self) -> None:
        lower = _readonly_array(self.lower_A).reshape(-1)
        upper = _readonly_array(self.upper_A).reshape(-1)
        if len(lower) == 0 or len(lower) != len(upper):
            raise ValueError("Current bounds must contain one lower/upper value per channel.")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
            raise ValueError("Current bounds must be finite and satisfy lower <= upper.")
        resistance = None
        if self.resistance_ohm is not None:
            resistance = _readonly_array(self.resistance_ohm).reshape(-1)
            if len(resistance) != len(lower) or np.any(~np.isfinite(resistance)) or np.any(resistance <= 0.0):
                raise ValueError("Channel resistances must be positive, finite, and match the channel count.")
        power = self.total_power_limit_W
        if power is not None and (not np.isfinite(power) or power <= 0.0):
            raise ValueError("total_power_limit_W must be finite and positive when provided.")
        if power is not None and resistance is None:
            raise ValueError("A total power limit requires per-channel resistance_ohm values.")
        object.__setattr__(self, "lower_A", lower)
        object.__setattr__(self, "upper_A", upper)
        object.__setattr__(self, "resistance_ohm", resistance)

    @classmethod
    def symmetric(
        cls,
        channel_count: int,
        limit_A: float = 2.0,
        *,
        resistance_ohm: Sequence[float] | None = None,
        total_power_limit_W: float | None = None,
    ) -> "ChannelCurrentLimits":
        if channel_count < 1 or not np.isfinite(limit_A) or limit_A <= 0.0:
            raise ValueError("channel_count and limit_A must be positive.")
        resistance = None if resistance_ohm is None else np.asarray(resistance_ohm, dtype=float)
        return cls(
            lower_A=np.full(channel_count, -float(limit_A)),
            upper_A=np.full(channel_count, float(limit_A)),
            resistance_ohm=resistance,
            total_power_limit_W=total_power_limit_W,
        )

    @property
    def channel_count(self) -> int:
        return len(self.lower_A)

    @property
    def absolute_limits_A(self) -> np.ndarray:
        return np.maximum(np.abs(self.lower_A), np.abs(self.upper_A))

    @property
    def bidirectional_limits_A(self) -> np.ndarray:
        """Largest symmetric current interval contained in each channel bound."""

        return np.maximum(np.minimum(-self.lower_A, self.upper_A), 0.0)


@dataclass(frozen=True, slots=True)
class CohortFitOptions:
    center_frequency: bool = True
    current_regularization: float = 0.0
    saturation_relative_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_regularization) or self.current_regularization < 0.0:
            raise ValueError("current_regularization must be finite and non-negative.")
        if not np.isfinite(self.saturation_relative_tolerance) or self.saturation_relative_tolerance < 0.0:
            raise ValueError("saturation_relative_tolerance must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class ScenarioFitResult:
    scenario_id: str
    channel_currents_A: np.ndarray
    auxiliary_coefficients: Mapping[str, np.ndarray]
    auxiliary_field_T: np.ndarray
    frequency_offset_T: float
    residual_T: np.ndarray
    rms_before_T: float
    rms_after_T: float
    saturated_channels: np.ndarray
    power_W: float | None
    solver: str
    status: str


@dataclass(frozen=True, slots=True)
class CohortFitResult:
    scenario_results: tuple[ScenarioFitResult, ...]
    channel_count: int

    @property
    def current_matrix_A(self) -> np.ndarray:
        return np.vstack([result.channel_currents_A for result in self.scenario_results])

    @property
    def rms_before_T(self) -> np.ndarray:
        return np.asarray([result.rms_before_T for result in self.scenario_results])

    @property
    def rms_after_T(self) -> np.ndarray:
        return np.asarray([result.rms_after_T for result in self.scenario_results])

    @property
    def saturation_fraction(self) -> float:
        saturated = np.vstack([result.saturated_channels for result in self.scenario_results])
        return float(np.mean(np.any(saturated, axis=1)))

    def summary(self) -> dict[str, float | int]:
        before = self.rms_before_T
        after = self.rms_after_T
        fractions = np.divide(after, before, out=np.zeros_like(after), where=before > 0.0)
        return {
            "scenario_count": len(self.scenario_results),
            "channel_count": self.channel_count,
            "median_rms_before_T": float(np.median(before)),
            "median_rms_after_T": float(np.median(after)),
            "p95_rms_after_T": float(np.quantile(after, 0.95)),
            "worst_rms_after_T": float(np.max(after)),
            "median_residual_fraction": float(np.median(fractions)),
            "p95_residual_fraction": float(np.quantile(fractions, 0.95)),
            "saturated_scenario_fraction": self.saturation_fraction,
        }


@dataclass(frozen=True, slots=True)
class RegisteredSurfaceProblem:
    """A surface operator together with the mask pose used to build it."""

    problem: StreamFunctionOptimizationProblem
    scanner_from_mask: RigidTransform
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scanner_from_mask.target_frame != "scanner":
            raise ValueError("A registered surface problem must map the mask into scanner coordinates.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SurfaceShimScenario:
    scenario: ShimScenario
    registered_surface: RegisteredSurfaceProblem

    @property
    def problem(self) -> StreamFunctionOptimizationProblem:
        return self.registered_surface.problem

    def __post_init__(self) -> None:
        matrix = self.problem.unweighted_reduced_sensitivity
        if matrix is None:
            raise ValueError("A cohort surface problem requires an unweighted reduced sensitivity matrix.")
        if np.asarray(matrix).shape[0] != self.scenario.sample_count:
            raise ValueError("Scenario samples do not match the surface sensitivity rows.")
        target_coords = self.problem.target_coords
        if target_coords is None or np.asarray(target_coords).shape != self.scenario.coords_scanner_m.shape:
            raise ValueError("The registered surface problem must retain the scenario target coordinates.")
        if not np.allclose(target_coords, self.scenario.coords_scanner_m, atol=1.0e-9, rtol=0.0):
            raise ValueError("Surface sensitivity coordinates do not match the scenario scanner coordinates.")
        expected = self.scenario.scanner_from_mask.matrix_m
        actual = self.registered_surface.scanner_from_mask.matrix_m
        if (
            self.scenario.scanner_from_mask.source_frame != self.registered_surface.scanner_from_mask.source_frame
            or not np.allclose(actual, expected, atol=1.0e-10, rtol=0.0)
        ):
            raise ValueError("Surface operator and scenario use different mask-to-scanner registrations.")


@dataclass(frozen=True, slots=True)
class ReferenceStreamFunctionResult:
    scenario_ids: tuple[str, ...]
    reduced_stream_functions: np.ndarray
    diagnostics: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ResistanceWeightedPODResult:
    modes: np.ndarray
    singular_values: np.ndarray
    coefficients: np.ndarray
    explained_energy_fraction: np.ndarray
    cumulative_energy_fraction: np.ndarray
    resistance_orthonormality_error: float
    resistance_rank: int
    centered: bool
    mean_stream_function: np.ndarray


@dataclass(frozen=True, slots=True)
class PODCalibrationResult:
    channel_basis_per_A: np.ndarray
    required_scale_per_A: np.ndarray
    applied_scale_per_A: np.ndarray
    coefficient_quantile_A: np.ndarray
    calibrated_currents_A: np.ndarray
    peak_current_density_A_per_m_per_A: np.ndarray | None
    density_limited_channels: np.ndarray
    quantile_within_current_limit: np.ndarray


@dataclass(frozen=True, slots=True)
class CohortChannelDesignOptions:
    channel_count: int = 4
    current_limit_A: float = 2.0
    coefficient_quantile: float = 0.95
    preferred_current_fraction: float = 0.90
    center_reference_stream_functions: bool = False
    fit_options: CohortFitOptions = field(default_factory=CohortFitOptions)
    refinement_iterations: int = 0
    refinement_resistance_weight: float = 1.0e-3
    refinement_relative_tolerance: float = 1.0e-4
    scenario_weighting: str = "subject_balanced"
    scenario_weights: Mapping[str, float] | None = None
    channel_resistance_ohm: tuple[float, ...] | None = None
    total_power_limit_W: float | None = None
    sectorization: AdaptiveSectorizationOptions = field(default_factory=AdaptiveSectorizationOptions)

    def __post_init__(self) -> None:
        if self.channel_count < 1:
            raise ValueError("channel_count must be positive.")
        if not np.isfinite(self.current_limit_A) or self.current_limit_A <= 0.0:
            raise ValueError("current_limit_A must be finite and positive.")
        if not 0.0 < self.coefficient_quantile <= 1.0:
            raise ValueError("coefficient_quantile must lie in (0, 1].")
        if not 0.0 < self.preferred_current_fraction <= 1.0:
            raise ValueError("preferred_current_fraction must lie in (0, 1].")
        if self.refinement_iterations < 0:
            raise ValueError("refinement_iterations must be non-negative.")
        if not np.isfinite(self.refinement_resistance_weight) or self.refinement_resistance_weight < 0.0:
            raise ValueError("refinement_resistance_weight must be finite and non-negative.")
        if not np.isfinite(self.refinement_relative_tolerance) or self.refinement_relative_tolerance < 0.0:
            raise ValueError("refinement_relative_tolerance must be finite and non-negative.")
        weighting = str(self.scenario_weighting).strip().lower().replace("-", "_")
        if weighting not in {"subject_balanced", "uniform_scenario"}:
            raise ValueError("scenario_weighting must be 'subject_balanced' or 'uniform_scenario'.")
        explicit_weights = None
        if self.scenario_weights is not None:
            explicit_weights = {str(key): float(value) for key, value in self.scenario_weights.items()}
            if any(not np.isfinite(value) or value <= 0.0 for value in explicit_weights.values()):
                raise ValueError("Explicit scenario weights must be finite and positive.")
        object.__setattr__(self, "scenario_weighting", weighting)
        object.__setattr__(self, "scenario_weights", explicit_weights)
        if not isinstance(self.sectorization, AdaptiveSectorizationOptions):
            raise TypeError("sectorization must be an AdaptiveSectorizationOptions instance.")


@dataclass(frozen=True, slots=True)
class CohortChannelDesignResult:
    reference: ReferenceStreamFunctionResult
    pod: ResistanceWeightedPODResult
    calibration: PODCalibrationResult
    current_limits: ChannelCurrentLimits
    scenario_weights: np.ndarray
    channel_basis_per_A: np.ndarray
    full_channel_basis_per_A: np.ndarray
    fit: CohortFitResult
    refinement_history: tuple[Mapping[str, Any], ...]
    sectorization: AdaptiveSectorizationResult | None = None


def fit_scenario_currents(
    scenario: ShimScenario,
    channel_fields_T_per_A: np.ndarray,
    limits: ChannelCurrentLimits,
    *,
    options: CohortFitOptions | None = None,
    auxiliary_actuators: Sequence[LinearShimActuator] = (),
) -> ScenarioFitResult:
    """Fit one scenario using bounded channel currents and optional auxiliaries."""

    fit_options = options or CohortFitOptions()
    channel_fields = np.asarray(channel_fields_T_per_A, dtype=float)
    if channel_fields.shape != (scenario.sample_count, limits.channel_count):
        raise ValueError("channel_fields_T_per_A must have shape (scenario samples, channels).")
    if not np.all(np.isfinite(channel_fields)):
        raise ValueError("Channel fields must be finite.")

    matrices = [channel_fields]
    lower = [np.asarray(limits.lower_A)]
    upper = [np.asarray(limits.upper_A)]
    auxiliary_slices: list[tuple[LinearShimActuator, slice]] = []
    column_offset = limits.channel_count
    for actuator in auxiliary_actuators:
        if actuator.sample_count != scenario.sample_count:
            raise ValueError(f"Actuator {actuator.name!r} does not match scenario sample count.")
        if actuator.sample_coords_m is not None and not np.allclose(
            actuator.sample_coords_m,
            scenario.coords_scanner_m,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise ValueError(f"Actuator {actuator.name!r} coordinates do not match the scenario.")
        matrix = actuator.component_matrix(
            "BdotB0",
            b0_direction=scenario.b0_direction_scanner,
        )
        matrices.append(matrix)
        lower.append(np.asarray(actuator.lower_bounds, dtype=float))
        upper.append(np.asarray(actuator.upper_bounds, dtype=float))
        actuator_slice = slice(column_offset, column_offset + actuator.channel_count)
        auxiliary_slices.append((actuator, actuator_slice))
        column_offset += actuator.channel_count

    design = np.column_stack(matrices)
    target = scenario.cancellation_target_T
    weights = _scenario_weights(scenario)
    transformed_design, transformed_target = _weighted_centered_system(
        design,
        target,
        weights,
        center=fit_options.center_frequency,
    )
    lower_values = np.concatenate(lower)
    upper_values = np.concatenate(upper)
    field_scale = max(_weighted_rms(scenario.disturbance_T, weights, fit_options.center_frequency), 1.0e-15)
    if fit_options.current_regularization > 0.0:
        current_scale = np.maximum(limits.absolute_limits_A, 1.0e-12)
        regularizer = np.zeros((limits.channel_count, design.shape[1]), dtype=float)
        regularizer[:, : limits.channel_count] = np.diag(
            np.sqrt(fit_options.current_regularization) * field_scale / current_scale
        )
        transformed_design = np.vstack((transformed_design, regularizer))
        transformed_target = np.concatenate((transformed_target, np.zeros(limits.channel_count)))

    if limits.total_power_limit_W is None:
        solution, solver, status = _bounded_least_squares(
            transformed_design,
            transformed_target,
            lower_values,
            upper_values,
        )
    else:
        solution, solver, status = _power_constrained_least_squares(
            transformed_design,
            transformed_target,
            lower_values,
            upper_values,
            limits,
        )

    currents = np.asarray(solution[: limits.channel_count], dtype=float)
    auxiliary_coefficients: dict[str, np.ndarray] = {}
    auxiliary_field = np.zeros(scenario.sample_count, dtype=float)
    for actuator, values_slice in auxiliary_slices:
        values = np.asarray(solution[values_slice], dtype=float)
        auxiliary_coefficients[actuator.name] = values
        auxiliary_field += (
            actuator.component_matrix(
                "BdotB0",
                b0_direction=scenario.b0_direction_scanner,
            )
            @ values
        )
    raw_residual = scenario.disturbance_T + channel_fields @ currents + auxiliary_field
    frequency_offset = 0.0
    if fit_options.center_frequency:
        frequency_offset = -float(weights @ raw_residual) / float(np.sum(weights))
    residual = raw_residual + frequency_offset
    tolerance = fit_options.saturation_relative_tolerance * np.maximum(
        limits.absolute_limits_A,
        np.finfo(float).eps,
    )
    saturated = (currents - limits.lower_A <= tolerance) | (limits.upper_A - currents <= tolerance)
    power = None
    if limits.resistance_ohm is not None:
        power = float(np.sum(limits.resistance_ohm * currents**2))
    return ScenarioFitResult(
        scenario_id=scenario.scenario_id,
        channel_currents_A=_readonly_array(currents),
        auxiliary_coefficients={name: _readonly_array(values) for name, values in auxiliary_coefficients.items()},
        auxiliary_field_T=_readonly_array(auxiliary_field),
        frequency_offset_T=frequency_offset,
        residual_T=_readonly_array(residual),
        rms_before_T=_weighted_rms(scenario.disturbance_T, weights, fit_options.center_frequency),
        rms_after_T=_weighted_rms(residual, weights, center=False),
        saturated_channels=_readonly_array(saturated, dtype=bool),
        power_W=power,
        solver=solver,
        status=status,
    )


def fit_cohort_currents(
    dataset: PopulationShimDataset,
    channel_fields_by_scenario: Mapping[str, np.ndarray],
    limits: ChannelCurrentLimits,
    *,
    options: CohortFitOptions | None = None,
    auxiliary_actuators_by_scenario: Mapping[str, Sequence[LinearShimActuator]] | None = None,
) -> CohortFitResult:
    """Fit every scenario and preserve patient/scenario separation."""

    auxiliary = auxiliary_actuators_by_scenario or {}
    results: list[ScenarioFitResult] = []
    for index, scenario in enumerate(dataset.scenarios):
        if scenario.scenario_id not in channel_fields_by_scenario:
            raise KeyError(f"Missing channel fields for scenario {scenario.scenario_id!r}.")
        with progress_scope(100 * index / len(dataset.scenarios), 100 * (index + 1) / len(dataset.scenarios)):
            ProgressReporter(label="Cohort current fitting").report(None, f"Solving currents: {scenario.scenario_id}")
            results.append(
                fit_scenario_currents(
                    scenario,
                    channel_fields_by_scenario[scenario.scenario_id],
                    limits,
                    options=options,
                    auxiliary_actuators=auxiliary.get(scenario.scenario_id, ()),
                )
            )
    return CohortFitResult(tuple(results), limits.channel_count)


def solve_reference_stream_functions(
    surface_scenarios: Sequence[SurfaceShimScenario],
    config: CoilGenConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ReferenceStreamFunctionResult:
    """Solve one physically constrained cancellation problem per scenario."""

    started = perf_counter()
    scenarios = tuple(surface_scenarios)
    if not scenarios:
        raise ValueError("At least one surface scenario is required.")
    dof_count = scenarios[0].problem.reduced_sensitivity.shape[1]
    sample_count = scenarios[0].scenario.sample_count
    _emit_progress(
        progress_callback,
        f"Reference setup: {len(scenarios)} scenarios, {sample_count} target samples, "
        f"{dof_count} reduced stream-function DOFs.",
    )
    reference_resistance = np.asarray(scenarios[0].problem.reduced_resistance, dtype=float)
    stream_functions = np.empty((dof_count, len(scenarios)), dtype=float)
    diagnostics: list[Mapping[str, Any]] = []
    expected_direction = _scalar_component_direction(config)
    scenario_problems: list[StreamFunctionOptimizationProblem] = []
    for item in scenarios:
        problem = item.problem
        if not np.allclose(
            item.scenario.b0_direction_scanner,
            expected_direction,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError(
                f"Scenario {item.scenario.scenario_id!r} B0 direction does not match "
                f"CoilGen's {config.target_field_component!r} sensitivity component."
            )
        if problem.reduced_sensitivity.shape[1] != dof_count:
            raise ValueError("All cohort problems must use the same reduced stream-function coordinates.")
        if not np.allclose(problem.reduced_resistance, reference_resistance, atol=1.0e-10, rtol=1.0e-8):
            raise ValueError("All rigid-mask cohort problems must use the same reduced resistance metric.")
        scenario_problems.append(_problem_with_scalar_target(problem, item.scenario.cancellation_target_T, config))

    if _can_use_shared_fixed_tikhonov_solve(scenarios, config):
        _emit_progress(
            progress_callback,
            "Reference solve: common operator and fixed Tikhonov detected; using one "
            f"factorization with {len(scenarios)} right-hand sides.",
        )
        stream_functions, diagnostics = _solve_shared_fixed_tikhonov_references(
            tuple(scenario_problems),
            config,
            progress_callback=progress_callback,
        )
    else:
        _emit_progress(
            progress_callback,
            "Reference solve: shared factorization is unavailable; solving each scenario independently.",
        )
        for index, scenario_problem in enumerate(scenario_problems):
            scenario_id = scenarios[index].scenario.scenario_id
            _emit_progress(
                progress_callback,
                f"Reference solve {index + 1}/{len(scenarios)}: {scenario_id}...",
            )
            scenario_started = perf_counter()
            with progress_scope(100 * index / len(scenarios), 100 * (index + 1) / len(scenarios)):
                solution = solve_stream_function_problem(scenario_problem, config)
            stream_functions[:, index] = solution.reduced_stream_function
            diagnostic = dict(solution.diagnostics)
            scenario_seconds = perf_counter() - scenario_started
            diagnostic["cohort_reference_solve_seconds"] = scenario_seconds
            diagnostics.append(diagnostic)
            _emit_progress(
                progress_callback,
                f"Reference solve {index + 1}/{len(scenarios)} complete in {scenario_seconds:.2f} s.",
            )
    elapsed = perf_counter() - started
    _emit_progress(
        progress_callback,
        f"Reference stream functions complete in {elapsed:.2f} s.",
    )
    return ReferenceStreamFunctionResult(
        scenario_ids=tuple(item.scenario.scenario_id for item in scenarios),
        reduced_stream_functions=_readonly_array(stream_functions),
        diagnostics=tuple(diagnostics),
    )


def _can_use_shared_fixed_tikhonov_solve(
    scenarios: Sequence[SurfaceShimScenario],
    config: CoilGenConfig,
) -> bool:
    """Return whether one factorization is mathematically identical to all solves."""

    method = str(getattr(config, "sf_opt_method", "tikhonov")).strip().lower().replace("-", "_")
    selection = str(getattr(config, "regularization_selection_method", "l_curve")).strip().lower().replace("-", "_")
    if method != "tikhonov" or selection != "fixed":
        return False
    if float(getattr(config, "stream_function_smoothness_factor", 0.0)) > 0.0:
        return False
    if float(getattr(config, "turn_spacing_regularization_factor", 0.0)) > 0.0:
        return False
    if float(getattr(config, "tikonov_reg_factor", 1.0)) <= 0.0:
        return False
    # Identity is intentional: a shared RegisteredSurfaceProblem guarantees the
    # exact same sampling, sensitivity, reducer, resistance, and mask pose.
    reference_problem = scenarios[0].problem
    return all(item.problem is reference_problem for item in scenarios)


def _solve_shared_fixed_tikhonov_references(
    scenario_problems: Sequence[StreamFunctionOptimizationProblem],
    config: CoilGenConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[Mapping[str, Any]]]:
    """Solve common-operator fixed-Tikhonov targets with one factorization.

    For target columns B = [b_1, ..., b_s], this evaluates

        (A.T A + lambda^2 R) X = A.T B

    once. Each column of X is the corresponding independent fixed-Tikhonov
    solution, apart from normal floating-point differences in a multi-RHS solve.
    """

    reference = scenario_problems[0]
    sensitivity = np.asarray(reference.reduced_sensitivity, dtype=float)
    resistance = np.asarray(reference.reduced_resistance, dtype=float)
    targets = np.column_stack([np.asarray(problem.target_z, dtype=float) for problem in scenario_problems])
    scale = reference.regularization_scale
    factor = float(getattr(config, "tikonov_reg_factor", 1.0))
    lambda_value = factor * scale
    assembly_started = perf_counter()
    _emit_progress(
        progress_callback,
        f"Reference assembly: forming A.T@A and A.T@B from A={sensitivity.shape} and B={targets.shape}...",
    )
    normal_matrix = sensitivity.T @ sensitivity
    resistance = 0.5 * (resistance + resistance.T)
    lhs = normal_matrix + lambda_value * lambda_value * resistance
    rhs = sensitivity.T @ targets
    assembly_seconds = perf_counter() - assembly_started
    matrix_memory_mib = lhs.nbytes / (1024.0**2)
    _emit_progress(
        progress_callback,
        f"Reference assembly complete in {assembly_seconds:.2f} s; solving "
        f"{lhs.shape[0]} x {lhs.shape[1]} system ({matrix_memory_mib:.1f} MiB) with "
        f"{rhs.shape[1]} RHS...",
    )
    solve_started = perf_counter()
    solutions, solver_name, relative_residual = _solve_shared_linear_system(lhs, rhs, config)
    solve_seconds = perf_counter() - solve_started
    _emit_progress(
        progress_callback,
        f"Reference linear solve complete in {solve_seconds:.2f} s via {solver_name}; "
        f"maximum relative residual {float(np.max(relative_residual)):.3e}.",
    )

    diagnostics: list[Mapping[str, Any]] = []
    symmetry = dict(reference.symmetry_diagnostics or {})
    diagnostic_started = perf_counter()
    _emit_progress(progress_callback, "Reference diagnostics: evaluating field and resistance norms...")
    reference_residual_norms = np.linalg.norm(sensitivity @ solutions - targets, axis=0)
    resistance_products = resistance @ solutions
    reference_resistance_norms = np.sqrt(
        np.maximum(np.einsum("is,is->s", solutions, resistance_products, optimize=True), 0.0)
    )
    for index in range(len(scenario_problems)):
        diagnostics.append(
            {
                "selection_method": "fixed",
                "candidate_solver": "shared_multi_rhs",
                "linear_solver": solver_name,
                "cohort_shared_factorization": True,
                "cohort_shared_rhs_count": len(scenario_problems),
                "cohort_shared_assembly_seconds": assembly_seconds,
                "cohort_shared_linear_solve_seconds": solve_seconds,
                "linear_system_relative_residual": float(relative_residual[index]),
                "factor_scale": scale,
                "selected_index": 0,
                "selected_factor": factor,
                "selected_lambda": lambda_value,
                "candidate_factors": [factor],
                "candidate_lambdas": [lambda_value],
                "residual_norms": [],
                "regularization_norms": [],
                "curvature": [],
                "stream_function_scheme": "tikhonov",
                "stream_function_cost_function": "field_error_plus_resistance",
                "cost_function_formula": "||A x - b||_2^2 + lambda^2 x^T R x",
                "unknown_count": int(sensitivity.shape[1]),
                "target_sample_count": int(sensitivity.shape[0]),
                "reference_residual_norm_T": float(reference_residual_norms[index]),
                "reference_resistance_norm": float(reference_resistance_norms[index]),
                **symmetry,
            }
        )
    _emit_progress(
        progress_callback,
        f"Reference diagnostics complete in {perf_counter() - diagnostic_started:.2f} s.",
    )
    return solutions, diagnostics


def _solve_shared_linear_system(
    lhs: np.ndarray,
    rhs: np.ndarray,
    config: CoilGenConfig,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Apply CoilGen's solve/auto/pinv policy to a matrix right-hand side."""

    solver = str(getattr(config, "optimization_solver", "auto")).strip().lower()
    if solver not in {"auto", "solve", "pinv"}:
        raise ValueError("optimization_solver must be 'auto', 'solve', or 'pinv'.")
    if solver == "pinv":
        solution = np.linalg.pinv(lhs) @ rhs
        solver_name = "pinv"
    else:
        try:
            solution = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            if solver == "solve":
                raise
            solution = np.linalg.pinv(lhs) @ rhs
            solver_name = "pinv_fallback"
        else:
            solver_name = "solve"

    residual = lhs @ solution - rhs
    denominator = np.maximum(np.linalg.norm(rhs, axis=0), 1.0e-30)
    relative_residual = np.linalg.norm(residual, axis=0) / denominator
    tolerance = max(float(getattr(config, "optimization_solver_residual_tolerance", 1.0e-8)), 0.0)
    valid = np.all(np.isfinite(solution)) and np.all(np.isfinite(relative_residual))
    if solver_name == "solve" and (not valid or float(np.max(relative_residual)) > tolerance):
        if solver == "solve":
            raise np.linalg.LinAlgError(
                "Direct shared solve residual "
                f"{float(np.max(relative_residual)):.3e} exceeds tolerance {tolerance:.3e}."
            )
        solution = np.linalg.pinv(lhs) @ rhs
        residual = lhs @ solution - rhs
        relative_residual = np.linalg.norm(residual, axis=0) / denominator
        solver_name = "pinv_residual_fallback"
    return np.asarray(solution, dtype=float), solver_name, np.asarray(relative_residual, dtype=float)


def resistance_weighted_pod(
    reference_stream_functions: np.ndarray,
    resistance_matrix: np.ndarray,
    channel_count: int,
    *,
    scenario_weights: Sequence[float] | None = None,
    center: bool = False,
    eigenvalue_relative_tolerance: float = 1.0e-11,
    progress_callback: ProgressCallback | None = None,
) -> ResistanceWeightedPODResult:
    """Return POD modes orthonormal in the stream-function resistance metric."""

    references = np.asarray(reference_stream_functions, dtype=float)
    resistance = np.asarray(resistance_matrix, dtype=float)
    if references.ndim != 2 or references.shape[1] < 1:
        raise ValueError("reference_stream_functions must have shape (dofs, scenarios).")
    if resistance.shape != (references.shape[0], references.shape[0]):
        raise ValueError("resistance_matrix must be square and match the stream-function DOFs.")
    if channel_count < 1 or channel_count > min(references.shape):
        raise ValueError("channel_count exceeds the available reference rank dimensions.")
    resistance = 0.5 * (resistance + resistance.T)
    eigensystem_started = perf_counter()
    _emit_progress(
        progress_callback,
        f"Channel basis: diagonalizing the {resistance.shape[0]} x {resistance.shape[1]} resistance metric...",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(resistance)
    largest = float(max(np.max(eigenvalues), 0.0))
    threshold = max(largest * eigenvalue_relative_tolerance, np.finfo(float).eps)
    retained = eigenvalues > threshold
    if int(np.count_nonzero(retained)) < channel_count:
        raise ValueError("The positive resistance subspace is smaller than the requested channel count.")
    _emit_progress(
        progress_callback,
        f"Resistance eigensystem complete in {perf_counter() - eigensystem_started:.2f} s; "
        f"positive rank {int(np.count_nonzero(retained))}/{len(eigenvalues)}.",
    )
    positive_values = eigenvalues[retained]
    positive_vectors = eigenvectors[:, retained]
    sqrt_resistance = np.sqrt(positive_values)[:, None] * positive_vectors.T
    inverse_sqrt_resistance = positive_vectors / np.sqrt(positive_values)[None, :]

    weights = np.ones(references.shape[1], dtype=float)
    if scenario_weights is not None:
        weights = np.asarray(scenario_weights, dtype=float).reshape(-1)
        if len(weights) != references.shape[1] or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("scenario_weights must be finite, non-negative, and match the scenarios.")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("scenario_weights must contain positive total weight.")
    weights = weights / np.sum(weights)
    mean = references @ weights if center else np.zeros(references.shape[0], dtype=float)
    centered_references = references - mean[:, None]
    whitened = sqrt_resistance @ centered_references
    weighted_whitened = whitened * np.sqrt(weights)[None, :]
    svd_started = perf_counter()
    _emit_progress(
        progress_callback,
        f"Channel basis: computing resistance-weighted POD/SVD of "
        f"{weighted_whitened.shape[0]} x {weighted_whitened.shape[1]} reference matrix...",
    )
    left, singular_values, _right = np.linalg.svd(weighted_whitened, full_matrices=False)
    singular_threshold = max(
        float(singular_values[0]) * eigenvalue_relative_tolerance,
        np.finfo(float).eps,
    )
    data_rank = int(np.count_nonzero(singular_values > singular_threshold))
    if data_rank < channel_count:
        raise ValueError(
            "The cohort reference solutions have numerical rank "
            f"{data_rank}, smaller than the requested {channel_count} channels."
        )
    _emit_progress(
        progress_callback,
        f"Resistance-weighted POD complete in {perf_counter() - svd_started:.2f} s; "
        f"reference rank {data_rank}, retaining {channel_count} channels.",
    )
    modes = inverse_sqrt_resistance @ left[:, :channel_count]
    coefficients = modes.T @ resistance @ centered_references
    energy = singular_values**2
    total_energy = float(np.sum(energy))
    explained = energy[:channel_count] / total_energy if total_energy > 0.0 else np.zeros(channel_count)
    cumulative = np.cumsum(energy) / total_energy if total_energy > 0.0 else np.zeros_like(energy)
    orthonormality = modes.T @ resistance @ modes
    return ResistanceWeightedPODResult(
        modes=_readonly_array(modes),
        singular_values=_readonly_array(singular_values[:channel_count]),
        coefficients=_readonly_array(coefficients),
        explained_energy_fraction=_readonly_array(explained),
        cumulative_energy_fraction=_readonly_array(cumulative[:channel_count]),
        resistance_orthonormality_error=float(np.linalg.norm(orthonormality - np.eye(channel_count))),
        resistance_rank=int(np.count_nonzero(retained)),
        centered=bool(center),
        mean_stream_function=_readonly_array(mean),
    )


def calibrate_pod_modes(
    pod: ResistanceWeightedPODResult,
    reference_stream_functions: np.ndarray,
    resistance_matrix: np.ndarray,
    limits: ChannelCurrentLimits,
    *,
    coefficient_quantile: float = 0.95,
    preferred_current_fraction: float = 0.90,
    scenario_weights: Sequence[float] | None = None,
    surface_problem: StreamFunctionOptimizationProblem | None = None,
    surface_current_density_limit_A_per_m: float | None = None,
) -> PODCalibrationResult:
    """Scale POD modes into stream function per ampere under the 2 A envelope."""

    if pod.modes.shape[1] != limits.channel_count:
        raise ValueError("POD mode count and current-limit channel count differ.")
    if not 0.0 < coefficient_quantile <= 1.0 or not 0.0 < preferred_current_fraction <= 1.0:
        raise ValueError("Calibration quantile/fraction must lie in (0, 1].")
    references = np.asarray(reference_stream_functions, dtype=float)
    resistance = np.asarray(resistance_matrix, dtype=float)
    centered = references - pod.mean_stream_function[:, None]
    coefficients = pod.modes.T @ resistance @ centered
    coefficient_quantiles = _scenario_quantiles(
        np.abs(coefficients),
        coefficient_quantile,
        scenario_weights,
    )
    symmetric_current = limits.bidirectional_limits_A
    if np.any(symmetric_current <= 0.0):
        raise ValueError("POD calibration requires every channel bound to include positive and negative current.")
    usable_current = symmetric_current * preferred_current_fraction
    required_scale = np.divide(
        coefficient_quantiles,
        usable_current,
        out=np.zeros_like(coefficient_quantiles),
        where=usable_current > 0.0,
    )
    required_scale = np.maximum(required_scale, np.finfo(float).eps)
    applied_scale = required_scale.copy()
    peak_density = None
    density_limited = np.zeros(limits.channel_count, dtype=bool)
    if surface_problem is not None and surface_current_density_limit_A_per_m is not None:
        density_limit = float(surface_current_density_limit_A_per_m)
        if not np.isfinite(density_limit) or density_limit <= 0.0:
            raise ValueError("surface_current_density_limit_A_per_m must be positive when provided.")
        peak_density = np.asarray(
            [surface_problem.max_surface_current_density(pod.modes[:, index]) for index in range(limits.channel_count)]
        )
        maximum_scale = np.divide(
            density_limit,
            limits.absolute_limits_A * peak_density,
            out=np.full_like(peak_density, np.inf),
            where=(limits.absolute_limits_A * peak_density) > 0.0,
        )
        density_limited = required_scale > maximum_scale
        applied_scale = np.minimum(required_scale, maximum_scale)
    basis = pod.modes * applied_scale[None, :]
    calibrated_currents = np.divide(
        coefficients,
        applied_scale[:, None],
        out=np.zeros_like(coefficients),
        where=applied_scale[:, None] > 0.0,
    )
    calibrated_quantile = _scenario_quantiles(
        np.abs(calibrated_currents),
        coefficient_quantile,
        scenario_weights,
    )
    within = calibrated_quantile <= symmetric_current * (1.0 + 1.0e-10)
    return PODCalibrationResult(
        channel_basis_per_A=_readonly_array(basis),
        required_scale_per_A=_readonly_array(required_scale),
        applied_scale_per_A=_readonly_array(applied_scale),
        coefficient_quantile_A=_readonly_array(calibrated_quantile),
        calibrated_currents_A=_readonly_array(calibrated_currents),
        peak_current_density_A_per_m_per_A=None
        if peak_density is None
        else _readonly_array(peak_density * applied_scale),
        density_limited_channels=_readonly_array(density_limited, dtype=bool),
        quantile_within_current_limit=_readonly_array(within, dtype=bool),
    )


def design_cohort_channels(
    surface_scenarios: Sequence[SurfaceShimScenario],
    config: CoilGenConfig,
    options: CohortChannelDesignOptions | None = None,
    *,
    auxiliary_actuators_by_scenario: Mapping[str, Sequence[LinearShimActuator]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CohortChannelDesignResult:
    """Design and boundedly evaluate a population-derived channel basis."""

    design_started = perf_counter()
    design_options = options or CohortChannelDesignOptions()
    scenarios = tuple(surface_scenarios)
    if len(scenarios) < design_options.channel_count:
        raise ValueError("The cohort needs at least as many scenarios as requested channels.")
    refinement_weight = 15 if design_options.refinement_iterations else 0
    sectorization_weight = 10 if design_options.sectorization.enabled else 0
    scale = 100 / (75 + refinement_weight + sectorization_weight)
    with progress_scope(0, 45 * scale):
        reference = solve_reference_stream_functions(
            scenarios,
            config,
            progress_callback=progress_callback,
        )
    resistance = np.asarray(scenarios[0].problem.reduced_resistance, dtype=float)
    scenario_weights = _cohort_design_scenario_weights(scenarios, design_options)
    _emit_progress(
        progress_callback,
        f"Channel basis: computing {design_options.channel_count} subject-weighted modes "
        f"from {len(scenarios)} reference solutions.",
    )
    with progress_scope(45 * scale, 60 * scale):
        pod = resistance_weighted_pod(
            reference.reduced_stream_functions,
            resistance,
            design_options.channel_count,
            scenario_weights=scenario_weights,
            center=design_options.center_reference_stream_functions,
            progress_callback=progress_callback,
        )
    resistances = design_options.channel_resistance_ohm
    limits = ChannelCurrentLimits.symmetric(
        design_options.channel_count,
        design_options.current_limit_A,
        resistance_ohm=resistances,
        total_power_limit_W=design_options.total_power_limit_W,
    )
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    _emit_progress(
        progress_callback,
        "Channel calibration: applying cohort coefficient quantile, current bounds, and surface-current-density limit.",
    )
    calibration_started = perf_counter()
    with progress_scope(60 * scale, 65 * scale):
        calibration = calibrate_pod_modes(
            pod,
            reference.reduced_stream_functions,
            resistance,
            limits,
            coefficient_quantile=design_options.coefficient_quantile,
            preferred_current_fraction=design_options.preferred_current_fraction,
            scenario_weights=scenario_weights,
            surface_problem=scenarios[0].problem if density_limit > 0.0 else None,
            surface_current_density_limit_A_per_m=density_limit if density_limit > 0.0 else None,
        )
    _emit_progress(
        progress_callback,
        f"Channel calibration complete in {perf_counter() - calibration_started:.2f} s; "
        f"density-limited channels {np.flatnonzero(calibration.density_limited_channels).size}/"
        f"{limits.channel_count}.",
    )
    basis = np.asarray(calibration.channel_basis_per_A, dtype=float)
    dataset = PopulationShimDataset(tuple(item.scenario for item in scenarios))
    _emit_progress(
        progress_callback,
        f"Bounded current fit: evaluating {len(scenarios)} scenarios against the calibrated basis...",
    )
    fit_started = perf_counter()
    fields = _channel_fields_for_surface_scenarios(scenarios, basis)
    with progress_scope(65 * scale, 75 * scale):
        fit = fit_cohort_currents(
            dataset,
            fields,
            limits,
            options=design_options.fit_options,
            auxiliary_actuators_by_scenario=auxiliary_actuators_by_scenario,
        )
    _emit_progress(
        progress_callback,
        f"Bounded current fit complete in {perf_counter() - fit_started:.2f} s; "
        f"median residual RMS {float(np.median(fit.rms_after_T)):.6g} T, "
        f"saturated-scenario fraction {fit.saturation_fraction:.3f}.",
    )
    history: list[Mapping[str, Any]] = []
    if design_options.refinement_iterations:
        _emit_progress(
            progress_callback,
            f"Channel refinement: starting up to {design_options.refinement_iterations} alternating steps.",
        )
        with progress_scope(75 * scale, (75 + refinement_weight) * scale):
            basis, fit, history = refine_cohort_channel_basis(
                scenarios,
                basis,
                limits,
                config,
                options=design_options.fit_options,
                auxiliary_actuators_by_scenario=auxiliary_actuators_by_scenario,
                scenario_weights=scenario_weights,
                maximum_iterations=design_options.refinement_iterations,
                resistance_weight=design_options.refinement_resistance_weight,
                relative_tolerance=design_options.refinement_relative_tolerance,
            )
    sectorization = None
    if design_options.sectorization.enabled:
        _emit_progress(
            progress_callback,
            "Adaptive sectorization: localizing channel support and enforcing the final hard partition...",
        )
        with progress_scope((75 + refinement_weight) * scale, 100):
            basis, fit, sectorization = adaptively_sectorize_cohort_channels(
                scenarios,
                basis,
                limits,
                config,
                design_options.sectorization,
                fit_options=design_options.fit_options,
                auxiliary_actuators_by_scenario=auxiliary_actuators_by_scenario,
                scenario_weights=scenario_weights,
                discovery_basis_reduced=reference.reduced_stream_functions,
                progress_callback=progress_callback,
            )
    full_basis = np.column_stack(
        [scenarios[0].problem.expand(basis[:, index]) for index in range(limits.channel_count)]
    )
    _emit_progress(
        progress_callback,
        f"Cohort channel design stage complete in {perf_counter() - design_started:.2f} s; "
        f"median residual RMS {float(np.median(fit.rms_after_T)):.6g} T.",
    )
    return CohortChannelDesignResult(
        reference=reference,
        pod=pod,
        calibration=calibration,
        current_limits=limits,
        scenario_weights=_readonly_array(scenario_weights),
        channel_basis_per_A=_readonly_array(basis),
        full_channel_basis_per_A=_readonly_array(full_basis),
        fit=fit,
        refinement_history=tuple(history),
        sectorization=sectorization,
    )


def save_cohort_channel_stream_functions(
    directory: str | Path,
    result: CohortChannelDesignResult,
    *,
    prefix: str = "cohort_channel",
    contour_current_A: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    """Save one preoptimization-compatible NPZ file per physical channel.

    ``stream_function`` is scaled to the stated contour current.  For a 2 A
    channel, CoilGen should therefore contour that array with a 2 A wire-current
    setting.  ``stream_function_per_A`` retains the field-calibrated basis used
    by the cohort current optimizer.
    """

    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename_prefix = str(prefix).strip()
    if not filename_prefix or Path(filename_prefix).name != filename_prefix:
        raise ValueError("prefix must be a non-empty filename prefix, not a path.")
    full_basis = np.asarray(result.full_channel_basis_per_A, dtype=float)
    reduced_basis = np.asarray(result.channel_basis_per_A, dtype=float)
    channel_count = result.current_limits.channel_count
    if full_basis.ndim != 2 or reduced_basis.ndim != 2:
        raise ValueError("Cohort channel bases must be two-dimensional.")
    if full_basis.shape[1] != channel_count or reduced_basis.shape[1] != channel_count:
        raise ValueError("Cohort channel basis and current-limit dimensions differ.")
    if contour_current_A is None:
        currents = np.asarray(result.current_limits.absolute_limits_A, dtype=float)
    else:
        currents = np.asarray(contour_current_A, dtype=float).reshape(-1)
    if len(currents) != channel_count or np.any(~np.isfinite(currents)) or np.any(currents <= 0.0):
        raise ValueError("contour_current_A must contain one positive finite value per channel.")
    if np.any(currents > result.current_limits.absolute_limits_A * (1.0 + 1.0e-12)):
        raise ValueError("contour_current_A exceeds a configured channel hardware limit.")

    outputs: list[Path] = []
    sectorization = result.sectorization
    for index, current_A in enumerate(currents):
        path = output_directory / f"{filename_prefix}_{index + 1:02d}.npz"
        payload: dict[str, Any] = {
            "stream_function": full_basis[:, index] * current_A,
            "stream_function_per_A": full_basis[:, index],
            "reduced_stream_function_per_A": reduced_basis[:, index],
            "contour_current_A": np.asarray(current_A),
            "current_lower_A": np.asarray(result.current_limits.lower_A[index]),
            "current_upper_A": np.asarray(result.current_limits.upper_A[index]),
            "channel_index": np.asarray(index),
            "adaptive_sectorization_enabled": np.asarray(sectorization is not None),
        }
        if sectorization is not None:
            payload.update(
                {
                    "sector_face_labels": np.asarray(sectorization.face_labels, dtype=np.int64),
                    "sector_active_face_mask": np.asarray(sectorization.active_face_masks[:, index], dtype=bool),
                    "sector_guard_face_mask": np.asarray(sectorization.guard_face_mask, dtype=bool),
                    "sector_max_inactive_density_relative": np.asarray(
                        sectorization.max_inactive_density_relative[index]
                    ),
                    "sector_local_dof_count": np.asarray(sectorization.local_dof_counts[index], dtype=np.int64),
                }
            )
        np.savez_compressed(path, **payload)
        outputs.append(path)
    return tuple(outputs)


def _cohort_design_scenario_weights(
    scenarios: Sequence[SurfaceShimScenario],
    options: CohortChannelDesignOptions,
) -> np.ndarray:
    scenario_ids = tuple(item.scenario.scenario_id for item in scenarios)
    if options.scenario_weights is not None:
        supplied = set(options.scenario_weights)
        expected = set(scenario_ids)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"Explicit scenario weights have missing IDs {missing} and extra IDs {extra}.")
        weights = np.asarray([options.scenario_weights[scenario_id] for scenario_id in scenario_ids], dtype=float)
    elif options.scenario_weighting == "uniform_scenario":
        weights = np.ones(len(scenarios), dtype=float)
    else:
        subject_counts: dict[str, int] = {}
        for item in scenarios:
            subject = item.scenario.subject_id
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        weights = np.asarray(
            [1.0 / subject_counts[item.scenario.subject_id] for item in scenarios],
            dtype=float,
        )
    weights /= np.sum(weights)
    return weights


def refine_cohort_channel_basis(
    surface_scenarios: Sequence[SurfaceShimScenario],
    initial_basis_per_A: np.ndarray,
    limits: ChannelCurrentLimits,
    config: CoilGenConfig,
    *,
    options: CohortFitOptions | None = None,
    auxiliary_actuators_by_scenario: Mapping[str, Sequence[LinearShimActuator]] | None = None,
    scenario_weights: Sequence[float] | None = None,
    maximum_iterations: int = 2,
    resistance_weight: float = 1.0e-3,
    relative_tolerance: float = 1.0e-4,
) -> tuple[np.ndarray, CohortFitResult, list[Mapping[str, Any]]]:
    """Bound-aware alternating refinement of a physically scaled channel basis.

    Current fitting is convex for fixed channels.  The channel update is convex
    for fixed currents and constrains every per-ampere channel so its sheet
    current at the maximum hardware current respects CoilGen's configured limit.
    The overall bilinear problem is not globally convex, so the returned history
    explicitly records each alternating step.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("CVXPY is required for cohort channel refinement.") from exc
    scenarios = tuple(surface_scenarios)
    if not scenarios:
        raise ValueError("At least one surface scenario is required.")
    basis = np.asarray(initial_basis_per_A, dtype=float).copy()
    dof_count = scenarios[0].problem.reduced_sensitivity.shape[1]
    if basis.shape != (dof_count, limits.channel_count):
        raise ValueError("initial_basis_per_A has inconsistent dimensions.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    dataset = PopulationShimDataset(tuple(item.scenario for item in scenarios))
    if scenario_weights is None:
        normalized_scenario_weights = np.full(len(scenarios), 1.0 / len(scenarios))
    else:
        normalized_scenario_weights = np.asarray(scenario_weights, dtype=float).reshape(-1)
        if (
            len(normalized_scenario_weights) != len(scenarios)
            or np.any(~np.isfinite(normalized_scenario_weights))
            or np.any(normalized_scenario_weights <= 0.0)
        ):
            raise ValueError("scenario_weights must be finite, positive, and match the cohort scenarios.")
        normalized_scenario_weights /= np.sum(normalized_scenario_weights)
    fit_options = options or CohortFitOptions()
    auxiliary = auxiliary_actuators_by_scenario or {}
    history: list[Mapping[str, Any]] = []
    previous_objective = np.inf
    for iteration in range(maximum_iterations):
        fields = _channel_fields_for_surface_scenarios(scenarios, basis)
        start, span = 100 * iteration / maximum_iterations, 100 / maximum_iterations
        with progress_scope(start, start + 0.2 * span):
            fit = fit_cohort_currents(
                dataset,
                fields,
                limits,
                options=fit_options,
                auxiliary_actuators_by_scenario=auxiliary,
            )
        currents = fit.current_matrix_A
        variable = cp.Variable(basis.shape, name="cohort_channel_basis_per_A")
        baseline_scale = max(float(np.median(fit.rms_before_T)), 1.0e-15)
        residual_terms: list[Any] = []
        for scenario_weight, item, fit_result, scenario_currents in zip(
            normalized_scenario_weights,
            scenarios,
            fit.scenario_results,
            currents,
        ):
            matrix = np.asarray(item.problem.unweighted_reduced_sensitivity, dtype=float)
            disturbance = np.asarray(item.scenario.disturbance_T, dtype=float) + fit_result.auxiliary_field_T
            weights = _scenario_weights(item.scenario)
            prediction = matrix @ variable @ scenario_currents + disturbance
            if fit_options.center_frequency:
                prediction = prediction - (weights @ prediction) / float(np.sum(weights))
            residual_terms.append(
                float(scenario_weight) * cp.sum_squares(cp.multiply(np.sqrt(weights / np.sum(weights)), prediction))
            )
        residual_objective = sum(residual_terms) / baseline_scale**2

        resistance = np.asarray(scenarios[0].problem.reduced_resistance, dtype=float)
        resistance_sqrt = _positive_semidefinite_square_root(resistance)
        mean_current_square = np.maximum(np.mean(currents**2, axis=0), 1.0e-12)
        initial_energy = float(
            np.sum(mean_current_square * np.einsum("pk,pq,qk->k", basis, resistance, basis, optimize=True))
        )
        energy_expression = cp.sum_squares(resistance_sqrt @ variable @ np.diag(np.sqrt(mean_current_square))) / max(
            initial_energy, 1.0e-30
        )
        objective = cp.Minimize(residual_objective + float(resistance_weight) * energy_expression)
        constraints: list[Any] = []
        density_limit = engineering_surface_current_density_limit_A_per_m(config)
        density_constraint_count = 0
        if density_limit > 0.0:
            per_amp_limit = density_limit / float(np.max(limits.absolute_limits_A))
            for channel_index in range(limits.channel_count):
                channel_constraints, count = _cvxpy_surface_current_density_constraints(
                    cp,
                    scenarios[0].problem,
                    variable[:, channel_index],
                    per_amp_limit,
                )
                constraints.extend(channel_constraints)
                density_constraint_count += count
        cvx_problem = cp.Problem(objective, constraints)
        with progress_scope(start + 0.2 * span, start + 0.8 * span):
            ProgressReporter(label="Cohort refinement").report(
                None, f"Solving alternating refinement step {iteration + 1}/{maximum_iterations}"
            )
            status, solver = _solve_cvxpy(cvx_problem, cp, config)
        if variable.value is None:
            raise RuntimeError("Cohort channel refinement returned no channel basis.")
        candidate = np.asarray(variable.value, dtype=float)
        candidate_fields = _channel_fields_for_surface_scenarios(scenarios, candidate)
        with progress_scope(start + 0.8 * span, start + span):
            candidate_fit = fit_cohort_currents(
                dataset,
                candidate_fields,
                limits,
                options=fit_options,
                auxiliary_actuators_by_scenario=auxiliary,
            )
        score = float(normalized_scenario_weights @ (candidate_fit.rms_after_T**2))
        relative_improvement = (
            (previous_objective - score) / max(previous_objective, 1.0e-30)
            if np.isfinite(previous_objective)
            else np.inf
        )
        history.append(
            {
                "iteration": iteration + 1,
                "solver": solver,
                "status": status,
                "mean_squared_residual_T2": score,
                "relative_improvement": float(relative_improvement),
                "saturated_scenario_fraction": candidate_fit.saturation_fraction,
                "density_constraint_count": density_constraint_count,
            }
        )
        basis = candidate
        fit = candidate_fit
        if np.isfinite(relative_improvement) and 0.0 <= relative_improvement < relative_tolerance:
            break
        previous_objective = score
    return _readonly_array(basis), fit, history


def _channel_fields_for_surface_scenarios(
    scenarios: Sequence[SurfaceShimScenario],
    basis_per_A: np.ndarray,
) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for item in scenarios:
        matrix = np.asarray(item.problem.unweighted_reduced_sensitivity, dtype=float)
        fields[item.scenario.scenario_id] = matrix @ basis_per_A
    return fields


def _problem_with_scalar_target(
    problem: StreamFunctionOptimizationProblem,
    target_T: np.ndarray,
    config: CoilGenConfig,
) -> StreamFunctionOptimizationProblem:
    target = np.asarray(target_T, dtype=float).reshape(-1)
    unweighted = problem.unweighted_target_z
    if unweighted is None or len(target) != len(unweighted):
        raise ValueError("Cohort target does not match the base problem target rows.")
    if problem.target_suppressed is not None and np.any(problem.target_suppressed):
        raise NotImplementedError("Cohort reference solves currently require DSV-only target rows.")
    weights = np.asarray(problem.target_weights, dtype=float).reshape(-1)
    if len(weights) != len(target):
        raise ValueError("Base problem weights do not match the cohort target.")
    direction = _scalar_component_direction(config)
    target_b = target[:, None] * direction[None, :]
    return replace(
        problem,
        target_z=target * np.sqrt(np.maximum(weights, 0.0)),
        unweighted_target_z=target,
        target_b=target_b,
    )


def _scalar_component_direction(config: CoilGenConfig) -> np.ndarray:
    component = str(getattr(config, "target_field_component", "Bz")).strip().lower()
    if component in {"bx", "x", "b_x"}:
        return np.asarray((1.0, 0.0, 0.0))
    if component in {"by", "y", "b_y"}:
        return np.asarray((0.0, 1.0, 0.0))
    if component in {"bz", "z", "b_z"}:
        return np.asarray((0.0, 0.0, 1.0))
    if component in {"bdotb0", "b0", "b_dot_b0", "b_parallel", "parallel_b0"}:
        return normalized_b0_direction(getattr(config, "b_0_direction", (0.0, 0.0, 1.0)))
    raise NotImplementedError("Cohort reference solves require a scalar target-field component.")


def _bounded_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    # lsq_linear requires strict bounds. Eliminate commands fixed by the caller,
    # including disabled channels, and retain their field in the residual target.
    fixed = lower == upper
    solution = np.zeros(matrix.shape[1], dtype=float)
    solution[fixed] = lower[fixed]
    if np.all(fixed):
        return solution, "fixed_commands", "All commands are fixed by their bounds."
    free_matrix = matrix[:, ~fixed]
    free_target = target - matrix[:, fixed] @ solution[fixed]
    scaled_matrix, scaled_target = _normalized_current_fit_system(free_matrix, free_target)
    try:
        from scipy.optimize import lsq_linear
    except ImportError as exc:
        raise RuntimeError("SciPy is required for bounded cohort-current fitting.") from exc
    result = lsq_linear(
        scaled_matrix,
        scaled_target,
        bounds=(lower[~fixed], upper[~fixed]),
        method="trf",
        lsmr_tol="auto",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Bounded cohort-current fitting failed: {result.message}")
    solution[~fixed] = np.asarray(result.x, dtype=float)
    return solution, "scipy_lsq_linear", str(result.message)


def _normalized_current_fit_system(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the least-squares optimum while removing tesla-scale tolerances."""

    scale = max(float(np.max(np.abs(matrix), initial=0.0)), float(np.max(np.abs(target), initial=0.0)))
    if scale == 0.0:
        return matrix, target
    return matrix / scale, target / scale


def _power_constrained_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    limits: ChannelCurrentLimits,
) -> tuple[np.ndarray, str, str]:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("CVXPY is required when total shim power is constrained.") from exc
    variable = cp.Variable(matrix.shape[1], name="cohort_channel_commands")
    constraints: list[Any] = []
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    if np.any(finite_lower):
        constraints.append(variable[finite_lower] >= lower[finite_lower])
    if np.any(finite_upper):
        constraints.append(variable[finite_upper] <= upper[finite_upper])
    currents = variable[: limits.channel_count]
    constraints.append(
        cp.sum(cp.multiply(np.asarray(limits.resistance_ohm, dtype=float), cp.square(currents)))
        <= float(limits.total_power_limit_W)
    )
    scaled_matrix, scaled_target = _normalized_current_fit_system(matrix, target)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(scaled_matrix @ variable - scaled_target)), constraints)
    errors: list[str] = []
    for solver in ("CLARABEL", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except DesignCancelled:
            raise
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and variable.value is not None:
            return np.asarray(variable.value, dtype=float), solver, str(problem.status)
        errors.append(f"{solver}: {problem.status}")
    raise RuntimeError("Power-constrained current fit failed (" + "; ".join(errors) + ").")


def _solve_cvxpy(
    problem: Any,
    cp: Any,
    config: CoilGenConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str]:
    errors: list[str] = []
    solver_verbose = bool(
        getattr(
            config,
            "constrained_optimization_verbose",
            getattr(config, "progress_flag", False),
        )
    )
    for solver in _cvxpy_conic_solver_order(cp, config):
        _emit_progress(progress_callback, f"Adaptive CVXPY: canonicalizing for {solver}...")
        compilation_started = perf_counter()
        try:
            solver_data = problem.get_problem_data(solver)[0]
        except DesignCancelled:
            raise
        except Exception as exc:
            errors.append(f"{solver} canonicalization: {exc}")
            _emit_progress(
                progress_callback,
                f"Adaptive CVXPY: {solver} canonicalization failed after "
                f"{perf_counter() - compilation_started:.2f} s: {exc}",
            )
            continue
        compilation_seconds = perf_counter() - compilation_started
        matrix_summaries: list[str] = []
        for name in ("P", "A"):
            matrix = solver_data.get(name)
            if matrix is not None and hasattr(matrix, "shape"):
                stored_nnz = getattr(matrix, "nnz", None)
                nnz = int(stored_nnz if stored_nnz is not None else np.count_nonzero(matrix))
                matrix_summaries.append(f"{name}={matrix.shape}, nnz={nnz}")
        summary = "; ".join(matrix_summaries) or "solver matrices unavailable"
        requested_threads = int(getattr(config, "constrained_optimization_threads", 0) or 0)
        _emit_progress(
            progress_callback,
            f"Adaptive CVXPY: {solver} canonicalization complete in {compilation_seconds:.2f} s "
            f"({summary}); starting numerical solve with requested threads={requested_threads}, "
            f"verbose={solver_verbose}...",
        )
        solve_started = perf_counter()
        try:
            problem.solve(
                solver=solver,
                verbose=solver_verbose,
                **_cvxpy_solver_options(solver, config),
            )
        except DesignCancelled:
            raise
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            _emit_progress(
                progress_callback,
                f"Adaptive CVXPY: {solver} failed after {perf_counter() - solve_started:.2f} s: {exc}",
            )
            continue
        solve_seconds = perf_counter() - solve_started
        _emit_progress(
            progress_callback,
            f"Adaptive CVXPY: {solver} returned {problem.status} in {solve_seconds:.2f} s.",
        )
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return str(problem.status), str(solver)
        errors.append(f"{solver}: {problem.status}")
    raise RuntimeError("Cohort channel update failed (" + "; ".join(errors) + ").")


def _weighted_centered_system(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    center: bool,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / np.sum(weights)
    transformed_matrix = np.asarray(matrix, dtype=float)
    transformed_target = np.asarray(target, dtype=float)
    if center:
        transformed_matrix = transformed_matrix - normalized @ transformed_matrix
        transformed_target = transformed_target - float(normalized @ transformed_target)
    sqrt_weights = np.sqrt(normalized)
    return transformed_matrix * sqrt_weights[:, None], transformed_target * sqrt_weights


def _weighted_rms(values: np.ndarray, weights: np.ndarray, center: bool) -> float:
    normalized = weights / np.sum(weights)
    field = np.asarray(values, dtype=float)
    if center:
        field = field - float(normalized @ field)
    return float(np.sqrt(normalized @ (field**2)))


def _scenario_quantiles(
    values_by_channel: np.ndarray,
    quantile: float,
    scenario_weights: Sequence[float] | None,
) -> np.ndarray:
    values = np.asarray(values_by_channel, dtype=float)
    if values.ndim != 2:
        raise ValueError("Scenario quantiles require shape (channels, scenarios).")
    if scenario_weights is None:
        return np.quantile(values, quantile, axis=1)
    weights = np.asarray(scenario_weights, dtype=float).reshape(-1)
    if len(weights) != values.shape[1] or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("scenario_weights must be finite, positive, and match the calibration scenarios.")
    weights /= np.sum(weights)
    result = np.empty(values.shape[0], dtype=float)
    for channel_index, channel_values in enumerate(values):
        order = np.argsort(channel_values, kind="stable")
        sorted_values = channel_values[order]
        sorted_weights = weights[order]
        probability = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        result[channel_index] = np.interp(
            quantile,
            probability,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    return result


def _scenario_weights(scenario: ShimScenario) -> np.ndarray:
    if scenario.weights is None:
        return np.ones(scenario.sample_count, dtype=float)
    return np.asarray(scenario.weights, dtype=float)


def _positive_semidefinite_square_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * 1.0e-12, np.finfo(float).eps)
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("Resistance matrix is not positive semidefinite.")
    retained = eigenvalues > tolerance
    return np.sqrt(eigenvalues[retained])[:, None] * eigenvectors[:, retained].T


def _readonly_array(values: Any, dtype: Any = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.flags.writeable:
        array = array.copy()
        array.setflags(write=False)
    return array
