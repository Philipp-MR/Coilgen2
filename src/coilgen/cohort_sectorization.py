"""Adaptive single-layer support discovery for cohort shim channels.

The continuous localization stage uses the physically meaningful sheet-current
density ``K = n x grad_s(psi)`` on every triangle.  It is deliberately only a
support-discovery stage: a final graph partition and exact zero-current
constraints are required before the result is described as sectorized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import combinations
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import CoilGenConfig
from .engineering_limits import engineering_surface_current_density_limit_A_per_m
from .field_components import sensitivity_component_matrix
from .models import Mesh


@dataclass(frozen=True, slots=True)
class AdaptiveSectorizationOptions:
    """Controls automatic discovery and hardening of disjoint channel supports."""

    enabled: bool = False
    discovery_basis_dimension: int = 96
    enforce_density_during_discovery: bool = False
    localization_iterations: int = 4
    overlap_penalty_initial: float = 0.05
    overlap_penalty_growth: float = 2.0
    resistance_weight: float = 1.0e-3
    basis_anchor_weight: float = 2.0e-1
    activity_quantile: float = 0.90
    activity_threshold_fraction: float = 0.75
    ownership_power: float = 4.0
    activity_damping: float = 0.30
    minimum_localization_overlap_reduction: float = 0.005
    maximum_localization_rms_ratio: float = 1.15
    minimum_channel_field_norm_ratio: float = 0.25
    minimum_discovery_field_projection_ratio: float = 0.75
    maximum_localized_overlap_fraction: float = 0.15
    maximum_hardening_rms_ratio: float = 1.25
    maximum_normalized_field_condition_number: float = 10.0
    partition_smoothness: float = 2.0
    partition_compactness: float = 1.0
    minimum_sector_area_fraction: float = 0.10
    maximum_sector_area_fraction: float = 0.45
    clearance_m: float = 0.0
    hard_support_relative_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.localization_iterations < 0:
            raise ValueError("localization_iterations must be non-negative.")
        if self.discovery_basis_dimension < 1:
            raise ValueError("discovery_basis_dimension must be positive.")
        for name in (
            "overlap_penalty_initial",
            "resistance_weight",
            "basis_anchor_weight",
            "ownership_power",
            "partition_smoothness",
            "partition_compactness",
            "clearance_m",
            "hard_support_relative_tolerance",
            "minimum_localization_overlap_reduction",
            "maximum_localization_rms_ratio",
            "minimum_channel_field_norm_ratio",
            "minimum_discovery_field_projection_ratio",
            "maximum_localized_overlap_fraction",
            "maximum_hardening_rms_ratio",
            "maximum_normalized_field_condition_number",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not np.isfinite(self.overlap_penalty_growth) or self.overlap_penalty_growth < 1.0:
            raise ValueError("overlap_penalty_growth must be finite and at least one.")
        if not 0.0 < self.activity_quantile <= 1.0:
            raise ValueError("activity_quantile must lie in (0, 1].")
        if not 0.0 < self.activity_threshold_fraction <= 1.0:
            raise ValueError("activity_threshold_fraction must lie in (0, 1].")
        if self.ownership_power < 1.0:
            raise ValueError("ownership_power must be at least one.")
        if not 0.0 <= self.activity_damping < 1.0:
            raise ValueError("activity_damping must lie in [0, 1).")
        if not 0.0 <= self.minimum_localization_overlap_reduction < 1.0:
            raise ValueError("minimum_localization_overlap_reduction must lie in [0, 1).")
        if self.maximum_localization_rms_ratio < 1.0:
            raise ValueError("maximum_localization_rms_ratio must be at least one.")
        if not 0.0 < self.minimum_channel_field_norm_ratio <= 1.0:
            raise ValueError("minimum_channel_field_norm_ratio must lie in (0, 1].")
        if not 0.0 < self.minimum_discovery_field_projection_ratio <= 1.0:
            raise ValueError(
                "minimum_discovery_field_projection_ratio must lie in (0, 1]."
            )
        if not 0.0 < self.maximum_localized_overlap_fraction <= 1.0:
            raise ValueError("maximum_localized_overlap_fraction must lie in (0, 1].")
        if self.maximum_hardening_rms_ratio < 1.0:
            raise ValueError("maximum_hardening_rms_ratio must be at least one.")
        if self.maximum_normalized_field_condition_number < 1.0:
            raise ValueError(
                "maximum_normalized_field_condition_number must be at least one."
            )
        if not 0.0 < self.minimum_sector_area_fraction < 1.0:
            raise ValueError("minimum_sector_area_fraction must lie in (0, 1).")
        if not self.minimum_sector_area_fraction < self.maximum_sector_area_fraction < 1.0:
            raise ValueError(
                "maximum_sector_area_fraction must exceed the minimum and remain below one."
            )
        if self.enabled and self.hard_support_relative_tolerance <= 0.0:
            raise ValueError("Enabled sectorization requires a positive hard-support tolerance.")


@dataclass(frozen=True, slots=True)
class AdaptiveSectorizationResult:
    """Audit data for an automatically discovered hard channel partition."""

    face_labels: np.ndarray
    guard_face_mask: np.ndarray
    active_face_masks: np.ndarray
    localized_occupancy: np.ndarray
    final_current_density_A_per_m_per_A: np.ndarray
    initial_overlap_fraction: float
    localized_overlap_fraction: float
    final_continuous_overlap_fraction: float
    max_inactive_density_relative: np.ndarray
    sector_area_fraction: np.ndarray
    active_area_fraction: np.ndarray
    local_dof_counts: np.ndarray
    terminal_face_indices: np.ndarray
    channel_field_rms_T_per_A: np.ndarray
    channel_field_norm_ratio: np.ndarray
    intersector_boundary_length_m: float
    performance_rms_ratio: float
    normalized_field_condition_number: float
    maximum_normalized_field_correlation: float
    history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    single_layer_support_enforced: bool = True

    def __post_init__(self) -> None:
        for name, dtype in (
            ("face_labels", np.int64),
            ("guard_face_mask", bool),
            ("active_face_masks", bool),
            ("localized_occupancy", float),
            ("final_current_density_A_per_m_per_A", float),
            ("max_inactive_density_relative", float),
            ("sector_area_fraction", float),
            ("active_area_fraction", float),
            ("local_dof_counts", np.int64),
            ("terminal_face_indices", np.int64),
            ("channel_field_rms_T_per_A", float),
            ("channel_field_norm_ratio", float),
        ):
            values = np.asarray(getattr(self, name), dtype=dtype).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        labels = self.face_labels
        active = self.active_face_masks
        face_count = len(labels)
        if labels.ndim != 1 or self.guard_face_mask.shape != (face_count,):
            raise ValueError("Sector face labels and guard mask must contain one value per face.")
        if active.ndim != 2 or active.shape[0] != face_count or active.shape[1] < 1:
            raise ValueError("active_face_masks must have shape (faces, channels).")
        channel_count = active.shape[1]
        if self.localized_occupancy.shape != active.shape:
            raise ValueError("localized_occupancy must match the active face-mask dimensions.")
        if self.final_current_density_A_per_m_per_A.shape != active.shape:
            raise ValueError("Final current-density data must match the active face-mask dimensions.")
        for name in (
            "max_inactive_density_relative",
            "sector_area_fraction",
            "active_area_fraction",
            "local_dof_counts",
            "terminal_face_indices",
            "channel_field_rms_T_per_A",
            "channel_field_norm_ratio",
        ):
            if np.asarray(getattr(self, name)).shape != (channel_count,):
                raise ValueError(f"{name} must contain one value per channel.")
        if np.any(self.local_dof_counts < 1):
            raise ValueError("local_dof_counts must be positive for every channel.")
        if np.any(self.terminal_face_indices < 0) or np.any(
            self.terminal_face_indices >= face_count
        ):
            raise ValueError("terminal_face_indices must identify valid support-mesh faces.")
        if len(np.unique(self.terminal_face_indices)) != channel_count:
            raise ValueError("Every sector channel requires one unique boundary terminal face.")
        if np.any(labels < 0) or np.any(labels >= channel_count):
            raise ValueError("Sector face labels must identify a valid channel.")
        expected_active = np.column_stack(
            [(labels == channel) & ~self.guard_face_mask for channel in range(channel_count)]
        )
        if not np.array_equal(active, expected_active):
            raise ValueError("Active face masks must equal their labeled sector with guards removed.")
        if np.any(np.sum(active, axis=1) > 1):
            raise ValueError("Adaptive sector active-face masks must be disjoint.")
        for name in (
            "localized_occupancy",
            "final_current_density_A_per_m_per_A",
            "max_inactive_density_relative",
            "sector_area_fraction",
            "active_area_fraction",
            "channel_field_rms_T_per_A",
            "channel_field_norm_ratio",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite non-negative values.")
        for name in (
            "initial_overlap_fraction",
            "localized_overlap_fraction",
            "final_continuous_overlap_fraction",
            "intersector_boundary_length_m",
            "performance_rms_ratio",
            "normalized_field_condition_number",
            "maximum_normalized_field_correlation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        object.__setattr__(self, "history", tuple(dict(item) for item in self.history))

    @property
    def channel_count(self) -> int:
        return int(self.active_face_masks.shape[1])

    @property
    def guard_area_fraction(self) -> float:
        return float(max(0.0, 1.0 - np.sum(self.active_area_fraction)))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "single_layer_support_enforced": bool(self.single_layer_support_enforced),
            "channel_count": self.channel_count,
            "initial_overlap_fraction": float(self.initial_overlap_fraction),
            "localized_overlap_fraction": float(self.localized_overlap_fraction),
            "final_continuous_overlap_fraction": float(self.final_continuous_overlap_fraction),
            "max_inactive_density_relative": self.max_inactive_density_relative.tolist(),
            "sector_area_fraction": self.sector_area_fraction.tolist(),
            "active_area_fraction": self.active_area_fraction.tolist(),
            "local_dof_counts": self.local_dof_counts.tolist(),
            "terminal_face_indices": self.terminal_face_indices.tolist(),
            "channel_field_rms_T_per_A": self.channel_field_rms_T_per_A.tolist(),
            "channel_field_norm_ratio": self.channel_field_norm_ratio.tolist(),
            "guard_area_fraction": self.guard_area_fraction,
            "intersector_boundary_length_m": float(self.intersector_boundary_length_m),
            "performance_rms_ratio": float(self.performance_rms_ratio),
            "normalized_field_condition_number": float(
                self.normalized_field_condition_number
            ),
            "maximum_normalized_field_correlation": float(
                self.maximum_normalized_field_correlation
            ),
            "history": [dict(item) for item in self.history],
        }


def adaptively_sectorize_cohort_channels(
    surface_scenarios: Sequence[Any],
    initial_basis_per_A: np.ndarray,
    limits: Any,
    config: CoilGenConfig,
    options: AdaptiveSectorizationOptions,
    *,
    fit_options: Any,
    auxiliary_actuators_by_scenario: Mapping[str, Sequence[Any]] | None = None,
    scenario_weights: Sequence[float] | None = None,
    discovery_basis_reduced: np.ndarray | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, Any, AdaptiveSectorizationResult]:
    """Localize, partition, and hard-constrain a cohort channel basis.

    For fixed cohort currents and fixed overlap weights each stream-function
    update is convex.  The alternating localization and the discrete partition
    are not globally convex, so all iterations and the final support residual
    are returned for audit.
    """

    if not options.enabled:
        raise ValueError("adaptively_sectorize_cohort_channels requires options.enabled=True.")
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("CVXPY is required for adaptive channel sectorization.") from exc

    # Imported lazily to avoid a module cycle: cohort_shim publicly re-exports
    # this module's option/result classes.
    from .cohort_shim import (
        CohortFitOptions,
        PopulationShimDataset,
        _channel_fields_for_surface_scenarios,
        fit_cohort_currents,
    )

    scenarios = tuple(surface_scenarios)
    if not scenarios:
        raise ValueError("Adaptive sectorization requires at least one surface scenario.")
    problem = scenarios[0].problem
    basis = np.asarray(initial_basis_per_A, dtype=float).copy()
    dof_count = int(problem.reduced_sensitivity.shape[1])
    channel_count = int(limits.channel_count)
    if basis.shape != (dof_count, channel_count):
        raise ValueError("initial_basis_per_A has inconsistent dimensions.")
    _validate_common_sectorization_surface(scenarios)

    density_operators = _reduced_face_current_density_operators(problem)
    discovery_parameterization = _build_discovery_parameterization(
        basis,
        discovery_basis_reduced,
        maximum_dimension=options.discovery_basis_dimension,
        problem=problem,
        center_frequency=fit_options.center_frequency if fit_options is not None else True,
    )
    face_areas = _mesh_face_areas(problem.combined_mesh)
    if len(face_areas) < channel_count:
        raise ValueError("The support mesh has fewer faces than requested sectors.")
    adjacency = _face_adjacency(problem.combined_mesh)
    if progress_callback is not None:
        progress_callback(
            f"Adaptive sectorization setup: {len(face_areas)} mesh faces, {channel_count} channels, "
            f"{options.localization_iterations} localization iterations, "
            f"{discovery_parameterization.shape[1]}/{dof_count} discovery DOFs per channel; "
            "discovery density cones "
            f"{'enabled' if options.enforce_density_during_discovery else 'deferred to hardening'}."
        )
    normalized_scenario_weights = _normalized_weights(scenario_weights, len(scenarios))
    fit_options = fit_options or CohortFitOptions()
    dataset = PopulationShimDataset(tuple(item.scenario for item in scenarios))
    auxiliary = auxiliary_actuators_by_scenario or {}
    fields = _channel_fields_for_surface_scenarios(scenarios, basis)
    fit = fit_cohort_currents(
        dataset,
        fields,
        limits,
        options=fit_options,
        auxiliary_actuators_by_scenario=auxiliary,
    )
    initial_fit = fit
    anchor_basis = basis.copy()
    anchor_field_rms = _channel_field_rms(
        problem,
        anchor_basis,
        center_frequency=fit_options.center_frequency,
    )
    density = _face_current_density_magnitudes(density_operators, basis)
    occupancy, thresholds = _activity_occupancy(
        density,
        face_areas,
        np.asarray(limits.absolute_limits_A, dtype=float),
        options,
    )
    initial_overlap = _continuous_overlap_fraction(occupancy, face_areas)
    if progress_callback is not None:
        progress_callback(
            f"Adaptive sectorization initial activity overlap: {initial_overlap:.6f}; "
            f"median fitted RMS {float(np.median(fit.rms_after_T)):.6g} T."
        )
    history: list[Mapping[str, Any]] = [
        {
            "stage": "initial",
            "iteration": 0,
            "overlap_fraction": initial_overlap,
            "median_rms_after_T": float(np.median(fit.rms_after_T)),
        }
    ]

    overlap_penalty = float(options.overlap_penalty_initial)
    damped_occupancy = occupancy.copy()
    for iteration in range(options.localization_iterations):
        if progress_callback is not None:
            progress_callback(
                f"Adaptive localization {iteration + 1}/{options.localization_iterations}: "
                f"solving with overlap penalty {overlap_penalty:.6g}..."
            )
        overlap_weights = np.sum(damped_occupancy, axis=1, keepdims=True) - damped_occupancy
        candidate, solver, status, constraint_count = _solve_basis_update(
            cp,
            scenarios,
            fit,
            basis,
            anchor_basis,
            limits,
            config,
            fit_options,
            normalized_scenario_weights,
            density_operators,
            face_areas,
            resistance_weight=options.resistance_weight,
            basis_anchor_weight=options.basis_anchor_weight,
            overlap_weights=overlap_weights,
            overlap_penalty=overlap_penalty,
            active_face_masks=None,
            basis_parameterization=discovery_parameterization,
            enforce_density_constraints=options.enforce_density_during_discovery,
            minimum_field_projection_ratio=(
                options.minimum_discovery_field_projection_ratio
            ),
            progress_callback=progress_callback,
        )
        candidate_fields = _channel_fields_for_surface_scenarios(scenarios, candidate)
        candidate_fit = fit_cohort_currents(
            dataset,
            candidate_fields,
            limits,
            options=fit_options,
            auxiliary_actuators_by_scenario=auxiliary,
        )
        candidate_density = _face_current_density_magnitudes(density_operators, candidate)
        candidate_field_rms = _channel_field_rms(
            problem,
            candidate,
            center_frequency=fit_options.center_frequency,
        )
        field_norm_ratio = candidate_field_rms / np.maximum(anchor_field_rms, 1.0e-30)
        rejection_reasons: list[str] = []
        if float(np.min(field_norm_ratio)) < options.minimum_channel_field_norm_ratio:
            rejection_reasons.append(
                "minimum channel field-norm ratio "
                f"{float(np.min(field_norm_ratio)):.3f} < "
                f"{options.minimum_channel_field_norm_ratio:.3f}"
            )
        try:
            candidate_occupancy, candidate_thresholds = _activity_occupancy(
                candidate_density,
                face_areas,
                np.asarray(limits.absolute_limits_A, dtype=float),
                options,
            )
            overlap = _continuous_overlap_fraction(candidate_occupancy, face_areas)
        except RuntimeError as exc:
            candidate_occupancy = occupancy
            candidate_thresholds = thresholds
            overlap = float("inf")
            rejection_reasons.append(str(exc))
        previous_overlap = _continuous_overlap_fraction(occupancy, face_areas)
        required_overlap = previous_overlap * (
            1.0 - options.minimum_localization_overlap_reduction
        )
        if not np.isfinite(overlap) or overlap > required_overlap:
            rejection_reasons.append(
                f"overlap {overlap:.6f} did not improve below {required_overlap:.6f}"
            )
        localization_rms_ratio = float(
            np.median(candidate_fit.rms_after_T)
            / max(float(np.median(initial_fit.rms_after_T)), 1.0e-30)
        )
        if localization_rms_ratio > options.maximum_localization_rms_ratio:
            rejection_reasons.append(
                f"RMS ratio {localization_rms_ratio:.3f} > "
                f"{options.maximum_localization_rms_ratio:.3f}"
            )
        accepted = not rejection_reasons
        history.append(
            {
                "stage": "localization",
                "iteration": iteration + 1,
                "solver": solver,
                "status": status,
                "overlap_penalty": overlap_penalty,
                "overlap_fraction": overlap,
                "median_rms_after_T": float(np.median(candidate_fit.rms_after_T)),
                "field_norm_ratio": field_norm_ratio.tolist(),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
                "hard_constraint_scalar_count": constraint_count,
            }
        )
        if progress_callback is not None:
            disposition = "accepted" if accepted else "rejected: " + "; ".join(rejection_reasons)
            progress_callback(
                f"Adaptive localization {iteration + 1}/{options.localization_iterations} complete; "
                f"{disposition} via {solver}: overlap {overlap:.6f}, median RMS "
                f"{float(np.median(candidate_fit.rms_after_T)):.6g} T."
            )
        if accepted:
            damped_occupancy = (
                options.activity_damping * damped_occupancy
                + (1.0 - options.activity_damping) * candidate_occupancy
            )
            basis = candidate
            fit = candidate_fit
            density = candidate_density
            occupancy = candidate_occupancy
            thresholds = candidate_thresholds
        overlap_penalty *= options.overlap_penalty_growth

    localized_overlap = _continuous_overlap_fraction(occupancy, face_areas)
    if localized_overlap > options.maximum_localized_overlap_fraction:
        raise RuntimeError(
            "Adaptive localization did not produce sufficiently separable channel activity: "
            f"overlap {localized_overlap:.6f} exceeds "
            f"{options.maximum_localized_overlap_fraction:.6f}."
        )
    if progress_callback is not None:
        progress_callback("Adaptive partition: assigning connected face sectors from localized activity...")
    ownership = _competitive_ownership(occupancy)
    labels, seeds, partition_unary = _automatic_face_partition(
        problem.combined_mesh,
        ownership,
        face_areas,
        adjacency,
        options,
    )
    labels, guard = _harden_partition_with_clearance(
        labels,
        seeds,
        partition_unary,
        face_areas,
        adjacency,
        options,
    )
    active_face_masks = np.column_stack(
        [(labels == channel) & ~guard for channel in range(channel_count)]
    )
    _validate_hardened_partition(
        labels,
        seeds,
        guard,
        active_face_masks,
        face_areas,
        adjacency,
        options,
    )
    if progress_callback is not None:
        active_counts = np.count_nonzero(active_face_masks, axis=0).tolist()
        progress_callback(
            f"Adaptive partition complete: active faces per channel {active_counts}; "
            f"guard faces {int(np.count_nonzero(guard))}."
        )

    if progress_callback is not None:
        progress_callback("Adaptive hardening: enforcing zero sheet current outside every sector...")
    hardened, solver, status, constraint_count, local_dof_counts = _solve_sector_local_basis_update(
        cp,
        scenarios,
        fit,
        basis,
        anchor_basis,
        limits,
        config,
        fit_options,
        normalized_scenario_weights,
        active_face_masks,
        resistance_weight=options.resistance_weight,
        basis_anchor_weight=options.basis_anchor_weight,
        progress_callback=progress_callback,
    )
    hardened_fields = _channel_fields_for_surface_scenarios(scenarios, hardened)
    hardened_fit = fit_cohort_currents(
        dataset,
        hardened_fields,
        limits,
        options=fit_options,
        auxiliary_actuators_by_scenario=auxiliary,
    )
    final_density = _face_current_density_magnitudes(density_operators, hardened)
    final_occupancy, _final_thresholds = _activity_occupancy(
        final_density,
        face_areas,
        np.asarray(limits.absolute_limits_A, dtype=float),
        options,
        thresholds=thresholds,
    )
    inactive_relative = np.zeros(channel_count, dtype=float)
    for channel in range(channel_count):
        maximum = max(float(np.max(final_density[:, channel])), 1.0e-30)
        inactive = ~active_face_masks[:, channel]
        inactive_relative[channel] = (
            float(np.max(final_density[inactive, channel])) / maximum if np.any(inactive) else 0.0
        )
    maximum_inactive = float(np.max(inactive_relative))
    if maximum_inactive > options.hard_support_relative_tolerance:
        raise RuntimeError(
            "Adaptive sectorization did not satisfy the hard zero-current support constraint: "
            f"maximum relative inactive density {maximum_inactive:.3e} exceeds "
            f"{options.hard_support_relative_tolerance:.3e}."
        )

    total_area = max(float(np.sum(face_areas)), 1.0e-30)
    sector_area_fraction = np.asarray(
        [np.sum(face_areas[labels == channel]) / total_area for channel in range(channel_count)]
    )
    active_area_fraction = np.asarray(
        [np.sum(face_areas[active_face_masks[:, channel]]) / total_area for channel in range(channel_count)]
    )
    boundary_length = _intersector_boundary_length(labels, adjacency)
    performance_ratio = float(
        np.median(hardened_fit.rms_after_T)
        / max(float(np.median(initial_fit.rms_after_T)), 1.0e-30)
    )
    hardened_field_rms, normalized_field_condition, maximum_field_correlation = (
        _normalized_channel_field_diagnostics(
            problem,
            hardened,
            center_frequency=fit_options.center_frequency,
        )
    )
    hardened_field_norm_ratio = hardened_field_rms / np.maximum(anchor_field_rms, 1.0e-30)
    quality_failures: list[str] = []
    if performance_ratio > options.maximum_hardening_rms_ratio:
        quality_failures.append(
            f"median RMS ratio {performance_ratio:.3f} exceeds "
            f"{options.maximum_hardening_rms_ratio:.3f}"
        )
    minimum_hardened_field_ratio = float(np.min(hardened_field_norm_ratio))
    if minimum_hardened_field_ratio < options.minimum_channel_field_norm_ratio:
        quality_failures.append(
            f"minimum channel field-norm ratio {minimum_hardened_field_ratio:.3f} is below "
            f"{options.minimum_channel_field_norm_ratio:.3f}"
        )
    if normalized_field_condition > options.maximum_normalized_field_condition_number:
        quality_failures.append(
            f"normalized field condition number {normalized_field_condition:.3f} exceeds "
            f"{options.maximum_normalized_field_condition_number:.3f}"
        )
    if quality_failures:
        raise RuntimeError(
            "Adaptive sectorization failed final physical quality gates: "
            + "; ".join(quality_failures)
            + "."
        )
    final_overlap = _continuous_overlap_fraction(final_occupancy, face_areas)
    history.append(
        {
            "stage": "hardening",
            "iteration": 1,
            "solver": solver,
            "status": status,
            "overlap_fraction": final_overlap,
            "median_rms_after_T": float(np.median(hardened_fit.rms_after_T)),
            "performance_rms_ratio": performance_ratio,
            "channel_field_rms_T_per_A": hardened_field_rms.tolist(),
            "channel_field_norm_ratio": hardened_field_norm_ratio.tolist(),
            "normalized_field_condition_number": normalized_field_condition,
            "maximum_normalized_field_correlation": maximum_field_correlation,
            "active_face_density_constraint_count": constraint_count,
            "local_dof_counts": local_dof_counts.tolist(),
            "max_inactive_density_relative": maximum_inactive,
        }
    )
    if progress_callback is not None:
        progress_callback(
            f"Adaptive hardening complete via {solver} ({status}): final overlap {final_overlap:.3e}, "
            f"maximum inactive-current ratio {maximum_inactive:.3e}, performance RMS ratio "
            f"{performance_ratio:.4f}, normalized field condition "
            f"{normalized_field_condition:.3f}."
        )
    result = AdaptiveSectorizationResult(
        face_labels=labels,
        guard_face_mask=guard,
        active_face_masks=active_face_masks,
        localized_occupancy=occupancy,
        final_current_density_A_per_m_per_A=final_density,
        initial_overlap_fraction=initial_overlap,
        localized_overlap_fraction=localized_overlap,
        final_continuous_overlap_fraction=final_overlap,
        max_inactive_density_relative=inactive_relative,
        sector_area_fraction=sector_area_fraction,
        active_area_fraction=active_area_fraction,
        local_dof_counts=local_dof_counts,
        terminal_face_indices=seeds,
        channel_field_rms_T_per_A=hardened_field_rms,
        channel_field_norm_ratio=hardened_field_norm_ratio,
        intersector_boundary_length_m=boundary_length,
        performance_rms_ratio=performance_ratio,
        normalized_field_condition_number=normalized_field_condition,
        maximum_normalized_field_correlation=maximum_field_correlation,
        history=tuple(history),
    )
    return _readonly(hardened), hardened_fit, result


def _build_discovery_parameterization(
    initial_basis: np.ndarray,
    reference_basis: np.ndarray | None,
    *,
    maximum_dimension: int,
    problem: Any | None = None,
    center_frequency: bool = True,
) -> np.ndarray:
    """Build field-producing and approximate-null spatial discovery modes."""

    initial = np.asarray(initial_basis, dtype=float)
    candidates = [initial]
    if reference_basis is not None:
        references = np.asarray(reference_basis, dtype=float)
        if references.ndim != 2 or references.shape[0] != initial.shape[0]:
            raise ValueError("discovery_basis_reduced must match the reduced stream-function DOFs.")
        candidates.append(references)
    columns = np.column_stack(candidates)
    norms = np.linalg.norm(columns, axis=0)
    retained_columns = np.isfinite(norms) & (norms > 1.0e-15)
    if np.count_nonzero(retained_columns) < initial.shape[1]:
        raise RuntimeError("Adaptive sectorization has too few nonzero discovery basis vectors.")
    normalized = columns[:, retained_columns] / norms[retained_columns][None, :]
    left, singular_values, _right = np.linalg.svd(normalized, full_matrices=False)
    threshold = max(float(singular_values[0]) * 1.0e-11, np.finfo(float).eps)
    rank = int(np.count_nonzero(singular_values > threshold))
    retained_dimension = min(int(maximum_dimension), rank)
    if retained_dimension < initial.shape[1]:
        raise RuntimeError(
            f"Adaptive discovery span has rank {rank}, below the {initial.shape[1]} channels."
        )
    parameterization = left[:, :retained_dimension]
    remaining = int(maximum_dimension) - retained_dimension
    if problem is not None and remaining > 0:
        null_modes = _localized_approximate_field_null_modes(
            problem,
            remaining,
            existing_modes=parameterization,
            center_frequency=center_frequency,
        )
        if null_modes.shape[1]:
            parameterization = np.column_stack((parameterization, null_modes))
    return _readonly(parameterization)


def _localized_approximate_field_null_modes(
    problem: Any,
    requested_count: int,
    *,
    existing_modes: np.ndarray,
    center_frequency: bool,
) -> np.ndarray:
    """Construct smooth localized modes and remove their sampled B-field response."""

    count = min(int(requested_count), int(problem.reduced_sensitivity.shape[1]))
    if count < 1:
        return np.empty((problem.reduced_sensitivity.shape[1], 0), dtype=float)
    expansion = np.abs(np.asarray(problem.reducer.expansion_matrix, dtype=float))
    vertices = np.asarray(problem.combined_mesh.vertices, dtype=float)
    row_weights = np.sum(expansion, axis=0)
    if np.any(row_weights <= 1.0e-15):
        raise ValueError("Reduced stream-function coordinates contain an unmapped DOF.")
    coordinates = (expansion.T @ vertices) / row_weights[:, None]
    center_indices = _farthest_point_indices(coordinates, count)
    centers = coordinates[center_indices]
    if len(centers) > 1:
        center_distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        center_distances[center_distances == 0.0] = np.inf
        spacing = float(np.median(np.min(center_distances, axis=1)))
    else:
        spacing = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    sigma = max(1.75 * spacing, 1.0e-12)
    squared_distance = np.sum(
        (coordinates[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    localized = np.exp(-0.5 * squared_distance / sigma**2)

    sensitivity = np.asarray(problem.unweighted_reduced_sensitivity, dtype=float)
    raw_weights = problem.target_weights
    weights = (
        np.ones(sensitivity.shape[0], dtype=float)
        if raw_weights is None
        else np.asarray(raw_weights, dtype=float).reshape(-1)
    )
    normalized_weights = weights / float(np.sum(weights))
    if center_frequency:
        sensitivity = sensitivity - np.sum(normalized_weights[:, None] * sensitivity, axis=0)
    weighted_sensitivity = np.sqrt(normalized_weights)[:, None] * sensitivity
    field_scale = max(float(np.linalg.norm(weighted_sensitivity)), np.finfo(float).tiny)
    weighted_sensitivity = weighted_sensitivity / field_scale
    field_gram = weighted_sensitivity @ weighted_sensitivity.T
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (field_gram + field_gram.T))
    cutoff = max(float(np.max(eigenvalues)) * 1.0e-10, np.finfo(float).eps)
    retained = eigenvalues > cutoff
    if np.any(retained):
        response = weighted_sensitivity @ localized
        coefficients = (
            eigenvectors[:, retained].T @ response
        ) / eigenvalues[retained, None]
        localized -= weighted_sensitivity.T @ (eigenvectors[:, retained] @ coefficients)
    existing = np.asarray(existing_modes, dtype=float)
    if existing.size:
        localized -= existing @ (existing.T @ localized)
    left, singular_values, _right = np.linalg.svd(localized, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((localized.shape[0], 0), dtype=float)
    threshold = max(float(singular_values[0]) * 1.0e-9, np.finfo(float).eps)
    rank = min(int(np.count_nonzero(singular_values > threshold)), count)
    return np.asarray(left[:, :rank], dtype=float)


def _farthest_point_indices(coordinates: np.ndarray, count: int) -> np.ndarray:
    """Deterministically sample spatially distributed reduced-coordinate centers."""

    points = np.asarray(coordinates, dtype=float)
    count = min(max(int(count), 1), len(points))
    centroid = np.mean(points, axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared_distance = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared_distance))
        distance = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, distance)
    return selected


def _sector_local_expansion(mesh: Mesh, active_face_mask: np.ndarray) -> Any:
    """Map local sector DOFs to full vertices with constant inactive faces."""

    from scipy import sparse

    faces = np.asarray(mesh.faces, dtype=np.int64)
    active = np.asarray(active_face_mask, dtype=bool).reshape(-1)
    if active.shape != (mesh.n_faces,) or not np.any(active):
        raise ValueError("A sector-local expansion requires a non-empty active face mask.")

    parent = np.arange(mesh.n_vertices, dtype=np.int64)
    component_parent = np.arange(mesh.n_vertices, dtype=np.int64)

    def find(values: np.ndarray, parents: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.int64).copy()
        for index in range(len(result)):
            value = int(result[index])
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = int(parents[value])
            result[index] = value
        return result

    def union(first: int, second: int, parents: np.ndarray) -> None:
        first_root = int(find(np.asarray([first]), parents)[0])
        second_root = int(find(np.asarray([second]), parents)[0])
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for face in faces:
        union(int(face[0]), int(face[1]), component_parent)
        union(int(face[1]), int(face[2]), component_parent)
    for face in faces[~active]:
        union(int(face[0]), int(face[1]), parent)
        union(int(face[1]), int(face[2]), parent)
    for boundary in mesh.boundaries:
        nodes = np.unique(np.asarray(boundary, dtype=np.int64))
        if len(nodes):
            for node in nodes[1:]:
                union(int(nodes[0]), int(node), parent)

    roots = find(np.arange(mesh.n_vertices), parent)
    component_roots = find(np.arange(mesh.n_vertices), component_parent)
    active_vertices = np.unique(faces[active].reshape(-1))
    active_roots = np.unique(roots[active_vertices])
    fixed_roots: set[int] = set()
    for component in np.unique(component_roots[active_vertices]):
        candidates = [
            int(root)
            for root in active_roots
            if np.any(component_roots[roots == root] == component)
        ]
        if candidates:
            fixed_roots.add(
                max(candidates, key=lambda root: int(np.count_nonzero(roots == root)))
            )
    free_roots = [int(root) for root in active_roots if int(root) not in fixed_roots]
    if not free_roots:
        raise RuntimeError("Automatic sector has no non-gauge local stream-function DOFs.")
    root_to_column = {root: column for column, root in enumerate(free_roots)}
    rows = np.asarray(
        [vertex for vertex, root in enumerate(roots) if int(root) in root_to_column],
        dtype=np.int64,
    )
    columns = np.asarray([root_to_column[int(roots[vertex])] for vertex in rows], dtype=np.int64)
    expansion = sparse.coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(mesh.n_vertices, len(free_roots)),
    ).tocsr()
    return expansion


def _sector_local_expansions(mesh: Mesh, active_face_masks: np.ndarray) -> tuple[Any, ...]:
    masks = np.asarray(active_face_masks, dtype=bool)
    return tuple(_sector_local_expansion(mesh, masks[:, channel]) for channel in range(masks.shape[1]))


def _solve_basis_update(
    cp: Any,
    scenarios: Sequence[Any],
    fit: Any,
    current_basis: np.ndarray,
    anchor_basis: np.ndarray,
    limits: Any,
    config: CoilGenConfig,
    fit_options: Any,
    scenario_weights: np.ndarray,
    density_operators: tuple[Any, Any, Any],
    face_areas: np.ndarray,
    *,
    resistance_weight: float,
    basis_anchor_weight: float,
    overlap_weights: np.ndarray | None,
    overlap_penalty: float,
    active_face_masks: np.ndarray | None,
    basis_parameterization: np.ndarray | None = None,
    enforce_density_constraints: bool = True,
    minimum_field_projection_ratio: float = 0.0,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, str, str, int]:
    """Solve one convex channel update with frozen currents/support weights."""

    model_started = perf_counter()
    reduced_dof_count, channel_count = current_basis.shape
    parameterization = (
        np.eye(reduced_dof_count, dtype=float)
        if basis_parameterization is None
        else np.asarray(basis_parameterization, dtype=float)
    )
    if (
        parameterization.ndim != 2
        or parameterization.shape[0] != reduced_dof_count
        or parameterization.shape[1] < channel_count
        or np.any(~np.isfinite(parameterization))
    ):
        raise ValueError("basis_parameterization must have shape (reduced DOFs, discovery DOFs).")
    discovery_dof_count = parameterization.shape[1]
    coefficient_slices = [
        slice(channel * discovery_dof_count, (channel + 1) * discovery_dof_count)
        for channel in range(channel_count)
    ]
    variable = cp.Variable(
        discovery_dof_count * channel_count,
        name="sectorized_channel_basis_coefficients_per_A",
    )
    projected_density_operators = tuple(
        operator @ parameterization for operator in density_operators
    )
    constraints: list[Any] = []
    projected_sensitivities_by_problem: dict[int, np.ndarray] = {}
    currents = np.asarray(fit.current_matrix_A, dtype=float)
    baseline_scale = max(float(np.median(fit.rms_before_T)), 1.0e-15)
    total_coefficient_count = discovery_dof_count * channel_count
    quadratic = np.zeros((total_coefficient_count, total_coefficient_count), dtype=float)
    linear = np.zeros(total_coefficient_count, dtype=float)
    residual_groups: dict[tuple[int, bytes], dict[str, Any]] = {}
    from .cohort_shim import _scenario_weights, _solve_cvxpy

    for scenario_weight, item, fit_result, scenario_currents in zip(
        scenario_weights,
        scenarios,
        fit.scenario_results,
        currents,
    ):
        problem_key = id(item.problem)
        matrix = projected_sensitivities_by_problem.get(problem_key)
        if matrix is None:
            matrix = (
                np.asarray(item.problem.unweighted_reduced_sensitivity, dtype=float)
                @ parameterization
            )
            projected_sensitivities_by_problem[problem_key] = matrix
        disturbance = np.asarray(item.scenario.disturbance_T, dtype=float) + fit_result.auxiliary_field_T
        weights = _scenario_weights(item.scenario)
        normalized_weights = weights / float(np.sum(weights))
        metric_disturbance = normalized_weights * disturbance
        if fit_options.center_frequency:
            metric_disturbance -= normalized_weights * float(normalized_weights @ disturbance)
        scale = float(scenario_weight) / baseline_scale**2
        group_key = (problem_key, normalized_weights.tobytes())
        group = residual_groups.get(group_key)
        if group is None:
            group = {
                "sensitivity": matrix,
                "weights": normalized_weights,
                "current_second_moment": np.zeros((channel_count, channel_count), dtype=float),
                "disturbance_rhs": [
                    np.zeros_like(normalized_weights) for _ in range(channel_count)
                ],
            }
            residual_groups[group_key] = group
        group["current_second_moment"] += scale * np.outer(
            scenario_currents,
            scenario_currents,
        )
        for channel in range(channel_count):
            group["disturbance_rhs"][channel] += (
                2.0 * scale * float(scenario_currents[channel]) * metric_disturbance
            )

    for group in residual_groups.values():
        matrix = group["sensitivity"]
        normalized_weights = group["weights"]
        metric_hessian = matrix.T @ (normalized_weights[:, None] * matrix)
        if fit_options.center_frequency:
            weighted_mean = matrix.T @ normalized_weights
            metric_hessian -= np.outer(weighted_mean, weighted_mean)
        current_second_moment = group["current_second_moment"]
        for left_channel in range(channel_count):
            left_slice = coefficient_slices[left_channel]
            linear[left_slice] += matrix.T @ group["disturbance_rhs"][left_channel]
            for right_channel in range(left_channel, channel_count):
                coefficient = float(current_second_moment[left_channel, right_channel])
                if coefficient == 0.0:
                    continue
                right_slice = coefficient_slices[right_channel]
                quadratic[left_slice, right_slice] += coefficient * metric_hessian
                if right_channel != left_channel:
                    quadratic[right_slice, left_slice] += coefficient * metric_hessian.T

    resistance = np.asarray(scenarios[0].problem.reduced_resistance, dtype=float)
    resistance = 0.5 * (resistance + resistance.T)
    if np.any(~np.isfinite(resistance)):
        raise ValueError("Channel resistance matrix must be finite.")
    projected_resistance = parameterization.T @ resistance @ parameterization
    projected_resistance = 0.5 * (projected_resistance + projected_resistance.T)
    mean_current_square = np.maximum(np.mean(currents**2, axis=0), 1.0e-12)
    initial_energy = float(
        np.sum(
            mean_current_square
            * np.einsum("pk,pq,qk->k", current_basis, resistance, current_basis, optimize=True)
        )
    )
    for channel in range(channel_count):
        coefficient_slice = coefficient_slices[channel]
        quadratic[coefficient_slice, coefficient_slice] += (
            float(resistance_weight)
            * float(mean_current_square[channel])
            / max(initial_energy, 1.0e-30)
            * projected_resistance
        )

    reference_problem = scenarios[0].problem
    weighted_sensitivity = np.asarray(reference_problem.reduced_sensitivity, dtype=float)
    anchor_denominator = max(float(np.sum((weighted_sensitivity @ anchor_basis) ** 2)), 1.0e-30)
    anchor_sensitivity = weighted_sensitivity @ parameterization
    anchor_fields = weighted_sensitivity @ anchor_basis
    anchor_scale = float(basis_anchor_weight) / anchor_denominator
    anchor_hessian = anchor_sensitivity.T @ anchor_sensitivity
    for channel in range(channel_count):
        coefficient_slice = coefficient_slices[channel]
        quadratic[coefficient_slice, coefficient_slice] += anchor_scale * anchor_hessian
        linear[coefficient_slice] -= (
            2.0 * anchor_scale * anchor_sensitivity.T @ anchor_fields[:, channel]
        )

    if overlap_weights is not None and overlap_penalty > 0.0:
        weights_array = np.asarray(overlap_weights, dtype=float)
        if weights_array.shape != (len(face_areas), limits.channel_count):
            raise ValueError("overlap_weights have inconsistent face/channel dimensions.")
        reference_density = _face_current_density_magnitudes(density_operators, current_basis)
        reference_energy = np.maximum(
            np.sum(face_areas[:, None] * reference_density**2, axis=0),
            1.0e-30,
        )
        for channel in range(limits.channel_count):
            row_weights = np.maximum(face_areas * weights_array[:, channel], 0.0)
            overlap_quadratic = np.zeros(
                (parameterization.shape[1], parameterization.shape[1]),
                dtype=float,
            )
            for operator in projected_density_operators:
                weighted_operator = _row_scaled_matrix(operator, np.sqrt(row_weights))
                overlap_quadratic += np.asarray(
                    (weighted_operator.T @ weighted_operator).toarray()
                    if hasattr(weighted_operator, "toarray")
                    else weighted_operator.T @ weighted_operator,
                    dtype=float,
                )
            overlap_quadratic = 0.5 * (overlap_quadratic + overlap_quadratic.T)
            coefficient_slice = coefficient_slices[channel]
            quadratic[coefficient_slice, coefficient_slice] += (
                float(overlap_penalty)
                / max(limits.channel_count, 1)
                / reference_energy[channel]
                * overlap_quadratic
            )

    quadratic = 0.5 * (quadratic + quadratic.T)
    if np.any(~np.isfinite(quadratic)) or np.any(~np.isfinite(linear)):
        raise RuntimeError("Adaptive discovery produced a non-finite condensed objective.")
    objective = cp.Minimize(
        cp.quad_form(variable, quadratic, assume_PSD=True) + linear @ variable
    )
    scalar_constraint_count = 0
    if minimum_field_projection_ratio > 0.0:
        raw_sensitivity = np.asarray(
            reference_problem.unweighted_reduced_sensitivity,
            dtype=float,
        )
        raw_weights = reference_problem.target_weights
        field_weights = (
            np.ones(raw_sensitivity.shape[0], dtype=float)
            if raw_weights is None
            else np.asarray(raw_weights, dtype=float).reshape(-1)
        )
        field_weights /= float(np.sum(field_weights))
        discovery_fields = raw_sensitivity @ parameterization
        preserved_fields = raw_sensitivity @ anchor_basis
        if fit_options.center_frequency:
            discovery_fields -= np.sum(
                field_weights[:, None] * discovery_fields,
                axis=0,
            )
            preserved_fields -= np.sum(
                field_weights[:, None] * preserved_fields,
                axis=0,
            )
        for channel in range(channel_count):
            anchor_field = preserved_fields[:, channel]
            denominator = max(
                float(np.sum(field_weights * anchor_field**2)),
                1.0e-30,
            )
            projection = discovery_fields.T @ (field_weights * anchor_field) / denominator
            constraints.append(
                projection @ variable[coefficient_slices[channel]]
                >= float(minimum_field_projection_ratio)
            )
            scalar_constraint_count += 1
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    for channel in range(limits.channel_count):
        if density_limit > 0.0 and enforce_density_constraints:
            per_amp_limit = density_limit / max(float(limits.absolute_limits_A[channel]), 1.0e-30)
            components = cp.vstack(
                [
                    operator @ variable[coefficient_slices[channel]] / per_amp_limit
                    for operator in projected_density_operators
                ]
            ).T
            constraints.append(cp.norm(components, 2, axis=1) <= 1.0)
            scalar_constraint_count += len(face_areas)
        if active_face_masks is not None:
            inactive = ~np.asarray(active_face_masks[:, channel], dtype=bool)
            reference_scale = max(
                float(np.max(_face_current_density_magnitudes(density_operators, current_basis)[:, channel])),
                1.0e-12,
            )
            for operator in projected_density_operators:
                inactive_operator = _matrix_rows(operator, inactive)
                constraints.append(
                    (inactive_operator @ variable[coefficient_slices[channel]])
                    / reference_scale
                    == 0.0
                )
                scalar_constraint_count += int(np.count_nonzero(inactive))
    problem = cp.Problem(objective, constraints)
    if progress_callback is not None:
        metrics = problem.size_metrics
        progress_callback(
            f"Adaptive CVXPY model built in {perf_counter() - model_started:.2f} s: "
            f"{metrics.num_scalar_variables} scalar basis variables, "
            f"{metrics.num_scalar_eq_constr} equalities, "
            f"{metrics.num_scalar_leq_constr} scalar inequalities, "
            f"{scalar_constraint_count} modeled scalar support/field constraints."
        )
    status, solver = _solve_cvxpy(
        problem,
        cp,
        config,
        progress_callback=progress_callback,
    )
    if variable.value is None:
        raise RuntimeError("Adaptive sectorization returned no channel basis.")
    coefficient_values = np.asarray(variable.value, dtype=float)
    candidate = parameterization @ np.column_stack(
        [coefficient_values[coefficient_slice] for coefficient_slice in coefficient_slices]
    )
    if candidate.shape != current_basis.shape or not np.all(np.isfinite(candidate)):
        raise RuntimeError("Adaptive sectorization returned an invalid channel basis.")
    return candidate, solver, status, scalar_constraint_count


def _solve_sector_local_basis_update(
    cp: Any,
    scenarios: Sequence[Any],
    fit: Any,
    current_basis: np.ndarray,
    anchor_basis: np.ndarray,
    limits: Any,
    config: CoilGenConfig,
    fit_options: Any,
    scenario_weights: np.ndarray,
    active_face_masks: np.ndarray,
    *,
    resistance_weight: float,
    basis_anchor_weight: float,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, str, str, int, np.ndarray]:
    """Solve the hard partition using only each sector's physical local DOFs."""

    from .cohort_shim import _scenario_weights, _solve_cvxpy

    model_started = perf_counter()
    reference_problem = scenarios[0].problem
    mesh = reference_problem.combined_mesh
    masks = np.asarray(active_face_masks, dtype=bool)
    expansions = _sector_local_expansions(mesh, masks)
    local_dof_counts = np.asarray([expansion.shape[1] for expansion in expansions], dtype=np.int64)
    if progress_callback is not None:
        progress_callback(
            "Adaptive hardening local parameterization: channel DOFs "
            f"{local_dof_counts.tolist()} ({int(np.sum(local_dof_counts))} total versus "
            f"{current_basis.size} global channel DOFs)."
        )

    local_offsets = np.concatenate(([0], np.cumsum(local_dof_counts, dtype=np.int64)))
    local_slices = [
        slice(int(local_offsets[channel]), int(local_offsets[channel + 1]))
        for channel in range(limits.channel_count)
    ]
    total_local_dofs = int(local_offsets[-1])
    variable = cp.Variable(total_local_dofs, name="sector_local_stream_functions")
    constraints: list[Any] = []
    local_sensitivities_by_problem: dict[int, tuple[np.ndarray, ...]] = {}

    def local_sensitivities(problem: Any) -> tuple[np.ndarray, ...]:
        problem_key = id(problem)
        matrices = local_sensitivities_by_problem.get(problem_key)
        if matrices is None:
            full_sensitivity = sensitivity_component_matrix(
                problem.sensitivity_matrix,
                config.target_field_component,
                b0_direction=config.b_0_direction,
            )
            matrices = tuple(
                np.asarray(full_sensitivity @ expansion, dtype=float)
                for expansion in expansions
            )
            local_sensitivities_by_problem[problem_key] = matrices
        return matrices

    currents = np.asarray(fit.current_matrix_A, dtype=float)
    baseline_scale = max(float(np.median(fit.rms_before_T)), 1.0e-15)
    quadratic = np.zeros((total_local_dofs, total_local_dofs), dtype=float)
    linear = np.zeros(total_local_dofs, dtype=float)
    residual_groups: dict[tuple[int, bytes], dict[str, Any]] = {}
    for scenario_weight, item, fit_result, scenario_currents in zip(
        scenario_weights,
        scenarios,
        fit.scenario_results,
        currents,
    ):
        channel_sensitivities = local_sensitivities(item.problem)
        disturbance = np.asarray(item.scenario.disturbance_T, dtype=float) + fit_result.auxiliary_field_T
        weights = _scenario_weights(item.scenario)
        normalized_weights = weights / float(np.sum(weights))
        metric_disturbance = normalized_weights * disturbance
        if fit_options.center_frequency:
            metric_disturbance -= normalized_weights * float(normalized_weights @ disturbance)
        scale = float(scenario_weight) / baseline_scale**2
        group_key = (id(item.problem), normalized_weights.tobytes())
        group = residual_groups.get(group_key)
        if group is None:
            group = {
                "sensitivities": channel_sensitivities,
                "weights": normalized_weights,
                "current_second_moment": np.zeros(
                    (limits.channel_count, limits.channel_count), dtype=float
                ),
                "disturbance_rhs": [
                    np.zeros_like(normalized_weights) for _ in range(limits.channel_count)
                ],
            }
            residual_groups[group_key] = group
        group["current_second_moment"] += scale * np.outer(
            scenario_currents,
            scenario_currents,
        )
        for channel in range(limits.channel_count):
            group["disturbance_rhs"][channel] += (
                2.0 * scale * float(scenario_currents[channel]) * metric_disturbance
            )

    for group in residual_groups.values():
        channel_sensitivities = group["sensitivities"]
        normalized_weights = group["weights"]
        current_second_moment = group["current_second_moment"]
        weighted_means = [
            channel_sensitivity.T @ normalized_weights
            for channel_sensitivity in channel_sensitivities
        ]
        for left_channel, left_sensitivity in enumerate(channel_sensitivities):
            left_slice = local_slices[left_channel]
            linear[left_slice] += left_sensitivity.T @ group["disturbance_rhs"][left_channel]
            for right_channel in range(left_channel, limits.channel_count):
                coefficient = float(current_second_moment[left_channel, right_channel])
                if coefficient == 0.0:
                    continue
                right_sensitivity = channel_sensitivities[right_channel]
                block = left_sensitivity.T @ (
                    normalized_weights[:, None] * right_sensitivity
                )
                if fit_options.center_frequency:
                    block -= np.outer(
                        weighted_means[left_channel],
                        weighted_means[right_channel],
                    )
                right_slice = local_slices[right_channel]
                quadratic[left_slice, right_slice] += coefficient * block
                if right_channel != left_channel:
                    quadratic[right_slice, left_slice] += coefficient * block.T

    full_resistance = np.asarray(reference_problem.resistance_matrix, dtype=float)
    full_resistance = 0.5 * (full_resistance + full_resistance.T)
    if np.any(~np.isfinite(full_resistance)):
        raise ValueError("Full channel resistance matrix must be finite.")
    original_expansion = np.asarray(reference_problem.reducer.expansion_matrix, dtype=float)
    current_full_basis = original_expansion @ current_basis
    mean_current_square = np.maximum(np.mean(currents**2, axis=0), 1.0e-12)
    initial_energy = float(
        np.sum(
            mean_current_square
            * np.einsum(
                "pk,pq,qk->k",
                current_full_basis,
                full_resistance,
                current_full_basis,
                optimize=True,
            )
        )
    )
    for channel, expansion in enumerate(expansions):
        local_resistance = np.asarray(expansion.T @ full_resistance @ expansion, dtype=float)
        local_resistance = 0.5 * (local_resistance + local_resistance.T)
        local_slice = local_slices[channel]
        quadratic[local_slice, local_slice] += (
            float(resistance_weight)
            * float(mean_current_square[channel])
            / max(initial_energy, 1.0e-30)
            * local_resistance
        )

    reference_sensitivities = local_sensitivities(reference_problem)
    anchor_fields = np.asarray(reference_problem.unweighted_reduced_sensitivity, dtype=float) @ anchor_basis
    raw_target_weights = reference_problem.target_weights
    target_weights = (
        np.ones(anchor_fields.shape[0], dtype=float)
        if raw_target_weights is None
        else np.asarray(raw_target_weights, dtype=float).reshape(-1)
    )
    target_weights = np.maximum(target_weights, 0.0)
    anchor_denominator = max(
        float(np.sum((np.sqrt(target_weights)[:, None] * anchor_fields) ** 2)),
        1.0e-30,
    )
    anchor_scale = float(basis_anchor_weight) / anchor_denominator
    for channel, channel_sensitivity in enumerate(reference_sensitivities):
        local_slice = local_slices[channel]
        weighted_channel_sensitivity = target_weights[:, None] * channel_sensitivity
        quadratic[local_slice, local_slice] += anchor_scale * (
            channel_sensitivity.T @ weighted_channel_sensitivity
        )
        linear[local_slice] -= (
            2.0
            * anchor_scale
            * channel_sensitivity.T
            @ (target_weights * anchor_fields[:, channel])
        )

    full_density_operators = _full_face_current_density_operators(reference_problem)
    density_limit = engineering_surface_current_density_limit_A_per_m(config)
    density_constraint_count = 0
    if density_limit > 0.0:
        for channel, expansion in enumerate(expansions):
            active = masks[:, channel]
            per_amp_limit = density_limit / max(float(limits.absolute_limits_A[channel]), 1.0e-30)
            components = cp.vstack(
                [
                    (_matrix_rows(operator, active) @ expansion)
                    @ variable[local_slices[channel]]
                    / per_amp_limit
                    for operator in full_density_operators
                ]
            ).T
            constraints.append(cp.norm(components, 2, axis=1) <= 1.0)
            density_constraint_count += int(np.count_nonzero(active))

    quadratic = 0.5 * (quadratic + quadratic.T)
    if np.any(~np.isfinite(quadratic)) or np.any(~np.isfinite(linear)):
        raise RuntimeError("Adaptive local hardening produced a non-finite condensed objective.")
    objective = cp.Minimize(
        cp.quad_form(variable, quadratic, assume_PSD=True) + linear @ variable
    )
    problem = cp.Problem(objective, constraints)
    if progress_callback is not None:
        metrics = problem.size_metrics
        progress_callback(
            f"Adaptive local hardening model built in {perf_counter() - model_started:.2f} s: "
            f"{metrics.num_scalar_variables} scalar variables, "
            f"{metrics.num_scalar_eq_constr} equalities, "
            f"{metrics.num_scalar_leq_constr} scalar inequalities, "
            f"{density_constraint_count} active-face density cones."
        )
    status, solver = _solve_cvxpy(
        problem,
        cp,
        config,
        progress_callback=progress_callback,
    )
    if variable.value is None:
        raise RuntimeError("Adaptive local hardening returned no channel basis.")
    local_values = np.asarray(variable.value, dtype=float)
    full_basis = np.column_stack(
        [
            np.asarray(expansion @ local_values[local_slice], dtype=float)
            for expansion, local_slice in zip(expansions, local_slices)
        ]
    )
    reduced_basis = _project_full_basis_to_reduced(reference_problem, full_basis)
    return reduced_basis, solver, status, density_constraint_count, local_dof_counts


def _project_full_basis_to_reduced(problem: Any, full_basis: np.ndarray) -> np.ndarray:
    """Recover the original reduced coordinates, allowing only gauge offsets."""

    expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
    component_gauges = _vertex_component_indicators(problem.combined_mesh)
    augmented = np.column_stack((expansion, component_gauges))
    gram = augmented.T @ augmented
    right_hand_side = augmented.T @ np.asarray(full_basis, dtype=float)
    try:
        coefficients = np.linalg.solve(gram, right_hand_side)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, right_hand_side, rcond=1.0e-12)[0]
    reduced = coefficients[: expansion.shape[1]]
    reconstructed = expansion @ reduced
    full_operators = _full_face_current_density_operators(problem)
    difference = reconstructed - full_basis
    maximum_density_error = max(
        float(np.max(np.abs(operator @ difference))) for operator in full_operators
    )
    reference_density = max(
        max(float(np.max(np.abs(operator @ full_basis))) for operator in full_operators),
        1.0e-30,
    )
    if maximum_density_error / reference_density > 1.0e-8:
        raise RuntimeError(
            "Sector-local stream function could not be mapped back to the original reduced "
            "coordinates without changing sheet current."
        )
    return np.asarray(reduced, dtype=float)


def _vertex_component_indicators(mesh: Mesh) -> np.ndarray:
    """Return one constant stream-function gauge column per connected mesh component."""

    parent = np.arange(mesh.n_vertices, dtype=np.int64)

    def find(vertex: int) -> int:
        value = int(vertex)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for face in np.asarray(mesh.faces, dtype=np.int64):
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
    roots = np.asarray([find(vertex) for vertex in range(mesh.n_vertices)], dtype=np.int64)
    unique_roots, component_indices = np.unique(roots, return_inverse=True)
    indicators = np.zeros((mesh.n_vertices, len(unique_roots)), dtype=float)
    indicators[np.arange(mesh.n_vertices), component_indices] = 1.0
    return indicators


def _validate_common_sectorization_surface(scenarios: Sequence[Any]) -> None:
    reference = scenarios[0].problem
    reference_faces = np.asarray(reference.combined_mesh.faces, dtype=np.int64)
    reference_vertices = np.asarray(reference.combined_mesh.vertices, dtype=float)
    reference_expansion = np.asarray(reference.reducer.expansion_matrix, dtype=float)
    for item in scenarios[1:]:
        problem = item.problem
        if problem is reference:
            continue
        if not np.array_equal(problem.combined_mesh.faces, reference_faces):
            raise ValueError("Adaptive sectorization requires identical surface face topology for every scenario.")
        if not np.allclose(problem.combined_mesh.vertices, reference_vertices, atol=1.0e-12, rtol=0.0):
            raise ValueError("Adaptive sectorization requires one common registered support geometry.")
        if not np.allclose(problem.reducer.expansion_matrix, reference_expansion, atol=1.0e-12, rtol=0.0):
            raise ValueError("Adaptive sectorization requires identical reduced stream-function coordinates.")


def _reduced_face_current_density_operators(problem: Any) -> tuple[Any, Any, Any]:
    """Return sparse face operators mapping reduced stream function to Kx/Ky/Kz."""

    from scipy import sparse

    expansion = sparse.csr_matrix(np.asarray(problem.reducer.expansion_matrix, dtype=float))
    full_operators = _full_face_current_density_operators(problem)
    operators = tuple((operator @ expansion).tocsr() for operator in full_operators)
    return operators[0], operators[1], operators[2]


def _full_face_current_density_operators(problem: Any) -> tuple[Any, Any, Any]:
    """Return sparse face operators mapping full vertex stream function to K."""

    try:
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError(
            "Adaptive sectorization requires SciPy sparse matrices; a dense face-by-vertex "
            "operator is unsafe for production surface meshes."
        ) from exc
    part_lengths = tuple(
        len(values)
        for values in (
            problem.part_vertex_counts,
            problem.part_current_density_mats,
            problem.part_faces,
            problem.part_face_basis_currents,
        )
    )
    if not part_lengths or len(set(part_lengths)) != 1 or part_lengths[0] < 1:
        raise ValueError("Surface-current-density metadata must contain one consistent entry per part.")
    if sum(problem.part_vertex_counts) != problem.combined_mesh.n_vertices:
        raise ValueError("Surface part vertices do not match the combined support mesh.")
    part_component_operators: list[list[Any]] = [[], [], []]
    for vertex_count, current_density_mat, faces, face_basis_currents in zip(
        problem.part_vertex_counts,
        problem.part_current_density_mats,
        problem.part_faces,
        problem.part_face_basis_currents,
    ):
        face_array = np.asarray(faces, dtype=np.int64)
        if current_density_mat is not None:
            density_matrix = np.asarray(current_density_mat, dtype=float)
            if density_matrix.shape != (vertex_count, len(face_array), 3):
                raise ValueError("Surface-current-density matrix has inconsistent dimensions.")
            component_arrays = [
                sparse.csr_matrix(density_matrix[:, :, component].T)
                for component in range(3)
            ]
        else:
            local = np.asarray(face_basis_currents, dtype=float)
            if local.shape != (len(face_array), 3, 3):
                raise ValueError("Surface-current-density operators have inconsistent dimensions.")
            component_arrays = []
            rows = np.repeat(np.arange(len(face_array), dtype=np.int64), 3)
            columns = face_array.reshape(-1)
            for component in range(3):
                values = local[:, :, component].reshape(-1)
                operator = sparse.coo_matrix(
                    (values, (rows, columns)),
                    shape=(len(face_array), vertex_count),
                ).tocsr()
                component_arrays.append(operator)
        for component, operator in enumerate(component_arrays):
            part_component_operators[component].append(sparse.csr_matrix(operator))
    operators: list[Any] = []
    for component_parts in part_component_operators:
        operators.append(sparse.block_diag(component_parts, format="csr"))
    if any(
        operator.shape != (problem.combined_mesh.n_faces, problem.combined_mesh.n_vertices)
        for operator in operators
    ):
        raise ValueError("Combined face-current-density operator does not match the support mesh.")
    return operators[0], operators[1], operators[2]


def _face_current_density_magnitudes(
    density_operators: tuple[Any, Any, Any],
    basis_per_A: np.ndarray,
) -> np.ndarray:
    components = np.stack(
        [np.asarray(operator @ basis_per_A, dtype=float) for operator in density_operators],
        axis=2,
    )
    return np.linalg.norm(components, axis=2)


def _channel_field_rms(
    problem: Any,
    basis_per_A: np.ndarray,
    *,
    center_frequency: bool,
) -> np.ndarray:
    """Return target-weighted field RMS per ampere for each physical channel."""

    rms, _condition, _correlation = _normalized_channel_field_diagnostics(
        problem,
        basis_per_A,
        center_frequency=center_frequency,
    )
    return rms


def _normalized_channel_field_diagnostics(
    problem: Any,
    basis_per_A: np.ndarray,
    *,
    center_frequency: bool,
) -> tuple[np.ndarray, float, float]:
    """Measure channel strength and independence in the weighted target space.

    The condition number is evaluated after independently normalizing every
    channel field.  It therefore detects field collinearity without confusing
    it with the per-ampere scaling chosen during channel calibration.
    """

    sensitivity = np.asarray(problem.unweighted_reduced_sensitivity, dtype=float)
    fields = sensitivity @ np.asarray(basis_per_A, dtype=float)
    raw_weights = problem.target_weights
    weights = (
        np.ones(fields.shape[0], dtype=float)
        if raw_weights is None
        else np.asarray(raw_weights, dtype=float).reshape(-1)
    )
    normalized_weights = weights / float(np.sum(weights))
    if center_frequency:
        fields = fields - np.sum(normalized_weights[:, None] * fields, axis=0)
    rms = np.sqrt(np.sum(normalized_weights[:, None] * fields**2, axis=0))
    if np.any(rms <= 1.0e-30):
        return rms, float("inf"), 1.0
    normalized_fields = np.sqrt(normalized_weights[:, None]) * fields / rms[None, :]
    singular_values = np.linalg.svd(normalized_fields, compute_uv=False)
    condition = float(
        singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny)
    )
    correlation = normalized_fields.T @ normalized_fields
    off_diagonal = np.abs(correlation - np.eye(correlation.shape[0]))
    maximum_correlation = float(np.max(off_diagonal)) if len(correlation) > 1 else 0.0
    return rms, condition, maximum_correlation


def _activity_occupancy(
    density_per_A: np.ndarray,
    face_areas: np.ndarray,
    current_limits_A: np.ndarray,
    options: AdaptiveSectorizationOptions,
    *,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hardware_density = np.asarray(density_per_A, dtype=float) * current_limits_A[None, :]
    if thresholds is None:
        thresholds = np.asarray(
            [
                options.activity_threshold_fraction
                * _weighted_quantile(hardware_density[:, channel], face_areas, options.activity_quantile)
                for channel in range(hardware_density.shape[1])
            ]
        )
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    if len(thresholds) != hardware_density.shape[1] or np.any(~np.isfinite(thresholds)):
        raise ValueError("Activity thresholds do not match the channel basis.")
    peak = np.max(hardware_density, axis=0)
    if np.any(peak <= 1.0e-15):
        missing = (np.flatnonzero(peak <= 1.0e-15) + 1).tolist()
        raise RuntimeError(f"Adaptive sectorization found inactive channel(s): {missing}.")
    thresholds = np.maximum(thresholds, np.maximum(peak * 1.0e-9, 1.0e-15))
    relative = hardware_density / thresholds[None, :]
    scores = np.minimum(relative, 1.0e6) ** float(options.ownership_power)
    occupancy = scores / (1.0 + scores)
    return occupancy, thresholds


def _competitive_ownership(activity: np.ndarray) -> np.ndarray:
    """Convert independent activity into channel ownership plus implicit background."""

    values = np.clip(np.asarray(activity, dtype=float), 0.0, 1.0 - 1.0e-12)
    odds = values / np.maximum(1.0 - values, 1.0e-12)
    return odds / (1.0 + np.sum(odds, axis=1, keepdims=True))


def _continuous_overlap_fraction(occupancy: np.ndarray, face_areas: np.ndarray) -> float:
    channel_count = occupancy.shape[1]
    if channel_count < 2:
        return 0.0
    pair_sum = np.zeros(len(face_areas), dtype=float)
    for first, second in combinations(range(channel_count), 2):
        pair_sum += occupancy[:, first] * occupancy[:, second]
    denominator = max(float(np.sum(face_areas)) * (channel_count * (channel_count - 1) / 2.0), 1.0e-30)
    return float(np.sum(face_areas * pair_sum) / denominator)


def _automatic_face_partition(
    mesh: Mesh,
    occupancy: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    options: AdaptiveSectorizationOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_count = occupancy.shape[1]
    seeds = _boundary_channel_seeds(
        mesh,
        occupancy,
        clearance_m=options.clearance_m,
    )
    unary = -np.log(np.clip(occupancy, 1.0e-12, 1.0))
    if options.partition_compactness > 0.0:
        distance_columns = []
        for seed in seeds:
            distances = _graph_distances(adjacency, (int(seed),))
            finite = distances[np.isfinite(distances)]
            scale = max(float(np.quantile(finite, 0.90)) if finite.size else 1.0, 1.0e-30)
            distances = np.where(np.isfinite(distances), distances / scale, 2.0)
            distance_columns.append(distances)
        unary += options.partition_compactness * np.column_stack(distance_columns)
    effective_smoothness = _scaled_partition_smoothness(
        unary,
        options.partition_smoothness,
    )
    labels = _balanced_connected_region_growing(
        unary,
        seeds,
        face_areas,
        adjacency,
        minimum_area_fraction=options.minimum_sector_area_fraction,
        maximum_area_fraction=options.maximum_sector_area_fraction,
        smoothness=effective_smoothness,
        port_radius_m=max(float(options.clearance_m), 0.0),
    )
    total_area = max(float(np.sum(face_areas)), 1.0e-30)
    fractions = np.asarray(
        [np.sum(face_areas[labels == channel]) / total_area for channel in range(channel_count)]
    )
    if np.any(fractions < options.minimum_sector_area_fraction):
        failed = (np.flatnonzero(fractions < options.minimum_sector_area_fraction) + 1).tolist()
        raise RuntimeError(f"Connected region growing produced undersized channel sector(s) {failed}.")
    if np.any(fractions > options.maximum_sector_area_fraction):
        failed = (np.flatnonzero(fractions > options.maximum_sector_area_fraction) + 1).tolist()
        raise RuntimeError(f"Connected region growing produced oversized channel sector(s) {failed}.")
    return labels, seeds, unary


def _balanced_connected_region_growing(
    unary: np.ndarray,
    seeds: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    *,
    minimum_area_fraction: float,
    maximum_area_fraction: float,
    smoothness: float,
    port_radius_m: float,
) -> np.ndarray:
    """Grow port-connected channel regions while enforcing area capacities."""

    channel_count = unary.shape[1]
    labels = np.full(len(unary), -1, dtype=np.int64)
    port_labels = _port_core_labels(
        seeds,
        adjacency,
        radius_m=port_radius_m,
    )
    total_area = max(float(np.sum(face_areas)), 1.0e-30)
    minimum_area = float(minimum_area_fraction) * total_area
    maximum_area = float(maximum_area_fraction) * total_area
    target_area = min(maximum_area, max(minimum_area, total_area / channel_count))
    areas = np.zeros(channel_count, dtype=float)
    for face, channel in port_labels.items():
        if labels[face] >= 0 and labels[face] != channel:
            raise RuntimeError("Boundary terminal cores overlap after nearest-port assignment.")
        labels[face] = channel
    for channel in range(channel_count):
        areas[channel] = float(np.sum(face_areas[labels == channel]))

    edge_lengths = [entry[1] for neighbors in adjacency for entry in neighbors]
    edge_scale = max(float(np.median(edge_lengths)) if edge_lengths else 1.0, 1.0e-30)
    frontiers: list[list[tuple[float, int]]] = [[] for _ in range(channel_count)]
    queued: list[set[int]] = [set() for _ in range(channel_count)]

    def push(channel: int, face: int) -> None:
        if labels[face] >= 0 or face in queued[channel]:
            return
        same_boundary = sum(
            shared_length
            for neighbor, shared_length, _travel in adjacency[face]
            if labels[neighbor] == channel
        )
        cost = float(unary[face, channel]) - float(smoothness) * same_boundary / edge_scale
        heapq.heappush(frontiers[channel], (cost, int(face)))
        queued[channel].add(int(face))

    for face, channel in port_labels.items():
        for neighbor, _shared, _travel in adjacency[face]:
            push(channel, int(neighbor))

    def grow_one(channel: int) -> bool:
        while frontiers[channel]:
            _cost, face = heapq.heappop(frontiers[channel])
            if labels[face] >= 0:
                continue
            labels[face] = channel
            areas[channel] += face_areas[face]
            for neighbor, _shared, _travel in adjacency[face]:
                push(channel, int(neighbor))
            return True
        return False

    # First protect one manufacturable, port-connected core per channel.  Growing
    # all channels to the equal-area target here can let an early front encircle
    # another seed on narrow or annular meshes, even though a valid balanced
    # partition exists.  The remaining area is allocated competitively below.
    for _step in range(len(labels) * channel_count):
        deficient = np.flatnonzero(areas < minimum_area)
        if deficient.size == 0:
            break
        channel = int(
            deficient[
                np.argmax((minimum_area - areas[deficient]) / max(minimum_area, 1.0e-30))
            ]
        )
        if not grow_one(channel):
            raise RuntimeError(
                f"Channel {channel + 1} cannot reach its minimum port-connected area."
            )

    while np.any(labels < 0):
        eligible = [
            channel
            for channel in range(channel_count)
            if areas[channel] < maximum_area and frontiers[channel]
        ]
        if not eligible:
            eligible = [channel for channel in range(channel_count) if frontiers[channel]]
        if not eligible:
            raise RuntimeError("Connected region growing left unreachable support faces.")
        channel = min(
            eligible,
            key=lambda item: (
                areas[item] / max(target_area, 1.0e-30),
                frontiers[item][0][0],
                item,
            ),
        )
        if not grow_one(channel):
            continue
    return labels


def _unique_channel_seeds(occupancy: np.ndarray) -> np.ndarray:
    channel_count = occupancy.shape[1]
    try:
        from scipy.optimize import linear_sum_assignment

        channels, faces = linear_sum_assignment(-occupancy.T)
        seeds = np.empty(channel_count, dtype=np.int64)
        seeds[channels] = faces
    except Exception:
        seeds = np.full(channel_count, -1, dtype=np.int64)
        used: set[int] = set()
        channel_order = np.argsort(-np.max(occupancy, axis=0))
        for channel in channel_order:
            for face in np.argsort(-occupancy[:, channel]):
                if int(face) not in used:
                    seeds[channel] = int(face)
                    used.add(int(face))
                    break
    if np.any(seeds < 0) or len(set(int(value) for value in seeds)) != channel_count:
        raise RuntimeError("Could not identify one unique active seed face per channel.")
    return seeds


def _boundary_channel_seeds(
    mesh: Mesh,
    ownership: np.ndarray,
    *,
    clearance_m: float,
) -> np.ndarray:
    """Choose distinct, separated boundary ports for every physical channel."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(first), int(second)), max(int(first), int(second)))
            edge_faces.setdefault(edge, []).append(face_index)
    boundary_faces = np.asarray(
        sorted({incident[0] for incident in edge_faces.values() if len(incident) == 1}),
        dtype=np.int64,
    )
    channel_count = ownership.shape[1]
    if len(boundary_faces) < channel_count:
        raise RuntimeError("Adaptive sectorization requires one boundary terminal face per channel.")
    centroids = np.mean(vertices[faces], axis=1)
    channel_order = np.argsort(-np.max(ownership[boundary_faces], axis=0), kind="stable")
    seeds = np.full(channel_count, -1, dtype=np.int64)
    selected: list[int] = []
    minimum_separation = max(4.0 * float(clearance_m), 1.0e-12)
    for channel in channel_order:
        candidates = boundary_faces.copy()
        scores = np.asarray(ownership[candidates, channel], dtype=float)
        if selected:
            separation = np.min(
                np.linalg.norm(
                    centroids[candidates, None, :] - centroids[np.asarray(selected)][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            feasible = separation >= minimum_separation
            if np.any(feasible):
                candidates = candidates[feasible]
                scores = scores[feasible]
            else:
                scores = scores * np.maximum(separation / minimum_separation, 1.0e-6)
        if selected:
            unused = ~np.isin(candidates, np.asarray(selected, dtype=np.int64))
            candidates = candidates[unused]
            scores = scores[unused]
        selected_face = int(candidates[int(np.argmax(scores))])
        seeds[int(channel)] = selected_face
        selected.append(selected_face)
    if np.any(seeds < 0) or len(set(int(seed) for seed in seeds)) != channel_count:
        raise RuntimeError("Could not choose unique boundary terminal faces for all channels.")
    return seeds


def _port_core_labels(
    seeds: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    *,
    radius_m: float,
) -> dict[int, int]:
    """Reserve a clearance-width same-label neighborhood around each terminal face."""

    distances = np.column_stack(
        [_graph_distances(adjacency, (int(seed),)) for seed in np.asarray(seeds, dtype=np.int64)]
    )
    nearest = np.argmin(distances, axis=1)
    minimum = np.min(distances, axis=1)
    forced = {
        int(face): int(nearest[face])
        for face in np.flatnonzero(minimum <= max(float(radius_m), 0.0))
    }
    for channel, seed in enumerate(np.asarray(seeds, dtype=np.int64)):
        forced[int(seed)] = channel
    return forced


def _merge_disconnected_label_islands(
    labels: np.ndarray,
    seeds: np.ndarray,
    unary: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
) -> np.ndarray:
    """Attach every orphan label component to an already seed-connected region.

    Reassigning all islands in one pass can merely transfer an island to a
    different disconnected label.  Growing only from seed-connected cores
    makes every accepted move connectivity preserving and strictly reduces the
    number of orphan components.
    """

    labels = labels.copy()
    channel_count = unary.shape[1]
    labels[seeds] = np.arange(channel_count, dtype=np.int64)
    for _step in range(max(len(labels) * channel_count, 1)):
        core = np.zeros(len(labels), dtype=bool)
        orphans: list[tuple[int, np.ndarray]] = []
        for channel in range(channel_count):
            components = _face_components(labels == channel, adjacency)
            seed_component = next(
                (
                    component
                    for component in components
                    if int(seeds[channel]) in component
                ),
                None,
            )
            if seed_component is None:
                raise RuntimeError("Automatic sector seed ownership was lost during partitioning.")
            core[np.asarray(seed_component, dtype=np.int64)] = True
            orphans.extend(
                (channel, np.asarray(component, dtype=np.int64))
                for component in components
                if component is not seed_component
            )
        if not orphans:
            return labels

        moves: list[tuple[float, int, np.ndarray]] = []
        for source, component in orphans:
            destinations: set[int] = set()
            for face in component:
                destinations.update(
                    int(labels[neighbor])
                    for neighbor, _shared, _travel in adjacency[int(face)]
                    if core[neighbor] and labels[neighbor] != source
                )
            for destination in destinations:
                cost = float(np.mean(unary[component, destination]))
                moves.append((cost, destination, component))
        if not moves:
            raise RuntimeError(
                "Automatic sector partition contains a disconnected mesh region that cannot be "
                "attached to any channel seed. Split or repair the support mesh first."
            )
        _cost, destination, component = min(
            moves,
            key=lambda item: (item[0], item[1], int(item[2][0])),
        )
        labels[component] = destination
    raise RuntimeError("Automatic sector connectivity cleanup did not converge.")


def _partition_guard_faces(
    labels: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    clearance_m: float,
) -> np.ndarray:
    guard = np.zeros(len(labels), dtype=bool)
    if clearance_m <= 0.0:
        return guard
    boundary_faces: set[int] = set()
    for face, neighbors in enumerate(adjacency):
        for neighbor, _shared, _travel in neighbors:
            if labels[face] != labels[neighbor]:
                boundary_faces.update((face, neighbor))
    if not boundary_faces:
        return guard
    distances = _graph_distances(adjacency, tuple(sorted(boundary_faces)))
    guard = distances <= 0.5 * float(clearance_m)
    guard[np.asarray(tuple(boundary_faces), dtype=np.int64)] = True
    return guard


def _harden_partition_with_clearance(
    labels: np.ndarray,
    seeds: np.ndarray,
    unary: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    options: AdaptiveSectorizationOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance connected labels until their clearance-eroded cores meet the area floor."""

    balanced_labels = np.asarray(labels, dtype=np.int64).copy()
    total_area = max(float(np.sum(face_areas)), 1.0e-30)
    minimum_area = options.minimum_sector_area_fraction * total_area
    minimum_label_areas = np.full(unary.shape[1], minimum_area, dtype=float)
    effective_smoothness = _scaled_partition_smoothness(
        unary,
        options.partition_smoothness,
    )
    for _iteration in range(20):
        guard = _partition_guard_faces(balanced_labels, adjacency, options.clearance_m)
        guard = _retain_connected_sector_cores(
            balanced_labels,
            seeds,
            guard,
            face_areas,
            adjacency,
        )
        active_areas = np.asarray(
            [
                np.sum(face_areas[(balanced_labels == channel) & ~guard])
                for channel in range(unary.shape[1])
            ],
            dtype=float,
        )
        deficits = np.maximum(minimum_area - active_areas, 0.0)
        if np.all(deficits <= max(1.0e-12 * total_area, 1.0e-30)):
            return balanced_labels, guard
        label_areas = np.asarray(
            [np.sum(face_areas[balanced_labels == channel]) for channel in range(unary.shape[1])],
            dtype=float,
        )
        # A boundary layer added to a deficient label is itself eroded. Growing
        # by twice the measured active-area deficit normally supplies both the
        # new guard layer and the requested usable area; repeated evaluation
        # handles nonuniform meshes and narrow geometry exactly.
        target_areas = minimum_label_areas.copy()
        deficient = deficits > max(1.0e-12 * total_area, 1.0e-30)
        target_areas[deficient] = label_areas[deficient] + 2.0 * deficits[deficient]
        balanced_labels = _grow_connected_sectors_to_area(
            balanced_labels,
            seeds,
            unary,
            face_areas,
            adjacency,
            target_areas,
            minimum_label_areas,
            partition_smoothness=effective_smoothness,
        )
    raise RuntimeError(
        "Automatic sector partition could not satisfy the usable-area floor after clearance "
        "while preserving connected sectors. Reduce clearance or the minimum area fraction."
    )


def _scaled_partition_smoothness(unary: np.ndarray, dimensionless_weight: float) -> float:
    """Scale Potts regularization to the observed channel-preference margin."""

    values = np.asarray(unary, dtype=float)
    if values.shape[1] < 2 or dimensionless_weight <= 0.0:
        return 0.0
    ordered = np.partition(values, 1, axis=1)
    margins = ordered[:, 1] - ordered[:, 0]
    positive = margins[np.isfinite(margins) & (margins > 0.0)]
    reference_margin = float(np.median(positive)) if positive.size else 1.0
    return float(dimensionless_weight) * max(reference_margin, 1.0e-12)


def _grow_connected_sectors_to_area(
    labels: np.ndarray,
    seeds: np.ndarray,
    unary: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    target_areas: np.ndarray,
    minimum_donor_areas: np.ndarray,
    *,
    partition_smoothness: float,
) -> np.ndarray:
    """Grow undersized sectors with frontier heaps, then repair connectivity."""

    grown = np.asarray(labels, dtype=np.int64).copy()
    channel_count = unary.shape[1]
    targets = np.asarray(target_areas, dtype=float).reshape(-1)
    donor_minimums = np.asarray(minimum_donor_areas, dtype=float).reshape(-1)
    if targets.shape != (channel_count,) or donor_minimums.shape != (channel_count,):
        raise ValueError("Sector area targets must contain one value per channel.")
    areas = np.asarray(
        [np.sum(face_areas[grown == channel]) for channel in range(channel_count)],
        dtype=float,
    )
    seed_faces = set(int(seed) for seed in np.asarray(seeds, dtype=np.int64))
    edge_lengths = [entry[1] for neighbors in adjacency for entry in neighbors]
    edge_scale = max(float(np.median(edge_lengths)) if edge_lengths else 1.0, 1.0e-30)
    tolerance = max(1.0e-12 * float(np.sum(face_areas)), 1.0e-30)
    move_count = 0
    maximum_moves = max(len(grown) * channel_count, 1)
    for _round in range(max(channel_count * 8, 1)):
        deficient = np.flatnonzero(areas + tolerance < targets)
        if deficient.size == 0:
            cleaned = _merge_disconnected_label_islands(
                grown,
                np.asarray(seeds, dtype=np.int64),
                unary,
                adjacency,
            )
            if np.array_equal(cleaned, grown):
                return grown
            grown = cleaned
            areas = np.asarray(
                [np.sum(face_areas[grown == item]) for item in range(channel_count)],
                dtype=float,
            )
            continue
        channel = int(
            deficient[
                np.argmax((targets[deficient] - areas[deficient]) / np.maximum(targets[deficient], tolerance))
            ]
        )
        frontier: list[tuple[float, int]] = []
        queued: set[int] = set()

        def push_candidate(face: int) -> None:
            if face in queued or grown[face] == channel or face in seed_faces:
                return
            donor = int(grown[face])
            if areas[donor] - face_areas[face] < donor_minimums[donor] - tolerance:
                return
            before_boundary = 0.0
            after_boundary = 0.0
            for neighbor, shared_length, _travel in adjacency[face]:
                before_boundary += shared_length * float(grown[neighbor] != donor)
                after_boundary += shared_length * float(grown[neighbor] != channel)
            cost = float(unary[face, channel] - unary[face, donor])
            cost += (
                float(partition_smoothness)
                * (after_boundary - before_boundary)
                / edge_scale
            )
            heapq.heappush(frontier, (cost, int(face)))
            queued.add(int(face))

        for owned_face in np.flatnonzero(grown == channel):
            for neighbor, _shared, _travel in adjacency[int(owned_face)]:
                push_candidate(int(neighbor))
        while areas[channel] + tolerance < targets[channel] and frontier:
            _cost, face = heapq.heappop(frontier)
            if grown[face] == channel:
                continue
            donor = int(grown[face])
            if areas[donor] - face_areas[face] < donor_minimums[donor] - tolerance:
                continue
            grown[face] = channel
            areas[channel] += face_areas[face]
            areas[donor] -= face_areas[face]
            move_count += 1
            if move_count > maximum_moves:
                raise RuntimeError("Automatic connected sector area balancing did not converge.")
            for neighbor, _shared, _travel in adjacency[face]:
                push_candidate(int(neighbor))
        if areas[channel] + tolerance < targets[channel]:
            raise RuntimeError(
                f"Cannot grow channel {channel + 1} to its minimum area without violating "
                "another channel's area floor."
            )
    raise RuntimeError("Automatic connected sector area balancing did not converge.")


def _retain_connected_sector_cores(
    labels: np.ndarray,
    seeds: np.ndarray,
    guard: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
) -> np.ndarray:
    """Move clearance-created islands into the guard without reducing spacing."""

    connected_guard = np.asarray(guard, dtype=bool).copy()
    for channel, seed in enumerate(np.asarray(seeds, dtype=np.int64)):
        active = (labels == channel) & ~connected_guard
        components = _face_components(active, adjacency)
        if len(components) <= 1:
            continue
        retained = next(
            (component for component in components if int(seed) in component),
            None,
        )
        if retained is None:
            retained = max(
                components,
                key=lambda component: (
                    float(np.sum(face_areas[np.asarray(component, dtype=np.int64)])),
                    -min(component),
                ),
            )
        retained_faces = set(int(face) for face in retained)
        discarded = [
            int(face)
            for component in components
            for face in component
            if int(face) not in retained_faces
        ]
        connected_guard[np.asarray(discarded, dtype=np.int64)] = True
    return connected_guard


def _validate_hardened_partition(
    labels: np.ndarray,
    seeds: np.ndarray,
    guard: np.ndarray,
    active_masks: np.ndarray,
    face_areas: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
    options: AdaptiveSectorizationOptions,
) -> None:
    total_area = max(float(np.sum(face_areas)), 1.0e-30)
    for channel in range(active_masks.shape[1]):
        active = active_masks[:, channel]
        fraction = float(np.sum(face_areas[active]) / total_area)
        if np.count_nonzero(active) < 3 or fraction < options.minimum_sector_area_fraction:
            raise RuntimeError(
                f"Clearance erosion leaves channel {channel + 1} with only "
                f"{np.count_nonzero(active)} faces ({100.0 * fraction:.3f}% of area)."
            )
        components = _face_components(active, adjacency)
        if len(components) != 1:
            raise RuntimeError(
                f"Clearance erosion disconnects channel {channel + 1} into {len(components)} regions; "
                "reduce clearance or increase partition compactness."
            )
        if labels[int(seeds[channel])] != channel:
            raise RuntimeError("Automatic sector terminal ownership was lost during partition hardening.")
        if not active[int(seeds[channel])]:
            raise RuntimeError(
                f"Clearance erosion removed channel {channel + 1}'s boundary terminal access."
            )
    if np.any(np.sum(active_masks, axis=1) > 1):
        raise RuntimeError("Hardened sector masks overlap, which violates the single-layer contract.")
    if np.any(guard & np.any(active_masks, axis=1)):
        raise RuntimeError("Guard faces must not be active in any channel sector.")


def _face_adjacency(mesh: Mesh) -> list[list[tuple[int, float, float]]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    centroids = np.mean(vertices[faces], axis=1)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(int(first), int(second)), max(int(first), int(second)))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency: list[list[tuple[int, float, float]]] = [[] for _ in range(len(faces))]
    for (first_vertex, second_vertex), incident in edge_faces.items():
        if len(incident) < 2:
            continue
        shared_length = float(np.linalg.norm(vertices[first_vertex] - vertices[second_vertex]))
        for first_face, second_face in combinations(incident, 2):
            travel = float(np.linalg.norm(centroids[first_face] - centroids[second_face]))
            travel = max(travel, 0.5 * shared_length, 1.0e-15)
            adjacency[first_face].append((second_face, shared_length, travel))
            adjacency[second_face].append((first_face, shared_length, travel))
    return adjacency


def _graph_distances(
    adjacency: list[list[tuple[int, float, float]]],
    sources: Sequence[int],
) -> np.ndarray:
    distances = np.full(len(adjacency), np.inf, dtype=float)
    queue: list[tuple[float, int]] = []
    for source in sources:
        distances[int(source)] = 0.0
        heapq.heappush(queue, (0.0, int(source)))
    while queue:
        distance, face = heapq.heappop(queue)
        if distance > distances[face]:
            continue
        for neighbor, _shared, travel in adjacency[face]:
            candidate = distance + travel
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _face_components(
    mask: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
) -> list[list[int]]:
    remaining = set(int(value) for value in np.flatnonzero(mask))
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        stack = [start]
        while stack:
            face = stack.pop()
            for neighbor, _shared, _travel in adjacency[face]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _intersector_boundary_length(
    labels: np.ndarray,
    adjacency: list[list[tuple[int, float, float]]],
) -> float:
    total = 0.0
    for face, neighbors in enumerate(adjacency):
        for neighbor, shared_length, _travel in neighbors:
            if face < neighbor and labels[face] != labels[neighbor]:
                total += shared_length
    return float(total)


def _mesh_face_areas(mesh: Mesh) -> np.ndarray:
    triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("Adaptive sectorization requires finite positive triangle areas.")
    return areas


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        raise ValueError("Weighted activity quantile requires positive total face area.")
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _normalized_weights(values: Sequence[float] | None, count: int) -> np.ndarray:
    if values is None:
        return np.full(count, 1.0 / count, dtype=float)
    weights = np.asarray(values, dtype=float).reshape(-1)
    if len(weights) != count or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("scenario_weights must be finite, positive, and match the cohort.")
    return weights / np.sum(weights)


def _row_scaled_matrix(matrix: Any, scale: np.ndarray) -> Any:
    try:
        from scipy import sparse

        if sparse.issparse(matrix):
            return sparse.diags(np.asarray(scale, dtype=float), format="csr") @ matrix
    except ImportError:
        pass
    return np.asarray(scale, dtype=float)[:, None] * np.asarray(matrix, dtype=float)


def _matrix_rows(matrix: Any, mask: np.ndarray) -> Any:
    return matrix[np.asarray(mask, dtype=bool)]


def _readonly(values: Any, dtype: Any = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array
