"""Hardware-facing coil compiler utilities.

This module keeps user-facing coil design inputs physical: current, power,
wire spacing, conductor geometry, and manufacturability limits. Optimizer
choices such as SVD truncation and Tikhonov factors stay internal to the
compiler and are represented in diagnostics rather than as required user
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .config import CoilGenConfig
from .conversion import coil_design_problem_to_config
from .field_components import component_sample_values, field_component_values
from .metrics import dsv_target_mask
from .models import CoilGenResult
from .optimization import StreamFunctionOptimizationProblem
from .problem import CoilDesignProblem
from .solver import SolverResult


CompilerEffort = Literal["quick", "balanced", "thorough"]


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Physical hardware contract for a manufacturable coil.

    The stream-function compiler treats these as constraints. The user should
    provide values here instead of selecting regularization factors, SVD cutoffs,
    or contour counts.
    """

    wire_current_A: float
    max_power_W: float | None = None
    max_current_A: float | None = None
    wire_radius_m: float | None = None
    wire_width_m: float = 0.002
    wire_height_m: float = 0.002
    resistivity_ohm_m: float = 1.724e-8
    min_wire_spacing_m: float = 0.0
    min_bend_radius_m: float = 0.0
    max_turns: int | None = None
    keepout_region_count: int = 0
    label: str = "hardware"

    def __post_init__(self) -> None:
        _require_positive(self.wire_current_A, "wire_current_A")
        if self.max_power_W is not None:
            _require_positive(self.max_power_W, "max_power_W")
        if self.max_current_A is not None:
            _require_positive(self.max_current_A, "max_current_A")
            if self.wire_current_A > self.max_current_A * (1.0 + 1e-12):
                raise ValueError("wire_current_A cannot exceed max_current_A.")
        if self.wire_radius_m is not None:
            _require_positive(self.wire_radius_m, "wire_radius_m")
        _require_positive(self.wire_width_m, "wire_width_m")
        _require_positive(self.wire_height_m, "wire_height_m")
        _require_positive(self.resistivity_ohm_m, "resistivity_ohm_m")
        _require_nonnegative(self.min_wire_spacing_m, "min_wire_spacing_m")
        _require_nonnegative(self.min_bend_radius_m, "min_bend_radius_m")
        if self.max_turns is not None and int(self.max_turns) < 1:
            raise ValueError("max_turns must be positive when provided.")
        if int(self.keepout_region_count) < 0:
            raise ValueError("keepout_region_count must be non-negative.")

    @classmethod
    def from_problem(
        cls,
        problem: CoilDesignProblem,
        *,
        wire_current_A: float,
        max_power_W: float | None = None,
        max_current_A: float | None = None,
        min_wire_spacing_m: float | None = None,
        min_bend_radius_m: float | None = None,
        max_turns: int | None = None,
        label: str = "hardware",
    ) -> "HardwareSpec":
        """Build a hardware spec from a high-level problem's conductor/layout data."""

        conductor = problem.conductor
        spacing = max(float(conductor.min_turn_spacing_m), float(problem.optimization.min_turn_spacing_m))
        bend_radius = float(problem.layout.min_bend_radius_m)
        return cls(
            wire_current_A=wire_current_A,
            max_power_W=max_power_W,
            max_current_A=max_current_A,
            wire_radius_m=conductor.equivalent_round_radius_m,
            wire_width_m=float(conductor.width_m),
            wire_height_m=float(conductor.height_m),
            resistivity_ohm_m=float(conductor.resistivity_ohm_m),
            min_wire_spacing_m=spacing if min_wire_spacing_m is None else float(min_wire_spacing_m),
            min_bend_radius_m=bend_radius if min_bend_radius_m is None else float(min_bend_radius_m),
            max_turns=max_turns,
            keepout_region_count=int(problem.layout.keepout_region_count),
            label=label,
        )

    @property
    def conductor_area_m2(self) -> float:
        if self.wire_radius_m is not None:
            return float(math.pi * self.wire_radius_m * self.wire_radius_m)
        return float(self.wire_width_m * self.wire_height_m)

    @property
    def current_density_limit_A_per_m(self) -> float:
        if self.min_wire_spacing_m <= 0.0:
            return float("inf")
        return float(self.wire_current_A / self.min_wire_spacing_m)

    @property
    def contour_step_A(self) -> float:
        return float(self.wire_current_A)

    def resistance_ohm(self, length_m: float) -> float:
        return float(self.resistivity_ohm_m * float(length_m) / max(self.conductor_area_m2, 1e-30))

    def wire_power_W(self, length_m: float, *, current_A: float | None = None) -> float:
        current = self.wire_current_A if current_A is None else float(current_A)
        return float(current * current * self.resistance_ohm(length_m))

    def as_summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "wire_current_A": float(self.wire_current_A),
            "max_current_A": None if self.max_current_A is None else float(self.max_current_A),
            "max_power_W": None if self.max_power_W is None else float(self.max_power_W),
            "min_wire_spacing_m": float(self.min_wire_spacing_m),
            "current_density_limit_A_per_m": float(self.current_density_limit_A_per_m),
            "min_bend_radius_m": float(self.min_bend_radius_m),
            "max_turns": None if self.max_turns is None else int(self.max_turns),
            "conductor_area_m2": float(self.conductor_area_m2),
            "resistivity_ohm_m": float(self.resistivity_ohm_m),
        }


@dataclass(frozen=True, slots=True)
class HardwareCompilerOptions:
    """High-level compiler controls.

    These options describe computational effort and acceptance goals. They do
    not expose low-level optimizer parameters.
    """

    effort: CompilerEffort = "balanced"
    field_error_goal_percent: float | None = None
    allow_cvxpy: bool = True
    validate_discrete_layout: bool = True

    def __post_init__(self) -> None:
        if self.effort not in {"quick", "balanced", "thorough"}:
            raise ValueError("effort must be 'quick', 'balanced', or 'thorough'.")
        if self.field_error_goal_percent is not None:
            _require_positive(self.field_error_goal_percent, "field_error_goal_percent")


@dataclass(frozen=True, slots=True)
class ProblemAnalysis:
    target_sample_count: int
    unknown_count: int
    singular_values: tuple[float, ...]
    numerical_rank: int
    condition_number: float
    target_norm: float
    target_rms: float
    target_max_abs: float
    rank_limited_residual_norm: float
    rank_limited_residual_percent: float
    suggested_svd_ranks: tuple[int, ...]

    def as_summary(self) -> dict[str, Any]:
        return {
            "target_sample_count": int(self.target_sample_count),
            "unknown_count": int(self.unknown_count),
            "numerical_rank": int(self.numerical_rank),
            "condition_number": float(self.condition_number),
            "target_norm": float(self.target_norm),
            "target_rms": float(self.target_rms),
            "target_max_abs": float(self.target_max_abs),
            "rank_limited_residual_norm": float(self.rank_limited_residual_norm),
            "rank_limited_residual_percent": float(self.rank_limited_residual_percent),
            "suggested_svd_ranks": [int(rank) for rank in self.suggested_svd_ranks],
            "singular_value_max": float(self.singular_values[0]) if self.singular_values else 0.0,
            "singular_value_min_retained": float(self.singular_values[self.numerical_rank - 1])
            if self.numerical_rank > 0
            else 0.0,
        }


@dataclass(frozen=True, slots=True)
class ContinuousCandidateMetrics:
    field_rms_error_percent: float
    field_max_error_percent: float
    target_amplitude_fraction: float
    power_proxy_W: float
    current_density_peak_A_per_m: float
    predicted_min_wire_spacing_m: float
    stream_function_range_A: float
    predicted_turn_count: float
    hard_constraint_violations: Mapping[str, float] = field(default_factory=dict)
    constraint_margins: Mapping[str, float] = field(default_factory=dict)
    active_constraints: tuple[str, ...] = ()
    discrete_checks_required: tuple[str, ...] = ()

    @property
    def satisfies_hard_constraints(self) -> bool:
        return not self.hard_constraint_violations

    def as_summary(self) -> dict[str, Any]:
        return {
            "field_rms_error_percent": float(self.field_rms_error_percent),
            "field_max_error_percent": float(self.field_max_error_percent),
            "target_amplitude_fraction": float(self.target_amplitude_fraction),
            "power_proxy_W": float(self.power_proxy_W),
            "current_density_peak_A_per_m": float(self.current_density_peak_A_per_m),
            "predicted_min_wire_spacing_m": float(self.predicted_min_wire_spacing_m),
            "stream_function_range_A": float(self.stream_function_range_A),
            "predicted_turn_count": float(self.predicted_turn_count),
            "hard_constraint_violations": dict(self.hard_constraint_violations),
            "constraint_margins": dict(self.constraint_margins),
            "active_constraints": list(self.active_constraints),
            "discrete_checks_required": list(self.discrete_checks_required),
        }


@dataclass(frozen=True, slots=True)
class ContinuousCandidate:
    name: str
    strategy: str
    reduced_stream_function: np.ndarray
    full_stream_function: np.ndarray
    metrics: ContinuousCandidateMetrics
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.metrics.satisfies_hard_constraints

    def as_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "feasible": bool(self.feasible),
            "metrics": self.metrics.as_summary(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class HardwareCompilerReport:
    status: Literal["ok", "best_effort", "infeasible"]
    hardware: HardwareSpec
    analysis: ProblemAnalysis
    best_candidate: ContinuousCandidate
    candidates: tuple[ContinuousCandidate, ...]
    limiting_constraints: tuple[str, ...]
    feedback_log: tuple[str, ...]
    suggested_hardware_changes: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hardware": self.hardware.as_summary(),
            "analysis": self.analysis.as_summary(),
            "best_candidate": self.best_candidate.as_summary(),
            "candidate_count": int(len(self.candidates)),
            "limiting_constraints": list(self.limiting_constraints),
            "feedback_log": list(self.feedback_log),
            "suggested_hardware_changes": dict(self.suggested_hardware_changes),
        }


@dataclass(frozen=True, slots=True)
class DiscreteHardwareMetrics:
    hard_constraints_satisfied: bool
    power_W: float
    max_power_W: float | None
    current_A: float
    max_current_A: float | None
    wire_length_m: float
    resistance_ohm: float
    min_wire_spacing_m: float
    required_min_wire_spacing_m: float
    min_bend_radius_m: float
    required_min_bend_radius_m: float
    turn_count: int
    max_turns: int | None
    field_rms_error_percent: float
    field_max_error_percent: float
    violations: Mapping[str, float] = field(default_factory=dict)

    def as_summary(self) -> dict[str, Any]:
        return {
            "hard_constraints_satisfied": bool(self.hard_constraints_satisfied),
            "power_W": float(self.power_W),
            "max_power_W": None if self.max_power_W is None else float(self.max_power_W),
            "current_A": float(self.current_A),
            "max_current_A": None if self.max_current_A is None else float(self.max_current_A),
            "wire_length_m": float(self.wire_length_m),
            "resistance_ohm": float(self.resistance_ohm),
            "min_wire_spacing_m": float(self.min_wire_spacing_m),
            "required_min_wire_spacing_m": float(self.required_min_wire_spacing_m),
            "min_bend_radius_m": float(self.min_bend_radius_m),
            "required_min_bend_radius_m": float(self.required_min_bend_radius_m),
            "turn_count": int(self.turn_count),
            "max_turns": None if self.max_turns is None else int(self.max_turns),
            "field_rms_error_percent": float(self.field_rms_error_percent),
            "field_max_error_percent": float(self.field_max_error_percent),
            "violations": dict(self.violations),
        }


@dataclass(slots=True)
class HardwareConstrainedCoilDesignSolver:
    """Solver adapter that maps a high-level problem to hardware-aware settings."""

    hardware: HardwareSpec
    options: HardwareCompilerOptions = field(default_factory=HardwareCompilerOptions)
    name: str = "hardware_constrained_compiler"
    project_root: str | Path | None = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def solve(self, problem: CoilDesignProblem) -> SolverResult:
        config = compile_problem_to_hardware_config(
            problem,
            self.hardware,
            self.options,
            config_overrides=self.config_overrides,
        )
        diagnostics = {
            "hardware": self.hardware.as_summary(),
            "compiler_options": {
                "effort": self.options.effort,
                "field_error_goal_percent": self.options.field_error_goal_percent,
                "validate_discrete_layout": self.options.validate_discrete_layout,
            },
            "config_bridge": _hardware_config_bridge_summary(config, self.hardware),
        }
        if self.dry_run:
            return SolverResult(
                status="ok",
                solution=config,
                diagnostics=diagnostics,
                message="Converted hardware-constrained problem to CoilGenConfig.",
            )

        from .pipeline import run_coilgen

        result = run_coilgen(config, project_root=Path(self.project_root) if self.project_root is not None else None)
        if self.options.validate_discrete_layout:
            discrete_metrics = evaluate_discrete_hardware(result, self.hardware)
            diagnostics["discrete_hardware"] = discrete_metrics.as_summary()
            status = "ok" if discrete_metrics.hard_constraints_satisfied else "best_effort"
            message = "Pipeline run completed and hardware constraints were evaluated."
        else:
            status = "ok"
            message = "Pipeline run completed; discrete hardware validation was skipped."
        return SolverResult(status=status, solution=result, diagnostics=diagnostics, message=message)


def analyze_stream_function_problem(problem: StreamFunctionOptimizationProblem) -> ProblemAnalysis:
    """Analyze conditioning and rank-limited reachability of a reduced inverse problem."""

    matrix = np.asarray(problem.reduced_sensitivity, dtype=float)
    target = np.asarray(problem.target_z, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(target):
        raise ValueError("Reduced sensitivity matrix and target vector dimensions do not match.")
    if matrix.size == 0:
        singular_values = np.zeros(0, dtype=float)
        rank = 0
        residual = float(np.linalg.norm(target))
    else:
        u_matrix, singular_values, _vt = np.linalg.svd(matrix, full_matrices=False)
        rank = _numerical_rank(singular_values, matrix.shape)
        if rank > 0:
            projection = u_matrix[:, :rank] @ (u_matrix[:, :rank].T @ target)
            residual = float(np.linalg.norm(target - projection))
        else:
            residual = float(np.linalg.norm(target))
    target_norm = float(np.linalg.norm(target))
    target_rms = float(np.sqrt(np.mean(target * target))) if len(target) else 0.0
    target_max = float(np.max(np.abs(target))) if len(target) else 0.0
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank > 0 and singular_values[rank - 1] > 0.0
        else float("inf")
    )
    ranks = _suggest_svd_ranks(singular_values, rank)
    return ProblemAnalysis(
        target_sample_count=int(matrix.shape[0]),
        unknown_count=int(matrix.shape[1]),
        singular_values=tuple(float(value) for value in singular_values),
        numerical_rank=int(rank),
        condition_number=condition,
        target_norm=target_norm,
        target_rms=target_rms,
        target_max_abs=target_max,
        rank_limited_residual_norm=residual,
        rank_limited_residual_percent=100.0 * residual / max(target_norm, 1e-30),
        suggested_svd_ranks=ranks,
    )


def compile_stream_function_problem(
    problem: StreamFunctionOptimizationProblem,
    hardware: HardwareSpec,
    options: HardwareCompilerOptions | None = None,
) -> HardwareCompilerReport:
    """Generate, score, and diagnose stream-function candidates against hardware.

    This is the continuous compiler stage. It does not claim manufacturability
    until a discrete contour/layout result is separately validated.
    """

    active_options = HardwareCompilerOptions() if options is None else options
    analysis = analyze_stream_function_problem(problem)
    candidates: list[ContinuousCandidate] = []
    feedback: list[str] = []

    for name, strategy, vector, diagnostics in _candidate_vectors(problem, analysis, hardware, active_options):
        candidates.append(_build_candidate(problem, hardware, name, strategy, vector, diagnostics))

    if not candidates:
        raise RuntimeError("Hardware compiler did not generate any stream-function candidates.")

    feedback.extend(_feedback_messages(candidates, hardware))
    candidates.extend(_feedback_candidates(problem, hardware, active_options, candidates))
    candidates = _dedupe_candidates(candidates)
    best = min(candidates, key=lambda candidate: _candidate_score(candidate, active_options))
    limiting = _limiting_constraints(best, candidates)
    suggestions = _suggest_hardware_changes(best, candidates, hardware)
    status = _compiler_status(best, active_options)

    return HardwareCompilerReport(
        status=status,
        hardware=hardware,
        analysis=analysis,
        best_candidate=best,
        candidates=tuple(candidates),
        limiting_constraints=limiting,
        feedback_log=tuple(feedback),
        suggested_hardware_changes=suggestions,
    )


def compile_problem_to_hardware_config(
    problem: CoilDesignProblem,
    hardware: HardwareSpec,
    options: HardwareCompilerOptions | None = None,
    *,
    base_config: CoilGenConfig | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> CoilGenConfig:
    """Bridge a physical hardware contract into the current CoilGen pipeline."""

    active_options = HardwareCompilerOptions() if options is None else options
    problem.validate()
    effort = _effort_settings(active_options.effort)
    requested_levels = int(problem.layout.contour_count)
    if hardware.max_turns is not None:
        requested_levels = max(1, min(requested_levels, int(hardware.max_turns)))
    max_current = (
        float(hardware.max_current_A) if hardware.max_current_A is not None else float(hardware.wire_current_A)
    )
    wire_current = float(hardware.wire_current_A)

    overrides: dict[str, Any] = {
        "contour_discretization_method": "engineering" if hardware.min_wire_spacing_m > 0.0 else "level_count",
        "max_current_A": max_current,
        "max_wire_current_A": wire_current,
        "min_wire_distance_m": float(hardware.min_wire_spacing_m),
        "min_turn_distance": float(hardware.min_wire_spacing_m),
        "turn_spacing_density_percentile": 100.0,
        "turn_spacing_regularization_factor": 1.0 if hardware.min_wire_spacing_m > 0.0 else 0.0,
        "turn_spacing_max_iterations": int(effort["feedback_iterations"]),
        "regularization_auto_range": True,
        "regularization_auto_range_max_contour_step_A": wire_current,
        "field_evaluation_current_source": "stream_function",
        "levels": int(requested_levels),
        "engineering_contour_max_levels": int(requested_levels),
        "conductor_cross_section_width": float(hardware.wire_width_m),
        "conductor_cross_section_height": float(hardware.wire_height_m),
        "specific_conductivity_conductor": float(hardware.resistivity_ohm_m),
        "sweep_min_bend_radius": float(hardware.min_bend_radius_m),
        "layout_keepout_region_count": int(hardware.keepout_region_count),
        "contourability_validation": "auto",
        "contourability_repair_enabled": True,
        "contourability_repair_max_levels": max(int(requested_levels), int(effort["max_contour_repair_levels"])),
        "skip_calculation_min_winding_distance": False if hardware.min_wire_spacing_m > 0.0 else True,
    }
    if active_options.field_error_goal_percent is not None:
        overrides.update(
            {
                "sf_opt_method": "constrained_quadratic"
                if active_options.allow_cvxpy
                else "target_error_constrained_efficiency",
                "target_error_tolerance_percent": float(active_options.field_error_goal_percent),
                "target_error_infeasible_fallback": "minimum_residual",
                "target_error_infeasible_residual_margin_percent": 0.0,
            }
        )
    if config_overrides:
        overrides.update(dict(config_overrides))
    return coil_design_problem_to_config(problem, base_config=base_config, config_overrides=overrides)


def evaluate_discrete_hardware(result: CoilGenResult, hardware: HardwareSpec) -> DiscreteHardwareMetrics:
    """Evaluate an actual routed/discrete CoilGen result against hardware."""

    resistance = _result_resistance_ohm(result, hardware)
    length = _result_wire_length_m(result)
    current = float(hardware.wire_current_A)
    power = float(current * current * resistance)
    spacing = _result_min_spacing_m(result)
    bend_radius = _result_min_bend_radius_m(result)
    turn_count = int(sum(part.physical_turn_count for part in result.coil_parts))
    violations: dict[str, float] = {}
    physical_realization_available = all(
        part.contour_sparsification is None or bool(part.contour_sparsification.realized_turn_contours)
        for part in result.coil_parts
    )
    if not physical_realization_available:
        violations["physical_turn_realization"] = 1.0
    if hardware.max_current_A is not None and current > hardware.max_current_A * (1.0 + 1e-12):
        violations["current"] = float(current / max(hardware.max_current_A, 1e-30) - 1.0)
    if hardware.max_power_W is not None and power > hardware.max_power_W * (1.0 + 1e-9):
        violations["power"] = float(power / max(hardware.max_power_W, 1e-30) - 1.0)
    if hardware.min_wire_spacing_m > 0.0 and np.isfinite(spacing) and spacing < hardware.min_wire_spacing_m:
        violations["wire_spacing"] = float(hardware.min_wire_spacing_m / max(spacing, 1e-30) - 1.0)
    if hardware.min_bend_radius_m > 0.0 and np.isfinite(bend_radius) and bend_radius < hardware.min_bend_radius_m:
        violations["bend_radius"] = float(hardware.min_bend_radius_m / max(bend_radius, 1e-30) - 1.0)
    if hardware.max_turns is not None and turn_count > int(hardware.max_turns):
        violations["turn_count"] = float(turn_count / max(int(hardware.max_turns), 1) - 1.0)
    field_rms_error_percent, field_max_error_percent = _result_field_error_percent(result)

    return DiscreteHardwareMetrics(
        hard_constraints_satisfied=not violations,
        power_W=power,
        max_power_W=hardware.max_power_W,
        current_A=current,
        max_current_A=hardware.max_current_A,
        wire_length_m=length,
        resistance_ohm=resistance,
        min_wire_spacing_m=spacing,
        required_min_wire_spacing_m=float(hardware.min_wire_spacing_m),
        min_bend_radius_m=bend_radius,
        required_min_bend_radius_m=float(hardware.min_bend_radius_m),
        turn_count=turn_count,
        max_turns=hardware.max_turns,
        field_rms_error_percent=field_rms_error_percent,
        field_max_error_percent=field_max_error_percent,
        violations=violations,
    )


def _candidate_vectors(
    problem: StreamFunctionOptimizationProblem,
    analysis: ProblemAnalysis,
    hardware: HardwareSpec,
    options: HardwareCompilerOptions,
) -> list[tuple[str, str, np.ndarray, dict[str, Any]]]:
    matrix = np.asarray(problem.reduced_sensitivity, dtype=float)
    target = np.asarray(problem.target_z, dtype=float).reshape(-1)
    candidates: list[tuple[str, str, np.ndarray, dict[str, Any]]] = []

    minimum_error = _least_squares(matrix, target)
    candidates.append(("minimum_error", "least_squares_full_rank", minimum_error, {}))

    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    for rank in analysis.suggested_svd_ranks:
        if rank <= 0:
            continue
        vector = _truncated_svd_solution(u_matrix, singular_values, vt_matrix, target, int(rank))
        candidates.append(
            (
                f"svd_rank_{int(rank)}",
                "svd_reduced_reachable_modes",
                vector,
                {"retained_rank": int(rank)},
            )
        )

    resistance = _safe_quadratic(problem.reduced_resistance, matrix.shape[1])
    density_metric = _reduced_current_density_metric(problem)
    for lambda_value in _internal_regularization_lambdas(singular_values, options):
        vector = _quadratic_regularized_solution(matrix, target, resistance, lambda_value)
        candidates.append(
            (
                f"regularized_{lambda_value:.3e}",
                "internal_tikhonov_scan",
                vector,
                {"lambda": float(lambda_value)},
            )
        )

    if hardware.max_power_W is not None:
        vector, diagnostics = _power_limited_solution(matrix, target, resistance, hardware.max_power_W)
        candidates.append(("power_limited", "minimum_field_error_under_power_proxy", vector, diagnostics))

    if density_metric is not None and np.isfinite(hardware.current_density_limit_A_per_m):
        for weight in _internal_penalty_weights(options):
            vector = _quadratic_regularized_solution(matrix, target, weight * density_metric, 1.0)
            candidates.append(
                (
                    f"current_density_penalty_{weight:.3e}",
                    "current_density_peak_feedback",
                    vector,
                    {"density_penalty_weight": float(weight)},
                )
            )

    scaled, scale_diagnostics = _scaled_to_hardware(minimum_error, problem, hardware)
    if scale_diagnostics["scale"] < 1.0:
        candidates.append(
            (
                "scaled_to_hardware_constraints",
                "maximum_target_amplitude_under_hardware_scaling",
                scaled,
                scale_diagnostics,
            )
        )
    return candidates


def _build_candidate(
    problem: StreamFunctionOptimizationProblem,
    hardware: HardwareSpec,
    name: str,
    strategy: str,
    reduced_stream_function: np.ndarray,
    diagnostics: Mapping[str, Any],
) -> ContinuousCandidate:
    reduced = np.asarray(reduced_stream_function, dtype=float).reshape(-1)
    full = problem.expand(reduced)
    metrics = _continuous_metrics(problem, reduced, full, hardware)
    return ContinuousCandidate(
        name=name,
        strategy=strategy,
        reduced_stream_function=reduced,
        full_stream_function=full,
        metrics=metrics,
        diagnostics=dict(diagnostics),
    )


def _continuous_metrics(
    problem: StreamFunctionOptimizationProblem,
    reduced: np.ndarray,
    full: np.ndarray,
    hardware: HardwareSpec,
) -> ContinuousCandidateMetrics:
    matrix = np.asarray(problem.reduced_sensitivity, dtype=float)
    target = np.asarray(problem.target_z, dtype=float).reshape(-1)
    prediction = matrix @ reduced
    residual = prediction - target
    target_rms = float(np.sqrt(np.mean(target * target))) if len(target) else 0.0
    target_max = float(np.max(np.abs(target))) if len(target) else 0.0
    residual_rms = float(np.sqrt(np.mean(residual * residual))) if len(residual) else 0.0
    residual_max = float(np.max(np.abs(residual))) if len(residual) else 0.0
    field_rms_error = 100.0 * residual_rms / max(target_rms, 1e-30)
    field_max_error = 100.0 * residual_max / max(target_max, 1e-30)
    target_denominator = float(target @ target)
    amplitude = float((prediction @ target) / target_denominator) if target_denominator > 1e-30 else 0.0
    resistance = _safe_quadratic(problem.reduced_resistance, len(reduced))
    power = _quadratic_value(resistance, reduced)
    current_density_peak = _peak_current_density(problem, full)
    predicted_spacing = (
        float(hardware.wire_current_A / current_density_peak)
        if current_density_peak > 0.0 and np.isfinite(current_density_peak)
        else float("inf")
    )
    stream_range = float(np.ptp(full)) if len(full) else 0.0
    predicted_turn_count = float(stream_range / max(hardware.wire_current_A, 1e-30))
    violations, margins = _constraint_violations(
        hardware,
        power_proxy_W=power,
        current_density_peak_A_per_m=current_density_peak,
        predicted_min_wire_spacing_m=predicted_spacing,
        predicted_turn_count=predicted_turn_count,
    )
    active = tuple(name for name, margin in margins.items() if np.isfinite(margin) and margin <= 1.05)
    discrete_checks = []
    if hardware.min_bend_radius_m > 0.0:
        discrete_checks.append("bend_radius")
    if hardware.keepout_region_count > 0:
        discrete_checks.append("keepout_regions")
    return ContinuousCandidateMetrics(
        field_rms_error_percent=field_rms_error,
        field_max_error_percent=field_max_error,
        target_amplitude_fraction=amplitude,
        power_proxy_W=power,
        current_density_peak_A_per_m=current_density_peak,
        predicted_min_wire_spacing_m=predicted_spacing,
        stream_function_range_A=stream_range,
        predicted_turn_count=predicted_turn_count,
        hard_constraint_violations=violations,
        constraint_margins=margins,
        active_constraints=active,
        discrete_checks_required=tuple(discrete_checks),
    )


def _feedback_candidates(
    problem: StreamFunctionOptimizationProblem,
    hardware: HardwareSpec,
    options: HardwareCompilerOptions,
    candidates: list[ContinuousCandidate],
) -> list[ContinuousCandidate]:
    generated: list[ContinuousCandidate] = []
    best = min(candidates, key=lambda candidate: _candidate_score(candidate, options))
    violations = set(best.metrics.hard_constraint_violations)
    if violations:
        scaled, diagnostics = _scaled_to_hardware(best.reduced_stream_function, problem, hardware)
        if diagnostics["scale"] < 1.0:
            generated.append(
                _build_candidate(
                    problem,
                    hardware,
                    "feedback_scaled_best",
                    "active_constraint_scaling_feedback",
                    scaled,
                    diagnostics,
                )
            )
    return generated


def _feedback_messages(candidates: list[ContinuousCandidate], hardware: HardwareSpec) -> list[str]:
    messages: list[str] = []
    names = {name for candidate in candidates for name in candidate.metrics.hard_constraint_violations}
    if "power" in names:
        messages.append("Power proxy was active; generated a power-limited candidate family.")
    if "wire_spacing" in names:
        messages.append("Wire-spacing/current-density limit was active; generated density-penalized candidates.")
    if "turn_count" in names:
        messages.append("Turn-count range was active; generated hardware-scaled candidates.")
    if hardware.min_bend_radius_m > 0.0:
        messages.append("Bend radius requires discrete contour validation after wire extraction.")
    if hardware.keepout_region_count > 0:
        messages.append("Keepout constraints require discrete geometry validation after routing.")
    return messages


def _candidate_score(candidate: ContinuousCandidate, options: HardwareCompilerOptions) -> tuple[float, ...]:
    metrics = candidate.metrics
    violation_count = float(len(metrics.hard_constraint_violations))
    violation_mag = float(sum(max(value, 0.0) for value in metrics.hard_constraint_violations.values()))
    goal_penalty = 0.0
    if options.field_error_goal_percent is not None:
        goal_penalty = max(metrics.field_rms_error_percent - float(options.field_error_goal_percent), 0.0)
    return (
        violation_count,
        violation_mag,
        goal_penalty,
        float(metrics.field_rms_error_percent),
        float(metrics.field_max_error_percent),
        float(metrics.power_proxy_W),
        float(metrics.predicted_turn_count),
    )


def _compiler_status(
    best: ContinuousCandidate,
    options: HardwareCompilerOptions,
) -> Literal["ok", "best_effort", "infeasible"]:
    if not best.metrics.satisfies_hard_constraints:
        return "infeasible"
    field_goal = 5.0 if options.field_error_goal_percent is None else float(options.field_error_goal_percent)
    if best.metrics.field_rms_error_percent <= field_goal:
        return "ok"
    return "best_effort"


def _limiting_constraints(
    best: ContinuousCandidate,
    candidates: list[ContinuousCandidate],
) -> tuple[str, ...]:
    if best.metrics.hard_constraint_violations:
        return tuple(sorted(best.metrics.hard_constraint_violations))
    active = list(best.metrics.active_constraints)
    if active:
        return tuple(sorted(active))
    all_violations = sorted({name for candidate in candidates for name in candidate.metrics.hard_constraint_violations})
    return tuple(all_violations)


def _suggest_hardware_changes(
    best: ContinuousCandidate,
    candidates: list[ContinuousCandidate],
    hardware: HardwareSpec,
) -> dict[str, float]:
    suggestions: dict[str, float] = {}
    minimum_error = next((candidate for candidate in candidates if candidate.name == "minimum_error"), None)
    reference = minimum_error or best
    metrics = reference.metrics
    if hardware.max_power_W is not None and metrics.power_proxy_W > hardware.max_power_W:
        suggestions["max_power_W_for_unconstrained_field_proxy"] = float(metrics.power_proxy_W)
    if hardware.min_wire_spacing_m > 0.0 and metrics.current_density_peak_A_per_m > 0.0:
        suggestions["min_wire_spacing_m_for_unconstrained_field"] = float(
            hardware.wire_current_A / metrics.current_density_peak_A_per_m
        )
        suggestions["wire_current_A_for_requested_spacing"] = float(
            metrics.current_density_peak_A_per_m * hardware.min_wire_spacing_m
        )
    if hardware.max_turns is not None and metrics.predicted_turn_count > hardware.max_turns:
        suggestions["max_turns_for_unconstrained_field"] = float(math.ceil(metrics.predicted_turn_count))
    if best.metrics.target_amplitude_fraction < 0.999:
        suggestions["achievable_target_amplitude_fraction"] = float(max(best.metrics.target_amplitude_fraction, 0.0))
    return suggestions


def _constraint_violations(
    hardware: HardwareSpec,
    *,
    power_proxy_W: float,
    current_density_peak_A_per_m: float,
    predicted_min_wire_spacing_m: float,
    predicted_turn_count: float,
) -> tuple[dict[str, float], dict[str, float]]:
    violations: dict[str, float] = {}
    margins: dict[str, float] = {}
    if hardware.max_current_A is not None:
        margins["current"] = float(hardware.max_current_A / max(hardware.wire_current_A, 1e-30))
        if margins["current"] < 1.0 - 1e-12:
            violations["current"] = float(1.0 / max(margins["current"], 1e-30) - 1.0)
    if hardware.max_power_W is not None:
        margins["power"] = float(hardware.max_power_W / max(power_proxy_W, 1e-30))
        if power_proxy_W > hardware.max_power_W * (1.0 + 1e-9):
            violations["power"] = float(power_proxy_W / max(hardware.max_power_W, 1e-30) - 1.0)
    if hardware.min_wire_spacing_m > 0.0:
        margins["wire_spacing"] = float(predicted_min_wire_spacing_m / hardware.min_wire_spacing_m)
        limit = hardware.current_density_limit_A_per_m
        if current_density_peak_A_per_m > limit * (1.0 + 1e-9):
            violations["wire_spacing"] = float(current_density_peak_A_per_m / max(limit, 1e-30) - 1.0)
    if hardware.max_turns is not None:
        margins["turn_count"] = float(hardware.max_turns / max(predicted_turn_count, 1e-30))
        if predicted_turn_count > hardware.max_turns * (1.0 + 1e-9):
            violations["turn_count"] = float(predicted_turn_count / max(float(hardware.max_turns), 1e-30) - 1.0)
    return violations, margins


def _scaled_to_hardware(
    reduced: np.ndarray,
    problem: StreamFunctionOptimizationProblem,
    hardware: HardwareSpec,
) -> tuple[np.ndarray, dict[str, float | list[str]]]:
    full = problem.expand(np.asarray(reduced, dtype=float).reshape(-1))
    resistance = _safe_quadratic(problem.reduced_resistance, len(reduced))
    power = _quadratic_value(resistance, reduced)
    density_peak = _peak_current_density(problem, full)
    stream_range = float(np.ptp(full)) if len(full) else 0.0
    scale = 1.0
    active: list[str] = []
    if hardware.max_power_W is not None and power > hardware.max_power_W > 0.0:
        scale = min(scale, math.sqrt(hardware.max_power_W / max(power, 1e-30)))
        active.append("power")
    if (
        np.isfinite(hardware.current_density_limit_A_per_m)
        and density_peak > hardware.current_density_limit_A_per_m > 0.0
    ):
        scale = min(scale, hardware.current_density_limit_A_per_m / max(density_peak, 1e-30))
        active.append("wire_spacing")
    if hardware.max_turns is not None:
        max_range = float(hardware.max_turns) * hardware.wire_current_A
        if stream_range > max_range > 0.0:
            scale = min(scale, max_range / max(stream_range, 1e-30))
            active.append("turn_count")
    scale = max(min(scale, 1.0), 0.0)
    return np.asarray(reduced, dtype=float) * scale, {"scale": float(scale), "active_constraints": active}


def _least_squares(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    solution, *_rest = np.linalg.lstsq(matrix, target, rcond=None)
    return np.asarray(solution, dtype=float).reshape(-1)


def _truncated_svd_solution(
    u_matrix: np.ndarray,
    singular_values: np.ndarray,
    vt_matrix: np.ndarray,
    target: np.ndarray,
    rank: int,
) -> np.ndarray:
    retained = max(0, min(int(rank), len(singular_values)))
    if retained == 0:
        return np.zeros(vt_matrix.shape[1], dtype=float)
    coefficients = (u_matrix[:, :retained].T @ target) / singular_values[:retained]
    return vt_matrix[:retained, :].T @ coefficients


def _quadratic_regularized_solution(
    matrix: np.ndarray,
    target: np.ndarray,
    quadratic: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    unknowns = matrix.shape[1]
    ata = matrix.T @ matrix
    rhs = matrix.T @ target
    lhs = ata + float(lambda_value) * float(lambda_value) * _safe_quadratic(quadratic, unknowns)
    if lambda_value == 0.0 and quadratic is not None and quadratic.shape == (unknowns, unknowns):
        lhs = ata + _safe_psd_jitter(quadratic)
    return _solve_or_pinv(lhs, rhs)


def _power_limited_solution(
    matrix: np.ndarray,
    target: np.ndarray,
    resistance: np.ndarray,
    max_power_W: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    minimum_error = _least_squares(matrix, target)
    power = _quadratic_value(resistance, minimum_error)
    if power <= max_power_W:
        return minimum_error, {"power_limited": False, "lambda": 0.0}
    low = 0.0
    high = max(float(np.linalg.norm(matrix, ord=2)) if matrix.size else 1.0, 1.0)
    for _ in range(80):
        candidate = _quadratic_regularized_solution(matrix, target, resistance, high)
        if _quadratic_value(resistance, candidate) <= max_power_W:
            break
        high *= 2.0
    candidate = minimum_error
    for _ in range(80):
        mid = 0.5 * (low + high)
        trial = _quadratic_regularized_solution(matrix, target, resistance, mid)
        if _quadratic_value(resistance, trial) <= max_power_W:
            candidate = trial
            high = mid
        else:
            low = mid
    return candidate, {"power_limited": True, "lambda": float(high)}


def _reduced_current_density_metric(problem: StreamFunctionOptimizationProblem) -> np.ndarray | None:
    full_count = int(sum(problem.part_vertex_counts))
    if full_count <= 0 or not problem.part_current_density_mats:
        return None
    metrics: list[np.ndarray] = []
    for vertex_count, current_density_mat in zip(problem.part_vertex_counts, problem.part_current_density_mats):
        if current_density_mat is None:
            return None
        matrix = np.asarray(current_density_mat, dtype=float)
        if matrix.shape[0] != int(vertex_count):
            return None
        flattened = matrix.reshape(matrix.shape[0], -1)
        metrics.append(flattened @ flattened.T)
    full_metric = _block_diag(metrics)
    expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
    if expansion.shape[0] != full_metric.shape[0]:
        return None
    return _safe_quadratic(expansion.T @ full_metric @ expansion, expansion.shape[1])


def _peak_current_density(problem: StreamFunctionOptimizationProblem, full_stream_function: np.ndarray) -> float:
    offset = 0
    peaks: list[float] = []
    for vertex_count, current_density_mat in zip(problem.part_vertex_counts, problem.part_current_density_mats):
        part_stream = np.asarray(full_stream_function[offset : offset + int(vertex_count)], dtype=float)
        offset += int(vertex_count)
        if current_density_mat is None:
            continue
        density = np.tensordot(part_stream, np.asarray(current_density_mat, dtype=float), axes=(0, 0))
        density = np.asarray(density, dtype=float)
        if density.ndim == 1:
            norms = np.abs(density)
        else:
            norms = np.linalg.norm(density.reshape(density.shape[0], -1), axis=1)
        finite = norms[np.isfinite(norms)]
        if finite.size:
            peaks.append(float(np.max(finite)))
    return max(peaks) if peaks else 0.0


def _dedupe_candidates(candidates: list[ContinuousCandidate]) -> tuple[ContinuousCandidate, ...]:
    unique: list[ContinuousCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = tuple(np.round(candidate.reduced_stream_function, decimals=12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _hardware_config_bridge_summary(config: CoilGenConfig, hardware: HardwareSpec) -> dict[str, Any]:
    return {
        "sf_opt_method": config.sf_opt_method,
        "levels": int(config.levels),
        "min_turn_distance": float(config.min_turn_distance),
        "regularization_auto_range_max_contour_step_A": float(config.regularization_auto_range_max_contour_step_A),
        "field_evaluation_current_source": str(config.field_evaluation_current_source),
        "conductor_cross_section_width": float(config.conductor_cross_section_width),
        "conductor_cross_section_height": float(config.conductor_cross_section_height),
        "specific_conductivity_conductor": float(config.specific_conductivity_conductor),
        "hardware_max_power_W_report_only": hardware.max_power_W,
    }


def _result_resistance_ohm(result: CoilGenResult, hardware: HardwareSpec) -> float:
    if result.resistance is not None:
        values = np.asarray(result.resistance, dtype=float)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size:
            return float(np.sum(finite))
    part_resistance = [float(part.coil_resistance) for part in result.coil_parts if part.coil_resistance > 0.0]
    if part_resistance:
        return float(sum(part_resistance))
    return hardware.resistance_ohm(_result_wire_length_m(result))


def _result_wire_length_m(result: CoilGenResult) -> float:
    if result.coil_lengths is not None:
        values = np.asarray(result.coil_lengths, dtype=float)
        finite = values[np.isfinite(values) & (values >= 0.0)]
        if finite.size:
            return float(np.sum(finite))
    return float(sum(max(float(part.coil_length), 0.0) for part in result.coil_parts))


def _result_min_spacing_m(result: CoilGenResult) -> float:
    if any(
        part.contour_sparsification is not None and not part.contour_sparsification.realized_turn_contours
        for part in result.coil_parts
    ):
        return float("nan")
    distances = [
        float(part.minimal_contour_distance)
        for part in result.coil_parts
        if np.isfinite(part.minimal_contour_distance) and part.minimal_contour_distance > 0.0
    ]
    return min(distances) if distances else float("nan")


def _result_min_bend_radius_m(result: CoilGenResult) -> float:
    radii: list[float] = []
    for part in result.coil_parts:
        if part.wire_path is not None:
            radii.append(_polyline_min_bend_radius(part.wire_path.v))
        for contour in part.contour_lines:
            radii.append(_polyline_min_bend_radius(contour.v))
    finite = [radius for radius in radii if np.isfinite(radius) and radius > 0.0]
    return min(finite) if finite else float("nan")


def _result_field_error_percent(result: CoilGenResult) -> tuple[float, float]:
    component = getattr(result.input_data, "target_field_component", "Bz")
    b0_direction = getattr(result.input_data, "b_0_direction", (0.0, 0.0, 1.0))
    target = field_component_values(result.target_field.b, component, b0_direction=b0_direction)
    layout = field_component_values(result.field_by_layout, component, b0_direction=b0_direction)
    mask = component_sample_values(dsv_target_mask(result.target_field), component).astype(bool)
    target = np.asarray(target, dtype=float).reshape(-1)
    layout = np.asarray(layout, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(target) != len(layout) or len(mask) != len(target):
        return float("nan"), float("nan")
    finite = np.isfinite(target) & np.isfinite(layout) & mask
    if not np.any(finite):
        return float("nan"), float("nan")
    residual = layout[finite] - target[finite]
    target_values = target[finite]
    target_rms = float(np.sqrt(np.mean(target_values * target_values)))
    target_max = float(np.max(np.abs(target_values)))
    rms = float(np.sqrt(np.mean(residual * residual)))
    max_abs = float(np.max(np.abs(residual)))
    return 100.0 * rms / max(target_rms, 1e-30), 100.0 * max_abs / max(target_max, 1e-30)


def _polyline_min_bend_radius(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return float("nan")
    radii: list[float] = []
    for idx in range(1, len(pts) - 1):
        a = float(np.linalg.norm(pts[idx] - pts[idx - 1]))
        b = float(np.linalg.norm(pts[idx + 1] - pts[idx]))
        c = float(np.linalg.norm(pts[idx + 1] - pts[idx - 1]))
        area2 = float(np.linalg.norm(np.cross(pts[idx] - pts[idx - 1], pts[idx + 1] - pts[idx - 1])))
        if a <= 0.0 or b <= 0.0 or c <= 0.0 or area2 <= 1e-30:
            continue
        radii.append(float(a * b * c / (2.0 * area2)))
    return min(radii) if radii else float("inf")


def _effort_settings(effort: CompilerEffort) -> dict[str, int]:
    if effort == "quick":
        return {
            "regularization_count": 5,
            "penalty_count": 2,
            "feedback_iterations": 2,
            "max_contour_repair_levels": 40,
        }
    if effort == "thorough":
        return {
            "regularization_count": 17,
            "penalty_count": 7,
            "feedback_iterations": 6,
            "max_contour_repair_levels": 240,
        }
    return {"regularization_count": 9, "penalty_count": 4, "feedback_iterations": 4, "max_contour_repair_levels": 120}


def _internal_regularization_lambdas(singular_values: np.ndarray, options: HardwareCompilerOptions) -> np.ndarray:
    settings = _effort_settings(options.effort)
    finite = np.asarray(singular_values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return np.zeros(0, dtype=float)
    scale = float(finite[0])
    return np.geomspace(scale * 1e-8, scale * 1e2, int(settings["regularization_count"]))


def _internal_penalty_weights(options: HardwareCompilerOptions) -> np.ndarray:
    settings = _effort_settings(options.effort)
    return np.geomspace(1e-4, 1e4, int(settings["penalty_count"]))


def _numerical_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = np.finfo(float).eps * max(shape) * float(singular_values[0])
    return int(np.count_nonzero(singular_values > tolerance))


def _suggest_svd_ranks(singular_values: np.ndarray, rank: int) -> tuple[int, ...]:
    if rank <= 0:
        return ()
    retained = np.asarray(singular_values[:rank], dtype=float)
    energy = np.cumsum(retained * retained)
    total = float(energy[-1]) if energy.size else 0.0
    ranks = {1, rank}
    if total > 0.0:
        for fraction in (0.90, 0.99, 0.999):
            ranks.add(int(np.searchsorted(energy / total, fraction) + 1))
    return tuple(sorted(value for value in ranks if 1 <= value <= rank))


def _safe_quadratic(matrix: np.ndarray | None, size: int) -> np.ndarray:
    if matrix is None:
        return np.eye(size, dtype=float)
    quadratic = np.asarray(matrix, dtype=float)
    if quadratic.shape != (size, size) or not np.all(np.isfinite(quadratic)):
        return np.eye(size, dtype=float)
    return 0.5 * (quadratic + quadratic.T)


def _safe_psd_jitter(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    scale = max(float(np.max(np.abs(np.diag(matrix)))) if matrix.ndim == 2 else 0.0, 1.0)
    return 1e-12 * scale * np.eye(matrix.shape[0], dtype=float)


def _quadratic_value(matrix: np.ndarray, vector: np.ndarray) -> float:
    x = np.asarray(vector, dtype=float).reshape(-1)
    if matrix.shape != (len(x), len(x)):
        return float(np.dot(x, x))
    value = float(x @ matrix @ x)
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)) * float(np.dot(x, x)), 1.0)
    if value < 0.0 and abs(value) <= 1e-10 * scale:
        value = 0.0
    return float(max(value, 0.0))


def _solve_or_pinv(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs
    residual = np.linalg.norm(lhs @ solution - rhs) / max(np.linalg.norm(rhs), 1e-30)
    if np.all(np.isfinite(solution)) and residual <= 1e-7:
        return np.asarray(solution, dtype=float).reshape(-1)
    return np.linalg.pinv(lhs) @ rhs


def _block_diag(matrices: list[np.ndarray]) -> np.ndarray:
    rows = sum(matrix.shape[0] for matrix in matrices)
    cols = sum(matrix.shape[1] for matrix in matrices)
    out = np.zeros((rows, cols), dtype=float)
    row = 0
    col = 0
    for matrix in matrices:
        out[row : row + matrix.shape[0], col : col + matrix.shape[1]] = matrix
        row += matrix.shape[0]
        col += matrix.shape[1]
    return out


def _require_positive(value: float, name: str) -> None:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")


def _require_nonnegative(value: float, name: str) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
