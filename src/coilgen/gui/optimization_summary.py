"""Read-only description of the configured stream-function formulation.

This module reads settings only. It never assembles operators, discovers solver
installations, or imports Qt/CVXPY. Numerical normalization and actual backend
selection remain recorded solve diagnostics, rather than inferred UI results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class OptimizationSummary:
    title: str
    equation: str
    terms: tuple[str, ...]
    notes: tuple[str, ...]
    solver_text: str
    formulation: str
    is_draft: bool = False


_METHODS = {
    **dict.fromkeys(("tikkonov", "tikhonov", "tikhonov_regularized_least_squares"), "tikhonov"),
    **dict.fromkeys(
        (
            "target_error_constrained_efficiency",
            "efficiency_with_target_error_constraint",
            "fixed_error_efficiency",
            "minimum_resistance_subject_to_field_error",
        ),
        "target_error_constrained_efficiency",
    ),
    **dict.fromkeys(
        (
            "constrained_quadratic",
            "cvxpy_constrained",
            "minimum_resistance_constrained",
            "minimum_resistance_subject_to_field_error_socp",
            "direct_fixed_error",
            "socp",
            "second_order_cone",
        ),
        "constrained_quadratic",
    ),
}
_MODES = {
    **dict.fromkeys(("balanced", "resistance", "resistance_plus_penalties"), "balanced"),
    **dict.fromkeys(("smoothness_priority", "smoothest_feasible", "contourability_priority"), "smoothness_priority"),
    "minimax_current_density": "minimax_current_density",
}


def _key(value):
    return str(value).strip().lower().replace("-", "_")


def normalize_optimization_method(name) -> str:
    """Return the engine's canonical method key without changing the config."""
    key = _key(name)
    return _METHODS.get(key, key)


def normalize_field_budget_metric(name) -> str:
    key = _key(name)
    return {
        "l2": "norm",
        "l2_norm": "norm",
        "residual_norm": "norm",
        "root_mean_square": "rms",
        "standard_deviation": "std",
        "homogeneity": "std",
    }.get(key, key)


def optimization_field_budget_text(config, draft=None) -> str:
    """Describe the effective native DSV budget, without needing its operators."""
    overrides = {} if draft is None else draft

    def get(name, default=None):
        return overrides.get(name, _read(config, name, default))

    physics = get("research_physics")
    if physics is not None and _read(physics, "include_default_field_constraints", True) is False:
        return "Default field budgets disabled; explicit physical constraints remain."
    metric = normalize_field_budget_metric(get("target_error_constraint_metric", "norm"))
    limit = get("target_error_" + metric + "_limit_T", 0) if metric in {"rms", "std"} else 0
    if _positive(limit):
        text = f"{metric.upper()} ≤ {_text(float(limit) * 1e9)} nT; explicit limit overrides relative/absolute budgets"
    else:
        absolute = get("target_error_tolerance_absolute_T", 0)
        text = f"{metric.upper()} budget {_text(get('target_error_tolerance_percent', 5))}% of target reference"
        if _positive(absolute):
            text += f" + {_text(float(absolute) * 1e9)} nT"
        if metric == "norm":
            text += " (weighted residual norm)"
    if physics is None and normalize_optimization_method(get("sf_opt_method", "tikhonov")) == "tikhonov":
        text = "Post-fit check, not a hard solve constraint: " + text
    return text


def _positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _text(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.6g}"
    return str(value)


def _read(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _has_values(value):
    try:
        return len(value) > 0
    except TypeError:
        return False


def build_optimization_summary(config, draft=None) -> OptimizationSummary:
    """Describe a config or a parsed draft of config overrides, without mutation.

    ``draft`` replaces complete config fields, including ``research_physics``;
    it is not a partial merge of a nested physics specification. Unsupported
    choices remain visible and require normal config validation before solving.
    """
    overrides = {} if draft is None else draft
    if not isinstance(overrides, Mapping):
        raise TypeError("Optimization draft must be a mapping of config overrides.")

    def get(name, default=None):
        if name in overrides:
            return overrides[name]
        return _read(config, name, default)

    physics = get("research_physics")
    source = str(get("preoptimized_stream_function_file", "none")).strip()
    if not source or source.lower() == "none":
        source = str(get("sf_source_file", "none")).strip()
    if source and source.lower() != "none":
        imported_notes = [
            "Objective, regularization, solver and basis-selection settings do not run on this imported path. Mesh matching, field evaluation, layout and engineering review still apply.",
            f"Requested winding-current evaluation policy: {get('field_evaluation_current_source', 'target_fit')}; use saved source/current evidence when comparing field maps.",
        ]
        if physics is not None:
            imported_notes.append(
                "Incompatible settings: an imported stream-function source cannot be combined with research_physics; config validation must reject this combination."
            )
        return OptimizationSummary(
            "Imported stream function · inverse solve bypassed",
            "ψ is loaded from saved vertex values; no optimization cost is minimized.",
            (f"Source: {source}; field name {get('preoptimized_stream_function_field_name', 'stream_function')}.",),
            tuple(imported_notes),
            "No inverse solver is invoked.",
            "preoptimized",
            draft is not None,
        )
    method = normalize_optimization_method(get("sf_opt_method", "tikhonov"))
    mode = _MODES.get(
        _key(get("constrained_quadratic_objective_mode", "balanced")),
        _key(get("constrained_quadratic_objective_mode", "balanced")),
    )
    physical = physics is not None
    terms, notes = [], []
    smooth = get("stream_function_smoothness_factor", 0)
    spacing = get("turn_spacing_regularization_factor", 0)
    contour = _key(get("contour_discretization_method", "level_count"))
    distance = get("min_wire_distance_m", 0)
    if not _positive(distance):
        distance = get("min_turn_distance", 0)
    current = get("max_wire_current_A", 0)
    if not _positive(current):
        current = get("max_current_A", 0)
    engineering = contour in {
        "engineering",
        "current_spacing",
        "current_limited",
        "wire_current_spacing",
        "wire_current",
        "physical",
    }
    engineering |= contour in {"", "auto"} and _positive(distance) and _positive(current)
    threshold = distance if engineering else get("min_turn_distance", 0)
    pure_peak = not physical and method == "constrained_quadratic" and mode == "minimax_current_density"
    spacing_active = not physical and not pure_peak and _positive(threshold) and (_positive(spacing) or engineering)
    smooth_active = not physical and not pure_peak and _positive(smooth)
    penalties = " + η xᵀH̃x" if smooth_active else ""
    penalties += " + μ xᵀS̃x" if spacing_active else ""

    if physical:
        title, equation, physical_terms, physical_notes = _physical_objective(physics)
        terms.extend(physical_terms)
        notes.extend(physical_notes)
        formulation = "research_physical_convex"
        notes.append(
            "research_physics overrides sf_opt_method. Native cost, native objective mode, Tikhonov selection/factor, linear solver and native scan fallback are inactive."
        )
        if _positive(smooth) or _positive(spacing):
            notes.append(
                "Incompatible settings: physical objectives reject positive legacy smoothness or spacing factors; use an explicit physical mixture."
            )
    elif method == "tikhonov":
        title, formulation = "Native Tikhonov · soft field fit", method
        equation = "min_x ‖√W(Ax − b)‖₂² + λ² xᵀRx" + penalties
        cost = _key(get("stream_function_cost_function", "field_error_plus_resistance"))
        terms.append(
            "Native cost: field_error_plus_resistance; standard and least_squares_plus_resistance are aliases of the same cost."
        )
        if cost not in {"standard", "field_error_plus_resistance", "least_squares_plus_resistance"}:
            notes.append(f"Unsupported native cost '{cost}'; the engine does not implement this selection.")
        selection = _key(get("regularization_selection_method", "l_curve"))
        if selection == "fixed":
            terms.append(
                f"Fixed λ = max(factor, 0) × N_rows/N_reduced_unknowns; configured factor {_text(get('tikonov_reg_factor', 1))}."
            )
            notes.append("L-curve range, sampling and automatic-range controls are inactive with fixed regularization.")
        else:
            terms.append(
                f"Regularization selection: {'l_curve' if selection == 'auto' else selection}; factor {_text(get('tikonov_reg_factor', 1))} seeds the range/fallback, rather than fixing λ."
            )
            notes.append(
                "The selected λ, matrix conditioning and any L-curve fallback are known only after assembly/solve."
            )
            if selection not in {"l_curve", "auto"}:
                notes.append("Unsupported regularization selection; choose fixed, l_curve or auto before solving.")
            if _has_values(get("l_curve_regularization_factors", ())):
                notes.append(
                    "Positive finite explicit L-curve factors override the automatic factor range; invalid-only candidates fall back to the generated range."
                )
            terms.append(
                f"L-curve candidate solver: {get('l_curve_scan_method', 'normal_equations')}; automatic/transformed SVD falls back to normal equations when fixed extra penalties are present."
            )
        notes.append(
            "Field budgets are reviewed after this soft fit; they are not hard constraints in the Tikhonov objective. Native conic objective/backend settings are inactive."
        )
    elif method == "target_error_constrained_efficiency":
        title, formulation = "Native fixed-error efficiency · restricted scan", method
        equation = (
            "Choose lowest xᵀRx within field budgets along x(λ) = argmin_x [‖√W(Ax − b)‖₂² + λ²xᵀRx" + penalties + "]"
        )
        notes.append(
            "This searches the scalar Tikhonov family; it is not a general optimum with multiple independent constraints. Native conic objective/backend and Tikhonov fixed/L-curve selection are inactive during the primary scan."
        )
        notes.append(
            "The restricted scan always uses normal equations; l_curve_scan_method is inactive. An L-curve infeasible fallback can apply curvature smoothing to the already scanned candidates."
        )
        terms.append(
            f"Target-error scan: {_text(get('target_error_constraint_factor_count', 61))} candidates before configured retries/fallback; factor {_text(get('tikonov_reg_factor', 1))} seeds the scan."
        )
    elif method == "constrained_quadratic":
        formulation = method
        if mode == "minimax_current_density":
            title, equation = (
                "Native minimax · peak surface current",
                "min_x max_f ‖K_f x‖₂ [A/m], subject to configured field and hardware bounds",
            )
            if _positive(smooth) or _positive(spacing):
                notes.append(
                    "Incompatible settings: pure native minimax rejects positive smoothness/spacing factors; the automatic engineering spacing penalty is disabled."
                )
        else:
            title = "Native constrained quadratic · " + mode.replace("_", " ")
            alpha = (
                _text(get("constrained_quadratic_resistance_tiebreak_factor", 1e-3))
                if mode == "smoothness_priority"
                else "1"
            )
            equation = f"min_x {alpha} xᵀRx" + penalties + ", subject to configured field and hardware bounds"
            if mode == "smoothness_priority" and not smooth_active:
                notes.append(
                    "Incompatible settings: smoothness-priority mode requires a positive stream-function smoothness factor in config validation."
                )
            if mode == "smoothness_priority":
                notes.append(
                    "Smoothness priority is a weighted quadratic tradeoff, not a lexicographic optimum; its resistance tie-break factor must be finite and positive."
                )
            if mode not in {"balanced", "smoothness_priority"}:
                notes.append(
                    f"Unsupported native objective mode '{mode}'; configuration must be corrected before solving."
                )
        notes.append(
            "Native cost selector, Tikhonov factor/selection and L-curve controls are inactive for the direct conic solve; fallback scans may use separate configured factors."
        )
        notes.append(
            f"Native fallback policy: {get('constrained_quadratic_fallback_policy', 'target_error_scan')}; a fallback scan does not certify the full constrained optimum. Strict forbids fallback; diagnostic can return an unmet contract."
        )
        if _key(get("constrained_quadratic_fallback_policy", "target_error_scan")) != "strict":
            notes.append(
                f"A permitted native fallback scan uses linear solver {get('optimization_solver', 'auto')} and its dimensionless residual guard; some regularization/scan controls then become active."
            )
    else:
        title, equation, formulation = (
            "Unsupported stream-function formulation",
            "No implemented cost equation for this selection.",
            method,
        )
        notes.append(
            f"The engine does not implement sf_opt_method='{method}'; choose a supported formulation before solving."
        )

    if not physical:
        terms.append(
            f"Smoothness {'requested' if smooth_active else 'inactive'}: factor {_text(smooth)}; cotangent FEM biharmonic H̃ is scaled to the mean resistance-matrix diagonal."
        )
        factor = spacing if _positive(spacing) else 1 if engineering and spacing_active else 0
        terms.append(
            f"Spacing penalty {'requested' if spacing_active else 'inactive'}: threshold {_text(threshold)} m, effective starting factor {_text(factor)}; S̃ is scaled to the mean AᵀWA diagonal."
        )
        if spacing_active:
            notes.append(
                f"The spacing weight uses factor × (threshold/median mesh-edge length)² × growth^attempt; configured growth {_text(get('turn_spacing_penalty_growth', 10))}, maximum attempts {_text(get('turn_spacing_max_iterations', 4))}. Growth ≤1 falls back to 10; attempt count is at least 1. It is a soft current-density surrogate, not a routed-clearance certificate."
            )
        notes.append(
            "ψ = E x is the nodal stream function [A]; A maps reduced x to the selected field component [T], and R = EᵀR_sheetE. W contains effective nonnegative sample weights. xᵀRx is sheet Joule loss [W], not magnetic energy or wire resistance [Ω]. A composite native quadratic cost has numerical weighting, not a single physical unit."
        )

    component = get("target_field_component", "Bz")
    terms.append(
        f"Field component: {component}; DSV residual metric {get('target_error_constraint_metric', 'norm')}, budget grouping {get('target_error_constraint_mode', 'separate')}."
    )
    native_fields = not physical or _read(physics, "include_default_field_constraints", True)
    if (method != "tikhonov" or physical) and native_fields:
        notes.append(
            "Requested residual/suppression, gradient, current-density and current-step bounds depend on available sample/operator data; actual enforced bounds are recorded after compilation and solve."
        )
        metric = _key(get("target_error_constraint_metric", "norm"))
        if metric in {"std", "standard_deviation", "homogeneity"}:
            notes.append(
                "STD centers the residual and does not constrain its DC offset; an additional RMS/absolute-field bound is needed when that offset matters."
            )
        if _key(get("target_error_constraint_mode", "separate")) in {
            "combined",
            "weighted",
            "weighted_combined",
            "legacy",
        } and metric not in {"norm", "l2", "l2_norm", "residual_norm"}:
            notes.append("Incompatible field constraints: combined grouping supports the norm metric only.")
        terms.append("DSV budget: " + optimization_field_budget_text(config, overrides))
        notes.append("Positive explicit RMS/STD limits replace the relative and absolute budgets for their metric.")
    basis = _key(get("stream_function_basis", "vertex"))
    spectral = basis in {"cotangent_spectral_lowpass", "spectral_lowpass", "laplace_beltrami_lowpass"}
    terms.append(
        f"Basis: {basis}; {'spectral cutoff ' + _text(get('stream_function_spectral_cutoff_length_m', 0)) + ' m and mode limits requested' if spectral else 'spectral cutoff/policy/mode limits inactive'}. Boundary, gauge and any valid symmetry reduction apply before solving."
    )
    if physical:
        notes.append(
            "Physical acceptance uses research acceptance_relative_tolerance and acceptance_absolute_by_unit; native acceptance tolerance fields are inactive. Physical winding evaluation uses original stream-function current, without post-solve rescaling."
        )
    else:
        notes.append(
            f"Requested winding-current evaluation policy: {get('field_evaluation_current_source', 'target_fit')}; target_fit can refit current after contouring, so compare sources at their recorded current."
        )
        if method in {"target_error_constrained_efficiency", "constrained_quadratic"}:
            notes.append(
                f"Native numerical acceptance: relative {_text(get('target_error_constraint_acceptance_tolerance', 0))} (0 derives max(1e-8, 10 × conic tolerance)); absolute {_text(get('target_error_constraint_acceptance_absolute_T', 1e-7))} T is added in transformed residual coordinates. For RMS/STD, its physical metric allowance is divided by √Σweights."
            )
    if _key(get("mri_shim_metric_mode", "off")) != "off":
        notes.append(
            "MRI shim residual/CVaR/gradient objective weights belong to joint-shim optimization; they do not change this sheet objective."
        )
    notes.append(
        "Configured solver tolerances govern numerical termination; engineering acceptance and independent field validation remain separate."
    )
    solver = _solver_text(get, physical or method == "constrained_quadratic", physical)
    notes.extend(_solver_notes(physical or method == "constrained_quadratic"))
    if not physical and method not in {"tikhonov", "target_error_constrained_efficiency", "constrained_quadratic"}:
        solver = "No implemented inverse solver for this method selection."
    return OptimizationSummary(title, equation, tuple(terms), tuple(notes), solver, formulation, draft is not None)


def _physical_objective(physics):
    def get(name, default=None):
        return _read(physics, name, default)

    objective = get("objective", "power")
    expressions = {
        "power": "P(x) = xᵀRx [W]",
        "magnetic_energy": "E(x) = xᵀM_E x [J]",
        "minimax": "K_peak(x) = max_f ‖K_f x‖₂ [A/m]",
    }
    terms, notes = [], []
    if objective == "mixture":
        equation = "min_x Σ_j w_j F_j(x)/s_j, subject to configured physical bounds"
        weights, scales = get("objective_weights", {"power": 1, "magnetic_energy": 1}), get("objective_scales", {})
        if not isinstance(weights, Mapping) or not isinstance(scales, Mapping):
            weights, scales = {}, {}
            notes.append("Invalid mixture weight/scale mappings; correct the specification before solving.")
        for name, weight in weights.items():
            if _positive(weight):
                terms.append(
                    f"{expressions.get(name, name)}; weight {_text(weight)}; scale {_text(scales[name]) if name in scales else 'minimum-power feasible reference'}."
                )
            if name not in expressions or not _nonnegative(weight):
                notes.append(
                    f"Invalid mixture term '{name}': objective weights must be finite, nonnegative and supported."
                )
        if not any(_positive(weight) for weight in weights.values()):
            notes.append("Invalid mixture: at least one objective weight must be positive.")
        if any(name not in expressions or not _positive(scale) for name, scale in scales.items()):
            notes.append(
                "Invalid physical objective scales: use positive finite SI values for supported objective keys."
            )
        notes.append(
            "Mixture cost is dimensionless. Scales are positive SI values (power W, energy J, minimax A/m); missing scales are measured at a minimum-power feasible reference, then retained for tolerance sweeps."
        )
        notes.append(
            f"Reference field-tolerance scale: {_text(get('reference_tolerance_scale', 1))}; any missing scale must be strictly positive at the feasible reference, otherwise supply an explicit scale."
        )
    else:
        equation = (
            "min_x "
            + expressions.get(objective, f"unsupported objective '{objective}'")
            + ", subject to configured physical bounds"
        )
        notes.append(
            "A single physical objective is numerically normalized by a positive scale. The normalized solver objective is dimensionless; reconstructed physical P/E/K quantities carry the stated SI units. Mixture weights and reference-tolerance scale are inactive for a single objective."
        )
        if objective not in expressions:
            notes.append("The physical objective is unsupported; correct the specification before solving.")
    terms.append(
        "Physical field/mechanical/power/energy bounds are requested by research_physics; only configured finite scenarios are covered, rather than a continuous uncertainty set."
    )
    if get("include_default_field_constraints", True) is False:
        notes.append(
            "Default native field-budget constraints are disabled; only explicitly supplied physical field constraints/scenarios and hardware bounds remain."
        )
    energy = (
        get("evaluate_energy", False) or get("magnetic_energy_limit_J") is not None or objective == "magnetic_energy"
    )
    energy |= objective == "mixture" and _positive(weights.get("magnetic_energy", 0))
    terms.append(
        "Energy quadrature settings are "
        + (
            "active: energy evaluation/objective/bound requested."
            if energy
            else "inactive: no magnetic-energy calculation requested."
        )
    )
    return "Physical convex · " + str(objective).replace("_", " "), equation, terms, notes


def _nonnegative(value):
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def _solver_text(get, conic, physical=False):
    if conic:
        backend = str(get("constrained_optimization_solver", "auto")).upper()
        route = "auto: installed CLARABEL → ECOS → SCS" if backend == "AUTO" else backend
        return (
            f"CVXPY conic · {route}; tolerance {_text(get('constrained_optimization_tolerance', 1e-7))}; "
            f"iterations {_text(get('constrained_optimization_max_iterations', 20000))}; "
            f"requested threads {_text(get('constrained_optimization_threads', 0))} (0: backend/runtime default); "
            + ("verbose inactive." if physical else f"verbose {get('constrained_optimization_verbose', False)}.")
        )
    backend = _key(get("optimization_solver", "auto"))
    description = {
        "auto": "NumPy direct solve with pseudoinverse fallback on factorization/residual failure",
        "solve": "NumPy direct solve; factorization/residual failure raises",
        "pinv": "NumPy pseudoinverse; direct-solve residual guard inactive",
    }.get(backend, "Unsupported linear solver " + backend)
    return (
        description
        + f"; algebraic residual tolerance {_text(get('optimization_solver_residual_tolerance', 1e-8))}. Conic backend inactive."
    )


def _solver_notes(conic):
    if conic:
        return (
            "CLARABEL maps tolerance to gap/feasibility tolerances and supports max_threads capped to logical cores; ECOS maps abstol/reltol/feastol, SCS maps eps, and both ignore this thread setting. Effective tolerance is floored at 1e-12 and iterations at 1.",
            "Installed/selected backend is determined at solve time; explicit unsupported/uninstalled choices fail. OSQP is unsupported for this general SOC formulation.",
        )
    return (
        "The linear guard measures dimensionless normal-equation residual (tolerance floored at 0), not field error. Pseudoinverse has no configurable rcond and bypasses the direct-solve residual guard. Conic backend iteration/thread controls are inactive for the primary linear solve; conic tolerance can still set derived native acceptance when explicit relative acceptance is 0.",
    )
