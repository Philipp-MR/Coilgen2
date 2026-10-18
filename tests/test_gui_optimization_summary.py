"""Configured objective descriptions preserve numerical and physical scope."""

from dataclasses import FrozenInstanceError, dataclass
import builtins
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.gui.optimization_summary import build_optimization_summary, normalize_optimization_method


def text(summary):
    return "\n".join((summary.title, summary.equation, *summary.terms, *summary.notes, summary.solver_text))


def test_default_describes_weighted_soft_cost_and_numerical_units():
    summary = build_optimization_summary(CoilGenConfig())
    assert summary.formulation == "tikhonov"
    assert "‖√W(Ax − b)‖₂² + λ² xᵀRx" in summary.equation
    assert "soft fit" in text(summary)
    assert "not hard constraints" in text(summary)
    assert "sheet Joule loss [W]" in text(summary)
    assert "wire resistance [Ω]" in text(summary)
    assert "not field error" in text(summary)
    assert "Conic backend iteration/thread controls are inactive" in text(summary)
    assert "known only after assembly/solve" in text(summary)


def test_fixed_factor_is_squared_after_row_unknown_normalization():
    summary = build_optimization_summary({"regularization_selection_method": "fixed", "tikonov_reg_factor": 3})
    assert "λ² xᵀRx" in summary.equation
    assert "N_rows/N_reduced_unknowns" in text(summary)
    assert "configured factor 3" in text(summary)
    assert "L-curve range, sampling and automatic-range controls are inactive" in text(summary)
    assert "L-curve candidate solver" not in text(summary)


@pytest.mark.parametrize("cost", ["standard", "least_squares_plus_resistance", "field_error_plus_resistance"])
def test_native_cost_aliases_keep_exact_same_equation(cost):
    summary = build_optimization_summary({"stream_function_cost_function": cost})
    assert summary.equation == build_optimization_summary({}).equation
    assert "Unsupported native cost" not in text(summary)


def test_l_curve_explicit_numpy_candidates_and_unsupported_selection_are_visible():
    config = {"l_curve_regularization_factors": np.array([0.1, 1.0, 10.0]), "l_curve_scan_method": "auto"}
    summary = build_optimization_summary(config)
    assert "explicit L-curve factors override" in text(summary)
    assert "falls back to normal equations when fixed extra penalties" in text(summary)
    np.testing.assert_array_equal(config["l_curve_regularization_factors"], [0.1, 1.0, 10.0])
    assert "Unsupported regularization selection" in text(
        build_optimization_summary({"regularization_selection_method": "unknown"})
    )
    assert "Unsupported native cost" in text(build_optimization_summary({"stream_function_cost_function": "energy"}))


def test_scan_is_restricted_family_not_general_hard_constraint_optimum():
    summary = build_optimization_summary(
        {"sf_opt_method": "fixed_error_efficiency", "target_error_constraint_factor_count": 17}
    )
    assert summary.formulation == "target_error_constrained_efficiency"
    assert "Choose lowest xᵀRx" in summary.equation
    assert "17 candidates" in text(summary)
    assert "not a general optimum" in text(summary)
    assert "normal equations" in text(summary)
    assert "l_curve_scan_method is inactive" in text(summary)
    assert "curvature smoothing to the already scanned candidates" in text(summary)
    assert "NumPy" in summary.solver_text
    assert "0 derives max(1e-8, 10 × conic tolerance)" in text(summary)
    assert "physical metric allowance is divided by √Σweights" in text(summary)


@pytest.mark.parametrize("mode, coefficient", [("balanced", "1"), ("smoothest_feasible", "0.02")])
def test_direct_quadratic_coefficients_penalty_scope_and_fallback(mode, coefficient):
    summary = build_optimization_summary(
        {
            "sf_opt_method": "socp",
            "constrained_quadratic_objective_mode": mode,
            "stream_function_smoothness_factor": 2,
            "constrained_quadratic_resistance_tiebreak_factor": 0.02,
        }
    )
    assert summary.equation.startswith(f"min_x {coefficient} xᵀRx + η xᵀH̃x")
    assert "full constrained optimum" in text(summary)
    assert "fallback scan uses linear solver" in text(summary)
    assert "CLARABEL → ECOS → SCS" in summary.solver_text
    if mode == "smoothest_feasible":
        assert "not a lexicographic optimum" in text(summary)


def test_smoothness_priority_missing_positive_penalty_is_explicitly_invalid():
    summary = build_optimization_summary(
        {"sf_opt_method": "constrained_quadratic", "constrained_quadratic_objective_mode": "smoothness_priority"}
    )
    assert "requires a positive stream-function smoothness factor" in text(summary)


def test_engineering_spacing_defaults_to_one_but_minimax_disables_surrogate():
    config = {
        "sf_opt_method": "constrained_quadratic",
        "contour_discretization_method": "engineering",
        "min_wire_distance_m": 0.003,
        "max_wire_current_A": 2,
        "turn_spacing_penalty_growth": 7,
        "turn_spacing_max_iterations": 3,
    }
    summary = build_optimization_summary(config)
    assert "+ μ xᵀS̃x" in summary.equation
    assert "effective starting factor 1" in text(summary)
    assert "threshold/median mesh-edge length" in text(summary)
    assert "configured growth 7, maximum attempts 3" in text(summary)
    assert "not a routed-clearance certificate" in text(summary)
    peak = build_optimization_summary(config, {"constrained_quadratic_objective_mode": "minimax_current_density"})
    assert "max_f ‖K_f x‖₂ [A/m]" in peak.equation
    assert "Spacing penalty inactive" in text(peak)
    assert "μ" not in peak.equation
    bad = build_optimization_summary(
        config,
        {"constrained_quadratic_objective_mode": "minimax_current_density", "turn_spacing_regularization_factor": 1},
    )
    assert "rejects positive smoothness/spacing factors" in text(bad)


@pytest.mark.parametrize("objective, unit", [("power", "[W]"), ("magnetic_energy", "[J]"), ("minimax", "[A/m]")])
def test_physical_precedence_single_objective_units_and_normalized_solver_value(objective, unit):
    summary = build_optimization_summary(
        {
            "sf_opt_method": "tikhonov",
            "research_physics": {"objective": objective},
            "constrained_optimization_verbose": True,
        }
    )
    assert summary.formulation == "research_physical_convex"
    assert unit in summary.equation
    assert "numerically normalized" in text(summary)
    assert "research_physics overrides sf_opt_method" in text(summary)
    assert "native scan fallback are inactive" in text(summary)
    assert "verbose inactive" in summary.solver_text
    assert "normalized solver objective is dimensionless" in text(summary)
    assert "original stream-function current" in text(summary)
    assert "native acceptance tolerance fields are inactive" in text(summary)


def test_physical_mixture_scale_units_zero_energy_weight_and_reference_scope():
    physics = {
        "objective": "mixture",
        "objective_weights": {"power": 2, "magnetic_energy": 0, "minimax": 1},
        "objective_scales": {"power": 0.3},
        "reference_tolerance_scale": 0.8,
    }
    summary = build_optimization_summary({"research_physics": physics})
    assert "Σ_j w_j F_j(x)/s_j" in summary.equation
    assert "weight 2; scale 0.3" in text(summary)
    assert "minimum-power feasible reference" in text(summary)
    assert "Reference field-tolerance scale: 0.8" in text(summary)
    assert "strictly positive at the feasible reference" in text(summary)
    assert "Energy quadrature settings are inactive" in text(summary)
    assert not any("E(x)" in term for term in summary.terms)
    assert "Energy quadrature settings are active" in text(
        build_optimization_summary({"research_physics": physics | {"evaluate_energy": True}})
    )


def test_default_physical_mixture_requests_energy_but_explicit_power_only_does_not():
    assert "Energy quadrature settings are active" in text(
        build_optimization_summary({"research_physics": {"objective": "mixture"}})
    )
    assert "Energy quadrature settings are inactive" in text(
        build_optimization_summary({"research_physics": {"objective": "mixture", "objective_weights": {"power": 1}}})
    )


@dataclass(frozen=True, slots=True)
class SlottedPhysics:
    objective: str = "magnetic_energy"
    include_default_field_constraints: bool = False


def test_slotted_spec_is_read_without_asdict_or_losing_objective():
    summary = build_optimization_summary({"research_physics": SlottedPhysics()})
    assert "[J]" in summary.equation
    assert "Default native field-budget constraints are disabled" in text(summary)
    assert not any(term.startswith("DSV budget:") for term in summary.terms)


def test_physical_legacy_penalty_and_invalid_mixture_warnings_do_not_claim_execution():
    summary = build_optimization_summary(
        {
            "research_physics": {
                "objective": "mixture",
                "objective_weights": {"power": 0},
                "objective_scales": {"power": -1},
            },
            "stream_function_smoothness_factor": 1,
        }
    )
    assert "at least one objective weight must be positive" in text(summary)
    assert "Invalid physical objective scales" in text(summary)
    assert "reject positive legacy smoothness or spacing factors" in text(summary)
    assert "η" not in summary.equation


def test_std_alias_and_combined_dependency_preserve_mean_free_limitation():
    summary = build_optimization_summary(
        {
            "sf_opt_method": "constrained_quadratic",
            "target_error_constraint_metric": "homogeneity",
            "target_error_constraint_mode": "legacy",
        }
    )
    assert "does not constrain its DC offset" in text(summary)
    assert "combined grouping supports the norm metric only" in text(summary)
    assert "explicit RMS/STD limits replace" in text(summary)


def test_spectral_controls_active_only_for_spectral_basis_and_shim_joint_scope():
    summary = build_optimization_summary(
        {
            "stream_function_basis": "spectral_lowpass",
            "stream_function_spectral_cutoff_length_m": 0.05,
            "mri_shim_metric_mode": "optimize",
        }
    )
    assert "spectral cutoff 0.05 m" in text(summary)
    assert "joint-shim optimization" in text(summary)
    assert "spectral cutoff/policy/mode limits inactive" in text(build_optimization_summary({}))


def test_solver_settings_are_backend_specific_and_do_not_imply_installation():
    summary = build_optimization_summary(
        {
            "sf_opt_method": "socp",
            "constrained_optimization_solver": "SCS",
            "constrained_optimization_tolerance": 0,
            "constrained_optimization_max_iterations": 0,
            "constrained_optimization_threads": 8,
        }
    )
    assert "SCS" in summary.solver_text
    assert "requested threads 8" in summary.solver_text
    assert "both ignore this thread setting" in text(summary)
    assert "floored at 1e-12 and iterations at 1" in text(summary)
    assert "determined at solve time" in text(summary)
    assert "OSQP is unsupported" in text(summary)
    assert len(summary.solver_text) < 260
    pinv = build_optimization_summary({"optimization_solver": "pinv", "optimization_solver_residual_tolerance": 0.2})
    assert "direct-solve residual guard inactive" in pinv.solver_text
    assert "no configurable rcond" in text(pinv)


def test_draft_is_complete_field_overlay_immutable_and_does_not_retain_inputs():
    config = {
        "sf_opt_method": "tikhonov",
        "research_physics": {"objective": "mixture", "objective_weights": {"power": 2}},
    }
    draft = {"research_physics": {"objective": "minimax"}}
    summary = build_optimization_summary(config, draft)
    assert summary.is_draft
    assert "[A/m]" in summary.equation
    assert "weight 2" not in text(summary)
    config["research_physics"]["objective_weights"]["power"] = 99
    draft["research_physics"]["objective"] = "power"
    assert "[A/m]" in summary.equation
    with pytest.raises(FrozenInstanceError):
        summary.title = "changed"
    assert build_optimization_summary(config, {"research_physics": None}).formulation == "tikhonov"
    with pytest.raises(TypeError, match="mapping"):
        build_optimization_summary(config, [])


def test_imported_source_bypasses_solver_and_physical_conflict_remains_invalid():
    summary = build_optimization_summary({"sf_source_file": "external.npz", "research_physics": {"objective": "power"}})
    assert summary.formulation == "preoptimized"
    assert "inverse solve bypassed" in summary.title
    assert "No inverse solver is invoked" in summary.solver_text
    assert "cannot be combined with research_physics" in text(summary)
    assert build_optimization_summary({"sf_source_file": "NONE"}).formulation == "tikhonov"


def test_unknown_formulation_and_modes_are_reported_not_invented():
    assert (
        "Unsupported stream-function formulation"
        in build_optimization_summary({"sf_opt_method": "not_implemented"}).title
    )
    assert "Unsupported native objective mode" in text(
        build_optimization_summary({"sf_opt_method": "socp", "constrained_quadratic_objective_mode": "not_implemented"})
    )


@pytest.mark.parametrize(
    "name, canonical",
    [
        ("Tikkonov", "tikhonov"),
        ("direct-fixed-error", "constrained_quadratic"),
        ("socp", "constrained_quadratic"),
        ("fixed_error_efficiency", "target_error_constrained_efficiency"),
        ("unimplemented", "unimplemented"),
    ],
)
def test_public_normalizer_matches_execution_aliases_without_overwriting_choices(name, canonical):
    config = {"sf_opt_method": name}
    assert normalize_optimization_method(name) == canonical
    assert build_optimization_summary(config).formulation == canonical
    assert config["sf_opt_method"] == name


def test_isolated_module_import_and_build_need_no_qt_cvxpy_numpy_or_solver(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "src/coilgen/gui/optimization_summary.py"
    original_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        if name.split(".")[0] in {"PySide6", "cvxpy", "numpy", "coilgen"}:
            raise AssertionError("Summary must not import numerical or Qt implementation: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted_import)
    spec = importlib.util.spec_from_file_location("_isolated_optimization_summary", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    summary = module.build_optimization_summary({"research_physics": {"objective": "mixture"}})
    assert summary.formulation == "research_physical_convex"


@pytest.mark.parametrize("metric", ["rms", "standard_deviation"])
def test_summary_identifies_effective_explicit_field_limit_instead_of_inactive_percentage(metric):
    config = {
        "sf_opt_method": "socp",
        "target_error_constraint_metric": metric,
        "target_error_rms_limit_T": 1e-7,
        "target_error_std_limit_T": 1e-7,
        "target_error_tolerance_percent": 7,
    }
    summary = build_optimization_summary(config)
    budget = next(term for term in summary.terms if term.startswith("DSV budget:"))
    assert "100 nT" in budget and "overrides" in budget and "7%" not in budget
