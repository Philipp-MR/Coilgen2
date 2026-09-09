from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coilgen import (
    CoilDesignProblem,
    CoilGenConfig,
    CoilSupportSpec,
    ConductorSpec,
    Mesh,
    OptimizationSpec,
    SupportMesh,
    TargetFieldSpec,
)


CONTROL_VALUES = (
    ("constrained_objective_mode", "constrained_quadratic_objective_mode", "minimax_current_density"),
    ("conic_solver", "constrained_optimization_solver", "clarabel"),
    ("conic_tolerance", "constrained_optimization_tolerance", 2e-9),
    ("conic_max_iterations", "constrained_optimization_max_iterations", 1234),
    ("conic_fallback_policy", "constrained_quadratic_fallback_policy", "strict"),
    ("target_error_mode", "target_error_constraint_mode", "separate"),
    ("target_error_metric", "target_error_constraint_metric", "rms"),
    ("target_error_tolerance_absolute_T", "target_error_tolerance_absolute_T", 1e-9),
    ("target_error_constraint_acceptance_absolute_T", "target_error_constraint_acceptance_absolute_T", 1e-12),
    ("target_error_constraint_acceptance_tolerance", "target_error_constraint_acceptance_tolerance", 2e-8),
    ("target_error_rms_limit_T", "target_error_rms_limit_T", 3e-7),
    ("target_error_std_limit_T", "target_error_std_limit_T", 2e-7),
    ("suppression_rms_limit_T", "suppression_field_rms_limit_T", 4e-7),
    ("suppression_peak_limit_T", "suppression_field_peak_limit_T", 8e-7),
    ("surface_current_density_limit_A_per_m", "surface_current_density_limit_A_per_m", 1250.0),
    ("smoothness_factor", "stream_function_smoothness_factor", 0.03),
    ("basis", "stream_function_basis", "cotangent_spectral_lowpass"),
    ("spectral_cutoff_length_m", "stream_function_spectral_cutoff_length_m", 0.015),
    ("spectral_cutoff_policy", "stream_function_spectral_cutoff_policy", "minimum_modes"),
    ("spectral_min_modes", "stream_function_spectral_min_modes", 3),
    ("spectral_max_modes", "stream_function_spectral_max_modes", 40),
)


def _problem(
    optimization: OptimizationSpec | None = None,
    conductor: ConductorSpec | None = None,
) -> CoilDesignProblem:
    mesh = Mesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        faces=np.asarray([[0, 1, 2]]),
    )
    return CoilDesignProblem(
        target_field=TargetFieldSpec.from_arrays(np.asarray([[0.0, 0.0, 0.02]]), np.asarray([1e-5])),
        support=CoilSupportSpec("single_mesh", (SupportMesh(mesh, role="primary"),)),
        optimization=optimization or OptimizationSpec(),
        conductor=conductor or ConductorSpec(),
    )


def test_typed_optimization_controls_reach_pipeline_config() -> None:
    spec = OptimizationSpec(
        method="constrained_quadratic",
        **{name: value for name, _config_name, value in CONTROL_VALUES},
    )

    config = _problem(spec).to_config()

    assert config.sf_opt_method == "constrained_quadratic"
    for name, config_name, value in CONTROL_VALUES:
        assert getattr(config, config_name) == value, name


def test_unspecified_advanced_controls_preserve_base_config() -> None:
    base = CoilGenConfig(
        conductor_thickness=0.0007,
        **{config_name: value for _name, config_name, value in CONTROL_VALUES},
    )

    config = _problem().to_config(base_config=base)

    assert config.conductor_thickness == base.conductor_thickness
    for _name, config_name, value in CONTROL_VALUES:
        assert getattr(config, config_name) == value
        assert getattr(base, config_name) == value


def test_zero_controls_explicitly_disable_base_limits_and_regularization() -> None:
    spec = OptimizationSpec(
        target_error_tolerance_absolute_T=0.0,
        target_error_constraint_acceptance_absolute_T=0.0,
        target_error_rms_limit_T=0.0,
        target_error_std_limit_T=0.0,
        suppression_rms_limit_T=0.0,
        suppression_peak_limit_T=0.0,
        surface_current_density_limit_A_per_m=0.0,
        smoothness_factor=0.0,
        spectral_cutoff_length_m=0.0,
        spectral_max_modes=0,
    )
    base = CoilGenConfig(**{config_name: value for _name, config_name, value in CONTROL_VALUES})

    config = _problem(spec).to_config(base_config=base)

    for name, config_name, _value in CONTROL_VALUES:
        if getattr(spec, name) is not None:
            assert getattr(config, config_name) == 0.0, name


def test_numerical_acceptance_slack_is_separate_from_requested_field_error() -> None:
    spec = OptimizationSpec(
        method="constrained_quadratic",
        target_error_metric="rms",
        target_error_rms_limit_T=2e-8,
        target_error_tolerance_absolute_T=3e-9,
        target_error_constraint_acceptance_absolute_T=1e-12,
        target_error_constraint_acceptance_tolerance=1e-8,
    )

    config = _problem(spec).to_config()

    assert config.target_error_rms_limit_T == 2e-8
    assert config.target_error_tolerance_absolute_T == 3e-9
    assert config.target_error_constraint_acceptance_absolute_T == 1e-12
    assert config.target_error_constraint_acceptance_tolerance == 1e-8


def test_metadata_and_explicit_overrides_keep_precedence_over_typed_controls() -> None:
    spec = OptimizationSpec(
        constrained_objective_mode="minimax_current_density",
        conic_solver="clarabel",
        metadata={"config_overrides": {"constrained_optimization_solver": "scs", "sweep_insulation_thickness": 0.0003}},
    )
    problem = _problem(spec, ConductorSpec(insulation_thickness_m=0.0001, effective_sheet_thickness_m=0.0006))
    problem = replace(
        problem,
        evaluation=replace(problem.evaluation, metadata={"config_overrides": {"conductor_thickness": 0.0008}}),
    )

    metadata_config = problem.to_config()
    explicit_config = problem.to_config(
        config_overrides={
            "constrained_optimization_solver": "ecos",
            "constrained_quadratic_objective_mode": "balanced",
            "conductor_thickness": 0.001,
            "sweep_insulation_thickness": 0.0004,
        }
    )

    assert metadata_config.constrained_optimization_solver == "scs"
    assert metadata_config.constrained_quadratic_objective_mode == "minimax_current_density"
    assert metadata_config.conductor_thickness == 0.0008
    assert metadata_config.sweep_insulation_thickness == 0.0003
    assert explicit_config.constrained_optimization_solver == "ecos"
    assert explicit_config.constrained_quadratic_objective_mode == "balanced"
    assert explicit_config.conductor_thickness == 0.001
    assert explicit_config.sweep_insulation_thickness == 0.0004


def test_wire_dimensions_insulation_and_effective_sheet_thickness_remain_distinct() -> None:
    conductor = ConductorSpec(
        width_m=0.003,
        height_m=0.002,
        insulation_thickness_m=0.00015,
        effective_sheet_thickness_m=0.0004,
    )
    config = _problem(conductor=conductor).to_config(base_config=CoilGenConfig(conductor_thickness=0.005))

    assert config.conductor_cross_section_width == 0.003
    assert config.conductor_cross_section_height == 0.002
    assert config.sweep_insulation_thickness == 0.00015
    assert config.conductor_thickness == 0.0004
    assert conductor.cross_section_area_m2 == pytest.approx(6e-6)
    assert conductor.resistance_ohm(2.0) == pytest.approx(0.006)


@pytest.mark.parametrize("thickness", [0.0, -0.001, np.nan, np.inf, -np.inf])
def test_effective_sheet_thickness_requires_positive_finite_length(thickness: float) -> None:
    with pytest.raises(ValueError, match="Effective sheet thickness"):
        ConductorSpec(effective_sheet_thickness_m=thickness)
