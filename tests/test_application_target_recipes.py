from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application.acceptance import (
    field_acceptance_slack_overrides,
    field_acceptance_slack_T,
    independent_field_acceptance_slack_T,
)
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.targets import (
    SolidHarmonicTarget,
    recognize_solid_harmonic_target,
    solid_harmonic_target,
)
from coilgen.models import TargetField
from coilgen.research.convex import ResearchPhysicsSpec
from coilgen.research.validation import field_budget_check
from coilgen.spherical_harmonics import solid_spherical_harmonic_basis
from coilgen.target import define_target_field, evaluate_field_expression, solid_harmonic


@pytest.mark.parametrize(
    "degree,order", [(degree, order) for degree in range(5) for order in range(-degree, degree + 1)]
)
def test_analytic_harmonic_matches_existing_basis_for_every_mode_through_degree_four(degree, order):
    center = np.array([0.017, -0.012, 0.034])
    rng = np.random.default_rng(721)
    points = np.vstack([rng.normal(size=(40, 3)) * 0.055 + center, center, center + np.eye(3) * 0.07])
    radius, coefficient = 0.075, -8.4e-6
    basis, terms, _, _ = solid_spherical_harmonic_basis(points, degree, center=center, radius=radius)
    column = next(index for index, term in enumerate(terms) if (term.degree, term.order) == (degree, order))
    recipe = solid_harmonic_target(degree, order, coefficient, radius, center)
    actual = evaluate_field_expression(recipe.expression, points)
    np.testing.assert_allclose(actual, coefficient * basis[:, column], rtol=1e-13, atol=1e-20)
    assert np.isfinite(actual).all()


def test_z2_coefficient_and_condon_shortley_sign_are_explicit():
    points = np.array([[0.06, 0, 0], [0, 0.06, 0], [0, 0, 0.06], [0, 0, 0]])
    coefficient, radius = 2e-6, 0.06
    z2 = evaluate_field_expression(solid_harmonic_target(2, 0, coefficient, radius).expression, points)
    expected = (
        coefficient
        * np.sqrt(5 / (16 * np.pi))
        * (2 * points[:, 2] ** 2 - points[:, 0] ** 2 - points[:, 1] ** 2)
        / radius**2
    )
    np.testing.assert_allclose(z2, expected, rtol=1e-14, atol=1e-21)
    assert z2[2] == pytest.approx(0.6307831305050401 * coefficient)
    assert solid_harmonic(1, 1, radius, 0, 0, radius) == pytest.approx(-np.sqrt(3 / (4 * np.pi)))
    assert solid_harmonic(1, -1, 0, radius, 0, radius) == pytest.approx(-np.sqrt(3 / (4 * np.pi)))


def test_explicit_geometry_remains_invariant_under_translation_and_common_length_scale():
    points = np.array([[0.02, 0.01, -0.03], [-0.04, 0.02, 0.06], [0, 0, 0]])
    center = np.array([0.005, 0.003, -0.01])
    offset = np.array([0.13, -0.2, 0.07])
    original = solid_harmonic_target(4, -3, 3e-6, 0.07, center)
    scaled = solid_harmonic_target(4, -3, 3e-6, 0.07 * 2.5, center * 2.5 + offset)
    np.testing.assert_allclose(
        evaluate_field_expression(original.expression, points),
        evaluate_field_expression(scaled.expression, points * 2.5 + offset),
        rtol=1e-13,
        atol=1e-20,
    )


def test_recipe_roundtrip_reopens_exact_coefficient_and_prevents_gradient_rescaling(tmp_path):
    center = [0.011, -0.027, 0.04]
    recipe = SolidHarmonicTarget(2, -1, -3.5e-6, 0.08, center)
    center[0] = 9
    assert recipe.center_m == (0.011, -0.027, 0.04)
    config = preset_config()
    project = ProjectDocument.new("Absolute harmonic", config).update_config(recipe.to_config_overrides())
    project.save(tmp_path / "harmonic.coilgen")
    reopened = ProjectDocument.load(tmp_path / "harmonic.coilgen")
    assert recognize_solid_harmonic_target(reopened.config.field_shape_function) == recipe
    assert project.fingerprint == reopened.fingerprint
    assert reopened.config.target_gradient_normalization == "off"
    assert reopened.config.target_region_radius == config.target_region_radius
    target = define_target_field([], None, None, reopened.config)
    np.testing.assert_allclose(target.b[:, 2], evaluate_field_expression(recipe.expression, target.coords))
    assert target.target_diagnostics["target_gradient_normalization_scale"] == 1
    assert reopened.config.design_context == "mr_shim"


def test_recipe_clears_imported_target_precedence_without_redefining_roi():
    changes = solid_harmonic_target(0, 0, 1e-6, 0.04).to_config_overrides()
    assert changes["target_field_definition_file"] == "none"
    assert "target_mesh_file" not in changes
    assert "target_region_radius" not in changes
    assert "secondary_target_mesh_file" not in changes


@pytest.mark.parametrize(
    "changes",
    [
        {"degree": True},
        {"degree": 2.0},
        {"degree": -1},
        {"degree": 17},
        {"order": False},
        {"order": 3},
        {"order": -3},
        {"coefficient_T": np.inf},
        {"coefficient_T": True},
        {"coefficient_T": 2j},
        {"reference_radius_m": 0},
        {"reference_radius_m": -0.01},
        {"reference_radius_m": np.nan},
        {"reference_radius_m": True},
        {"center_m": [0, 0]},
        {"center_m": [0, 0, np.inf]},
        {"center_m": [0, 0, True]},
        {"center_m": None},
    ],
)
def test_recipe_rejects_ambiguous_or_nonfinite_parameters(changes):
    values = dict(degree=2, order=0, coefficient_T=1e-6, reference_radius_m=0.06)
    with pytest.raises(ValueError):
        SolidHarmonicTarget(**(values | changes))


@pytest.mark.parametrize(
    "expression",
    [
        "x*y",
        "solid_harmonic(2,0,x,y,z,0.06,0,0,0)",
        "1e-6*solid_harmonic(2,0,y,x,z,0.06,0,0,0)",
        "1e-6*solid_harmonic(2,0,x,y,z,r,0,0,0)",
        "1e-6*solid_harmonic(2,3,x,y,z,0.06,0,0,0)",
        "1e-6*solid_harmonic(2,0,x,y,z,0.06,0,0,0)+1",
        "1e-6*solid_harmonic(2,0,x,y,z,0.06,0,0,0,extra=1)",
        "__import__('os').getcwd()*solid_harmonic(2,0,x,y,z,0.06,0,0,0)",
        "1e-6*solid_harmonic(2,0,x,y,z,0.06,0,0,0).__class__",
        "inf*solid_harmonic(2,0,x,y,z,0.06,0,0,0)",
        "True*solid_harmonic(2,0,x,y,z,0.06,0,0,0)",
        "(" * 5000,
    ],
)
def test_recognizer_never_executes_or_reinterprets_other_expressions(expression):
    assert recognize_solid_harmonic_target(expression) is None


def test_callable_broadcasts_and_validates_parameters_and_coordinates():
    assert solid_harmonic(0, 0, 0, 0, 0, 0.06) == pytest.approx(1 / np.sqrt(4 * np.pi))
    assert solid_harmonic(2, 0, np.empty(0), np.empty(0), np.empty(0), 0.06).shape == (0,)
    with pytest.raises(ValueError, match="degree"):
        solid_harmonic(2.0, 0, 0, 0, 0, 0.06)
    with pytest.raises(ValueError, match="radius"):
        solid_harmonic(2, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="finite real"):
        solid_harmonic(2, 0, np.nan, 0, 0, 0.06)
    with pytest.raises(ValueError, match="finite real"):
        solid_harmonic(2, 0, 1j, 0, 0, 0.06)


def test_existing_expression_syntax_is_preserved():
    points = np.array([[1, 2, 3], [3, -2, 1]], dtype=float)
    np.testing.assert_array_equal(
        evaluate_field_expression("2xy + z.^2", points), 2 * points[:, 0] * points[:, 1] + points[:, 2] ** 2
    )


def test_harmonic_callable_cannot_expose_python_globals_or_execute_other_calls():
    points = np.zeros((2, 3))
    for expression in (
        "solid_harmonic.__globals__['__builtins__']",
        "solid_harmonic.__globals__.clear()",
        "(lambda: solid_harmonic)()",
        "solid_harmonic(0,0,x,y,z,0.06)+__import__('os').getcwd()",
        "1e1000*solid_harmonic(0,0,x,y,z,0.06)",
    ):
        with pytest.raises(ValueError, match="Could not evaluate"):
            evaluate_field_expression(expression, points)
    np.testing.assert_allclose(
        evaluate_field_expression("sin(x)+2*solid_harmonic(0,0,x,y,z,0.06)", points),
        np.full(2, 2 / np.sqrt(4 * np.pi)),
    )


def test_native_allowance_edit_is_isolated_from_physical_spec_and_independent_policy():
    config = CoilGenConfig(target_error_constraint_acceptance_absolute_T=3e-8)
    assert field_acceptance_slack_T(config) == 3e-8
    assert independent_field_acceptance_slack_T(config) == 1e-10
    changes = field_acceptance_slack_overrides(config, 2e-8)
    assert changes == {"target_error_constraint_acceptance_absolute_T": 2e-8}
    assert config.target_error_constraint_acceptance_absolute_T == 3e-8


@pytest.mark.parametrize(
    "physics", [{}, {"acceptance_absolute_by_unit": {"N": 8e-8}}, ResearchPhysicsSpec(objective="minimax")]
)
def test_physical_allowance_edit_preserves_spec_and_isolation(physics):
    config = CoilGenConfig(research_physics=physics)
    before = ResearchPhysicsSpec.from_dict(physics).to_dict()
    changes = field_acceptance_slack_overrides(config, 4e-9)
    after = ResearchPhysicsSpec.from_dict(changes["research_physics"]).to_dict()
    assert after["acceptance_absolute_by_unit"] == before["acceptance_absolute_by_unit"] | {"T": 4e-9}
    assert (after | {"acceptance_absolute_by_unit": before["acceptance_absolute_by_unit"]}) == before
    assert ResearchPhysicsSpec.from_dict(config.research_physics).to_dict() == before
    assert field_acceptance_slack_T(replace(config, **changes)) == 4e-9
    assert independent_field_acceptance_slack_T(replace(config, **changes)) == 4e-9


@pytest.mark.parametrize("value", [-1, np.inf, np.nan, True, 1j, "1e-9"])
def test_invalid_acceptance_slack_never_mutates_config(value):
    config = CoilGenConfig(research_physics={"objective": "power"})
    with pytest.raises(ValueError):
        field_acceptance_slack_overrides(config, value)
    assert config.research_physics == {"objective": "power"}


def test_physical_allowance_control_changes_actual_independent_acceptance():
    config = CoilGenConfig(research_physics={}, target_error_tolerance_percent=0)
    coords, target = np.zeros((2, 3)), np.zeros((2, 3))
    actual = target.copy()
    actual[:, 2] = 2e-9
    field = TargetField(coords, target, np.ones(2), np.zeros(2, bool), np.ones(2, int))
    result = SimpleNamespace(input_data=config, target_field=field)
    assert not field_budget_check(result, actual, target)["passed"]
    result.input_data = replace(config, **field_acceptance_slack_overrides(config, 3e-9))
    assert field_budget_check(result, actual, target)["passed"]
