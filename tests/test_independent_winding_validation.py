"""Independent winding evidence preserves recorded paths, currents and target scale."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.gradient_linearity import GradientLinearitySpec
from coilgen.models import TargetField
from coilgen.physics import biot_savart_wire_path
from coilgen.research.convex import FieldScenario, ResearchPhysicsSpec
from coilgen.research.validation import sample_independent_target, validate_winding_on_independent_grid
from coilgen.target_regions import sample_target_region


def winding_fixture(**changes):
    radius, current = 0.1, 2.0
    path = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float) * radius
    center_field = 4e-7 * np.sqrt(2) * current / radius
    config = CoilGenConfig(
        field_shape_function=str(center_field),
        target_gradient_normalization="off",
        target_region_radius=0.01,
        target_error_constraint_metric="rms",
        target_error_rms_limit_T=1e-6,
        research_physics={},
    )
    config = replace(config, **changes)
    coords = np.array([[0.002, 0.003, 0.004], [-0.002, -0.003, -0.004]])
    target = np.zeros_like(coords)
    target[:, 2] = center_field
    result = SimpleNamespace(
        input_data=config,
        target_field=TargetField(coords, target, np.ones(2), np.zeros(2, bool), np.ones(2, int)),
        regularization_diagnostics={"target_gradient_normalization_scale": 1.0},
        combined_mesh=SimpleNamespace(vertices=np.array([[-0.1, -0.1, 0], [0.1, 0.1, 0]])),
    )
    fabrication = SimpleNamespace(
        winding_steps=[{"body_id": "loop", "electrical_points_m": path, "circuit_complete": True}],
        electrical_metrics={"bodies": [{"body_id": "loop", "current_A": current}]},
    )
    return result, fabrication


def test_known_square_loop_field_and_exact_segment_invariance():
    result, fabrication = winding_fixture()
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    center = np.flatnonzero(np.all(report["points_m"] == 0, axis=1)).item()
    assert report["field_T"][center] == pytest.approx([0, 0, 4e-7 * np.sqrt(2) * 2 / 0.1], abs=1e-18)
    assert report["available"] and report["verified"] and report["passed"]
    assert report["source"] == "fabricated_electrical_paths_independent_grid_original_current"
    assert report["current_source"] == "original_contour_current_no_refit"
    assert report["currents_A"] == {"loop": 2.0} and report["current_refitted"] is False
    integration = report["integration_check"]
    assert integration["passed"]
    assert integration["max_deviation_T"] < 1e-18
    assert integration["method"] == "collinear_segment_bisection_invariance"
    assert not integration["geometry_changed"] and not integration["curve_discretization_convergence"]
    assert not integration["finite_cross_section_check"]


def test_each_signed_body_current_is_used_without_mutating_the_source():
    result, fabrication = winding_fixture()
    original = fabrication.winding_steps[0]["electrical_points_m"].copy()
    second = original + [0, 0, 0.04]
    fabrication.winding_steps.append({"body_id": "return", "electrical_points_m": second, "circuit_complete": True})
    fabrication.electrical_metrics["bodies"].append({"body_id": "return", "current_A": -0.7})
    saved_config, saved_target, saved_diagnostics = (
        deepcopy(result.input_data),
        result.target_field.b.copy(),
        deepcopy(result.regularization_diagnostics),
    )
    report = validate_winding_on_independent_grid(result, fabrication, resolution=4)
    expected = 2 * biot_savart_wire_path(original, report["points_m"]) - 0.7 * biot_savart_wire_path(
        second, report["points_m"]
    )
    np.testing.assert_allclose(report["field_T"], expected, rtol=1e-14, atol=1e-20)
    assert report["currents_A"] == {"loop": 2, "return": -0.7}
    np.testing.assert_array_equal(fabrication.winding_steps[0]["electrical_points_m"], original)
    np.testing.assert_array_equal(fabrication.winding_steps[1]["electrical_points_m"], second)
    np.testing.assert_array_equal(result.target_field.b, saved_target)
    assert result.input_data == saved_config and result.regularization_diagnostics == saved_diagnostics


def test_resolution_changes_only_the_validation_samples_and_current_is_fixed():
    result, fabrication = winding_fixture()
    coarse = validate_winding_on_independent_grid(result, fabrication, resolution=4)
    fine = validate_winding_on_independent_grid(result, fabrication, resolution=6)
    assert coarse["point_count"] < fine["point_count"]
    assert coarse["distinct_from_design_grid"] and fine["distinct_from_design_grid"]
    assert coarse["currents_A"] == fine["currents_A"] == {"loop": 2}
    assert coarse["sampling_weights"] == "equal_point_weights"


@pytest.mark.parametrize("unit,factor", [("T", 1), ("Hz", 1 / 42.57747892e6)])
def test_target_units_saved_normalization_and_world_center_are_reused(unit, factor):
    result, _fabrication = winding_fixture(
        field_shape_function="x + 2*y",
        target_field_units=unit,
        set_roi_into_mesh_center=True,
    )
    shift = np.array([0.07, -0.02, 0.04])
    result.combined_mesh.vertices += shift
    result.regularization_diagnostics["target_gradient_normalization_scale"] = 0.015
    target = sample_independent_target(result, resolution=4)
    assert target.coords.mean(axis=0) == pytest.approx(shift, abs=1e-15)
    assert target.b[:, 2] == pytest.approx((target.coords[:, 0] + 2 * target.coords[:, 1]) * 0.015 * factor)
    assert not np.any(target.b[:, :2])


def test_explicit_rotated_region_retains_world_expression_and_saved_target_scale():
    region = {
        "shape": "box",
        "size_m": [0.02, 0.04, 0.06],
        "center_m": [0.03, -0.02, 0.01],
        "rotation_axis": [0, 0, 1],
        "rotation_angle_rad": np.pi / 2,
    }
    result, _fabrication = winding_fixture(
        target_region_geometry=region,
        field_shape_function="x + 2*y",
        set_roi_into_mesh_center=True,
    )
    result.regularization_diagnostics["target_gradient_normalization_scale"] = 0.015
    result.combined_mesh.vertices += [0.2, 0.3, 0.4]
    target = sample_independent_target(result, resolution=4)
    local = sample_target_region({"shape": "box", "size_m": region["size_m"]}, 4, independent=True)
    expected = np.column_stack((-local[:, 1], local[:, 0], local[:, 2])) + region["center_m"]
    np.testing.assert_allclose(target.coords, expected, atol=1e-17)
    np.testing.assert_allclose(target.b[:, 2], 0.015 * (target.coords[:, 0] + 2 * target.coords[:, 1]))


def test_unsupported_field_units_are_unavailable():
    result, fabrication = winding_fixture(target_field_units="mT")
    report = validate_winding_on_independent_grid(result, fabrication)
    assert not report["available"] and report["passed"] is None
    assert "units" in report["reason"]


@pytest.mark.parametrize(
    "changes",
    [
        {"target_field_definition_file": "external-map.csv"},
        {"target_mesh_file": "external-roi.stl"},
    ],
)
def test_imported_targets_need_a_separate_independent_dataset(changes):
    result, fabrication = winding_fixture(**changes)
    report = validate_winding_on_independent_grid(result, fabrication)
    assert not report["available"] and not report["verified"] and report["passed"] is None
    assert "separately supplied" in report["reason"]


@pytest.mark.parametrize("kind", ["custom_only", "scenario", "suppression", "shield", "std"])
def test_unimplemented_contracts_never_promote_available_roi_fields_to_full_pass(kind):
    result, fabrication = winding_fixture()
    if kind == "custom_only":
        result.input_data.research_physics = {"include_default_field_constraints": False}
    elif kind == "scenario":
        result.input_data.research_physics = ResearchPhysicsSpec(
            scenarios=(FieldScenario("shifted", [[1]], [0], rms_limit_T=1e-6),)
        ).to_dict()
    elif kind == "suppression":
        result.target_field.suppressed[0] = True
    elif kind == "shield":
        result.input_data.shield_gradient_opposition_enabled = True
    else:
        result.input_data.target_error_constraint_metric = "std"
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and not report["verified"] and report["passed"] is None
    assert report["unknown_reasons"] and any(check["passed"] is None for check in report["checks"])


@pytest.mark.parametrize("kind", ["missing_current", "nan_current", "duplicate_body", "missing_body", "invalid_path"])
def test_invalid_recorded_path_or_current_is_unavailable(kind):
    result, fabrication = winding_fixture()
    if kind == "missing_current":
        fabrication.electrical_metrics["bodies"][0].pop("current_A")
    elif kind == "nan_current":
        fabrication.electrical_metrics["bodies"][0]["current_A"] = float("nan")
    elif kind == "duplicate_body":
        fabrication.winding_steps.append(deepcopy(fabrication.winding_steps[0]))
    elif kind == "missing_body":
        fabrication.electrical_metrics["bodies"] = []
    else:
        fabrication.winding_steps[0]["electrical_points_m"] = [[0, 0], [1, 0]]
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert not report["available"] and report["passed"] is None and report["reason"]


def test_partial_circuit_remains_inspectable_but_unverified():
    result, fabrication = winding_fixture()
    fabrication.winding_steps[0]["circuit_complete"] = False
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and not report["verified"] and report["passed"] is None
    assert any("complete modeled return" in reason for reason in report["unknown_reasons"])


def test_reordered_design_grid_cannot_be_claimed_as_independent():
    result, fabrication = winding_fixture()
    result.target_field.coords = sample_independent_target(result, resolution=3).coords[::-1].copy()
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert not report["available"] and "repeats" in report["reason"]
    assert report["new_point_count"] == 0
    assert report["design_grid_overlap_point_count"] == report["point_count"]


def test_validation_subset_of_design_samples_is_unavailable_even_with_roundoff():
    result, fabrication = winding_fixture()
    independent = sample_independent_target(result, resolution=3).coords
    result.target_field.coords = np.vstack((independent + 1e-18, [[0.009, 0, 0]]))
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert not report["available"] and report["passed"] is None
    assert report["new_point_count"] == 0 and report["overlap_tolerance_m"] > 1e-18


def test_shared_origin_is_reported_without_discarding_distinct_samples():
    result, fabrication = winding_fixture()
    result.target_field.coords = np.vstack((result.target_field.coords, [[0, 0, 0]]))
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and report["distinct_from_design_grid"]
    assert report["design_grid_overlap_point_count"] == 1
    assert report["new_point_count"] == report["point_count"] - 1


def test_singular_centerline_samples_are_unavailable_instead_of_kernel_zero():
    result, fabrication = winding_fixture()
    fabrication.winding_steps[0]["electrical_points_m"] = [[-0.02, 0, 0], [0.02, 0, 0], [-0.02, 0, 0]]
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert not report["available"] and "intersect" in report["reason"]


@pytest.mark.parametrize("strength_error,passed", [(0.0, True), (0.05, False)])
def test_requested_gradient_contract_uses_independent_field_at_original_current(monkeypatch, strength_error, passed):
    spec = GradientLinearitySpec(
        gradient_strength_tolerance_percent=2,
        rms_tolerance_percent=1,
        peak_tolerance_percent=1,
        cross_gradient_tolerance_percent=1,
    )
    result, fabrication = winding_fixture(
        field_shape_function="y", target_gradient_strength=0.02, **spec.to_config_overrides()
    )
    result.regularization_diagnostics["target_gradient_normalization_scale"] = 0.02

    def linear_field(_path, points, **_kwargs):
        field = np.zeros_like(points)
        field[:, 2] = 0.01 * (1 - strength_error) * points[:, 1]
        return field

    monkeypatch.setattr("coilgen.research.validation.biot_savart_wire_path", linear_field)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["verified"] and report["passed"] is passed
    assert report["linearity"]["linearity_achieved_gradient_T_per_m"] == pytest.approx(0.02 * (1 - strength_error))
    gradient = next(check for check in report["checks"] if check["name"] == "Independent winding gradient strength")
    assert gradient["passed"] is passed
    assert report["currents_A"] == {"loop": 2}


def test_noncartesian_gradient_contract_is_explicitly_unknown():
    result, fabrication = winding_fixture(target_field_component="BdotB0", gradient_strength_tolerance_percent=1)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and report["passed"] is None
    assert any("Cartesian Bz" in reason for reason in report["unknown_reasons"])


@pytest.mark.parametrize("current", [2.0, -2.0])
def test_disabled_gradient_contract_still_reports_fixed_current_affine_fit_without_acceptance(monkeypatch, current):
    result, fabrication = winding_fixture(field_shape_function="y", target_gradient_strength=0.02)
    result.regularization_diagnostics["target_gradient_normalization_scale"] = 0.02
    fabrication.electrical_metrics["bodies"][0]["current_A"] = current

    def linear_field(_path, points, **_kwargs):
        field = np.zeros_like(points)
        field[:, 2] = 0.01 * points[:, 1]
        return field

    monkeypatch.setattr("coilgen.research.validation.biot_savart_wire_path", linear_field)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    summary = report["linearity"]
    assert summary["linearity_achieved_gradient_T_per_m"] == pytest.approx(0.01 * current)
    assert summary["linearity_requested_gradient_T_per_m"] == pytest.approx(0.02)
    assert summary["linearity_rms_percent"] == pytest.approx(0, abs=1e-12)
    assert summary["linearity_cross_gradient_percent"] == pytest.approx(0, abs=1e-12)
    assert not summary["linearity_enabled"] and summary["linearity_status"] == "not_checked"
    assert all(value is None for key, value in summary.items() if key.endswith("_met"))
    assert report["currents_A"] == {"loop": current} and not report["current_refitted"]
    assert len(report["checks"]) == 2 and report["unknown_reasons"] == []
    assert report["verified"] and report["passed"] is (current > 0)


@pytest.mark.parametrize("expression", ["0", "2e-6", "2e-3*(x*x-y*y)", "y + 0.5*y*y"])
def test_disabled_gradient_contract_does_not_promote_constant_or_nonlinear_target_roundoff(expression):
    result, fabrication = winding_fixture(field_shape_function=expression)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=4)
    assert report["available"] and report["verified"]
    assert report["linearity"] == {} and report["unknown_reasons"] == []
    assert len(report["checks"]) == 2


@pytest.mark.parametrize(
    "changes",
    [{"target_field_component": "BdotB0"}, {"design_context": "mr_shim"}, {"linearity_region": "suppression"}],
)
def test_disabled_incompatible_gradient_fit_adds_no_unknown_contract(changes):
    result, fabrication = winding_fixture(field_shape_function="0.02*y", **changes)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and report["verified"]
    assert report["linearity"] == {} and report["unknown_reasons"] == []
    assert len(report["checks"]) == 2


def test_disabled_gradient_fit_on_degenerate_grid_adds_no_acceptance_blocker(monkeypatch):
    result, fabrication = winding_fixture(field_shape_function="0.02*y")
    points = np.column_stack((np.zeros(5), np.linspace(-0.004, 0.004, 5), np.zeros(5)))
    values = np.zeros_like(points)
    values[:, 2] = 0.02 * points[:, 1]
    target = TargetField(points, values, np.ones(5), np.zeros(5, bool), np.ones(5, int))
    monkeypatch.setattr("coilgen.research.validation.sample_independent_target", lambda *_a, **_k: target)
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and report["verified"]
    assert report["linearity"] == {} and report["unknown_reasons"] == []
    assert len(report["checks"]) == 2


@pytest.mark.parametrize("region", ["all", "any", "suppression"])
def test_linearity_contract_including_suppression_cannot_claim_roi_only_fit(region):
    result, fabrication = winding_fixture(linearity_region=region, gradient_strength_tolerance_percent=1)
    result.target_field.suppressed[0] = True
    report = validate_winding_on_independent_grid(result, fabrication, resolution=3)
    assert report["available"] and report["passed"] is None and not report["verified"]
    assert report["linearity"] == {}
    assert not any(check["name"] == "Independent winding gradient strength" for check in report["checks"])
    assert any("suppression-region winding linearity" in reason for reason in report["unknown_reasons"])


def test_effective_field_budget_uses_existing_margin_contract():
    result, fabrication = winding_fixture(target_error_rms_limit_T=1e-9)
    strict = validate_winding_on_independent_grid(result, fabrication, resolution=4)
    loose = validate_winding_on_independent_grid(result, fabrication, resolution=4, design_margin=0.5)
    assert loose["limit"] == pytest.approx(2 * strict["limit"])
    np.testing.assert_array_equal(strict["field_T"], loose["field_T"])


@pytest.mark.parametrize("resolution", [True, 1, 2.5])
def test_invalid_resolution_is_rejected(resolution):
    result, fabrication = winding_fixture()
    with pytest.raises(ValueError, match="resolution"):
        validate_winding_on_independent_grid(result, fabrication, resolution=resolution)


@pytest.mark.parametrize("margin", [0, -1, 1.1, float("nan")])
def test_invalid_margin_is_rejected(margin):
    result, fabrication = winding_fixture()
    with pytest.raises(ValueError, match="design_margin"):
        validate_winding_on_independent_grid(result, fabrication, design_margin=margin)
