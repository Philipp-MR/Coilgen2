from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig, run_coilgen
from coilgen.models import TargetField
from coilgen.physics import biot_savart_wire_path
from coilgen.research.convex import FieldScenario, ResearchPhysicsSpec
from coilgen.research.validation import (
    field_budget_check,
    validate_fabricated_constraints,
    validate_on_independent_grid,
    wire_force_torque,
)


def field_result(**config_values):
    config = CoilGenConfig(research_physics={}, **config_values)
    target = TargetField(np.zeros((4, 3)), np.zeros((4, 3)), np.ones(4), np.zeros(4, bool), np.ones(4, int))
    return SimpleNamespace(input_data=config, target_field=target)


def test_resolved_advanced_solve_budget_and_margin_define_final_acceptance():
    result = field_result(target_error_tolerance_percent=2.0)
    target = np.array([[0, 0, 1e-3]])
    passing = field_budget_check(result, target * 1.049, target, design_margin=0.4)
    failing = field_budget_check(result, target * 1.051, target, design_margin=0.4)
    assert passing["relative_limit"] == pytest.approx(0.05)
    assert passing["limit"] == pytest.approx(5e-5)
    assert passing["passed"]
    assert not failing["passed"]


def test_vector_rms_is_magnitude_with_equivalent_component_budget():
    result = field_result(
        target_field_component="vector", target_error_constraint_metric="rms", target_error_rms_limit_T=2e-6
    )
    target = np.zeros((2, 3))
    actual = np.full((2, 3), 2e-6)
    check = field_budget_check(result, actual, target)
    assert check["rms_error_T"] == pytest.approx(np.sqrt(3) * 2e-6)
    assert check["peak_error_T"] == pytest.approx(np.sqrt(3) * 2e-6)
    assert check["limit"] == pytest.approx(np.sqrt(3) * 2e-6)
    assert check["rms_convention"] == "vector_magnitude"
    assert check["passed"]
    assert check["relative_field_error"] is None


def test_absolute_norm_budget_retains_design_sample_normalization():
    result = field_result(target_error_tolerance_percent=0.0, target_error_tolerance_absolute_T=1e-6)
    target = np.zeros((100, 3))
    actual = target.copy()
    actual[:, 2] = 0.49e-6
    check = field_budget_check(result, actual, target)
    assert check["limit"] == pytest.approx(1e-6 / np.sqrt(4))
    assert check["passed"]


@pytest.mark.parametrize("kind", ["std", "custom", "imported"])
def test_unsupported_independent_contract_is_explicitly_unverified(kind):
    result = field_result()
    if kind == "std":
        result.input_data.target_error_constraint_metric = "std"
    elif kind == "custom":
        result.input_data.research_physics = {"include_default_field_constraints": False}
    else:
        result.input_data.target_field_definition_file = "validation_data.csv"
    check = validate_on_independent_grid(result)
    assert not check["available"]
    assert not check["verified"]
    assert check["passed"] is None
    assert check["reason"]


def test_nonfinite_field_cannot_be_accepted():
    result = field_result()
    check = field_budget_check(result, np.full((2, 3), np.nan), np.ones((2, 3)))
    assert check["verified"]
    assert check["passed"] is False


@pytest.mark.parametrize("limit", [{"sheet_power_limit_W": 10.0}, {"magnetic_energy_limit_J": 0.01}])
def test_postroute_power_energy_limits_need_the_actual_circuit_model(limit):
    result = field_result()
    result.input_data.research_physics = limit
    fabrication = SimpleNamespace(electrical_metrics={"bodies": []}, winding_steps=[])
    check = validate_fabricated_constraints(result, fabrication)
    assert not check["verified"]
    assert not check["passed"]
    assert any("power" in reason or "energy" in reason for reason in check["unknown_reasons"])


def test_independent_grid_rebuilds_cleared_basis_without_mutating_result():
    result = run_coilgen(
        CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
            field_shape_function="y",
            target_region_radius=0.035,
            target_region_resolution=5,
            levels=4,
            gauss_order=1,
            target_error_tolerance_percent=40,
            min_point_loop_number=3,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
            progress_flag=False,
            low_memory_mode=True,
            research_physics={"objective": "power"},
        )
    )
    assert all(not part.basis_elements for part in result.coil_parts)
    original = result.combined_mesh.stream_function.copy()
    validation = validate_on_independent_grid(result, resolution=6)
    assert validation["available"] and validation["verified"]
    assert np.all(np.isfinite(validation["field_T"]))
    assert np.linalg.norm(validation["field_T"]) > 0
    assert all(not part.basis_elements for part in result.coil_parts)
    assert result.combined_mesh.stream_function == pytest.approx(original)


def test_wire_force_torque_matches_analytic_loop_and_origin_transform():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    force, torque = wire_force_torque(points, 2.0, [0, 3, 0], [0, 0, 0])
    assert force == pytest.approx([0, 0, 0])
    assert torque == pytest.approx([-6, 0, 0])
    moved_force, moved_torque = wire_force_torque(points + [2, 3, 4], 2.0, [0, 3, 0], [2, 3, 4])
    assert moved_force == pytest.approx(force)
    assert moved_torque == pytest.approx(torque)


def fabricated_fixture(spec=ResearchPhysicsSpec()):
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    coords = np.array([[0.2, 0.3, 0.4]])
    target = 2 * biot_savart_wire_path(points, coords, worker_count=1)
    result = SimpleNamespace(
        input_data=CoilGenConfig(research_physics=spec.to_dict()),
        target_field=TargetField(coords, target, np.ones(1), np.zeros(1, bool), np.ones(1, int)),
    )
    fabrication = SimpleNamespace(
        winding_steps=[{"body_id": "loop", "electrical_points_m": points, "circuit_complete": True}],
        electrical_metrics={"bodies": [{"body_id": "loop", "current_A": 2.0}]},
    )
    return result, fabrication


def test_final_wire_torque_violation_blocks_acceptance_even_when_field_passes():
    result, fabrication = fabricated_fixture(ResearchPhysicsSpec(b0_T=(0, 3, 0), torque_limit_Nm=3.0))
    validation = validate_fabricated_constraints(result, fabrication)
    assert validation["verified"]
    assert not validation["passed"]
    assert validation["torque_Nm"] == pytest.approx([-6, 0, 0])
    assert validation["checks"][0]["passed"]
    assert any(row["name"] == "Fabricated wire torque" and not row["passed"] for row in validation["checks"])


def test_final_wire_force_and_torque_equalities_are_checked():
    result, fabrication = fabricated_fixture(
        ResearchPhysicsSpec(b0_T=(0, 3, 0), force_equality_N=(0, 0, 0), torque_equality_Nm=(-6, 0, 0))
    )
    validation = validate_fabricated_constraints(result, fabrication)
    assert validation["verified"] and validation["passed"]
    assert validation["force_model"] == "uniform_B0_complete_electrical_polyline"
    assert validation["partial_path_force_N"] is None


@pytest.mark.parametrize("claimed_complete", [False, True])
def test_open_path_loads_are_not_reported_as_complete_circuit_loads(claimed_complete):
    result, fabrication = fabricated_fixture(ResearchPhysicsSpec(force_limit_N=1.0, torque_limit_Nm=1.0))
    plan = fabrication.winding_steps[0]
    plan["electrical_points_m"] = plan["electrical_points_m"][:-1]
    plan["circuit_complete"] = claimed_complete
    validation = validate_fabricated_constraints(result, fabrication)
    assert not validation["verified"]
    assert not validation["passed"]
    assert validation["force_model"] == "uniform_B0_partial_electrical_paths"
    assert validation["force_N"] is None
    assert validation["torque_Nm"] is None
    assert validation["partial_path_force_N"] == pytest.approx([6.0, 0, 0])
    assert validation["partial_path_torque_Nm"] == pytest.approx([0, 0, -3.0])
    assert not any(row["name"] in {"Fabricated wire force", "Fabricated wire torque"} for row in validation["checks"])


@pytest.mark.parametrize("kind", ["scenario", "nonuniform", "return", "gradient", "false_closed"])
def test_missing_postroute_physics_produces_unknown_blockers(kind):
    result, fabrication = fabricated_fixture()
    if kind == "scenario":
        result.input_data.research_physics = ResearchPhysicsSpec(
            scenarios=(FieldScenario("displaced", [[1]], [0], rms_limit_T=1e-6),)
        ).to_dict()
    elif kind == "nonuniform":
        result.input_data.research_physics = ResearchPhysicsSpec(
            background_B_per_face_T=[[0, 1, 0]], torque_limit_Nm=1
        ).to_dict()
    elif kind == "return":
        fabrication.winding_steps[0]["circuit_complete"] = False
    elif kind == "gradient":
        result.input_data.gradient_strength_tolerance_percent = 1
    else:
        fabrication.winding_steps[0]["electrical_points_m"] = fabrication.winding_steps[0]["electrical_points_m"][:-1]
    validation = validate_fabricated_constraints(result, fabrication)
    assert not validation["verified"]
    assert not validation["passed"]
    assert validation["unknown_reasons"]
    assert any(row["passed"] is None for row in validation["checks"])
