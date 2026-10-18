from __future__ import annotations

from copy import deepcopy
import csv

import pytest

from coilgen import CoilGenConfig, result_quality_status, result_summary_dict, run_coilgen, write_design_summary_csv
from coilgen.reporting import result_performance_table_rows
from coilgen.models import CoilGenResult
from coilgen.study import DesignStudyTable, study_result_from_coilgen_result


@pytest.fixture(scope="module")
def legacy_result() -> CoilGenResult:
    """A normal fixed-regularization result has no conic acceptance contract."""
    return run_coilgen(
        CoilGenConfig(
            field_shape_function="y",
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
            target_region_radius=0.035,
            target_region_resolution=3,
            levels=4,
            pot_offset_factor=0.25,
            min_point_loop_number=3,
            gauss_order=1,
            regularization_selection_method="fixed",
            tikonov_reg_factor=20.0,
            plot_flag=False,
            save_stl_flag=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
            low_memory_mode=True,
            retain_heavy_fields=False,
        )
    )


@pytest.mark.parametrize(
    "diagnostics, expected_reason",
    [
        ({"constrained_solver_contract_met": False}, "full solver contract not met"),
        ({"constrained_solver_physical_constraints_met": False}, "physical constraints not met"),
        (
            {
                "constrained_solver_diagnostic_result": True,
                "constrained_solver_physical_constraints_met": True,
            },
            "diagnostic-only result",
        ),
    ],
)
def test_explicit_unaccepted_solver_result_fails_quality(
    legacy_result: CoilGenResult, diagnostics: dict, expected_reason: str
) -> None:
    result = deepcopy(legacy_result)
    result.regularization_diagnostics.update(diagnostics)

    quality = result_quality_status(result)

    assert quality.status == "FAIL"
    assert not quality.ok
    assert any(expected_reason in failure for failure in quality.failures)


def test_missing_or_successful_contract_preserves_existing_quality(legacy_result: CoilGenResult) -> None:
    baseline = result_quality_status(legacy_result)
    assert baseline.status in {"PASS", "WARNING"}
    assert "constrained_solver_contract_met" not in legacy_result.regularization_diagnostics
    assert result_summary_dict(legacy_result)["constrained_solver_contract_met"] is None

    result = deepcopy(legacy_result)
    result.regularization_diagnostics.update(
        constrained_solver_contract_met=True,
        constrained_solver_physical_constraints_met=True,
        constrained_solver_diagnostic_result=False,
    )
    assert result_quality_status(result) == baseline


def test_reporting_separates_completed_execution_from_failed_engineering_checks(legacy_result):
    result = deepcopy(legacy_result)
    result.coil_parts[0].route_diagnostics["route_quality_status"] = "fail"
    summary = result_summary_dict(result)
    assert summary["execution_status"] == "completed"
    assert summary["engineering_status"] == "FAIL"
    assert summary["quality_status"] == "FAIL"
    assert "Route-quality" in summary["first_failure"]
    rows = {(group, quantity): value for group, quantity, value, _unit in result_performance_table_rows(result)}
    assert rows[("Execution", "status")] == "completed"
    assert rows[("Quality", "engineering acceptance")] == "FAIL"
    assert rows[("Optimization", "linearity RMS constraint met")] == "inactive"
    assert rows[("Optimization", "suppression constraint met")] == "inactive"


def test_actual_design_csv_preserves_metadata_and_two_column_structure(legacy_result, tmp_path):
    result = deepcopy(legacy_result)
    result.input_data.layout_manufacturing_intent = 'Copper, "rolled"\nprototype'
    path = write_design_summary_csv(result, tmp_path / "design.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert all(len(row) == 2 for row in rows)
    summary = dict(rows[1:])
    assert summary["layout_manufacturing_intent"] == result.input_data.layout_manufacturing_intent
    assert summary["execution_status"] == "completed"
    assert summary["engineering_status"] == result_summary_dict(result)["engineering_status"]


def test_failed_contract_is_reported_and_excluded_from_default_study_ranking(
    legacy_result: CoilGenResult,
) -> None:
    diagnostic = deepcopy(legacy_result)
    diagnostic.regularization_factor = legacy_result.regularization_factor / 2.0
    diagnostic.regularization_diagnostics.update(
        constrained_solver_status="optimal_inaccurate",
        constrained_solver_contract_met=False,
        constrained_solver_physical_constraints_met=False,
        constrained_solver_diagnostic_result=True,
        constrained_solver_failed_constraints=["surface_current_density", "shield_sign"],
    )
    summary = result_summary_dict(diagnostic)
    assert summary["constrained_solver_contract_met"] is False
    assert summary["constrained_solver_physical_constraints_met"] is False
    assert summary["constrained_solver_diagnostic_result"] is True
    assert summary["constrained_solver_failed_constraints"] == "surface_current_density, shield_sign"

    normal_row = study_result_from_coilgen_result("normal", {}, legacy_result)
    diagnostic_row = study_result_from_coilgen_result("diagnostic", {}, diagnostic)
    assert diagnostic_row.status == "FAIL"
    assert any("surface_current_density, shield_sign" in note for note in diagnostic_row.notes)
    table = DesignStudyTable.from_results([normal_row, diagnostic_row])
    assert table.best_by("regularization_factor", maximize=False) is normal_row
    assert table.best_by("regularization_factor", maximize=False, allowed_statuses=None) is diagnostic_row


def test_accepted_nonregression_route_never_hides_inherited_fixed_current_field_failure(legacy_result):
    result = deepcopy(legacy_result)
    result.input_data.field_evaluation_current_source = "target_fit"
    result.coil_parts[0].joint_route_diagnostics = {
        "status": "accepted",
        "field_acceptance_policy": "non_regression",
        "selected_absolute_field_contract_met": False,
        "selected_absolute_field_contract_reasons": {
            "route": "gradient_strength_contract",
            "physical": "gradient_strength_contract",
        },
    }
    quality = result_quality_status(result)
    assert quality.status == "FAIL"
    assert any(
        "unchanged series current" in failure and "gradient_strength_contract" in failure
        for failure in quality.failures
    )
    summary = result_summary_dict(result)
    assert summary["routing_field_acceptance_policies"] == "non_regression"
    assert summary["routing_absolute_field_contract_status"] == "FAIL"
    assert summary["routing_absolute_field_contract_failed_part_count"] == 1
    assert summary["routing_absolute_field_contract_failure_reasons"] == "part 1: gradient_strength_contract"
    row = study_result_from_coilgen_result("inherited", {}, result)
    assert row.status == "FAIL"


def test_accepted_absolute_contract_and_legacy_routes_do_not_create_quality_failures(legacy_result):
    result = deepcopy(legacy_result)
    quality = result_quality_status(result)
    for flag in (None, True):
        result.coil_parts[0].joint_route_diagnostics = {
            "status": "accepted",
            "selected_absolute_field_contract_met": flag,
        }
        assert result_quality_status(result) == quality
    assert result_summary_dict(result)["routing_absolute_field_contract_status"] == "PASS"


def test_report_distinguishes_route_clearance_from_source_contour_spacing(legacy_result):
    result = deepcopy(legacy_result)
    result.coil_parts[0].joint_route_diagnostics = {
        "status": "accepted",
        "source_contour_clearance_m": 0.005,
        "routing_clearance_target_m": 0.0035,
        "selected_physical_minimum_centerline_distance_m": 0.0036,
        "selected_physical_clearance_lower_bound_m": 0.0035,
    }
    summary = result_summary_dict(result)
    assert summary["routing_source_contour_clearance_m"] == 0.005
    assert summary["routing_clearance_target_m"] == 0.0035
    assert summary["routing_physical_minimum_centerline_distance_m"] == 0.0036
    assert summary["routing_physical_clearance_lower_bound_m"] == 0.0035
