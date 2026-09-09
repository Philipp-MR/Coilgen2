from __future__ import annotations

from copy import deepcopy

import pytest

from coilgen import CoilGenConfig, result_quality_status, result_summary_dict, run_coilgen
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
