from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.engineering_metrics import (
    ConcomitantFieldReport,
    CurrentDensityReport,
    ReadinessReport,
    SuppressionComparison,
    drive_metrics_from_inductance,
    fit_bz_field_quality,
)
from coilgen.gradient_linearity import gradient_linearity_summary


def test_suppression_comparison_reports_normal_field_reduction() -> None:
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    unshielded = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    active = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])

    data = SuppressionComparison.from_fields(
        active,
        unshielded,
        active_normals=normals,
        unshielded_normals=normals,
    ).as_dict()

    assert data["active_rms_Bmag_T"] == pytest.approx(0.2)
    assert data["shielding_rms_Bz_suppression_factor"] == pytest.approx(5.0)
    assert data["shielding_rms_Bnormal_reduction_percent"] == pytest.approx(80.0)


def test_drive_metrics_from_inductance_uses_target_slew_rate() -> None:
    drive = drive_metrics_from_inductance(
        current_A=10.0,
        resistance_ohm=2.0,
        inductance_H=0.01,
        target_gradient_T_per_m=0.02,
        target_slew_rate_T_per_m_per_s=200.0,
    )

    assert drive.ramp_time_s == pytest.approx(1.0e-4)
    assert drive.di_dt_A_per_s == pytest.approx(1.0e5)
    assert drive.inductive_voltage_V == pytest.approx(1000.0)
    assert drive.resistive_voltage_V == pytest.approx(20.0)
    assert drive.drive_voltage_V == pytest.approx(1020.0)
    assert drive.resistive_power_W == pytest.approx(200.0)
    assert drive.stored_energy_J == pytest.approx(0.5)


def test_current_density_and_readiness_reports_are_explicit_screens() -> None:
    current_density = CurrentDensityReport.from_current_and_conductor(
        current_A=120.0,
        conductor_width_m=2.0e-3,
        conductor_height_m=2.0e-3,
    )
    readiness = ReadinessReport.from_status(magnetostatic_status="PASS", layout_status="WARNING")

    assert current_density.current_density_A_per_mm2 == pytest.approx(30.0)
    assert current_density.current_density_class == "pulsed_review_required"
    assert not readiness.finished_engineering_design_ready
    assert readiness.engineering_readiness_status == "screening_only"


def test_concomitant_field_report_uses_maxwell_transverse_terms() -> None:
    coords = np.array(
        [
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=float,
    )

    report = ConcomitantFieldReport.from_linear_gradient(
        coords,
        np.array([2.0, 0.0, 0.0], dtype=float),
        main_field_T=1.0,
    )
    values = ConcomitantFieldReport.field_values(
        coords,
        np.array([2.0, 0.0, 0.0], dtype=float),
        main_field_T=1.0,
    )

    assert values.tolist() == pytest.approx([0.02, 0.0, 0.02])
    assert report.sample_count == 3
    assert report.max_abs_concomitant_T == pytest.approx(0.02)
    assert report.rms_concomitant_T == pytest.approx(np.sqrt((0.02**2 + 0.02**2) / 3.0))
    assert report.max_abs_concomitant_ppm_of_B0 == pytest.approx(20000.0)


def test_fit_bz_field_quality_reports_desired_axis_and_cross_gradient() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T

    target = np.zeros((coords.shape[0], 3))
    target[:, 2] = 2.0 * coords[:, 1]
    actual = target.copy()
    actual[:, 2] += 0.5 * coords[:, 0] + 0.1 * coords[:, 2] ** 2

    fit = fit_bz_field_quality(
        actual,
        SimpleNamespace(coords=coords, b=target),
        sample_mask=np.ones(coords.shape[0], dtype=bool),
    )

    assert fit.desired_gradient_axis == "y"
    assert fit.desired_gradient_T_per_m == pytest.approx(2.0)
    assert fit.dbz_dx_T_per_m == pytest.approx(0.5)
    assert fit.dbz_dy_T_per_m == pytest.approx(2.0)
    assert fit.cross_gradient_percent_of_desired == pytest.approx(25.0)
    assert fit.qzz_T_per_m2 == pytest.approx(0.1)


def test_gradient_linearity_summary_normalizes_by_requested_gradient_times_radius() -> None:
    axis = np.array([-0.08, 0.0, 0.08])
    coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    target = np.zeros((coords.shape[0], 3), dtype=float)
    target[:, 2] = 0.01 * coords[:, 0]
    actual = target.copy()
    actual[:, 2] += 1.0e-5 * (coords[:, 2] / 0.08) ** 2
    target_field = SimpleNamespace(
        coords=coords,
        b=target,
        suppressed=np.zeros(coords.shape[0], dtype=bool),
    )
    config = SimpleNamespace(
        target_gradient_strength=0.01,
        linearity_region="dsv",
        linearity_reference_radius_m=0.08,
        linearity_rms_tolerance_percent=2.0,
        linearity_peak_tolerance_percent=2.0,
        linearity_cross_gradient_tolerance_percent=1.0,
        gradient_strength_tolerance_percent=1.0,
        linearity_normalization="requested_gradient_times_radius",
        linearity_fit_offset=True,
    )

    summary = gradient_linearity_summary(actual, target_field, config)

    assert summary["linearity_status"] == "PASS"
    assert summary["linearity_reference_scale_T"] == pytest.approx(8.0e-4)
    assert summary["linearity_peak_percent"] < 2.0
    assert summary["linearity_rms_percent"] < 2.0
