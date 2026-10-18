"""Conductor statistics retain saved units and separate material representations."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from coilgen.application import RunRecord
from coilgen.gui.coil_statistics import build_coil_statistics, mass_estimates


def metadata(*, complete=True, body_count=1):
    bodies, paths, pipeline = [], [], []
    for index in range(body_count):
        body = f"body{index}"
        bodies.append(
            {
                "body_id": body,
                "current_A": -2.0,
                "length_m": 4.0,
                "resistance_ohm": 0.08,
                "joule_power_W": 0.32,
                "cross_section_area_m2": 1e-6,
                "inductance_H": 8e-6,
                "inductance_kind": "closed_path_self" if complete else "open_path_partial",
                "circuit_complete": complete,
                "mutual_inductance_included": False,
            }
        )
        paths.append(
            {
                "body_id": body,
                "turn_multiplicity": 3,
                "length_m": 4.0,
                "closed": complete,
                "diagnostics": {
                    "absolute_volume_m3": 3.9e-6,
                    "sweep_expected_profile_volume_m3": 4e-6,
                    "sweep_nominal_cross_section_area_m2": 1e-6,
                },
            }
        )
        pipeline.append(
            {
                "body_id": "pipeline_" + body,
                "manufacturing_ok": False,
                "diagnostics": {"absolute_volume_m3": 30e-6, "sweep_expected_profile_volume_m3": 40e-6},
            }
        )
    return {
        "summary": {
            "scope": "fabrication_review",
            "resistance_ohm": 0.08,
            "inductance_h": 8e-6,
            "current_a": -2.0,
            "wire_length_m": 4.0 * body_count,
            "power_w": 0.02,
            "fabricated_power_w": 0.32 * body_count,
            "max_current_density_a_per_m": 210.0,
        },
        "config": {"specific_conductivity_conductor": 1.8e-8, "sweep_insulation_thickness": 0.0001},
        "fabrication": {"ready": False, "paths": paths, "electrical_metrics": {"bodies": bodies}},
        "pipeline_sweeps": {"checks_passed": False, "bodies": pipeline},
    }


def test_saved_single_circuit_units_current_and_failed_geometry():
    report = build_coil_statistics(metadata())
    assert report.has_values
    assert report.row("coil:resistance").value == 0.08
    assert report.row("coil:resistance").unit == "Ω"
    assert report.row("coil:inductance").value == 8e-6
    assert report.row("coil:inductance").unit == "H"
    assert report.row("coil:current").value == -2.0
    assert "excludes mutual coupling" in report.row("coil:inductance").scope
    assert report.row("coil:volume").value == 3.9e-6
    assert report.row("coil:volume").unit == "m³"
    assert "Geometry checks failed" in report.row("coil:volume").scope
    assert report.row("reviewed:body0:nominal_volume").value == 4e-6
    assert report.row("pipeline:pipeline_body0:nominal_volume").value == 40e-6
    assert report.row("material:specific_conductivity_conductor").quantity == "Conductor resistivity"
    assert report.row("model:max_current_density_a_per_m").unit == "A/m"
    assert "not volume current density" in report.row("model:max_current_density_a_per_m").scope
    assert report.row("coil:mass").value is None
    assert not report.density_available
    assert "density" in report.mass_unavailable_reason
    assert "self L, no mutual" in report.overview_scope
    assert "reviewed conductor mesh" in report.overview_scope
    assert "geometry failed" in report.overview_scope
    assert "insulation excluded" in report.overview_scope
    assert len(report.overview_scope) <= 130


@pytest.mark.parametrize("complete,count", [(False, 1), (True, 2), (False, 2)])
def test_open_and_multiple_paths_do_not_become_whole_coil_self_inductance(complete, count):
    report = build_coil_statistics(metadata(complete=complete, body_count=count))
    for key in ("coil:resistance", "coil:inductance", "coil:current"):
        assert report.row(key).value is None
    assert report.row("body:body0:inductance_H").value == 8e-6
    assert "excludes mutual coupling" in report.row("body:body0:inductance_H").scope
    if count > 1:
        assert report.row("coil:volume").value is None


def test_borrowed_metadata_only_no_array_io_or_witness_iteration(tmp_path):
    class ScalarOnlyRecord(RunRecord):
        @property
        def metadata(self):
            raise AssertionError("Copied scientific metadata")

        @property
        def arrays(self):
            raise AssertionError("Loaded scientific arrays")

        @property
        def array_names(self):
            raise AssertionError("Inspected scientific arrays")

    supplied = metadata()
    supplied["winding_steps"] = [
        {
            "body_id": "body0",
            "points_m": np.zeros((1000, 3)),
            "drive_feed_designed": False,
            "terminal_assumption": "Feed not modeled",
        }
    ]
    record = ScalarOnlyRecord(tmp_path, supplied, {})
    original = record.metadata_view
    report = build_coil_statistics(record)
    assert report.row("feed:body0:drive_feed_designed").value == "No"
    assert report.row("coil:resistance").value == 0.08
    assert record.metadata_view is original
    with pytest.raises(FrozenInstanceError):
        report.rows[0].value = 12.0
    with pytest.raises(FrozenInstanceError):
        report.message = "changed"


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, True, "unrecorded"])
def test_invalid_or_absent_quantitative_values_remain_unavailable(value):
    supplied = metadata()
    supplied["summary"]["inductance_h"] = value
    supplied["fabrication"]["electrical_metrics"]["bodies"][0]["inductance_H"] = value
    row = build_coil_statistics(supplied).row("coil:inductance")
    assert row.value is None
    assert row.status == "Unavailable"


def test_no_run_and_empty_metadata_do_not_create_recorded_coil_statistics():
    for source in (None, {}):
        report = build_coil_statistics(source)
        assert not report.has_values
        for key in ("coil:resistance", "coil:inductance", "coil:mass", "coil:volume"):
            assert report.row(key).value is None


def test_density_mass_uses_reviewed_nominal_volume_once_no_turn_expansion():
    base = build_coil_statistics(metadata(body_count=2))
    estimate = mass_estimates(base, 8900.0)
    assert estimate.density_available
    assert estimate.mass_unavailable_reason == ""
    assert estimate.row("coil:mass").value == pytest.approx(2 * 4e-6 * 8900.0)
    assert estimate.row("coil:mass").status == "Assumed density estimate"
    assert "unexpanded" in estimate.row("coil:mass").scope
    assert estimate.row("estimate:reviewed:body0:nominal_volume:mass").value == pytest.approx(4e-6 * 8900.0)
    assert estimate.row("estimate:reviewed:body0:mesh_volume:mass").value == pytest.approx(3.9e-6 * 8900.0)
    assert "Geometry checks failed" in estimate.row("estimate:reviewed:body0:mesh_volume:mass").scope
    assert estimate.row("estimate:coil:mesh_mass").value == pytest.approx(2 * 3.9e-6 * 8900.0)
    assert base.row("coil:mass").value is None
    assert not base.density_available


def test_density_estimate_preserves_explicit_recorded_mass_separately():
    supplied = metadata()
    supplied["summary"]["conductor_mass_kg"] = 0.04
    base = build_coil_statistics(supplied)
    assert base.mass_unavailable_reason == ""
    estimate = mass_estimates(base, 8900.0)
    assert estimate.row("coil:mass") == base.row("coil:mass")
    assert estimate.row("estimate:coil:mass").value == pytest.approx(4e-6 * 8900.0)


def test_density_falls_back_to_pipeline_only_when_reviewed_bodies_absent():
    supplied = metadata()
    supplied["fabrication"]["paths"] = []
    estimate = mass_estimates(build_coil_statistics(supplied), 8900.0)
    assert estimate.row("coil:mass").value == pytest.approx(40e-6 * 8900.0)
    assert "pipeline" in estimate.row("coil:mass").scope


@pytest.mark.parametrize("density", [None, 0.0])
def test_unspecified_density_does_not_infer_copper_or_modify_stats(density):
    base = build_coil_statistics(metadata())
    assert mass_estimates(base, density) is base


@pytest.mark.parametrize("density", [True, -1.0, float("nan"), float("inf"), "8900"])
def test_density_requires_explicit_finite_numeric_assumption(density):
    with pytest.raises(ValueError, match="density"):
        mass_estimates(build_coil_statistics(metadata()), density)


def test_actual_shim_saved_statistics_are_finite_recorded_values():
    # A small representative saved record, without binding tests to output paths.
    supplied = metadata()
    supplied["summary"].update(
        resistance_ohm=0.08762305275650142,
        inductance_h=8.237791027696818e-6,
        current_a=3.067607063189881,
        wire_length_m=4.86794737536119,
    )
    report = build_coil_statistics(RunRecord(Path("saved_xz"), supplied, {}))
    assert report.row("coil:inductance").value == 8.237791027696818e-6
    assert report.row("coil:resistance").value == 0.08762305275650142


def test_explicit_drive_and_thermal_values_preserve_scope_and_negative_headroom():
    supplied = metadata()
    supplied["engineering_review"] = {
        "drive": {
            "available": True,
            "scope": "saved duty waveform",
            "inductance_scope": "supplied full matrix",
            "bodies": [
                {
                    "body_id": "body0",
                    "peak_current_A": 3.0,
                    "current_margin_A": -1.0,
                    "voltage_margin_V": 12.0,
                    "rms_current_A": 1.5,
                    "average_joule_loss_W": 0.2,
                }
            ],
        },
        "thermal": {
            "available": True,
            "model": "measured_shared_lumped_RC_constant_electrical_resistance",
            "peak_temperature_K": 301.0,
            "temperature_margin_K": -2.0,
            "time_constant_s": 10.0,
        },
    }
    report = build_coil_statistics(supplied)
    assert report.row("drive:body0:current_margin_A").value == -1.0
    assert report.row("drive:body0:rms_current_A").unit == "A"
    assert "supplied full matrix" in report.row("drive:body0:peak_current_A").scope
    assert report.row("thermal:peak_temperature_K").value == 301.0
    assert report.row("thermal:peak_temperature_K").unit == "K"
    assert report.row("thermal:temperature_margin_K").value == -2.0
    assert "no thermal qualification claim" in report.row("thermal:peak_temperature_K").scope
