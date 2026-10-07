"""Example studies use explicit physical targets and qualified evidence scopes."""

from dataclasses import asdict

import numpy as np
import pytest

from coilgen.application.presets import EXAMPLE_TITLES, PRESET_NAMES, preset_config, preset_description, preset_study
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.cli import main
from coilgen.research.application import run_design
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import json_value


@pytest.mark.parametrize("kind", PRESET_NAMES)
def test_examples_define_material_and_validation_assumptions(kind):
    config = preset_config(kind)
    study = preset_study(kind)

    assert config.specific_conductivity_conductor == pytest.approx(18e-9)
    assert config.conductor_thickness == pytest.approx(0.001)
    gradient = kind in ("cylinder", "biplanar")
    assert config.conductor_cross_section_width == pytest.approx(0.002 if gradient else 0.001)
    assert config.conductor_cross_section_height == pytest.approx(0.002 if gradient else 0.001)
    assert config.sweep_min_bend_radius == pytest.approx(0.006 if gradient else 0.003)
    assert config.conductor_cross_section_width * config.conductor_cross_section_height == pytest.approx(
        4e-6 if gradient else 1e-6
    )
    assert config.b_0_strength_T == config.research_physics["b0_T"][2] == 3.0
    assert config.field_evaluation_current_source == "stream_function"
    assert config.constrained_quadratic_fallback_policy == "strict"
    assert config.target_error_tolerance_percent / study["design_margin"] == pytest.approx(5.0)
    assert study["pipeline_stage"] == "evaluated" and study["fabricate"]
    assert EXAMPLE_TITLES[kind] and "feed" in preset_description(kind).lower()


@pytest.mark.parametrize("kind", ("cylinder", "biplanar"))
def test_prepared_gradients_use_world_y_with_the_requested_si_gradient(kind, tmp_path):
    config = preset_config(kind)
    request = ResearchRequest.from_config(config, pipeline_stage="prepared")
    run_design(request, tmp_path)
    record = DesignService.load_run(tmp_path)

    points = record.arrays["target_points_m"]
    target = record.arrays["target_field_T"]
    assert len(points) == 257
    assert np.max(np.linalg.norm(points, axis=1)) <= config.target_region_radius + 1e-12
    np.testing.assert_array_equal(target[:, :2], 0.0)
    np.testing.assert_allclose(target[:, 2], 0.02 * points[:, 1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.linalg.lstsq(points, target[:, 2], rcond=None)[0], (0, 0.02, 0), atol=1e-14)
    if kind == "biplanar":
        assert set(np.round(record.arrays["vertices_m"][:, 0], 12)) == {-0.05, 0.05}
        assert config.inter_group_connection_path_method == "direct_3d"
        assert config.routing_circuit_scope == "per_part"
    else:
        np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].min(), -0.1, atol=1e-15)
        np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].max(), 0.1, atol=1e-15)


@pytest.mark.parametrize("factory", (preset_config, preset_description, preset_study))
def test_unknown_examples_are_rejected(factory):
    with pytest.raises(ValueError, match="Unknown example"):
        factory("unknown")


def test_prepared_conical_target_matches_absolute_z2_in_world_coordinates(tmp_path):
    config = preset_config("conical_harmonic")
    run_design(ResearchRequest.from_config(config, pipeline_stage="prepared"), tmp_path)
    record = DesignService.load_run(tmp_path)
    points = record.arrays["target_points_m"]
    x, y, z = points.T

    np.testing.assert_allclose(record.arrays["target_field_T"][:, 2], 0.002 * (2 * z**2 - x**2 - y**2), atol=1e-18)
    np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].min(), -0.15, atol=1e-15)
    np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].max(), 0.15, atol=1e-15)
    assert config.surface_is_cylinder_flag is False and config.target_gradient_normalization == "off"


def test_cli_can_create_the_same_independent_turn_project(tmp_path):
    path = tmp_path / "independent_turns.coilgen.json"

    assert main(["new", "--preset", "independent_turns", "--output", str(path)]) == 0

    project = ProjectDocument.load(path)
    assert json_value(asdict(project.config)) == json_value(asdict(preset_config("independent_turns")))
    assert {key: project.study[key] for key in preset_study("independent_turns")} == preset_study("independent_turns")


def test_evaluated_independent_turn_demo_saves_four_passing_sweeps_without_drive_feed_claim(tmp_path):
    request = ResearchRequest.from_config(preset_config("independent_turns"), **preset_study("independent_turns"))

    result = run_design(request, tmp_path)
    record = DesignService.load_run(tmp_path)

    assert result["pipeline_stage"] == "evaluated"
    assert result["summary"]["sheet_accepted"] is True
    assert result["validation"]["passed"] is True and result["validation"]["point_count"] == 3112
    assert len(record.arrays["target_points_m"]) == 257
    np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].min(), -0.1, atol=1e-15)
    np.testing.assert_allclose(record.arrays["vertices_m"][:, 2].max(), 0.1, atol=1e-15)
    assert result["summary"]["relative_field_error"] <= 0.04
    assert result["validation"]["relative_field_error"] <= 0.05
    np.testing.assert_array_equal(record.arrays["target_field_T"][:, :2], 0.0)
    np.testing.assert_allclose(record.arrays["target_field_T"][:, 2], 1e-4, rtol=0, atol=0)
    contour = result["wire_fields"]["contour"]
    assert contour["available"] and not contour["current_refitted"]
    assert 0 < contour["at_stream_current"]["relative_rms_error"] < 0.05
    assert contour["at_stream_current"]["current_A"] == contour["current_A"]
    assert record.metadata["pipeline_sweeps"] == result["pipeline_sweeps"]
    sweeps = result["pipeline_sweeps"]
    assert sweeps["scope"] == "pipeline_sweep_geometry"
    assert sweeps["body_count"] == sweeps["checked_body_count"] == 4
    assert sweeps["checks_passed"] is True
    for index, body in enumerate(sweeps["bodies"], 1):
        assert body["contour_index"] == index and body["source_kind"] == "contour"
        assert body["manufacturing_ok"] is True and body["mesh_solid_ok"] is True
        assert body["vertices_key"] in record.arrays and body["faces_key"] in record.arrays
    assert not any(key.startswith("route_") for key in record.arrays)
    assert result["wire_fields"]["routed"]["available"] is False
    assert result["fabrication"]["ready"] is True
    assert result["fabrication"]["assembly_clearance"]["checked_body_pairs"] == 6
    assert result["fabrication"]["assembly_clearance"]["passed"] is True
    assert result["fabrication"]["field_metrics"]["current_source"] == "original_contour_step_no_refit"
    assert result["fabrication"]["field_metrics"]["dsv_relative_residual_percent"] < 5
    assert result["summary"]["fabrication_review_performed"] is True
    assert result["summary"]["scope"] == "fabrication_review"
    assert result["summary"]["construction_released"] is False
    assert result["summary"]["accepted"] is result["summary"]["sheet_accepted"]
