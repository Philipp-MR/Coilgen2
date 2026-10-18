"""Smoothing is a scientific contour-stage change, before any interconnection."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, invalidated_stage


def _config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=5,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power"},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=5,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        levels=6,
        min_point_loop_number=3,
        min_turn_distance=0.01,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_calculation_min_winding_distance=True,
        skip_inductance_calculation=True,
        progress_flag=False,
        plot_flag=False,
    )


def test_smoothing_reuses_the_sheet_but_invalidates_old_contours_and_roundtrips(tmp_path, monkeypatch):
    original = _config()
    sheet = CoilGenPipeline(original).run_stage("sheet")
    baseline = CoilGenPipeline(original).run_stage("contours", sheet)
    smoothing = replace(
        original,
        contour_path_regularization_mode="parameterized_bspline",
        contour_smoothing_scale_mode="mesh_coupled",
        contour_smoothing_mesh_gain=0.1,
        contour_smoothing_max_displacement=0.003,
    )
    assert invalidated_stage(original, smoothing) is PipelineStage.CONTOURS

    def unexpected_sheet(*_args):
        pytest.fail("Changing contour smoothing must not resolve the current sheet")

    monkeypatch.setattr(CoilGenPipeline, "_solve_sheet", unexpected_sheet)
    with pytest.raises(ValueError, match="Cannot reuse contours"):
        CoilGenPipeline(smoothing).run_stage("routed", baseline)
    smoothed = CoilGenPipeline(smoothing).run_stage("contours", sheet)
    path = smoothed.save(tmp_path / "smoothed.cgstage")
    loaded = PipelineCheckpoint.load(path)
    np.testing.assert_array_equal(loaded.sf_b_field, sheet.sf_b_field)
    for first, second in zip(loaded.parts, baseline.parts):
        assert first.contour_step == second.contour_step
        assert len(first.contour_lines) == len(second.contour_lines)
        assert first.contour_level_diagnostics["contour_path_smoothing_applied"]
        assert first.contour_level_diagnostics["contour_path_smoothing_max_applied_displacement_m"] <= 0.003001
        assert first.contour_level_diagnostics["contour_path_smoothing_min_distance_after_m"] >= 0.01
        for smooth, raw in zip(first.contour_lines, second.contour_lines):
            assert smooth.potential == raw.potential
            assert smooth.current_orientation == raw.current_orientation
            np.testing.assert_array_equal(smooth.v[0], smooth.v[-1])

    # Turning the option off recovers the original winding inputs exactly.
    disabled = replace(smoothing, contour_path_regularization_mode="off")
    restored = CoilGenPipeline(disabled).run_stage("contours", sheet)
    for first, second in zip(restored.parts, baseline.parts):
        for actual, expected in zip(first.contour_lines, second.contour_lines):
            np.testing.assert_array_equal(actual.v, expected.v)
            np.testing.assert_array_equal(actual.segment_faces, expected.segment_faces)
            np.testing.assert_array_equal(actual.segment_barycentric, expected.segment_barycentric)


def test_interconnection_receives_the_smoothed_surface_contours(monkeypatch):
    config = replace(
        _config(),
        skip_postprocessing=False,
        contour_path_regularization_mode="parameterized_bspline",
        contour_smoothing_scale_mode="mesh_coupled",
        contour_smoothing_mesh_gain=0.1,
        contour_smoothing_max_displacement=0.003,
    )
    received = []

    def inspect_routing_inputs(_pipeline, state, _context):
        for part in state.parts:
            assert part.contour_level_diagnostics["contour_path_smoothing_applied"]
            assert part.contour_level_diagnostics["contour_path_smoothing_scope"] == "unconnected_contours"
            assert part.wire_path is None
            for contour in part.contour_lines:
                triangles = part.mesh.vertices[part.mesh.faces[contour.segment_faces]]
                endpoints = np.einsum("nkj,nji->nki", contour.segment_barycentric, triangles)
                np.testing.assert_allclose(endpoints[:, 0], contour.v[:-1], atol=1e-12)
                np.testing.assert_allclose(endpoints[:, 1], contour.v[1:], atol=1e-12)
            received.append(len(part.contour_lines))

    monkeypatch.setattr(CoilGenPipeline, "_route_and_realize", inspect_routing_inputs)
    checkpoint = CoilGenPipeline(config).run_stage("routed")
    assert checkpoint.stage is PipelineStage.ROUTED
    assert received and all(count > 0 for count in received)
