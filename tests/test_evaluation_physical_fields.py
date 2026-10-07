"""Final reports use the manufactured conductor at explicitly identified currents."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.evaluation import evaluate_field_errors, final_field_diagnostics
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import create_swept_coil_meshes
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.physics import biot_savart_wire_path


@pytest.fixture
def physical_case():
    angle = np.linspace(0, 2 * np.pi, 65)
    points = 0.025 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    part = CoilPart(mesh, contour_step=2.0, wire_path=WirePath(points), contour_lines=[ContourLine(points.copy(), 1.0)])
    config = CoilGenConfig(
        conductor_cross_section_width=0.0005,
        conductor_cross_section_height=0.002,
        conductor_mounting_mode="surface_outer",
        field_evaluation_current_source="stream_function",
        parallel_worker_count=1,
        progress_flag=False,
    )
    create_swept_coil_meshes([part], config)
    coords = np.array([[0, 0, 0.01], [0.003, 0, 0.015], [0, 0.004, 0.02]])
    original = biot_savart_wire_path(points, coords, worker_count=1)
    target = TargetField(coords, original * 2, np.ones(3), np.zeros(3, bool), np.ones(3, int))
    return part, config, target, original


def test_actual_swept_centroid_is_the_authoritative_final_field(physical_case):
    part, config, target, original = physical_case
    expected = biot_savart_wire_path(part.fabrication_path.wire_points_m, target.coords, worker_count=1)
    assert not np.allclose(expected, original, rtol=1e-5, atol=0)
    result = evaluate_field_errors([part], config, target, original * 2)
    np.testing.assert_allclose(result[2], expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result[0], expected * 2, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result[3], original, rtol=1e-12, atol=0)
    diagnostics = final_field_diagnostics([part], config, target, result[2], result[3], result[5])
    assert diagnostics["field_evaluation_geometry_source"] == "fabricated_centroid"
    assert diagnostics["layout_target_source"] == "fabricated_centroid"
    assert diagnostics["field_evaluation_current_source"] == "stream_function"
    assert diagnostics["field_evaluation_current_a"] == 2
    assert not diagnostics["field_evaluation_current_refitted"]
    assert diagnostics["routing_distortion_checked"]
    difference = (expected - original)[:, 2] * 2
    assert diagnostics["routing_distortion_longitudinal_rms_T"] == pytest.approx(np.sqrt(np.mean(difference**2)))
    assert diagnostics["routing_distortion_longitudinal_peak_T"] == pytest.approx(np.max(np.abs(difference)))


def test_requested_target_fit_does_not_replace_fixed_current_routing_metrics(physical_case):
    part, config, target, original = physical_case
    config.field_evaluation_current_source = "target_projection"
    expected = biot_savart_wire_path(part.fabrication_path.wire_points_m, target.coords, worker_count=1)
    target.b = expected * 5
    result = evaluate_field_errors([part], config, target, original * 2)
    assert result[5] == pytest.approx(5)
    np.testing.assert_allclose(result[0], target.b, rtol=1e-12, atol=0)
    assert result[4]["max_rel_error_layout_vs_stream_function_field"] > 0
    diagnostics = final_field_diagnostics([part], config, target, result[2], result[3], result[5])
    assert diagnostics["field_evaluation_current_a"] == pytest.approx(5)
    assert diagnostics["field_evaluation_current_refitted"]
    assert diagnostics["routing_distortion_current_a"] == 2
    assert not diagnostics["routing_distortion_current_refitted"]
    expected_rms = np.sqrt(np.mean(((expected - original)[:, 2] * 2) ** 2))
    assert diagnostics["routing_distortion_longitudinal_rms_T"] == pytest.approx(expected_rms)


@pytest.mark.parametrize("skip_postprocessing", [False, True])
def test_no_sweep_and_explicit_contour_only_modes_keep_their_geometry_source(physical_case, skip_postprocessing):
    part, config, target, original = physical_case
    config.skip_postprocessing = skip_postprocessing
    if not skip_postprocessing:
        part.fabrication_path = None
    result = evaluate_field_errors([part], config, target, original * 2)
    np.testing.assert_allclose(result[2], original, rtol=1e-12, atol=0)
    diagnostics = final_field_diagnostics([part], config, target, result[2], result[3], result[5])
    expected = "unconnected_contours" if skip_postprocessing else "connected_wire"
    assert diagnostics["field_evaluation_geometry_source"] == expected


def test_mixed_geometry_sources_are_named_and_summed_without_a_silent_fallback(physical_case):
    part, config, target, original = physical_case
    config.routing_circuit_scope = "per_part"
    other = deepcopy(part)
    other.fabrication_path = None
    expected = biot_savart_wire_path(part.fabrication_path.wire_points_m, target.coords, worker_count=1)
    result = evaluate_field_errors([part, other], config, target, original * 4)
    np.testing.assert_allclose(result[2], expected + original, rtol=1e-12, atol=0)
    diagnostics = final_field_diagnostics([part, other], config, target, result[2], result[3], result[5])
    assert diagnostics["field_evaluation_geometry_source"] == "mixed"
    assert diagnostics["field_evaluation_part_geometry_sources"] == "fabricated_centroid;connected_wire"
    other.contour_step = 3
    diagnostics = final_field_diagnostics([part, other], config, target, result[2], result[3], result[5])
    assert not diagnostics["routing_distortion_checked"]
    assert "common" in diagnostics["routing_distortion_reason"]


def test_empty_active_dsv_is_unchecked_not_zero_distortion(physical_case):
    part, config, target, original = physical_case
    target.weights[:] = 0
    diagnostics = final_field_diagnostics([part], config, target, original, original, 2)
    assert not diagnostics["routing_distortion_checked"]
    assert "routing_distortion_longitudinal_rms_T" not in diagnostics


def test_invalid_present_fabrication_trace_does_not_fall_back_to_the_raw_route(physical_case):
    part, config, target, original = physical_case
    part.fabrication_path.points_m = np.full((3, 3), np.nan)
    with pytest.raises(ValueError, match="fabricated field trace"):
        evaluate_field_errors([part], config, target, original * 2)
