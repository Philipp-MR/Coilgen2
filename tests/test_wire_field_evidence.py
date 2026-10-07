"""Wire maps retain their evaluated geometry, drive and global target-zero-safe errors."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.research.application import _wire_field_evidence, _wire_field_metrics


@pytest.fixture
def evaluated_result():
    route = np.array([[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0]])
    centroid = route + np.array([0, 0, 0.002])
    target = SimpleNamespace(
        b=np.array([[0, 0, 0], [0, 0, 10e-6], [0, 0, -5e-6], [0, 0, 1e-6]]),
        weights=np.array([1, 3, 7, 0]),
        suppressed=np.array([False, False, True, False]),
    )
    contour = np.array([[0, 0, 1e-6], [0, 0, 12e-6], [0, 0, 1], [0, 0, 1e9]])
    routed = contour.copy()
    routed[1, 2] = 11e-6
    return SimpleNamespace(
        input_data=CoilGenConfig(field_evaluation_current_source="target_fit"),
        target_field=target,
        coil_parts=[
            SimpleNamespace(
                contour_step=2.0,
                wire_path=SimpleNamespace(v=route),
                fabrication_path=SimpleNamespace(wire_points_m=centroid),
            )
        ],
        needed_current_layout=3.0,
        field_by_unconnected_loops=contour,
        field_loops_per1Amp=contour / 3,
        field_by_layout=routed,
        field_layout_per1Amp=routed / 3,
    )


def test_saved_wire_fields_preserve_reported_and_original_drive_without_recalculation(evaluated_result, monkeypatch):
    monkeypatch.setattr("coilgen.physics.biot_savart_wire_path", lambda *_a, **_k: pytest.fail("Field recalculated"))
    result = evaluated_result
    arrays, evidence = _wire_field_evidence(result)
    np.testing.assert_array_equal(arrays["contour_field_T"], result.field_by_unconnected_loops)
    np.testing.assert_array_equal(arrays["routed_field_T"], result.field_by_layout)
    np.testing.assert_array_equal(arrays["contour_field_at_stream_current_T"], result.field_loops_per1Amp * 2)
    np.testing.assert_array_equal(arrays["routed_field_at_stream_current_T"], result.field_layout_per1Amp * 2)
    np.testing.assert_array_equal(
        arrays["evaluated_winding_0_points_m"], result.coil_parts[0].fabrication_path.wire_points_m
    )
    assert evidence["routed"]["geometry_sources"] == ["fabricated_centroid"]
    assert evidence["contour"]["current_A"] == 3
    assert evidence["contour"]["current_refitted"]
    original = evidence["contour"]["at_stream_current"]
    assert original["current_A"] == 2 and original["current_refitted"] is False
    assert evidence["contour"]["sample_count"] == 2
    assert evidence["contour"]["rms_error_T"] == pytest.approx(np.sqrt(13 / 4) * 1e-6)
    assert evidence["contour"]["target_rms_T"] == pytest.approx(np.sqrt(75) * 1e-6)
    assert evidence["contour"]["relative_rms_error"] == pytest.approx(np.sqrt(13 / 300))
    assert evidence["contour"]["peak_error_T"] == pytest.approx(2e-6)
    assert all(
        entry["acceptance_claim"] is False and entry["independent_sampling"] is False for entry in evidence.values()
    )


@pytest.mark.parametrize("skipped", [False, True])
def test_unconnected_layout_fallback_never_becomes_routed_evidence(evaluated_result, skipped):
    result = evaluated_result
    if skipped:
        result.input_data.skip_postprocessing = True
    else:
        result.coil_parts[0].wire_path = None
        result.coil_parts[0].fabrication_path = None
    arrays, evidence = _wire_field_evidence(result)
    assert "contour_field_T" in arrays
    assert "routed_field_T" not in arrays and "evaluated_winding_0_points_m" not in arrays
    assert evidence["routed"]["available"] is False
    assert "copies contours" in evidence["routed"]["reason"]


def test_connected_route_is_labeled_when_no_centroid_was_realized(evaluated_result):
    result = evaluated_result
    result.coil_parts[0].fabrication_path = None
    result.input_data.research_physics = {"objective": "power"}
    result.needed_current_layout = 2
    arrays, evidence = _wire_field_evidence(result)
    assert evidence["routed"]["geometry_sources"] == ["connected_wire"]
    assert evidence["routed"]["current_source"] == "stream_function"
    assert evidence["routed"]["current_refitted"] is False
    np.testing.assert_array_equal(arrays["evaluated_winding_0_points_m"], result.coil_parts[0].wire_path.v)


def test_mixed_original_currents_are_explicitly_unavailable(evaluated_result):
    result = evaluated_result
    part = result.coil_parts[0]
    result.coil_parts.append(
        SimpleNamespace(contour_step=4, wire_path=part.wire_path, fabrication_path=part.fabrication_path)
    )
    arrays, evidence = _wire_field_evidence(result)
    assert "routed_field_T" in arrays
    assert "routed_field_at_stream_current_T" not in arrays
    assert evidence["routed"]["at_stream_current"]["available"] is False
    assert "common" in evidence["routed"]["at_stream_current"]["reason"]


def test_zero_target_reports_absolute_residual_and_undefined_global_relative_error(evaluated_result):
    result = evaluated_result
    result.target_field.b[:] = 0
    metrics = _wire_field_metrics(result.field_by_layout, result.target_field, result.input_data)
    assert metrics["available"] and np.isfinite(metrics["rms_error_T"])
    assert metrics["target_rms_T"] == 0 and metrics["relative_rms_error"] is None


def test_vector_global_rms_uses_magnitude_and_includes_active_target_nodes(evaluated_result):
    result = evaluated_result
    result.input_data.target_field_component = "vector"
    field = result.target_field.b.copy()
    field[0] = [3e-6, 0, 4e-6]
    field[1, 0] = 2e-6
    metrics = _wire_field_metrics(field, result.target_field, result.input_data)
    assert metrics["rms_convention"] == "vector_magnitude"
    assert metrics["rms_error_T"] == pytest.approx(np.sqrt(37 / 4) * 1e-6)
    assert metrics["peak_error_T"] == pytest.approx(5e-6)
    assert metrics["relative_rms_error"] == pytest.approx(np.sqrt(37 / 300))


def test_nonfinite_active_wire_field_is_unavailable(evaluated_result):
    result = evaluated_result
    result.field_by_layout[0, 2] = np.nan
    arrays, evidence = _wire_field_evidence(result)
    assert evidence["routed"]["available"] is False
    assert "routed_field_T" not in arrays
