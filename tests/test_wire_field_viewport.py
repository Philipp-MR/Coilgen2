"""Wire maps and residuals display their saved drive and sample grid."""

import numpy as np

import test_native_workbench as workbench_tests

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def wire_data():
    target = np.array([[0, 0, 0], [0, 0, 10e-6]])
    stored = np.array([[0, 0, 1e-6], [0, 0, 12e-6]])
    arrays = {
        "target_points_m": np.array([[0, 0, 0], [0, 0, 0.01]]),
        "target_field_T": target,
        "contour_field_T": stored,
        "contour_field_at_stream_current_T": stored * 2 / 3,
        "routed_field_T": stored,
        "routed_field_at_stream_current_T": stored * 2 / 3,
    }
    metrics = {
        "current_A": 3,
        "current_source": "target_fit",
        "current_refitted": True,
        "geometry_sources": ["oriented_unconnected_contours"],
        "rms_error_T": np.sqrt(2.5) * 1e-6,
        "relative_rms_error": np.sqrt(0.05),
        "component": "Bz",
        "at_stream_current": {"current_A": 2, "current_refitted": False},
    }
    return arrays, {
        "wire_fields": {"contour": metrics, "routed": dict(metrics, geometry_sources=["fabricated_centroid"])}
    }


def test_wire_viewport_original_drive_and_signed_residual_preserve_zero_nodes(scientific_viewport):
    viewport = scientific_viewport
    arrays, metadata = wire_data()
    viewport.display_run(arrays, metadata)
    viewport.units.setCurrentText("µT")
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("contour_field"))
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [1, 12])
    assert "stored refitted current 3 A" in viewport.caption.text()
    assert "global weighted Bz RMS" in viewport.caption.text()
    assert "no independent wire acceptance" in viewport.caption.text()
    viewport.wire_current.setCurrentIndex(1)
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [2 / 3, 8])
    assert "original contour current 2 A" in viewport.caption.text()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("contour_residual"))
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [2 / 3, -2])
    assert "ΔBz [µT]" == viewport._dataset_mesh.active_scalars_name
    np.testing.assert_array_equal(viewport._dataset_mesh.points, arrays["target_points_m"])
    np.testing.assert_array_equal(arrays["target_field_T"], [[0, 0, 0], [0, 0, 10e-6]])


def test_wire_vector_residual_is_magnitude_of_vector_error(scientific_viewport):
    viewport = scientific_viewport
    arrays, metadata = wire_data()
    arrays["routed_field_T"] = np.array([[3e-6, 0, 4e-6], [0, 0, 10e-6]])
    viewport.display_run(arrays, metadata)
    viewport.units.setCurrentText("µT")
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_residual"))
    viewport.component.setCurrentText("|B|")
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [5, 0])
    assert viewport._dataset_mesh.active_scalars_name == "|B − target| [µT]"
    assert "fabricated_centroid" in viewport.caption.text()


def test_original_current_dataset_never_silently_falls_back(scientific_viewport):
    viewport = scientific_viewport
    arrays, metadata = wire_data()
    arrays.pop("routed_field_at_stream_current_T")
    metadata["wire_fields"]["routed"]["at_stream_current"] = {"reason": "Common original current unavailable."}
    viewport.display_run(arrays, metadata)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_field"))
    assert viewport._dataset_mesh is not None
    errors = []
    viewport.error.connect(errors.append)
    viewport.wire_current.setCurrentIndex(1)
    assert viewport.wire_current.currentIndex() == 0
    assert not viewport.wire_current.model().item(1).isEnabled()
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, arrays["routed_field_T"][:, 2])
    assert errors and "Current selection unavailable" in errors[-1]


def test_evaluated_winding_geometry_is_separate_from_earlier_route(scientific_viewport):
    viewport = scientific_viewport
    route = np.array([[0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0]])
    centroid = route + [0, 0, 0.002]
    viewport.display_run({"route_0_points_m": route, "evaluated_winding_0_points_m": centroid})
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("evaluated_winding"))
    assert "evaluated winding field paths" in viewport.caption.text()
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, centroid)


def test_current_selector_is_inactive_for_fixed_reviewed_field_and_restored_for_wire_maps(scientific_viewport):
    viewport = scientific_viewport
    arrays, metadata = wire_data()
    arrays["fabricated_points_m"] = arrays["target_points_m"]
    arrays["fabricated_field_T"] = arrays["routed_field_at_stream_current_T"].copy()
    metadata["fabricated_physics"] = {"current_source": "original_contour_current_no_refit"}
    viewport.display_run(arrays, metadata)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_field"))
    viewport.wire_current.setCurrentIndex(1)
    assert viewport.wire_current.isEnabled()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("fabricated"))
    assert not viewport.wire_current.isEnabled()
    assert "fixed saved current policy" in viewport.wire_current.toolTip()
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, arrays["fabricated_field_T"][:, 2])
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_field"))
    assert viewport.wire_current.isEnabled() and viewport.wire_current.currentIndex() == 1
    np.testing.assert_array_equal(
        viewport._dataset_mesh.active_scalars, arrays["routed_field_at_stream_current_T"][:, 2]
    )
