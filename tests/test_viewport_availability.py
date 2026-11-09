"""Unavailable plots never replace a valid view, source, current, or camera."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_native_workbench as native_tests

from coilgen.application.presets import preset_config
from coilgen.gui.viewport import ComparisonDialog, ScientificViewport

application = native_tests.application
scientific_viewport = native_tests.scientific_viewport


def arrays():
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    return {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "target_points_m": points,
        "target_field_T": np.zeros((3, 3)),
    }


def item(viewport, key):
    return viewport.dataset.model().item(viewport.dataset.findData(key))


def test_empty_model_and_partial_model_catalog(scientific_viewport):
    viewport = scientific_viewport
    assert all(not viewport.dataset.model().item(index).isEnabled() for index in range(viewport.dataset.count()))
    viewport.render_config(replace(preset_config(), field_shape_function="none"))
    assert item(viewport, "support").isEnabled()
    assert not item(viewport, "preview_target").isEnabled()
    assert viewport.dataset.currentData() == "support"
    assert not viewport.inspection_actions["field"].isEnabled()
    viewport.clear_model()
    assert not item(viewport, "support").isEnabled()


def test_source_boundaries_enable_only_actual_data_and_keep_rejected_plot_view(scientific_viewport):
    viewport = scientific_viewport
    saved = arrays()
    viewport.display_run(saved)
    before = deepcopy(viewport.view_state())
    scalars = viewport._dataset_mesh.active_scalars.copy()
    errors = []
    viewport.error.connect(errors.append)
    assert item(viewport, "target").isEnabled()
    assert not item(viewport, "sheet").isEnabled()
    assert "Optimize stream function" in item(viewport, "sheet").toolTip()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("sheet"))
    assert viewport.view_state() == before
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, scalars)
    assert errors and "Plot selection unavailable" in errors[-1]
    saved.update(stream_function_A=np.zeros(3), sheet_field_T=np.zeros((3, 3)))
    viewport.display_stage_preview(saved, {"pipeline_stage": "sheet"})
    assert item(viewport, "sheet").isEnabled() and item(viewport, "stream_function").isEnabled()
    assert not item(viewport, "contours").isEnabled()
    viewport.clear_stage_preview()
    assert not item(viewport, "sheet").isEnabled()
    assert item(viewport, "target").isEnabled()


def test_current_policy_is_preserved_on_source_change_and_can_be_explicitly_recovered(scientific_viewport):
    viewport = scientific_viewport
    old = arrays()
    old.update(routed_field_T=np.zeros((3, 3)), routed_field_at_stream_current_T=np.zeros((3, 3)))
    viewport.display_run(old)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_field"))
    viewport.wire_current.setCurrentIndex(1)
    new = arrays()
    new.update(routed_field_T=np.ones((3, 3)) * 1e-6, route_0_points_m=new["vertices_m"])
    viewport.display_run(new)
    assert viewport.wire_current.currentIndex() == 1
    assert not viewport.wire_current.isHidden()
    assert viewport.field_options.isHidden() and not viewport.current_controls.isHidden()
    assert viewport.wire_current.model().item(0).isEnabled()
    assert not viewport.wire_current.model().item(1).isEnabled()
    assert not item(viewport, "routed_field").isEnabled()
    assert not viewport.inspect_task("field")
    assert viewport.dataset.currentData() == "routes"
    viewport.wire_current.setCurrentIndex(0)
    assert item(viewport, "routed_field").isEnabled()
    assert viewport.inspect_task("field")
    assert viewport.dataset.currentData() == "routed_residual"
    np.testing.assert_allclose(
        viewport._dataset_mesh.active_scalars, 1e6 * new["routed_field_T"][:, 2], rtol=1e-12
    )  # µT default


@pytest.mark.parametrize("follow", [True, False])
def test_inflight_geometry_updates_plot_choices_and_respects_follow(scientific_viewport, follow):
    viewport = scientific_viewport
    live = arrays()
    contours = live["vertices_m"]
    route = np.vstack([contours, [0.01, 0.01, 0.002]])
    live["contour_0_0_points_m"] = contours
    metadata = {"pipeline_stage": "contours", "request_sha256": "a" * 64}
    viewport.display_stage_preview(live, metadata)
    assert viewport.dataset.currentData() == "contours"
    assert not item(viewport, "routes").isEnabled() and not item(viewport, "swept").isEnabled()
    viewport.roi_layer.setChecked(False)
    viewport.plotter.camera_position = [[0.3, 0.2, 0.1], [0, 0, 0], [0, 0, 1]]
    camera = deepcopy(viewport.plotter.camera_position)

    live["route_0_points_m"] = route
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "interconnections"}, follow=follow)
    assert viewport.dataset.currentData() == ("routes" if follow else "contours")
    assert item(viewport, "routes").isEnabled() and not item(viewport, "swept").isEnabled()
    assert "interconnections complete" in viewport.caption.text()
    np.testing.assert_array_equal(viewport.plotter.meshes[-1][0].points, route if follow else contours)

    live["fabrication_0_points_m"] = route
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "realized_centerlines"}, follow=follow)
    assert viewport.dataset.currentData() == ("fabrication" if follow else "contours")
    assert "realized conductor centerlines complete" in viewport.caption.text()
    live.update(swept_0_0_vertices_m=live["vertices_m"], swept_0_0_faces=live["faces"])
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "sweep"}, follow=follow)
    assert viewport.dataset.currentData() == ("swept" if follow else "contours")
    assert all(item(viewport, key).isEnabled() for key in ("contours", "routes", "fabrication", "swept"))
    assert not item(viewport, "review_conductor").isEnabled()
    assert "conductor sweep complete" in viewport.caption.text()
    assert not viewport.roi_layer.isChecked() and viewport.plotter.camera_position == camera
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routes"))
    updated_route = route + [0, 0, 0.001]
    live["route_0_points_m"] = updated_route
    viewport.display_stage_preview(live, {**metadata, "pipeline_stage": "routed"})
    assert viewport.dataset.currentData() == "routes"
    np.testing.assert_array_equal(viewport.plotter.meshes[-1][0].points, updated_route)
    viewport.follow_latest_stage_preview()
    assert viewport.dataset.currentData() == "swept"


def test_background_preview_cannot_replace_different_live_or_saved_source(scientific_viewport):
    viewport = scientific_viewport
    live = arrays()
    live["route_0_points_m"] = live["vertices_m"]
    viewport.display_stage_preview(live, {"request_sha256": "a" * 64})
    incoming = {"request_sha256": "b" * 64}
    viewport.display_stage_preview(arrays(), incoming, follow=False)
    assert viewport.dataset.currentData() == "routes"
    assert viewport.stage_preview_metadata["request_sha256"] == "a" * 64

    viewport.display_run(arrays(), {"name": "Saved run"})
    viewport.display_stage_preview(live, incoming, follow=False)
    assert viewport.view_state()["source"] == "run" and not item(viewport, "routes").isEnabled()


def test_live_evaluation_and_review_follow_each_field_on_its_recorded_grid(scientific_viewport):
    viewport = scientific_viewport
    live = arrays()
    points = live["target_points_m"]
    independent = points + [0.001, 0.002, 0.003]
    field = np.arange(9).reshape(3, 3) * 1e-6
    live.update(swept_0_0_vertices_m=points, swept_0_0_faces=live["faces"])
    metadata = {"pipeline_stage": "evaluated", "request_sha256": "a" * 64}
    viewport.display_stage_preview(live, metadata)
    assert viewport.dataset.currentData() == "swept"
    live.update(contour_field_T=field / 2, routed_field_T=field, evaluated_winding_0_points_m=points)
    metadata["wire_fields"] = {"routed": {"current_A": -2, "current_refitted": False}}
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "evaluated_winding"})
    assert viewport.dataset.currentData() == "routed_field"
    assert viewport.roi_layer.isChecked() and list(viewport._actors.values()).count("target") == 1
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, 1e6 * field[:, 2], rtol=1e-12)  # µT default
    assert "-2 A" in viewport.caption.text()
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("contour_field"))
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "evaluated_winding"})
    assert viewport.dataset.currentData() == "contour_field"

    live.update(validation_points_m=independent, validation_field_T=2 * field)
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "independent_field"})
    assert viewport.dataset.currentData() == "validation"
    np.testing.assert_array_equal(viewport._dataset_mesh.points, independent)
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, 1e6 * 2 * field[:, 2], rtol=1e-12)  # µT default
    assert list(viewport._actors.values()).count("target") == 2

    live.update(review_conductor_0_points_m=points, review_electrical_0_points_m=independent)
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "review_conductor"}, follow=False)
    assert viewport.dataset.currentData() == "validation" and item(viewport, "review_conductor").isEnabled()
    viewport.follow_latest_stage_preview()
    assert viewport.dataset.currentData() == "review_conductor"
    live.update(fabricated_points_m=points, fabricated_field_T=3 * field)
    metadata["fabricated_physics"] = {"current_source": "original_contour_current_no_refit"}
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "fabricated_field"})
    assert viewport.dataset.currentData() == "fabricated"
    assert list(viewport._actors.values()).count("target") == 1
    assert "original signed per-body contour currents" in viewport.caption.text()
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, 1e6 * 3 * field[:, 2], rtol=1e-12)  # µT default

    live.update(
        winding_validation_points_m=independent, winding_validation_field_T=4 * field, winding_validation_target_T=field
    )
    viewport.display_stage_preview(live, {**metadata, "preview_feature": "winding_validation"})
    assert viewport.dataset.currentData() == "winding_validation" and item(viewport, "swept").isEnabled()
    np.testing.assert_array_equal(viewport._dataset_mesh.points, independent)
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, 1e6 * 4 * field[:, 2], rtol=1e-12)  # µT default
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("winding_validation_residual"))
    np.testing.assert_allclose(
        viewport._dataset_mesh.active_scalars, 1e6 * (4 * field - field)[:, 2], rtol=1e-12
    )  # µT default
    assert "independent" in viewport.caption.text() and "acceptance pending" in viewport.caption.text()


def test_latest_feature_prefers_available_wire_field_and_review_paths(scientific_viewport):
    viewport = scientific_viewport
    live = arrays()
    points = live["target_points_m"]
    live.update(contour_field_T=points, routed_field_T=np.ones((2, 3)))
    viewport.display_stage_preview(live, {"preview_feature": "evaluated_winding"})
    assert viewport.dataset.currentData() == "contour_field" and not item(viewport, "routed_field").isEnabled()
    live["review_electrical_0_points_m"] = points
    viewport.display_stage_preview(live, {"preview_feature": "review_conductor"})
    assert viewport.dataset.currentData() == "review_electrical"


@pytest.mark.parametrize(
    ("new_arrays", "expected"),
    [
        ({"contour_field_T"}, "contour_field"),
        ({"contour_field_T", "routed_field_T"}, "routed_field"),
        ({"validation_points_m", "validation_field_T"}, "validation"),
        ({"review_conductor_0_points_m", "validation_points_m", "validation_field_T"}, "review_conductor"),
        ({"review_conductor_0_points_m", "fabricated_field_T"}, "fabricated"),
        ({"fabricated_field_T", "winding_validation_points_m", "winding_validation_field_T"}, "winding_validation"),
    ],
)
def test_automatic_completed_handoff_selects_newest_feature_even_if_last_preview_was_missed(
    scientific_viewport, new_arrays, expected
):
    viewport = scientific_viewport
    live = arrays()
    points = live["target_points_m"]
    live.update(swept_0_0_vertices_m=points, swept_0_0_faces=live["faces"])
    viewport.display_stage_preview(live, {"preview_feature": "sweep"})
    assert viewport.dataset.currentData() == "swept"
    viewport.plotter.camera_position = [[0.3, 0.02, 0.6], [0, 0, 0], [0, 0, 1]]
    camera = deepcopy(viewport.plotter.camera_position)
    complete = {**live, **dict.fromkeys(new_arrays, points)}
    viewport.display_run(complete, follow_latest=True)
    assert viewport.view_state()["source"] == "run" and viewport.dataset.currentData() == expected
    assert viewport.plotter.camera_position == camera and viewport.roi_layer.isChecked()
    viewport.display_run(complete)
    assert viewport.dataset.currentData() == (
        "review_conductor" if "review_conductor_0_points_m" in complete else "swept"
    )


def test_invalid_saved_selection_is_rejected_atomically(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(arrays())
    before = deepcopy(viewport.view_state())
    assert not viewport.restore_view_state(dict(before, dataset="validation", camera=[[9, 9, 9], [0, 0, 0], [0, 0, 1]]))
    assert viewport.view_state() == before
    assert viewport._dataset_mesh is not None


@pytest.mark.parametrize("status", ["cancelled", "failed", "completed"])
def test_terminal_stage_snapshot_retains_arrays_without_live_claim_and_setter_is_idempotent(
    scientific_viewport, status
):
    viewport = scientific_viewport
    viewport.display_stage_preview(arrays(), {"pipeline_stage": "prepared"})
    original = viewport._dataset_mesh.points.copy()
    notifications = []
    viewport.context_changed.connect(lambda: notifications.append(True))
    viewport.set_stage_preview_status(status)
    viewport.set_stage_preview_status(status)
    assert len(notifications) == 1
    assert f"Run snapshot · {status}" in viewport.caption.text()
    assert "Live study preview" not in viewport.caption.text()
    assert viewport.view_state()["source"] == "live"
    assert not viewport.stage_preview_metadata["acceptance_claim"]
    assert (
        viewport.dataset.model()
        .item(viewport._dataset_headers["SAVED RUN · FIELD MAPS"])
        .text()
        .startswith("RUN SNAPSHOT")
    )
    np.testing.assert_array_equal(viewport._dataset_mesh.points, original)


def test_cached_run_verification_failure_is_explicit_without_changing_displayed_field(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(arrays())
    before = viewport._dataset_mesh.active_scalars.copy()
    notifications = []
    viewport.context_changed.connect(lambda: notifications.append(True))
    viewport.set_verification_error("Array digest mismatch")
    viewport.set_verification_error("Array digest mismatch")
    assert len(notifications) == 1
    assert "Cached run · verification failed" in viewport.caption.text()
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, before)
    assert item(viewport, "target").isEnabled()


def test_comparison_catalog_requires_both_sources_without_blank_fallback(scientific_viewport):
    left = arrays()
    left["routed_field_T"] = np.ones((3, 3)) * 1e-6
    right = arrays()
    records = [
        SimpleNamespace(arrays=value, metadata={}, directory=Path(name))
        for value, name in ((left, "left"), (right, "right"))
    ]
    dialog = ComparisonDialog(
        *records, viewport_factory=lambda parent: ScientificViewport(parent, synchronous_preview=True)
    )
    try:
        assert not item(dialog.left, "routed_field").isEnabled()
        assert "Other comparison run" in item(dialog.left, "routed_field").toolTip()
        dialog.left.dataset.setCurrentIndex(dialog.left.dataset.findData("routed_field"))
        assert dialog.left.dataset.currentData() == dialog.right.dataset.currentData() == "target"
        assert dialog.left._dataset_mesh is not None and dialog.right._dataset_mesh is not None
    finally:
        dialog.close()
