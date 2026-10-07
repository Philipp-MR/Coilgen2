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
    assert "Optimize sheet" in item(viewport, "sheet").toolTip()
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
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, new["routed_field_T"][:, 2])


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
