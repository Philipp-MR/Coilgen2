"""Graphical choices retain native inputs and scientific inspection scope."""

from copy import deepcopy

import numpy as np
import pytest

import test_native_workbench as native_tests
import test_workbench_target_policies as target_tests
from PySide6.QtWidgets import QPushButton
from coilgen.gui.visual_selectors import CircuitSelector, SurfaceGallery

application = target_tests.application
windows = target_tests.windows
scientific_viewport = native_tests.scientific_viewport


def test_surface_tile_changes_only_source_and_preserves_exact_undo(windows):
    window = windows()
    before = window.project.values
    gallery = window.findChild(SurfaceGallery, "support_type")
    gallery.buttons[gallery.findData("create conical frustum mesh")].click()
    expected = dict(before, coil_mesh_file="create conical frustum mesh", surface_is_cylinder_flag=False)
    assert window.project.values == expected
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert window.project.values == before
    current = window.findChild(SurfaceGallery, "support_type")
    assert current.currentData() == before["coil_mesh_file"]
    assert current.buttons[current.currentIndex()].isChecked()


def test_circuit_tile_is_an_explicit_draft_until_apply_and_remains_undoable(windows):
    window = windows()
    window.select_node("winding")
    before = window.project.values
    selector = window._draft_controls["winding_intent"]
    assert isinstance(selector, CircuitSelector)
    selector.buttons[selector.findText("Independent closed turns")].click()
    assert window.project.values == before
    assert window._draft_dirty
    assert window.apply_settings()
    assert window.project.values == dict(before, contour_only_sweep_mode=True)
    window.undo_stack.undo()
    assert window.project.values == before
    assert window._draft_controls["winding_intent"].currentText() == "Connected circuit"


def test_primary_and_navigation_controls_have_icons_and_labels(windows):
    window = windows()
    assert window.compute_button.text() == "Compute full design"
    assert window.compute_button.accessibleName()
    assert not window.compute_button.icon().isNull()
    assert len(window.findChildren(QPushButton, "primary_compute")) == 1
    assert window.tree.headerItem().text(0) == "Model Builder"
    assert not hasattr(window, "workflow")
    assert window.stage_selector.isHidden()
    assert not hasattr(window, "panels_button")
    assert not window.workspace.isAncestorOf(window.view_results_button)  # Results action belongs to Jobs.
    assert all(not item.icon(0).isNull() for item in window.tree_items.values())
    assert all(not item.icon(0).isNull() for item in window.stage_items.values())
    assert all(
        window.evidence.tabText(i) and not window.evidence.tabIcon(i).isNull() for i in range(window.evidence.count())
    )


def _field_run(viewport):
    arrays = {
        "target_points_m": np.array([[0, 0, 0], [0, 0.01, 0]]),
        "target_field_T": np.array([[0, 0, 1e-6], [0, 0, 2e-6]]),
        "routed_field_T": np.array([[0, 0, 1.1e-6], [0, 0, 2.1e-6]]),
    }
    viewport.display_run(arrays)
    viewport.inspection_buttons["field"].click()
    return deepcopy(viewport._arrays)


@pytest.mark.parametrize("control", ["units", "component", "wire_current", "roi_layer", "slice_position"])
def test_field_task_and_scope_survive_display_changes(scientific_viewport, control):
    viewport = scientific_viewport
    before = _field_run(viewport)
    note = viewport.inspection_note.text()
    widget = getattr(viewport, control)
    if control in {"units", "component", "wire_current"}:
        widget.setCurrentIndex((widget.currentIndex() + 1) % widget.count())
    elif control == "roi_layer":
        widget.setChecked(not widget.isChecked())
    else:
        viewport.slice_axis.setCurrentText("y")
        widget.setValue(60)
    assert viewport.inspection_task.currentData() == "field"
    assert viewport.inspection_buttons["field"].isChecked()
    assert not viewport.inspection_note.isHidden()
    assert viewport.inspection_note.text() == note
    for key, array in before.items():
        np.testing.assert_array_equal(viewport._arrays[key], array)


def test_explicit_dataset_and_new_run_clear_previous_inspection_scope(scientific_viewport):
    viewport = scientific_viewport
    _field_run(viewport)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    assert viewport.inspection_task.currentData() is None
    assert not any(button.isChecked() for button in viewport.inspection_buttons.values())
    assert viewport.inspection_note.isHidden()
    viewport.inspect_task("wire")
    viewport.display_run({})
    assert viewport.inspection_task.currentData() is None
    assert viewport.inspection_note.isHidden()


def test_failed_target_preview_clears_unavailable_inspection_scope(scientific_viewport):
    viewport = scientific_viewport
    viewport.render_config(native_tests.preset_config())
    viewport.inspection_buttons["target"].click()
    assert viewport.inspection_task.currentData() == "target"
    viewport._preview_generation = 3
    viewport._preview_finished(3, (None, {}, "Preview unavailable", {"available": False}))
    assert not viewport.inspection_buttons["target"].isChecked()
    assert not viewport.inspection_buttons["target"].isEnabled()
    assert viewport.inspection_task.currentData() is None
    assert viewport.inspection_note.isHidden()
    assert viewport.dataset.currentData() == "support"
