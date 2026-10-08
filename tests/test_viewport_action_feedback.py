"""View actions acknowledge the actual camera, picked layer and visibility."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_native_workbench as native
from test_viewport_mouse_navigation import mouse

from PySide6.QtCore import QEvent, Qt

from coilgen.application.presets import preset_config
from coilgen.gui.viewport import ComparisonDialog, ScientificViewport


application = native.application
scientific_viewport = native.scientific_viewport


def field_scene(viewport):
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.0, 0.0]])
    arrays = {
        "vertices_m": points + [0, 0, 0.1],
        "faces": np.array([[0, 1, 2]]),
        "target_points_m": points,
        "target_field_T": np.array([[0, 0, 1e-6], [0, 0, 2e-6], [0, 0, 3e-6]]),
        "sheet_field_T": np.array([[0, 0, 1.1e-6], [0, 0, 2.1e-6], [0, 0, 3.1e-6]]),
    }
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("sheet"))
    viewport.units.setCurrentText("µT")
    return arrays


@pytest.mark.parametrize("button", [Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton])
def test_manual_camera_gesture_marks_custom_without_reset(scientific_viewport, application, monkeypatch, button):
    viewport = scientific_viewport
    field_scene(viewport)
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda: pytest.fail("Manual camera was reset"))
    mouse(viewport, QEvent.Type.MouseButtonPress, button=button, buttons=button)
    moved = [[2, 1, 1], [0.1, 0.2, 0], [0, 0, 1]]
    viewport.plotter.camera_position = deepcopy(moved)
    mouse(viewport, QEvent.Type.MouseMove, (60, 60), buttons=button)
    mouse(viewport, QEvent.Type.MouseButtonRelease, (60, 60), button=button)
    application.processEvents()
    assert viewport.view.currentText() == "Custom"
    assert viewport.plotter.camera_position == moved
    assert viewport.view_state()["view"] == "Custom"


def test_clipping_gesture_does_not_claim_camera_changed(scientific_viewport, application):
    viewport = scientific_viewport
    field_scene(viewport)
    viewport.clip.setChecked(True)
    mouse(viewport, QEvent.Type.MouseButtonPress, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton)
    mouse(viewport, QEvent.Type.MouseMove, (60, 60), buttons=Qt.MouseButton.LeftButton)
    mouse(viewport, QEvent.Type.MouseButtonRelease, (60, 60), button=Qt.MouseButton.LeftButton)
    application.processEvents()
    assert viewport.view.currentText() == "Isometric"


def test_same_camera_preset_can_be_reselected_and_programmatic_presets_still_apply(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    calls = []
    monkeypatch.setattr(viewport.plotter, "view_isometric", lambda: calls.append("Isometric"))
    monkeypatch.setattr(viewport.plotter, "view_xy", lambda: calls.append("XY"))
    assert viewport.view.currentText() == "Isometric"
    viewport.view.activated.emit(viewport.view.currentIndex())
    assert calls == ["Isometric"]
    viewport.view.setCurrentText("XY")
    assert calls == ["Isometric", "XY"]
    viewport.view.activated.emit(viewport.view.currentIndex())
    assert calls == ["Isometric", "XY", "XY"]


def test_custom_camera_roundtrips_without_refitting(scientific_viewport, application, monkeypatch):
    viewport = scientific_viewport
    field_scene(viewport)
    mouse(viewport, QEvent.Type.MouseButtonPress, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton)
    viewport.plotter.camera_position = [[2, 1, 1], [0.1, 0.2, 0], [0, 0, 1]]
    mouse(viewport, QEvent.Type.MouseButtonRelease, button=Qt.MouseButton.LeftButton)
    application.processEvents()
    saved = viewport.view_state()
    assert saved["view"] == "Custom"
    viewport.view.setCurrentText("XY")
    monkeypatch.setattr(viewport.plotter, "reset_camera", lambda: pytest.fail("Saved custom camera was refitted"))
    assert viewport.restore_view_state(saved)
    assert viewport.view_state() == saved


def test_manual_comparison_camera_updates_both_preset_labels(scientific_viewport, application, monkeypatch):
    arrays = field_scene(scientific_viewport)
    record = SimpleNamespace(directory=Path("saved"), metadata={}, arrays=arrays)
    dialog = ComparisonDialog(
        record, record, viewport_factory=lambda parent: ScientificViewport(parent, synchronous_preview=True)
    )
    try:
        for pane in dialog.panes:
            monkeypatch.setattr(pane.plotter, "reset_camera", lambda: pytest.fail("Linked custom camera was reset"))
        pane = dialog.left
        mouse(pane, QEvent.Type.MouseButtonPress, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.LeftButton)
        pane.plotter.camera_position = [[2, 1, 1], [0.1, 0.2, 0], [0, 0, 1]]
        mouse(pane, QEvent.Type.MouseButtonRelease, button=Qt.MouseButton.LeftButton)
        application.processEvents()
        assert all(pane.view.currentText() == "Custom" for pane in dialog.panes)
        assert dialog.left.plotter.camera is dialog.right.plotter.camera
    finally:
        dialog.close()
        application.processEvents()


def test_primary_probe_reports_actual_dataset_distance_units_and_stored_sample(scientific_viewport):
    viewport = scientific_viewport
    arrays = field_scene(viewport)
    before = {name: values.copy() for name, values in viewport._arrays.items()}
    actor = viewport.plotter.meshes[-1][2]
    viewport.plotter.picked_point = [0, 0.009, 0]
    viewport._picked_actor(actor)
    text = viewport.probe.text()
    assert "Selected Continuous-sheet field" in text
    assert "Nearest displayed sample in Continuous-sheet field" in text
    assert "(0, 0.01, 0) m" in text and "distance 0.001 m" in text
    assert "Bz [µT]: 2.1" in text and "no interpolation" in text
    for name, values in before.items():
        np.testing.assert_array_equal(viewport._arrays[name], values)
    np.testing.assert_array_equal(viewport._arrays["sheet_field_T"], arrays["sheet_field_T"])


@pytest.mark.parametrize("layer", ["support", "roi"])
def test_context_layer_pick_does_not_borrow_primary_field_value(scientific_viewport, layer):
    viewport = scientific_viewport
    field_scene(viewport)
    viewport.roi_layer.setChecked(True)
    mesh, _, actor = viewport.plotter.meshes[0 if layer == "support" else 1]
    viewport.plotter.picked_point = mesh.points[0]
    viewport._picked_actor(actor)
    text = viewport.probe.text()
    assert f"Selected {'Support layer' if layer == 'support' else 'Target points layer'}" in text
    assert "viewing Continuous-sheet field" in text
    assert "no field value sampled at this position" in text
    assert "Bz [µT]" not in text and "Nearest displayed sample" not in text


def test_clipped_probe_uses_original_samples_not_renderer_interpolation(scientific_viewport, monkeypatch):
    viewport = scientific_viewport
    field_scene(viewport)
    actor = object()

    def clipped(mesh, **_options):
        interpolated = viewport.pv.PolyData(np.array([[0, 0.009, 0]]))
        interpolated["Bz [µT]"] = np.array([999.0])
        viewport.plotter.plane_clipped_meshes.append(interpolated)
        return actor

    monkeypatch.setattr(viewport.plotter, "add_mesh_clip_plane", clipped)
    viewport.clip.setChecked(True)
    viewport.plotter.picked_point = [0, 0.009, 0]
    viewport._picked_actor(actor)
    text = viewport.probe.text()
    assert "Nearest stored sample before clipping" in text
    assert "distance 0.001 m" in text and "Bz [µT]: 2.1" in text
    assert "999" not in text and "no interpolation" in text


def test_hidden_support_is_explained_and_can_be_restored_without_changing_model(scientific_viewport):
    viewport = scientific_viewport
    viewport.render_config(preset_config())
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("support"))
    before = viewport._support.points.copy()
    viewport.support_layer.setChecked(False)
    assert not viewport.hidden_geometry.isHidden()
    assert "support hidden" in viewport.caption.text()
    assert viewport.current_capability.available
    viewport.show_support.click()
    assert viewport.hidden_geometry.isHidden()
    assert viewport.support_layer.isChecked()
    assert viewport.dataset.currentData() == "support"
    np.testing.assert_array_equal(viewport._support.points, before)
