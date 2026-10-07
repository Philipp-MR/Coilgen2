"""On-demand viewport controls retain stored fields and display-state semantics."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QStyle, QStyleOptionButton

import test_native_workbench as native_tests

application = native_tests.application
scientific_viewport = native_tests.scientific_viewport


def field_run(viewport):
    points = np.array([[0, 0, 0.0], [0.01, 0, 0], [0, 0.01, 0]])
    fields = np.array([[0, 0, 1e-6], [0, 0, 2e-6], [0, 0, 3e-6]])
    arrays = {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "stream_function_A": np.arange(3, dtype=float),
        "target_points_m": points,
        "target_field_T": fields,
        "routed_field_T": fields + np.array([0, 0, 0.1e-6]),
    }
    viewport.display_run(arrays)
    viewport.inspect_task("field")
    return arrays


def test_secondary_controls_live_in_labelled_popups(scientific_viewport, application):
    viewport = scientific_viewport
    field_run(viewport)
    viewport.resize(600, 500)
    viewport.show()
    application.processEvents()
    assert viewport.layers_button.text() == "Layers"
    assert viewport.display_button.text() == "Display"
    assert viewport.layers_button.menu() is viewport.layers_menu
    assert viewport.display_button.menu() is viewport.display_menu
    assert viewport.layers_button.isVisible() and viewport.display_button.isVisible()
    assert not viewport.support_layer.isVisible()
    assert not viewport.roi_layer.isVisible()
    assert not viewport.orthographic.isVisible()
    assert not viewport.edges.isVisible()
    assert not viewport.clip.isVisible()
    assert not viewport.slice_axis.isVisible()
    assert viewport.field_controls.isVisible()
    assert viewport.component.isVisible() and viewport.units.isVisible() and viewport.wire_current.isVisible()
    assert "3 saved samples; no interpolation" in viewport.caption.text()


def test_layer_popup_edits_existing_checkbox_and_preserves_inspection(scientific_viewport, application):
    viewport = scientific_viewport
    arrays = field_run(viewport)
    viewport.show()
    viewport.layers_menu.popup(viewport.mapToGlobal(QPoint(10, 10)))
    application.processEvents()
    assert viewport.support_layer.isVisible()
    assert viewport.roi_layer.isVisible()
    assert viewport.support_layer.isChecked()
    option = QStyleOptionButton()
    option.initFrom(viewport.support_layer)
    indicator = viewport.support_layer.style().subElementRect(
        QStyle.SubElement.SE_CheckBoxIndicator, option, viewport.support_layer
    )
    assert not indicator.isEmpty()
    QTest.mouseClick(viewport.support_layer, Qt.MouseButton.LeftButton, pos=indicator.center())
    assert not viewport.support_layer.isChecked()
    assert viewport.inspection_task.currentData() == "field"
    assert viewport.inspection_buttons["field"].isChecked()
    viewport.layers_menu.hide()
    assert not viewport.support_layer.isVisible()
    assert viewport.view_state()["support_layer"] is False
    for name, values in arrays.items():
        np.testing.assert_array_equal(viewport._arrays[name], values)


def test_display_popup_exposes_actual_sampling_and_projection_controls(scientific_viewport, application):
    viewport = scientific_viewport
    field_run(viewport)
    viewport.show()
    before = deepcopy(viewport.view_state())
    viewport.display_menu.popup(viewport.mapToGlobal(QPoint(10, 10)))
    application.processEvents()
    assert viewport.orthographic.isVisible() and viewport.edges.isVisible() and viewport.clip.isVisible()
    assert viewport.slice_axis.isVisible() and viewport.slice_position.isVisible()
    assert viewport.view_state() == before
    viewport.slice_axis.setCurrentText("x")
    viewport.slice_position.setValue(0)
    assert viewport.slice_position.isEnabled()
    assert viewport._dataset_mesh.n_points == 2
    assert "x ≈ 0 m · 2 samples" in viewport.slice_label.text()
    assert "displayed slab: x ≈ 0 m · 2 samples; no interpolation" in viewport.caption.text()
    assert viewport.inspection_task.currentData() == "field"
    viewport.display_menu.hide()
    assert "2 samples" in viewport.caption.text()


def test_hidden_controls_roundtrip_existing_session_schema(scientific_viewport):
    viewport = scientific_viewport
    field_run(viewport)
    viewport.units.setCurrentText("µT")
    viewport.slice_axis.setCurrentText("y")
    viewport.slice_position.setValue(100)
    viewport.support_layer.setChecked(False)
    viewport.orthographic.setChecked(False)
    viewport.edges.setChecked(False)
    state = deepcopy(viewport.view_state())
    viewport.units.setCurrentText("mT")
    viewport.slice_axis.setCurrentText("All points")
    viewport.slice_position.setValue(50)
    viewport.support_layer.setChecked(True)
    viewport.orthographic.setChecked(True)
    viewport.edges.setChecked(True)
    assert viewport.restore_view_state(state)
    assert viewport.view_state() == state
    assert viewport._dataset_mesh.n_points == 1
    assert viewport.inspection_task.currentData() is None
    assert viewport.layers_menu.isHidden() and viewport.display_menu.isHidden()


def test_geometry_hides_sampling_but_keeps_display_controls_available(scientific_viewport, application):
    viewport = scientific_viewport
    field_run(viewport)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("stream_function"))
    viewport.show()
    viewport.display_menu.popup(viewport.mapToGlobal(QPoint(10, 10)))
    application.processEvents()
    assert viewport.slice_controls.isHidden()
    assert not viewport.slice_axis.isEnabled() and not viewport.slice_position.isEnabled()
    assert viewport.orthographic.isVisible() and viewport.edges.isVisible() and viewport.clip.isVisible()
    assert viewport.field_controls.isVisible()
    assert viewport.wire_current.isVisible()
    assert viewport.field_options.isHidden()
    assert not viewport.component.isVisible() and not viewport.units.isVisible()
    assert not viewport.component.isEnabled() and not viewport.units.isEnabled()
    viewport.display_menu.hide()


def test_popup_keyboard_focus_reaches_controls_and_skips_disabled_sampling(scientific_viewport, application):
    viewport = scientific_viewport
    field_run(viewport)
    viewport.show()
    viewport.display_menu.popup(viewport.mapToGlobal(QPoint(10, 10)))
    application.processEvents()
    assert application.focusWidget() is viewport.orthographic
    QTest.keyClick(viewport.orthographic, Qt.Key.Key_Tab)
    assert application.focusWidget() is viewport.edges
    QTest.keyClick(viewport.edges, Qt.Key.Key_Backtab)
    assert application.focusWidget() is viewport.orthographic
    QTest.keyClick(viewport.clip, Qt.Key.Key_Tab)
    assert application.focusWidget() is viewport.slice_axis
    assert not viewport.slice_position.isEnabled()
    QTest.keyClick(viewport.slice_axis, Qt.Key.Key_Tab)
    assert application.focusWidget() is viewport.orthographic
    QTest.keyClick(viewport.orthographic, Qt.Key.Key_Space)
    assert not viewport.orthographic.isChecked()
    assert viewport.view_state()["orthographic"] is False
    QTest.keyClick(viewport.orthographic, Qt.Key.Key_Escape)
    assert viewport.display_menu.isHidden()
