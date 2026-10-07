"""One inspection menu retains exact datasets, evidence scope, and empty-model safety."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import test_native_workbench as native_tests

from coilgen.application.presets import preset_config
from coilgen.gui.viewport import _preview_data

application = native_tests.application
scientific_viewport = native_tests.scientific_viewport


def _run_arrays():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    target = np.array([[0.0, 0.0, 1e-6], [0.0, 0.0, 2e-6], [0.0, 0.0, 3e-6]])
    return {
        "vertices_m": points,
        "faces": np.array([[0, 1, 2]]),
        "target_points_m": points,
        "target_field_T": target,
        "routed_field_T": target + np.array([0.0, 0.0, 0.1e-6]),
        "routed_field_at_stream_current_T": target + np.array([0.0, 0.0, 0.2e-6]),
        "route_0_points_m": points,
    }


def test_one_visible_inspection_menu_replaces_shortcut_row(scientific_viewport, application):
    viewport = scientific_viewport
    viewport.resize(600, 500)
    viewport.show()
    application.processEvents()
    assert viewport.inspection_button.text() == "Inspect…"
    assert viewport.inspection_button.isVisible()
    assert viewport.inspection_button.menu() is viewport.inspection_menu
    assert viewport.dataset.isVisible()
    assert viewport.inspection_task.isHidden()
    assert all(button.isHidden() for button in viewport.inspection_buttons.values())
    assert [action.text() for action in viewport.inspection_actions.values()] == [
        "Target and ROI",
        "Field accuracy",
        "Wire fit",
        "Connections",
    ]


def test_menu_click_selects_saved_evidence_once_without_changing_arrays(scientific_viewport, application):
    viewport = scientific_viewport
    viewport.display_run(_run_arrays())
    viewport.units.setCurrentText("µT")
    viewport.wire_current.setCurrentText("Original contour current")
    before = deepcopy(viewport._arrays)
    selections = []
    viewport.dataset.currentIndexChanged.connect(selections.append)
    viewport.show()
    viewport.inspection_menu.popup(viewport.mapToGlobal(QPoint(10, 10)))
    application.processEvents()
    assert viewport.inspection_task.currentData() is None
    action = viewport.inspection_actions["field"]
    QTest.mouseClick(
        viewport.inspection_menu,
        Qt.MouseButton.LeftButton,
        pos=viewport.inspection_menu.actionGeometry(action).center(),
    )
    assert viewport.inspection_menu.isHidden()
    assert viewport.dataset.currentData() == "routed_residual"
    assert viewport.inspection_task.currentData() == "field"
    assert action.isChecked()
    assert viewport.inspection_buttons["field"].isChecked()
    assert "Connected-winding residual" in viewport.inspection_note.text()
    assert viewport.units.currentText() == "µT"
    assert viewport.wire_current.currentText() == "Original contour current"
    assert viewport._dataset_mesh is not None
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [0.2, 0.2, 0.2])
    assert selections == [viewport.dataset.findData("routed_residual")]
    for name, values in before.items():
        np.testing.assert_array_equal(viewport._arrays[name], values)


def test_inspection_menu_and_legacy_shortcuts_share_selection(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(_run_arrays())
    viewport.inspection_buttons["wire"].click()
    assert viewport.dataset.currentData() == "routes"
    assert viewport.inspection_actions["wire"].isChecked()
    assert "centerlines" in viewport.inspection_note.text()
    viewport.inspection_actions["connections"].trigger()
    assert viewport.inspection_task.currentData() == "connections"
    assert viewport.dataset.currentData() == "routes"
    assert [key for key, action in viewport.inspection_actions.items() if action.isChecked()] == ["connections"]
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("target"))
    assert viewport.inspection_task.currentData() is None
    assert not any(action.isChecked() for action in viewport.inspection_actions.values())
    assert viewport.inspection_note.isHidden()


def test_empty_model_clears_actors_and_ignores_outstanding_preview(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(_run_arrays())
    viewport.inspect_task("field")
    assert viewport._actors
    viewport._preview_arrays = _run_arrays()
    viewport._support = viewport._mesh(viewport._arrays["vertices_m"], viewport._arrays["faces"])
    old_generation = viewport._preview_generation
    outstanding_worker = object()
    viewport._preview_worker = outstanding_worker
    viewport._pending_preview = object()
    preflight = []
    viewport.preflight_ready.connect(preflight.append)
    viewport.clear_model()
    assert viewport._preview_worker is outstanding_worker
    assert viewport._pending_preview is None
    assert viewport._preview_generation > old_generation
    assert viewport._preview_arrays == viewport._arrays == viewport._metadata == {}
    assert viewport._support is None and viewport._dataset_mesh is None
    assert viewport._actors == {} and viewport.plotter.meshes == []
    assert viewport.dataset.currentData() == "support"
    assert "No model loaded" in viewport.caption.text()
    assert preflight[-1]["available"] is False and preflight[-1]["target"]["available"] is False
    old_mesh = SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    viewport._preview_finished(old_generation, (old_mesh, _run_arrays(), "", {"available": True}))
    assert viewport._preview_worker is None
    assert viewport._preview_arrays == {} and viewport._support is None and viewport._actors == {}
    assert len(preflight) == 1


def test_new_preview_can_follow_cleared_model(scientific_viewport):
    viewport = scientific_viewport
    viewport.clear_model()
    new_mesh = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    viewport._preview_finished(viewport._preview_generation, (new_mesh, {}, "", {"available": True}))
    assert viewport._support is not None
    assert viewport._actors
    assert viewport._preview_error == ""


def test_unspecified_support_preview_needs_no_backend_worker(scientific_viewport, monkeypatch, tmp_path):
    import coilgen.geometry as geometry

    def unexpected(*_args, **_kwargs):
        raise AssertionError("An unspecified support needs no geometry loading or preview worker")

    monkeypatch.setattr(geometry, "load_mesh_from_config", unexpected)
    viewport = scientific_viewport
    monkeypatch.setattr(viewport, "_start_preview", unexpected)
    config = replace(preset_config(), coil_mesh_file="none", field_shape_function="none")
    preflight, errors = [], []
    viewport.preflight_ready.connect(preflight.append)
    viewport.error.connect(errors.append)
    viewport.render_config(config, tmp_path)
    assert viewport._preview_worker is None
    assert viewport._support is None and viewport._preview_arrays == {}
    assert "Choose a support surface" in viewport.caption.text()
    assert preflight[-1]["available"] is False
    assert errors == []


def test_partial_model_keeps_support_without_inventing_target(monkeypatch, tmp_path):
    import coilgen.geometry as geometry
    import coilgen.target as target_module

    mesh = SimpleNamespace(
        vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        metadata={"geometry_type": "cylinder"},
    )
    monkeypatch.setattr(geometry, "load_mesh_from_config", lambda *_args: mesh)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("An unspecified target must not be evaluated")

    monkeypatch.setattr(target_module, "define_target_field", unexpected)
    config = replace(preset_config(), field_shape_function="none")
    support, arrays, message, diagnostics = _preview_data(config, tmp_path)
    assert support is mesh
    assert arrays == {}
    assert "Choose a target field" in message
    assert diagnostics["available"] is True
    assert diagnostics["geometry"]["vertex_count"] == 3
    assert diagnostics["target"]["available"] is False


def test_imported_target_is_not_mistaken_for_missing_analytic_expression(monkeypatch, tmp_path):
    import coilgen.application.diagnostics as diagnostics_module
    import coilgen.geometry as geometry
    import coilgen.target as target_module

    mesh = object()
    target = SimpleNamespace(coords=np.zeros((1, 3)), b=np.zeros((1, 3)), suppressed=np.array([False]))
    calls = []
    monkeypatch.setattr(geometry, "load_mesh_from_config", lambda *_args: mesh)
    monkeypatch.setattr(geometry, "load_optional_target_mesh", lambda *_args: None)
    monkeypatch.setattr(geometry, "load_secondary_target_mesh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(target_module, "define_target_field", lambda *_args: calls.append(True) or target)
    monkeypatch.setattr(diagnostics_module, "preview_diagnostics", lambda *_args, **_kwargs: {"available": True})
    config = replace(preset_config(), field_shape_function="none", target_field_definition_file="target.mat")
    support, arrays, message, _ = _preview_data(config, tmp_path)
    assert support is mesh and calls == [True]
    assert message == ""
    np.testing.assert_array_equal(arrays["target_field_T"], target.b)
