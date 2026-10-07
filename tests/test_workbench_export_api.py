"""CAD selection/ownership remains attached to its exact immutable saved run."""

from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest
import test_workbench_usability as usability

from coilgen.gui.readiness import check_readiness
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application import cad
from PySide6.QtCore import QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

application = usability.application
windows = usability.windows


def test_export_page_updates_when_selected_saved_run_changes(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(cad, "geometry_catalog", lambda record: ())
    window = windows()
    window.navigate_workflow("export")
    original = window.geometry_export_panel
    record = SimpleNamespace(directory=tmp_path / "new", metadata={"name": "New saved design"}, arrays={})
    window.runs["new"] = record
    window.show_run("new")
    assert window.geometry_export_panel is not original
    assert "New saved design" in window.geometry_export_panel.scope.text()


def test_foreign_owned_run_requires_its_project_for_json_companion(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(cad, "geometry_catalog", lambda record: ())
    window = windows()
    record = SimpleNamespace(directory=tmp_path / "foreign", metadata={"project_id": "another-owner"}, arrays={})
    window.runs["foreign"] = record
    window.show_run("foreign")
    window.navigate_workflow("export")
    assert not window.geometry_export_panel.include_project.isEnabled()
    assert not window.geometry_export_panel.include_project.isChecked()
    with pytest.raises(ValueError, match="owning project"):
        window.export_selected_geometry(tmp_path / "cad")


def test_geometry_only_export_preserves_pending_model_draft(windows, tmp_path, monkeypatch):
    window = windows()
    record = SimpleNamespace(directory=tmp_path / "saved", metadata={}, arrays={})
    window.runs["saved"] = record
    window._selected_run = "saved"
    window._draft_controls["radius_mm"].setValue(100)
    project = window.project
    export = Mock(return_value="geometry-package")
    monkeypatch.setattr(cad, "export_geometry", export)
    assert window.export_selected_geometry(tmp_path / "cad", include_project=False) == "geometry-package"
    assert window.project is project and window._draft_dirty
    export.assert_called_once_with(record, tmp_path / "cad", None, unit="mm", project=None)


def test_undefined_cylinder_rotation_cannot_be_applied_or_submitted(windows):
    window = windows()
    original = window.project
    for axis in "xyz":
        window._draft_controls[f"rotation_axis_{axis}"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)
    assert not window.apply_settings()
    assert window.project is original
    assert "Rotation axis must be nonzero" in window._draft_error
    assert window.compute() is None
    assert not window.service.submissions


def test_loaded_undefined_rotation_is_actionable_before_compute():
    config = preset_config()
    values = list(config.cylinder_mesh_parameter_list)
    values[4:8] = [0, 0, 0, 0.5]
    config.cylinder_mesh_parameter_list = values
    report = check_readiness(ProjectDocument.from_config(config))
    issue = next(issue for issue in report.blocking if issue.title == "Undefined support rotation")
    assert issue.node == "geometry" and issue.key == "rotation_axis_x"
    values[7] = 0
    config.cylinder_mesh_parameter_list = values
    assert not any(
        issue.title == "Undefined support rotation"
        for issue in check_readiness(ProjectDocument.from_config(config)).blocking
    )


def test_gui_geometry_writer_runs_off_ui_thread_and_restores_action(windows, tmp_path, monkeypatch, application):
    option = cad.GeometryOption(
        "body.stl",
        "body",
        "stl",
        "fabrication_review_geometry",
        "passed",
        "m",
        ((0, 0, 0), (0.1, 0.1, 0.1)),
        False,
        "0" * 64,
    )
    monkeypatch.setattr(cad, "geometry_catalog", lambda record: (option,))
    directory = tmp_path / "package"
    package = SimpleNamespace(
        paths=(directory / "body.stl",), directory=directory, manifest_path=directory / "geometry_manifest.json"
    )
    calls = []

    def write(record, path, names, *, unit, project):
        calls.append((QThread.currentThread() == application.thread(), record, names, unit, project))
        return package

    monkeypatch.setattr(cad, "export_geometry", write)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(directory), ""))
    window = windows()
    record = SimpleNamespace(directory=tmp_path / "saved", metadata={}, arrays={})
    window.runs["saved"] = record
    window.show_run("saved")
    window.navigate_workflow("export")
    window._geometry_export_dialog((option.name,), "mm", False)
    assert not window.geometry_export_panel.export_button.isEnabled()
    deadline = time.monotonic() + 5
    while window._geometry_export_task is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert window._geometry_export_task is None
    assert calls == [(False, record, (option.name,), "mm", None)]
    assert window.geometry_export_panel.export_button.isEnabled()
    assert str(directory) in window.messages.toPlainText()
