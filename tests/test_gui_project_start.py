"""Blank creation never silently selects an engineering example."""

from dataclasses import asdict
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.config import CoilGenConfig
from coilgen.gui.project_start import NewProjectDialog, blank_project
from coilgen.gui.readiness import check_readiness
from coilgen.persistence import json_value


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["blank-project-test"])


@pytest.fixture
def dialog(application):
    window = NewProjectDialog()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    application.processEvents()
    yield window
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_blank_has_no_selected_support_target_or_results_and_retains_native_defaults(tmp_path):
    project = blank_project("Unconfigured gradient", project_root=tmp_path)
    expected = json_value(asdict(CoilGenConfig()))
    expected["field_shape_function"] = "none"
    assert project.values == expected
    assert project.name == "Unconfigured gradient" and project.revision == 1
    assert project.project_root == tmp_path.resolve()
    assert project.to_dict()["assets"] == {} and project.saved_runs == ()
    assert project.study == ProjectDocument.new().study
    # Repeated blank starts must be independent documents, not a hidden template.
    assert blank_project().id != project.id
    assert CoilGenConfig().field_shape_function == "x"


def test_blank_is_saveable_before_it_is_computable(tmp_path):
    project = blank_project("First design", project_root=tmp_path)
    report = check_readiness(project)
    assert {(issue.key, issue.title, issue.node) for issue in report.blocking} == {
        ("coil_mesh_file", "Missing support", "geometry"),
        ("field_shape_function", "Missing target", "target"),
    }
    assert not report.ready
    assert {issue.field for issue in DesignService.validate(project).errors} == {"coil_mesh_file"}
    path = project.save(tmp_path / "first.coilgen.json")
    loaded = ProjectDocument.load(path)
    assert loaded.to_dict() == project.to_dict()
    assert loaded.fingerprint == project.fingerprint
    assert loaded.model_fingerprint == project.model_fingerprint
    assert loaded.saved_runs == ()


@pytest.mark.parametrize("name", ["", " ", "\n", "a" * 121, None])
def test_blank_uses_the_existing_name_validation(name):
    with pytest.raises(ValueError, match="Project name"):
        blank_project(name)


def test_open_cancel_and_escape_do_not_construct_a_project(dialog, monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("Opening or cancelling New project must not construct a document")

    monkeypatch.setattr("coilgen.gui.project_start.blank_project", fail)
    assert dialog.project is None
    QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Rejected and dialog.project is None
    dialog.show()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.result() == QDialog.DialogCode.Rejected and dialog.project is None


def test_create_accepts_only_a_named_blank_project(dialog):
    dialog.name_edit.setText("  My shim study  ")
    QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.project.name == "My shim study"
    assert dialog.project.values["coil_mesh_file"] == "none"
    assert dialog.project.values["field_shape_function"] == "none"


def test_invalid_name_keeps_the_dialog_open_and_can_be_corrected_by_keyboard(dialog, application):
    dialog.name_edit.setText(" ")
    QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    assert dialog.isVisible() and dialog.project is None
    assert dialog.error_label.isVisible() and "1–120" in dialog.error_label.text()
    assert dialog.focusWidget() is dialog.name_edit
    dialog.name_edit.selectAll()
    QTest.keyClicks(dialog.name_edit, "Conical study")
    assert not dialog.error_label.isVisible()
    QTest.keyClick(dialog.name_edit, Qt.Key.Key_Return)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.project.name == "Conical study"
