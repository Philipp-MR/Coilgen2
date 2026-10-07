"""True empty startup and saveable, explicitly configured new designs."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QDockWidget, QPushButton, QToolBar

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.choice_dialogs import GraphicalChoiceCombo
from coilgen.gui.project_start import NewProjectDialog
from coilgen.gui.workbench import CoilGenWorkbench
from test_workbench_usability import Preview, Queue, Service


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workbench-empty-test"])


@pytest.fixture
def window(application, tmp_path):
    instance = CoilGenWorkbench(jobs=Queue(tmp_path / "jobs"), service=Service(), viewport_factory=Preview)
    instance.timer.stop()
    instance.show()
    application.processEvents()
    yield instance
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_empty_startup_has_no_hidden_default_document(window):
    assert window.project is None and window.assembly is None
    assert window._active_channel_id is None
    assert not hasattr(window.viewport, "project")
    assert window.workspace_stack.currentWidget() is window.welcome
    assert window.welcome.isVisible()
    assert {button.objectName() for button in window.welcome.findChildren(QPushButton)} == {
        "welcome_new_project",
        "welcome_open_project",
        "welcome_load_template",
    }
    assert window.findChild(QDockWidget, "model_dock").isHidden()
    assert window.findChild(QToolBar, "study_toolbar").isHidden()
    assert not window.compute_action.isEnabled()
    assert not window.compute_button.isEnabled()
    assert window.compute() is None
    assert window.service.submissions == []
    with pytest.raises(ValueError, match="Create or open a project"):
        window.save_project("unopened.coilgen.json")


@pytest.mark.parametrize("create", [False, True])
def test_file_new_is_an_explicit_cancelable_blank_creation(window, application, create):
    outcomes = []

    def respond():
        dialog = application.activeModalWidget()
        if not isinstance(dialog, NewProjectDialog):
            outcomes.append(f"Unexpected modal widget: {type(dialog).__name__}")
            if dialog is not None:
                dialog.reject()
            return
        assert dialog.project is None
        dialog.name_edit.setText("Named empty design")
        standard = QDialogButtonBox.StandardButton.Ok if create else QDialogButtonBox.StandardButton.Cancel
        QTest.mouseClick(dialog.buttons.button(standard), Qt.MouseButton.LeftButton)
        outcomes.append("created" if create else "canceled")

    QTimer.singleShot(0, respond)
    window.new_action.trigger()
    assert outcomes == ["created" if create else "canceled"]
    if create:
        assert window.project.name == "Named empty design"
        assert window.project.config.coil_mesh_file == "none"
        assert window.project.config.field_shape_function == "none"
        assert window.project.saved_runs == ()
    else:
        assert window.project is None and window.assembly is None
    assert window.service.submissions == []


def test_blank_design_roundtrip_and_close_return_to_empty(window, application, tmp_path):
    window.new_project(name="Blank before choices")
    before = window.project
    assert before.config.coil_mesh_file == before.config.field_shape_function == "none"
    path = window.save_project(tmp_path / "blank.coilgen.json")
    restored = ProjectDocument.load(path)
    assert restored.fingerprint == before.fingerprint
    assert restored.config.coil_mesh_file == restored.config.field_shape_function == "none"
    assert restored.saved_runs == ()
    assert window.close_project()
    application.processEvents()
    assert window.project is None and window.assembly is None
    assert window.workspace_stack.currentWidget() is window.welcome
    assert window._draft_controls == {} and window.runs == {}
    assert window.viewport.displayed is None
    window.open_project(path)
    assert window.project.fingerprint == restored.fingerprint
    assert window.project.config.coil_mesh_file == window.project.config.field_shape_function == "none"
    assert window.service.submissions == []


@pytest.mark.parametrize("first", ["target", "support"])
def test_new_design_accepts_inputs_in_either_order_without_implicit_template(window, tmp_path, first):
    window.new_project(name=f"{first.capitalize()} first")
    geometry = preset_config()
    support = {
        "coil_mesh_file": geometry.coil_mesh_file,
        "cylinder_mesh_parameter_list": geometry.cylinder_mesh_parameter_list,
        "surface_is_cylinder_flag": geometry.surface_is_cylinder_flag,
    }
    target = {"field_shape_function": "y", "target_gradient_axis": "y", "target_gradient_strength": 0.02}
    assert window.apply_config(target if first == "target" else support)
    if first == "target":
        assert window.project.config.coil_mesh_file == "none"
        assert window.project.config.field_shape_function == "y"
    else:
        assert window.project.config.coil_mesh_file == geometry.coil_mesh_file
        assert window.project.config.field_shape_function == "none"
    partial = window.save_project(tmp_path / f"{first}-partial.coilgen.json")
    assert ProjectDocument.load(partial).fingerprint == window.project.fingerprint
    assert window.apply_config(support if first == "target" else target)
    assert window.project.config.coil_mesh_file == geometry.coil_mesh_file
    assert window.project.config.field_shape_function == "y"
    assert window.service.submissions == []


def test_explicit_template_has_one_compute_and_tree_navigation(window, application):
    window.load_template("cylinder")
    application.processEvents()
    assert window.project is not None and window.assembly is not None
    assert window.workspace_stack.currentWidget() is window.workspace
    assert window.findChild(QDockWidget, "model_dock").isVisible()
    assert window.tree.isVisible()
    assert window.compute_button.isVisible()
    assert window.stage_selector.isHidden()
    assert [
        button
        for button in window.findChildren(QPushButton)
        if button.isVisible() and button.objectName() == "primary_compute"
    ] == [window.compute_button]
    assert not any(
        button.isVisible() and button.text() in {"Brief", "Jobs", "Results", "History"}
        for button in window.workspace.findChildren(QPushButton)
    )
    assert window.service.submissions == []


@pytest.mark.parametrize("family", ["Linear gradient", "Analytic expression", "Spherical harmonic"])
def test_blank_target_popup_only_changes_field_after_explicit_tile_choice(window, application, tmp_path, family):
    window.new_project(name="Target family first")
    window.select_node("target")
    before = window.project
    assert window._target_region_editor is None
    assert "target_region_shape" not in window._draft_controls
    assert list(window._settings_decisions) == ["Choose field"]
    section = window._settings_decisions["Choose field"]
    control = section.controls[0]
    assert isinstance(control, GraphicalChoiceCombo)
    assert control.currentIndex() == -1
    assert [control.itemText(index) for index in range(control.count())] == [
        "Linear gradient",
        "Analytic expression",
        "Spherical harmonic",
    ]
    QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert section.dialog.isVisible() and control.dialog.isHidden()
    assert window.project is before
    section.dialog.close()
    application.processEvents()
    assert window.project is before and window.project.config.field_shape_function == "none"
    QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    application.processEvents()
    index = control.findText(family)
    assert index >= 0 and control.buttons[index].isEnabled()
    QTest.mouseClick(control.buttons[index], Qt.MouseButton.LeftButton)
    application.processEvents()
    assert window.project.config.coil_mesh_file == "none"
    assert window.project.config.field_shape_function != "none"
    assert window.project.revision == before.revision + 1
    assert window._settings_decisions["Choose field"].controls[0].currentText() == family
    assert window._target_region_editor is not None
    assert window._settings_decisions["Choose field"].dialog.isVisible()
    assert len([dialog for dialog in window.findChildren(QDialog) if dialog.isVisible()]) == 1
    if family == "Analytic expression":
        assert window.project.config.field_shape_function == "1e-4"
        assert window.project.config.target_gradient_normalization == "none"
        assert window.project.config.target_field_units == "T"
    path = window.save_project(tmp_path / "field-first.coilgen.json")
    assert ProjectDocument.load(path).fingerprint == window.project.fingerprint
    assert window.service.submissions == []
