"""Channel editing stays discoverable independently of plot visibility."""

from pathlib import Path

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QToolBar

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application.system_presets import system_assembly


application = usability.application
windows = usability.windows


def _show(window, application, width):
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            window.setFont(QFont(families[0], 9))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(width, 800)
    window.show()
    window.activateWindow()
    application.processEvents()


@pytest.mark.parametrize("width", (800, 1040, 1440))
def test_system_channel_selection_is_labeled_visible_and_outside_toolbar(windows, application, width):
    window = windows(system_assembly())
    _show(window, application, width)

    assert window.channel_context.isVisible()
    assert window.channel_label.text() == "Editing channel"
    assert window.channel_label.buddy() is window.channel_selector
    assert window.channel_selector.accessibleName() == "Editing channel"
    assert window.channel_selector.height() >= 36
    assert window.channel_selector.width() >= 160
    assert window.channel_selector.isVisible()
    assert window.channel_context.geometry().bottom() < window.view_context.geometry().top()
    assert window.channel_overview_button.text() == "9 channels…"
    assert window.channel_selector.parentWidget() is window.channel_context
    assert all(
        action.defaultWidget() is not window.channel_selector
        for toolbar in window.findChildren(QToolBar)
        for action in toolbar.actions()
        if hasattr(action, "defaultWidget")
    )
    for control in (
        window.channel_label,
        window.channel_selector,
        window.channel_overview_button,
        window.channel_enabled,
    ):
        assert window.channel_context.contentsRect().contains(control.geometry())
    assert window.compute_button.isVisible()


def test_keyboard_channel_switch_preserves_independent_models_and_disabled_channel(windows, application):
    assembly = system_assembly()
    selected = assembly.channels[1]
    assembly = assembly.update_channel(selected.id, enabled=False)
    window = windows(assembly)
    _show(window, application, 1040)
    fingerprints = {channel.id: channel.project.fingerprint for channel in assembly.channels}

    window.channel_selector.setFocus()
    QTest.keyClick(window.channel_selector, Qt.Key.Key_Down)

    assert window._active_channel_id == selected.id
    assert window.project.id == selected.project.id
    assert window.channel_selector.currentText() == selected.name
    assert not window.channel_enabled.isChecked()
    assert window.channel_enabled.text() == "Include in system compute"
    assert "excluded from system compute" in window.channel_selector.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert window.focusWidget() is window.channel_selector
    assert {channel.id: channel.project.fingerprint for channel in window.assembly.channels} == fingerprints
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


def test_channel_selection_rejects_invalid_pending_inputs_without_losing_draft(windows, application):
    config = preset_config()
    config.coil_mesh_file = "create conical frustum mesh"
    config.surface_is_cylinder_flag = False
    project = ProjectDocument.from_config(config, name="Conical channel")
    assembly = ProjectAssembly.new("Draft protection", (project, ProjectDocument.from_config(preset_config())))
    window = windows(assembly)
    _show(window, application, 1040)
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)
    active = window._active_channel_id

    window.channel_selector.setFocus()
    QTest.keyClick(window.channel_selector, Qt.Key.Key_Down)

    assert window._active_channel_id == active
    assert window.channel_selector.currentData() == active
    assert window.project is project
    assert window._draft_controls["rotation_axis_x"].value() == 0
    assert window._draft_controls["rotation_angle_deg"].value() == 45
    assert window._draft_dirty
    assert "rotation" in window.draft_error_label.text().lower()
    assert window.undo_stack.count() == 0


def test_single_project_hides_context_but_disabled_channel_can_be_reenabled(windows, application):
    window = windows()
    _show(window, application, 800)
    assert window.channel_context.isHidden()
    channel = window.assembly.channels[0]
    window._set_assembly(window.assembly.update_channel(channel.id, enabled=False))
    application.processEvents()

    assert window.channel_context.isVisible()
    assert window.channel_selector.isVisible()
    assert not window.channel_enabled.isChecked()
    window.channel_enabled.setChecked(True)
    application.processEvents()
    assert window.assembly.channel(channel.id).enabled
    assert window._collection_mode
    assert window.channel_context.isVisible()
    assert window.channel_overview_button.text() == "1 channel…"
    window.undo_stack.undo()
    assert not window.assembly.channel(channel.id).enabled
    assert window.channel_context.isVisible()
