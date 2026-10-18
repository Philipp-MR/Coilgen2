"""Owned editors retain global activity without driving runtime or selection."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from coilgen.gui.activity_feedback import ComputationFeedback
from coilgen.gui.design_progress import ComputationIndicator


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def owned(app):
    windows = []

    def create():
        root = QMainWindow()
        root.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        main = ComputationIndicator(root)
        root.statusBar().addPermanentWidget(main, 1)
        monitor = ComputationFeedback(root, main)
        root.show()
        windows.append((root, monitor))
        return root, monitor, main

    yield create
    for root, monitor in windows:
        if isValid(monitor):
            monitor.close()
        if isValid(root):
            root.close()
            root.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _dialog(parent, layout_type=QVBoxLayout, *, modal=False):
    dialog = QDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    dialog.setModal(modal)
    layout = layout_type(dialog)
    edit = QLineEdit("unchanged", dialog)
    if isinstance(layout, QFormLayout):
        layout.addRow(edit)
    else:
        layout.addWidget(edit)
    dialog.show()
    return dialog, edit


def _footer(dialog):
    return dialog.findChild(ComputationIndicator, "window_computation_indicator")


def _state(**changes):
    return {"id": "gx", "name": "Gx", "status": "running", "progress": 0.45, "stage": "Field samples", **changes}


def test_open_and_nested_modal_editors_receive_latest_activity(owned, app):
    root, monitor, main = owned()
    monitor.set_activity((_state(),))
    first, _ = _dialog(root)
    nested, _ = _dialog(first, modal=True)
    app.processEvents()
    for dialog in (first, nested):
        footer = _footer(dialog)
        assert footer is not None and footer.isVisible()
        assert footer.label.text() == main.label.text() == "Computing (45%)"
        assert "Field samples" in footer.toolTip()
    assert len(monitor.indicators) == 2
    monitor.set_activity((_state(progress=0.63),))
    assert _footer(nested).progress.value() == 63


@pytest.mark.parametrize("layout_type", (QVBoxLayout, QFormLayout, QGridLayout))
def test_footers_follow_queue_work_terminal_and_hidden_window_updates(owned, app, layout_type):
    root, monitor, _ = owned()
    dialog, _ = _dialog(root, layout_type)
    app.processEvents()
    footer = _footer(dialog)
    assert footer is not None and footer.isHidden()
    monitor.set_activity((_state(status="queued", progress=0),))
    assert footer.label.text() == "Queued" and not footer.isHidden()
    assert footer.progress.isHidden()
    dialog.hide()
    monitor.set_activity((_state(progress=0.61),))
    dialog.show()
    app.processEvents()
    assert _footer(dialog) is footer and footer.progress.value() == 61
    assert len(monitor.indicators) == 1
    monitor.set_activity((_state(status="completed", progress=1),))
    assert footer.isHidden()
    monitor.set_activity(working=({"name": "Field analysis", "progress": 0.24},))
    assert footer.label.text() == "Computing (24%)"


def test_mirror_updates_do_not_change_focus_dialog_result_or_inputs(owned, app):
    root, monitor, _ = owned()
    dialog, edit = _dialog(root)
    app.processEvents()
    edit.setFocus()
    dialog.setResult(7)
    before = QApplication.focusWidget()
    monitor.set_activity((_state(),))
    monitor.set_activity((_state(progress=0.55),))
    assert QApplication.focusWidget() is before
    assert dialog.result() == 7 and edit.text() == "unchanged"


def test_footer_remains_a_compact_strip_in_a_tall_window(owned, app):
    root, monitor, _ = owned()
    dialog, _ = _dialog(root)
    dialog.resize(760, 500)
    monitor.set_activity((_state(),))
    app.processEvents()
    footer = _footer(dialog)
    assert footer.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert footer.height() == footer.sizeHint().height()
    assert footer.height() < dialog.height() / 5


def test_unrelated_workbench_dialogs_keep_independent_runtime(owned, app):
    first, first_monitor, _ = owned()
    second, second_monitor, _ = owned()
    first_dialog, _ = _dialog(first)
    second_dialog, _ = _dialog(second)
    unrelated = QDialog()
    unrelated.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    QVBoxLayout(unrelated)
    unrelated.show()
    app.processEvents()
    first_monitor.set_activity((_state(progress=0.32),))
    assert _footer(first_dialog).progress.value() == 32
    assert _footer(second_dialog).isHidden()
    assert _footer(unrelated) is None
    second_monitor.set_activity((_state(progress=0.87),))
    assert _footer(second_dialog).progress.value() == 87
    assert _footer(first_dialog).progress.value() == 32
    unrelated.deleteLater()


@pytest.mark.parametrize("dialog_type", (QFileDialog, QMessageBox))
def test_platform_dialogs_and_unknown_layouts_are_left_unchanged(owned, app, dialog_type):
    root, monitor, _ = owned()
    platform = dialog_type(root)
    if isinstance(platform, QFileDialog):
        platform.setOption(QFileDialog.Option.DontUseNativeDialog)
    platform.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    platform.show()
    horizontal, _ = _dialog(root, QHBoxLayout)
    empty = QDialog(root)
    empty.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    empty.show()
    app.processEvents()
    monitor.set_activity((_state(),))
    assert all(_footer(dialog) is None for dialog in (platform, horizontal, empty))


def test_destroyed_dialogs_and_closed_monitor_leave_no_stale_callbacks(owned, app):
    root, monitor, _ = owned()
    first, _ = _dialog(root)
    app.processEvents()
    assert len(monitor.indicators) == 1
    first.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(monitor.indicators) == 0
    second, _ = _dialog(root)
    monitor.set_activity((_state(),))
    footer = _footer(second)
    assert not footer.isHidden()
    monitor.close()
    assert footer.isHidden() and len(monitor.indicators) == 0
    third, _ = _dialog(root)
    app.processEvents()
    monitor.set_activity((_state(progress=0.8),))
    assert _footer(third) is None


def test_parent_destruction_removes_application_filter(owned, app):
    root, monitor, _ = owned()
    dialog, _ = _dialog(root)
    app.processEvents()
    root.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(root) and not isValid(monitor) and not isValid(dialog)
    unrelated = QWidget()
    unrelated.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    unrelated.show()
    app.processEvents()
    unrelated.deleteLater()
