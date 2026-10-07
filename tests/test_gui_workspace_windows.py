"""Owned scientific windows preserve draft widgets and honest open-state flags."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QLabel, QLineEdit, QVBoxLayout, QWidget
from shiboken6 import isValid

from coilgen.gui.workspace_windows import EvidenceWindows, PanelWindow


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workspace-windows-test"])


@pytest.fixture
def owner(application):
    widget = QWidget()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(1280, 900)
    widget.show()
    application.processEvents()
    yield widget
    if isValid(widget):
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def registry(owner):
    widget = EvidenceWindows(owner)
    for name in ("Jobs", "Results", "History"):
        widget.addTab(QLineEdit(name), name)
    return widget


def _flush(application):
    application.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_registry_keeps_original_page_api_and_opens_only_requested_window(registry, owner, application):
    assert registry.count() == 3
    assert registry.isHidden()
    assert registry.currentWidget() in [registry.widget(i) for i in range(3)]
    assert registry.widget(-1) is None
    assert registry.widget(3) is None
    assert not registry.any_open()
    page = registry.widget(1)
    registry.setCurrentWidget(page)
    assert registry.currentWidget() is page
    assert not registry.any_open()
    registry.open_current()
    _flush(application)
    window = registry.window_for(page)
    assert window.parentWidget() is owner
    assert window.content is page
    assert page.parentWidget() is window
    assert not window.isModal()
    assert window.windowModality() == Qt.WindowModality.NonModal
    assert not window.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert not window.isHidden()
    assert [registry.isTabVisible(i) for i in range(3)] == [False, True, False]


def test_close_reopen_preserves_original_widget_text_undo_history_and_signals(registry, application):
    page = registry.widget(0)
    page.setText("original")
    page.insert(" pending edit")
    assert page.isUndoAvailable()
    changes = []
    page.textChanged.connect(changes.append)
    registry.open_page(page)
    window = registry.window_for(page)
    closed = []
    window.closed.connect(lambda: closed.append(True))
    assert window.close()
    _flush(application)
    assert not registry.any_open()
    assert isValid(window) and isValid(page)
    assert page.text() == "original pending edit"
    assert page.isUndoAvailable()
    registry.open_page(page)
    _flush(application)
    assert registry.window_for(page) is window
    assert not window.isHidden()
    assert closed == [True]
    assert changes == []
    page.undo()
    assert page.text() == "original"


def test_closing_one_panel_leaves_other_panel_and_page_state_open(registry, application):
    first, second = registry.widget(0), registry.widget(1)
    registry.open_page(first)
    registry.open_page(second)
    second.setText("saved evidence selection")
    registry.window_for(first).close()
    _flush(application)
    assert not registry.isTabVisible(0)
    assert registry.isTabVisible(1)
    assert registry.any_open()
    assert not registry.window_for(second).isHidden()
    assert second.text() == "saved evidence selection"


@pytest.mark.parametrize("action", ["reject", "done", "hide", "escape"])
def test_all_user_close_paths_clear_portable_open_flag(registry, application, action):
    page = registry.widget(1)
    registry.open_page(page)
    window = registry.window_for(page)
    closed = []
    window.closed.connect(lambda: closed.append(True))
    if action == "reject":
        window.reject()
    elif action == "done":
        window.done(QDialog.DialogCode.Accepted)
    elif action == "hide":
        window.hide()
    else:
        QTest.keyClick(window, Qt.Key.Key_Escape)
    _flush(application)
    assert window.isHidden()
    assert not registry.isTabVisible(1)
    assert not registry.any_open()
    assert isValid(window) and isValid(page)
    assert closed == [True]
    registry.open_page(page)
    assert not window.isHidden()


def test_close_all_hides_without_deleting_or_replacing_any_scientific_widget(registry, application):
    pages = [registry.widget(i) for i in range(registry.count())]
    for index, page in enumerate(pages):
        registry.open_page(page)
        page.setText(f"immutable result {index}")
    windows = [registry.window_for(page) for page in pages]
    registry.close_all()
    _flush(application)
    assert not registry.any_open()
    assert all(window.isHidden() and isValid(window) for window in windows)
    assert all(isValid(page) for page in pages)
    assert registry.count() == 3
    for index, page in enumerate(pages):
        assert registry.widget(index) is page
        assert page.text() == f"immutable result {index}"
        registry.open_page(page)
        assert registry.window_for(page) is windows[index]


def test_jobs_wrapper_contains_original_page_and_retains_filter_controls(registry, application):
    page = registry.widget(0)
    container = QWidget()
    layout = QVBoxLayout(container)
    filter_control = QCheckBox("This session only")
    filter_control.setChecked(True)
    layout.addWidget(filter_control)
    layout.addWidget(page)
    registry.set_content(page, container)
    assert registry.widget(0) is page
    registry.open_page(page)
    window = registry.window_for(page)
    assert window.content is container
    assert container.parentWidget() is window
    assert page.parentWidget() is container
    assert page.isVisibleTo(window)
    assert not page.isHidden()
    registry.close_all()
    _flush(application)
    registry.open_page(page)
    assert registry.window_for(page) is window
    assert filter_control.isChecked()
    assert page.text() == "Jobs"


def test_index_insertion_retains_correct_page_identity(owner):
    registry = EvidenceWindows(owner)
    first, second, middle = QLabel("First"), QLabel("Second"), QLabel("Middle")
    registry.addTab(first, "First")
    registry.addTab(second, "Second")
    registry.insertTab(1, middle, "Middle")
    assert [registry.widget(i) for i in range(3)] == [first, middle, second]
    assert registry.indexOf(second) == 2
    registry.open_page(second)
    assert registry.window_for(second).windowTitle() == "Second — CoilGen2"


def test_invalid_registration_and_late_content_replacement_do_not_drop_widgets(registry):
    page = registry.widget(0)
    with pytest.raises(ValueError, match="already registered"):
        registry.addTab(page, "Duplicate")
    assert registry.count() == 3
    unknown = QWidget()
    with pytest.raises(ValueError, match="Unknown evidence window"):
        registry.open_page(unknown)
    with pytest.raises(ValueError, match="Unknown evidence window"):
        registry.set_content(unknown, unknown)
    unknown.deleteLater()
    registry.open_page(page)
    replacement = QWidget()
    with pytest.raises(ValueError, match="before opening"):
        registry.set_content(page, replacement)
    assert registry.window_for(page).content is page
    replacement.deleteLater()


def test_standalone_panel_close_preserves_pending_controls_and_is_modeless(owner, application):
    content = QLineEdit("unapplied exact value")
    panel = PanelWindow("Inspector", content, owner)
    panel.open_panel()
    _flush(application)
    panel.close()
    _flush(application)
    assert not panel.isModal()
    assert isValid(content) and isValid(panel)
    assert content.text() == "unapplied exact value"
    panel.open_panel()
    assert not panel.isHidden()


def test_opening_existing_minimized_panel_restores_it(owner, application):
    panel = PanelWindow("Inspector", QLineEdit("pending"), owner)
    panel.open_panel()
    panel.setWindowState(panel.windowState() | Qt.WindowState.WindowMinimized)
    _flush(application)
    assert panel.isMinimized()
    panel.open_panel()
    _flush(application)
    assert not panel.isMinimized()
    assert not panel.isHidden()
    assert panel.content.text() == "pending"


def test_minimized_registered_panel_remains_a_portable_open_window(registry, application):
    page = registry.widget(0)
    registry.open_page(page)
    panel = registry.window_for(page)
    panel.setWindowState(panel.windowState() | Qt.WindowState.WindowMinimized)
    _flush(application)
    assert panel.isMinimized()
    assert registry.isTabVisible(0)
    assert registry.any_open()
    registry.open_page(page)
    _flush(application)
    assert not panel.isMinimized()


def test_owner_destruction_cleans_owned_panels_and_original_widgets(owner, registry, application, capsys):
    page = registry.widget(0)
    registry.open_page(page)
    panel = registry.window_for(page)
    owner.deleteLater()
    _flush(application)
    assert not isValid(owner)
    assert not isValid(registry)
    assert not isValid(panel)
    assert not isValid(page)
    assert "RuntimeError" not in capsys.readouterr().err
