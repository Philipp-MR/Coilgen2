"""Owned modeless editors preserve registered draft widgets and evidence scope."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget
from shiboken6 import isValid

from coilgen.gui.controls import ScientificSpinBox
from coilgen.gui.focused_editors import group_form_controls, open_focused_control
from coilgen.gui.region_selectors import RegionSamplingSelector, TargetRegionGallery


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["focused-editor-test"])


@pytest.fixture
def section(application):
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    form = QFormLayout(page)
    control = ScientificSpinBox()
    control.setDecimals(15)
    control.setRange(-1e6, 1e6)
    control.setKeyboardTracking(False)
    control.setValue(1.234567891234567)
    wrapper = QWidget()
    row = QVBoxLayout(wrapper)
    row.addWidget(control)
    error = QLabel("Input must be complete")
    error.hide()
    row.addWidget(error)
    label = QLabel("Center x")
    label.setBuddy(control)
    form.addRow(label, wrapper)
    window = SimpleNamespace(_draft_controls={"center": control}, _focused_sections=[])
    editor = group_form_controls(window, form, "Placement", ["center"], summary=lambda: control.text())
    window._focused_sections.append(editor)
    yield page, form, control, error, editor, window
    if isValid(page):
        page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_focus_window_moves_whole_row_without_copying_widgets_or_values(section):
    page, form, control, error, editor, _window = section
    value = control.value()
    assert form.rowCount() == 1 and form.itemAt(0).widget() is editor
    assert editor.form.itemAt(0, QFormLayout.ItemRole.LabelRole).widget().buddy() is control
    assert error.parentWidget() is control.parentWidget()
    assert control.value() == value
    assert editor.dialog.parentWidget() is editor
    assert editor.parentWidget() is page
    assert not editor.dialog.isModal()


def test_reopening_same_window_keeps_incomplete_pending_input_and_error(section, application):
    _page, _form, control, error, editor, _window = section
    value = control.value()
    editor.open()
    application.processEvents()
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "1e")
    error.show()
    dialog = editor.dialog
    dialog.close()
    assert not dialog.isVisible()
    editor.open()
    application.processEvents()
    assert editor.dialog is dialog
    assert control.lineEdit().text().startswith("1e")
    assert not control.hasAcceptableInput()
    assert control.value() == value
    assert error.isVisible()
    assert editor.summary_label.text().startswith("1e")


def test_failed_control_opens_its_section_and_visible_error(section, application):
    _page, _form, control, error, editor, window = section
    error.show()
    assert open_focused_control(window, control)
    application.processEvents()
    assert editor.dialog.isVisible()
    assert error.isVisible()
    assert not open_focused_control(window, QLabel("unrelated"))


def test_page_teardown_removes_owned_windows_and_widgets(section, application):
    page, _form, control, _error, editor, _window = section
    dialog = editor.dialog
    editor.open()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    assert not isValid(dialog)
    assert not isValid(control)


def test_focused_window_height_scales_with_visible_rows_and_keeps_scrolling(application):
    pages = []

    def build(count):
        page = QWidget()
        pages.append(page)
        form = QFormLayout(page)
        controls = {}
        for index in range(count):
            control = QSpinBox()
            controls[str(index)] = control
            wrapper = QWidget()
            layout = QVBoxLayout(wrapper)
            layout.addWidget(control)
            form.addRow(QLabel(f"Dimension {index}"), wrapper)
        return group_form_controls(SimpleNamespace(_draft_controls=controls), form, "Dimensions", controls)

    try:
        short, medium, long = build(2), build(4), build(30)
        assert short.dialog.height() < medium.dialog.height() <= long.dialog.height()
        assert long.dialog.height() <= long.dialog.screen().availableGeometry().height()
        scroll = long.dialog.findChild(QScrollArea)
        assert scroll is not None and scroll.widgetResizable()
        long.open()
        application.processEvents()
        assert scroll.verticalScrollBar().maximum() > 0
        assert long.open_for(long.controls[-1])
        assert scroll.verticalScrollBar().value() > 0
    finally:
        for page in pages:
            page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_compact_region_picker_opens_on_demand_and_hides_unavailable_map_state(application):
    gallery = TargetRegionGallery(compact=True)
    gallery.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    gallery.show()
    application.processEvents()
    try:
        assert gallery.height() <= 40
        assert gallery.currentData() == "sphere"
        assert not gallery.dialog.isVisible()
        QTest.mouseClick(gallery.change_button, Qt.MouseButton.LeftButton)
        application.processEvents()
        assert gallery.dialog.isVisible() and not gallery.dialog.isModal()
        assert not gallery.buttons[gallery.findData("imported_map")].isVisible()
        events = []
        gallery.currentIndexChanged.connect(events.append)
        QTest.mouseClick(gallery.buttons[gallery.findData("box")], Qt.MouseButton.LeftButton)
        assert gallery.currentData() == "box" and events == [2]
        assert gallery.selected_label.text() == "Box"
        assert not gallery.dialog.isVisible()
        gallery.showPopup()
        assert gallery.dialog.isVisible()
        with QSignalBlocker(gallery):
            gallery.setCurrentIndex(gallery.findData("ellipsoid"))
        assert gallery.selected_label.text() == "Ellipsoid" and events == [2]
    finally:
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_compact_current_map_and_legacy_mesh_remain_truthful_description_states(application):
    gallery = TargetRegionGallery(compact=True)
    try:
        imported = gallery.findData("imported_map")
        with QSignalBlocker(gallery):
            gallery.setCurrentIndex(imported)
        gallery.showPopup()
        application.processEvents()
        assert gallery.buttons[imported].isVisible()
        assert not gallery.buttons[imported].isEnabled()
        legacy = gallery.add_legacy_mesh()
        with QSignalBlocker(gallery):
            gallery.setCurrentIndex(legacy)
        assert gallery.selected_label.text() == "Legacy mesh"
        assert not gallery.buttons[legacy].isEnabled()
    finally:
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_sampling_is_compact_segmented_control_with_original_values(application):
    selector = RegionSamplingSelector()
    try:
        assert selector.height() <= 42
        assert [selector.itemData(index) for index in range(selector.count())] == ["volume", "surface"]
        assert all(button.minimumHeight() <= 36 for button in selector.buttons)
        assert "do not establish field accuracy" in selector.buttons[1].toolTip()
    finally:
        selector.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
