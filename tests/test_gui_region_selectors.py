"""Target-region tiles preserve combo signals and explicit sampling scopes."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QVBoxLayout

from coilgen.gui.region_selectors import RegionSamplingSelector, TargetRegionGallery


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["region-selector-test"])


@pytest.fixture
def region(application):
    widget = TargetRegionGallery()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(350, widget.sizeHint().height())
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_region_values_are_stable_and_clicks_emit_one_existing_combo_signal(region):
    assert isinstance(region, QComboBox)
    assert [region.itemData(index) for index in range(region.count())] == [
        "sphere",
        "ellipsoid",
        "box",
        "cylinder",
        "enclosing_mesh",
        "imported_map",
    ]
    events = []
    region.currentIndexChanged.connect(events.append)
    QTest.mouseClick(region.buttons[2], Qt.MouseButton.LeftButton)
    assert region.currentText() == "Box"
    assert region.currentData() == "box"
    assert events == [2]
    QTest.mouseClick(region.buttons[2], Qt.MouseButton.LeftButton)
    assert events == [2]
    assert region.buttons[2].isChecked()


def test_blocked_loading_updates_selection_without_edit_signals(region):
    events = []
    region.currentIndexChanged.connect(events.append)
    with QSignalBlocker(region):
        region.setCurrentIndex(region.findData("ellipsoid"))
    assert not events
    assert region.buttons[1].isChecked()
    with QSignalBlocker(region):
        region.setCurrentText("Enclosing mesh")
    assert region.buttons[4].isChecked()
    assert not events


def test_imported_map_is_current_only_and_cannot_be_created_by_click(region):
    imported = region.findData("imported_map")
    assert not region.buttons[imported].isEnabled()
    assert "already loaded map" in region.buttons[imported].toolTip()
    QTest.mouseClick(region.buttons[imported], Qt.MouseButton.LeftButton)
    assert region.currentData() == "sphere"
    with QSignalBlocker(region):
        region.setCurrentIndex(imported)
    assert region.currentData() == "imported_map"
    assert region.buttons[imported].isChecked()
    assert not region.buttons[imported].isEnabled()


def test_loaded_legacy_mesh_is_an_opt_in_current_only_state(region):
    assert region.findData("legacy_mesh") == -1
    index = region.add_legacy_mesh()
    assert region.count() == 7
    assert region.add_legacy_mesh() == index and region.count() == 7
    assert not region.buttons[index].isEnabled()
    assert "vertex/bounding-box sampling" in region.buttons[index].toolTip()
    assert "not an enclosing-volume" in region.buttons[index].toolTip()
    events = []
    region.currentIndexChanged.connect(events.append)
    with QSignalBlocker(region):
        region.setCurrentIndex(index)
    assert region.currentData() == "legacy_mesh"
    assert region.buttons[index].isChecked()
    assert not events


def test_six_tiles_fit_two_rows_and_respond_to_a_narrow_inspector(region, application):
    assert region._columns == 3
    assert region.height() <= 170
    assert all(button.isVisible() for button in region.buttons)
    region.setCurrentIndex(region.findData("cylinder"))
    region.resize(210, region.height())
    application.processEvents()
    assert region._columns == 2
    assert region.height() <= 255
    assert region.buttons[3].isChecked()
    assert all(button.accessibleName() and button.accessibleDescription() for button in region.buttons)


def test_region_keyboard_navigation_preserves_shape_values(region):
    region.buttons[0].setFocus()
    QTest.keyClick(region.buttons[0], Qt.Key.Key_Right)
    assert region.currentData() == "ellipsoid"
    QTest.keyClick(region.buttons[1], Qt.Key.Key_Down)
    assert region.currentData() == "enclosing_mesh"
    QTest.keyClick(region.buttons[4], Qt.Key.Key_Right)
    assert region.currentData() == "enclosing_mesh"  # The imported-map tile stays current-only.
    region.showPopup()
    assert not region.view().isVisible()


def test_surface_sampling_states_its_limited_scope_and_preserves_combo_contract(application):
    widget = RegionSamplingSelector()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(350, widget.sizeHint().height())
    widget.show()
    application.processEvents()
    try:
        assert isinstance(widget, QComboBox)
        assert [widget.itemData(index) for index in range(widget.count())] == ["volume", "surface"]
        assert widget.height() <= 90
        events = []
        widget.currentIndexChanged.connect(events.append)
        QTest.mouseClick(widget.buttons[1], Qt.MouseButton.LeftButton)
        assert widget.currentData() == "surface" and events == [1]
        assert "only the region boundary" in widget.buttons[1].toolTip()
        assert "do not establish field accuracy throughout its volume" in widget.buttons[1].toolTip()
        with QSignalBlocker(widget):
            widget.setCurrentIndex(widget.findData("volume"))
        assert widget.buttons[0].isChecked()
        assert events == [1]
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def compact_region(application):
    widget = TargetRegionGallery(compact=True)
    widget.resize(340, 38)
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_compact_picker_scroll_preserves_shared_authoritative_model(compact_region, application):
    events = []
    activations = []
    compact_region.currentIndexChanged.connect(events.append)
    compact_region.activated.connect(activations.append)
    assert compact_region._picker.model() is compact_region.model()
    compact_region.showPopup()
    application.processEvents()
    assert compact_region.dialog.isVisible()
    assert not compact_region._picker.buttons[compact_region.findData("imported_map")].isVisible()
    QTest.mouseClick(compact_region.buttons[2], Qt.MouseButton.LeftButton)
    assert compact_region.currentData() == "box"
    assert events == [2]
    assert activations == [2]
    assert not compact_region.dialog.isVisible()
    compact_region.showPopup()
    application.processEvents()
    assert compact_region.buttons[2].isChecked()
    compact_region.hidePopup()
    assert events == [2]


def test_embedded_region_uses_same_tiles_and_never_opens_second_window(compact_region, application):
    original_picker = compact_region._picker
    original_buttons = tuple(compact_region.buttons)
    host = QDialog(compact_region)
    layout = QVBoxLayout(host)
    layout.addWidget(compact_region.scroll)
    opened = []

    def open_host():
        opened.append(True)
        host.show()

    compact_region.set_embedded_host(host, open_callback=open_host)
    events = []
    compact_region.currentIndexChanged.connect(events.append)
    compact_region.showPopup()
    application.processEvents()
    assert opened == [True]
    assert host.isVisible()
    assert not compact_region.dialog.isVisible()
    assert compact_region._picker is original_picker
    assert tuple(compact_region.buttons) == original_buttons
    QTest.mouseClick(compact_region.buttons[3], Qt.MouseButton.LeftButton)
    assert compact_region.currentData() == "cylinder"
    assert events == [3]
    assert host.isVisible()
    assert not compact_region.dialog.isVisible()
    assert compact_region.buttons[3].hasFocus()
    compact_region.hidePopup()
    assert host.isVisible()
    compact_region.setEnabled(False)
    assert host.isVisible()
    assert all(not button.isEnabled() for button in compact_region.buttons)
    QTest.keyClick(compact_region.buttons[1], Qt.Key.Key_Return)
    assert compact_region.currentData() == "cylinder"
    assert events == [3]
    compact_region.setEnabled(True)
    assert compact_region.buttons[3].isEnabled()
    host.hide()
    assert compact_region.currentData() == "cylinder"


def test_embedded_imported_and_legacy_states_remain_disabled_current_only(compact_region, application):
    host = QDialog(compact_region)
    QVBoxLayout(host).addWidget(compact_region.scroll)
    compact_region.set_embedded_host(host)
    legacy = compact_region.add_legacy_mesh()
    events = []
    compact_region.currentIndexChanged.connect(events.append)
    for value in ("imported_map", "legacy_mesh"):
        index = compact_region.findData(value)
        with QSignalBlocker(compact_region):
            compact_region.setCurrentIndex(index)
        compact_region.showPopup()
        application.processEvents()
        assert compact_region.currentData() == value
        assert compact_region.buttons[index].isVisible()
        assert compact_region.buttons[index].isChecked()
        assert not compact_region.buttons[index].isEnabled()
        QTest.mouseClick(compact_region.buttons[index], Qt.MouseButton.LeftButton)
        assert compact_region.currentData() == value
        other = legacy if value == "imported_map" else compact_region.findData("imported_map")
        assert not compact_region.buttons[other].isVisible()
        host.hide()
    assert events == []


def test_embedded_unknown_region_keeps_exact_saved_value_and_blocked_state(compact_region, application):
    compact_region.addItem("Custom experimental enclosure", "custom_region_v3")
    index = compact_region.count() - 1
    events = []
    compact_region.currentIndexChanged.connect(events.append)
    with QSignalBlocker(compact_region):
        compact_region.setCurrentIndex(index)
    host = QDialog(compact_region)
    QVBoxLayout(host).addWidget(compact_region.scroll)
    compact_region.set_embedded_host(host)
    compact_region.showPopup()
    application.processEvents()
    assert compact_region.currentData() == "custom_region_v3"
    assert compact_region.selected_label.text() == "Custom experimental enclosure"
    assert compact_region.buttons[index].accessibleName() == "Custom experimental enclosure"
    assert compact_region.buttons[index].isChecked()
    assert events == []
    compact_region.buttons[1].setFocus()
    QTest.keyClick(compact_region.buttons[1], Qt.Key.Key_Return)
    assert compact_region.currentData() == "ellipsoid"
    assert events == [1]
    assert host.isVisible()


def test_compact_region_picker_scrolls_on_narrow_window(compact_region, application):
    compact_region.showPopup()
    compact_region.dialog.resize(260, 230)
    application.processEvents()
    assert compact_region._picker._columns == 2
    assert compact_region.scroll.horizontalScrollBar().maximum() == 0
    assert compact_region.scroll.verticalScrollBar().maximum() > 0
    assert all(compact_region.buttons[index].isVisible() for index in range(5))
