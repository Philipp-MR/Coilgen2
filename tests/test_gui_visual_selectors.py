"""Graphical geometry/circuit choices preserve the existing configuration API."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.gui.editors import SUPPORT_EDITORS
from coilgen.gui.visual_selectors import CircuitSelector, SurfaceGallery
from coilgen.gui.winding_intent import WINDING_INTENTS


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["visual-selector-test"])


@pytest.fixture
def gallery(application):
    widget = SurfaceGallery()
    widget.resize(330, 82)
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_surface_buttons_change_exact_existing_source_and_emit_once(gallery):
    assert [(gallery.itemText(i), gallery.itemData(i)) for i in range(gallery.count())] == [
        (editor.label, editor.source) for editor in SUPPORT_EDITORS
    ]
    events = []
    gallery.currentIndexChanged.connect(events.append)
    QTest.mouseClick(gallery.buttons[2], Qt.MouseButton.LeftButton)
    assert gallery.currentText() == "Conical frustum"
    assert gallery.currentData() == "create conical frustum mesh"
    assert events == [2]
    assert [index for index, button in enumerate(gallery.buttons) if button.isChecked()] == [2]
    QTest.mouseClick(gallery.buttons[2], Qt.MouseButton.LeftButton)
    assert events == [2]
    assert gallery.buttons[2].isChecked()


def test_blocked_loading_still_updates_visible_choice(gallery):
    events = []
    gallery.currentIndexChanged.connect(events.append)
    with QSignalBlocker(gallery):
        gallery.setCurrentText("Biplanar")
    assert events == []
    assert gallery.currentData() == "create bi-planary mesh"
    assert gallery.buttons[1].isChecked()
    with QSignalBlocker(gallery):
        gallery.setCurrentIndex(0)
    assert gallery.buttons[0].isChecked()


def test_imported_geometry_is_a_distinct_choice_without_replacing_its_source(gallery):
    initial_count = gallery.count()
    gallery.addItem("Imported / custom support", "custom_surface.stl")
    imported = gallery.findData("custom_surface.stl")
    gallery.setCurrentIndex(imported)
    assert gallery.currentData() == "custom_surface.stl"
    assert gallery.buttons[imported].accessibleName() == "Imported / custom support"
    assert "Imported" in gallery.buttons[imported].text()
    assert gallery.buttons[imported].isChecked()
    gallery.setItemData(imported, "updated_surface.npz")
    assert gallery.currentData() == "updated_surface.npz"
    assert gallery.buttons[imported].isChecked()
    gallery.removeItem(imported)
    assert len(gallery.buttons) == initial_count


def test_tiles_are_keyboard_accessible_and_do_not_open_popup(gallery):
    gallery.buttons[0].setFocus()
    QTest.keyClick(gallery.buttons[0], Qt.Key.Key_Right)
    assert gallery.currentText() == "Biplanar"
    QTest.keyClick(gallery.buttons[1], Qt.Key.Key_Right)
    assert gallery.currentText() == "Conical frustum"
    gallery.showPopup()
    assert not gallery.view().isVisible()
    assert gallery.buttons[2].hasFocus()


def test_responsive_layout_retains_all_labels_and_selection(gallery, application):
    gallery.setCurrentIndex(2)
    gallery.resize(205, gallery.height())
    application.processEvents()
    assert gallery._columns == 2
    assert gallery.height() <= 205
    assert gallery.buttons[2].isVisible() and gallery.buttons[2].isChecked()
    gallery.resize(330, gallery.height())
    application.processEvents()
    assert gallery._columns == 3
    assert gallery.height() <= 120


def test_disabled_item_cannot_be_clicked(gallery):
    gallery.model().item(1).setEnabled(False)
    assert not gallery.buttons[1].isEnabled()
    assert not gallery.surface_actions[gallery.itemData(1)].isEnabled()
    QTest.mouseClick(gallery.buttons[1], Qt.MouseButton.LeftButton)
    assert gallery.currentIndex() == 0


def test_circuit_tiles_preserve_all_winding_intentions(application):
    widget = CircuitSelector()
    try:
        widget.resize(330, widget.sizeHint().height())
        widget.show()
        application.processEvents()
        assert [widget.itemText(i) for i in range(widget.count())] == list(WINDING_INTENTS)
        assert widget._columns == 2
        assert widget.height() <= 170
        QTest.mouseClick(widget.buttons[2], Qt.MouseButton.LeftButton)
        assert widget.currentText() == "Independent closed turns"
        assert "electrically separate" in widget.buttons[2].toolTip()
        assert all(button.accessibleName() == widget.itemText(index) for index, button in enumerate(widget.buttons))
        with QSignalBlocker(widget):
            widget.setCurrentText("Contours only")
        assert widget.buttons[3].isChecked()
        assert "without routing" in widget.buttons[3].toolTip()
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
