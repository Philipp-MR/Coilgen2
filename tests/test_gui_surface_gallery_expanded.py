"""The focused support choice exposes the full existing primitive catalogue."""

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.gui.editors import SUPPORT_EDITORS
from coilgen.gui.visual_selectors import SurfaceGallery


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["expanded-support-gallery-test"])


@pytest.fixture
def gallery(application):
    widget = SurfaceGallery()
    widget.set_expanded(True)
    widget.resize(520, widget.height())
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _grid_height(gallery):
    rows = math.ceil(gallery.count() / gallery._columns)
    return rows * gallery._tile_height + (rows - 1) * 6 + 4


def test_expanded_gallery_shows_every_primitive_without_more_row(gallery):
    assert gallery._shown_indices() == list(range(len(SUPPORT_EDITORS)))
    assert all(button.isVisible() for button in gallery.buttons)
    assert [(gallery.itemText(i), gallery.itemData(i)) for i in range(gallery.count())] == [
        (editor.label, editor.source) for editor in SUPPORT_EDITORS
    ]
    assert gallery.more_button.isHidden()
    assert gallery._grid.indexOf(gallery.more_button) == -1
    assert gallery._grid.count() == len(SUPPORT_EDITORS)
    assert gallery.height() == _grid_height(gallery)
    assert gallery.sizeHint().height() == _grid_height(gallery)


def test_expanded_gallery_resizes_and_retains_all_choices(gallery, application):
    gallery.setCurrentIndex(gallery.count() - 1)
    for width, columns in ((220, 2), (115, 1), (520, 3)):
        gallery.resize(width, gallery.height())
        application.processEvents()
        assert gallery._columns == columns
        assert gallery.height() == _grid_height(gallery)
        assert gallery._grid.count() == gallery.count()
        assert all(button.isVisible() for button in gallery.buttons)
        assert gallery.buttons[-1].isChecked()
        assert gallery.more_button.isHidden()
        assert gallery._grid.indexOf(gallery.more_button) == -1


def test_expanded_keyboard_up_down_follow_visible_column_stride(gallery, application):
    gallery.buttons[0].setFocus()
    events = []
    gallery.currentIndexChanged.connect(events.append)
    QTest.keyClick(gallery.buttons[0], Qt.Key.Key_Down)
    assert gallery.currentIndex() == 3
    assert gallery.buttons[3].hasFocus()
    QTest.keyClick(gallery.buttons[3], Qt.Key.Key_Up)
    assert gallery.currentIndex() == 0
    gallery.resize(220, gallery.height())
    application.processEvents()
    QTest.keyClick(gallery.buttons[0], Qt.Key.Key_Down)
    assert gallery.currentIndex() == 2
    QTest.keyClick(gallery.buttons[2], Qt.Key.Key_Up)
    assert gallery.currentIndex() == 0
    assert events == [3, 0, 2, 0]


def test_collapse_retains_active_primitive_and_restores_more(gallery, application):
    events = []
    gallery.currentIndexChanged.connect(events.append)
    with QSignalBlocker(gallery):
        gallery.setCurrentIndex(gallery.count() - 1)
    selected = gallery.currentData()
    gallery.set_expanded(False)
    application.processEvents()
    assert gallery._shown_indices() == [0, 1, gallery.count() - 1]
    assert gallery.currentData() == selected
    assert events == []
    assert not gallery.more_button.isHidden()
    assert gallery._grid.indexOf(gallery.more_button) >= 0
    gallery.set_expanded(True)
    application.processEvents()
    assert gallery.currentData() == selected
    assert events == []
    assert all(button.isVisible() for button in gallery.buttons)
    assert gallery.height() == _grid_height(gallery)
