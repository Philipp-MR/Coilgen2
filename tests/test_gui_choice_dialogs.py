"""On-demand choices retain the exact combo model and draft signal contract."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QWidget
from shiboken6 import isValid

from coilgen.gui.choice_dialogs import GraphicalChoiceCombo


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["graphical-choice-test"])


@pytest.fixture
def selector(application):
    widget = GraphicalChoiceCombo(
        choices={
            "gradient": ("target-gradient", "Linear gradient", "Choose a gradient direction and strength."),
            "harmonic": ("target-harmonic", "Spherical harmonic", "Choose the harmonic coefficients in the next step."),
            "map": ("target-map", "Field map", "Import a measured or simulated target field."),
        },
        title="Choose target field",
    )
    widget.addItem("gradient", "Gy")
    widget.addItem("harmonic", {"l": 2, "m": 0})
    widget.addItem("map", "target.npy")
    widget.resize(350, 38)
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_summary_is_compact_and_choices_open_without_mutation(selector, application):
    events = []
    selector.currentIndexChanged.connect(events.append)
    assert selector.height() == 38
    assert not selector.dialog.isVisible()
    QTest.mouseClick(selector.change_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert selector.dialog.isVisible()
    assert not selector.dialog.isModal()
    assert selector.dialog.parent() is selector
    assert selector.currentData() == "Gy"
    assert events == []
    assert selector.buttons[0].isChecked()
    assert selector.buttons[0].accessibleName() == "gradient"
    assert "strength" in selector.buttons[0].description_label.text()
    selector.dialog.reject()
    assert selector.currentData() == "Gy"
    assert events == []


def test_real_tile_click_commits_once_and_same_selection_is_stable(selector, application):
    events = []
    activations = []
    text_activations = []
    selector.currentIndexChanged.connect(events.append)
    selector.activated.connect(activations.append)
    selector.textActivated.connect(text_activations.append)
    selector.showPopup()
    application.processEvents()
    QTest.mouseClick(selector.buttons[1], Qt.MouseButton.LeftButton)
    assert selector.currentData() == {"l": 2, "m": 0}
    assert events == [1]
    assert activations == [1]
    assert text_activations == ["harmonic"]
    assert not selector.dialog.isVisible()
    assert selector.selected_button.accessibleName() == "harmonic"
    selector.showPopup()
    application.processEvents()
    assert selector.buttons[1].isChecked()
    QTest.mouseClick(selector.buttons[1], Qt.MouseButton.LeftButton)
    assert events == [1]
    assert activations == [1, 1]
    assert text_activations == ["harmonic", "harmonic"]
    assert selector.buttons[1].isChecked()


def test_blocked_model_loading_syncs_both_views(selector):
    events = []
    selector.currentIndexChanged.connect(events.append)
    with QSignalBlocker(selector):
        selector.setCurrentText("map")
    assert events == []
    assert selector.currentData() == "target.npy"
    assert selector.buttons[2].isChecked()
    assert selector.selected_button.accessibleName() == "map"
    with QSignalBlocker(selector):
        selector.setCurrentIndex(1)
    assert selector.buttons[1].isChecked()
    assert selector.selected_button.accessibleName() == "harmonic"
    assert events == []


def test_unknown_imported_value_stays_exact_and_distinct(selector):
    selector.addItem("research_solver_v7", {"custom": "verbatim"})
    with QSignalBlocker(selector):
        selector.setCurrentIndex(3)
    assert selector.currentText() == "research_solver_v7"
    assert selector.currentData() == {"custom": "verbatim"}
    assert selector.buttons[3].title_label.text() == "research_solver_v7"
    assert "Imported / custom" in selector.buttons[3].description_label.text()
    selector.showPopup()
    selector.hidePopup()
    assert selector.currentText() == "research_solver_v7"
    selector.setItemData(3, {"custom": "updated"})
    assert selector.buttons[3].isChecked()
    assert selector.currentData() == {"custom": "updated"}


def test_disabled_choice_is_visible_but_cannot_mutate(selector, application):
    selector.model().item(1).setEnabled(False)
    selector.showPopup()
    application.processEvents()
    assert not selector.buttons[1].isEnabled()
    QTest.mouseClick(selector.buttons[1], Qt.MouseButton.LeftButton)
    assert selector.currentIndex() == 0
    assert selector.dialog.isVisible()
    selector.setEnabled(False)
    assert not selector.dialog.isVisible()
    selector.showPopup()
    assert not selector.dialog.isVisible()


def test_arrow_navigation_only_moves_focus_enter_selects(selector, application):
    events = []
    selector.currentIndexChanged.connect(events.append)
    selector.setFocus()
    QTest.keyClick(selector, Qt.Key.Key_Space)
    application.processEvents()
    assert selector.dialog.isVisible()
    selector.buttons[0].setFocus()
    QTest.keyClick(selector.buttons[0], Qt.Key.Key_Right)
    assert selector.buttons[1].hasFocus()
    assert selector.currentIndex() == 0
    assert events == []
    QTest.keyClick(selector.buttons[1], Qt.Key.Key_Return)
    assert selector.currentIndex() == 1
    assert events == [1]


def test_keyboard_skips_disabled_choice_and_escape_keeps_selection(selector, application):
    selector.model().item(1).setEnabled(False)
    selector.showPopup()
    application.processEvents()
    selector.buttons[0].setFocus()
    QTest.keyClick(selector.buttons[0], Qt.Key.Key_Right)
    assert selector.buttons[2].hasFocus()
    assert selector.currentIndex() == 0
    QTest.keyClick(selector.buttons[2], Qt.Key.Key_Escape)
    assert not selector.dialog.isVisible()
    assert selector.currentIndex() == 0


def test_many_choices_scroll_on_small_dialog_without_losing_active_item(selector, application):
    for index in range(30):
        selector.addItem(f"custom_{index}", index)
    selector.setCurrentIndex(selector.count() - 1)
    selector.showPopup()
    selector.dialog.resize(285, 310)
    application.processEvents()
    assert selector._columns == 1
    assert selector.scroll.verticalScrollBar().maximum() > 0
    assert selector.scroll.horizontalScrollBar().maximum() == 0
    assert selector.buttons[-1].isChecked()
    assert selector.buttons[-1].isVisible()
    selector.scroll.ensureWidgetVisible(selector.buttons[-1])
    assert selector.currentData() == 29
    selector.dialog.resize(690, 310)
    application.processEvents()
    assert selector._columns == 3
    stretched_rows = [row for row in range(selector.grid.rowCount()) if selector.grid.rowStretch(row)]
    assert stretched_rows == [(len(selector.buttons) + 2) // 3]


def test_model_replacement_removal_and_reset_update_views(selector, application):
    model = QStandardItemModel(selector)
    first = QStandardItem("harmonic")
    first.setData("retained", Qt.ItemDataRole.UserRole)
    model.appendRow(first)
    model.appendRow(QStandardItem("custom"))
    selector.setModel(model)
    selector.setCurrentIndex(1)
    assert len(selector.buttons) == 2
    assert selector.selected_button.accessibleName() == "custom"
    model.removeRow(1)
    application.processEvents()
    assert len(selector.buttons) == 1
    assert selector.currentData() == "retained"
    model.clear()
    assert selector.buttons == []
    assert selector.currentIndex() == -1
    assert not selector.change_button.isEnabled()


def test_parent_cleanup_destroys_owned_dialog(application):
    parent = QWidget()
    widget = GraphicalChoiceCombo(parent)
    widget.addItem("custom")
    dialog = widget.dialog
    widget.showPopup()
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(widget)
    assert not isValid(dialog)


def test_embedded_tiles_use_the_same_model_and_keep_decision_open(selector, application):
    host = QDialog(selector)
    layout = QVBoxLayout(host)
    layout.addWidget(selector.scroll)
    opened = []

    def open_host():
        opened.append(True)
        host.show()

    selector.set_embedded_host(host, open_callback=open_host)
    events = []
    selector.currentIndexChanged.connect(events.append)
    selector.showPopup()
    application.processEvents()
    assert opened == [True]
    assert host.isVisible()
    assert not selector.dialog.isVisible()
    QTest.mouseClick(selector.buttons[1], Qt.MouseButton.LeftButton)
    assert selector.currentData() == {"l": 2, "m": 0}
    assert events == [1]
    assert host.isVisible()
    assert not selector.dialog.isVisible()
    assert selector.buttons[1].hasFocus()
    selector.setEnabled(False)
    assert host.isVisible()
    assert all(not button.isEnabled() for button in selector.buttons)
    selector.setEnabled(True)
    assert selector.buttons[1].isEnabled()
    host.hide()
    assert selector.currentIndex() == 1
    selector.showPopup()
    assert host.isVisible()
    assert opened == [True, True]
