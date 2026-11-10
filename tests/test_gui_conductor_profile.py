"""The point editor retains scientific coordinates and incomplete UI drafts."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLineEdit

from coilgen.gui.conductor_profile import ConductorProfileEditor


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["conductor-profile-test"])


@pytest.fixture
def editor(application):
    widget = ConductorProfileEditor()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(760, 520)
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_rectangle_mode_preserves_legacy_sentinel_and_uses_changed_dimensions(editor):
    signals = QSignalSpy(editor.changed)
    assert editor.profile() == ((0.0, 0.0),)
    assert editor.table.isHidden() or editor.table.parentWidget().isHidden()
    editor.set_rectangle_dimensions(4, 1)
    assert signals.count() == 0
    assert editor.profile() == ((0.0, 0.0),)
    assert editor.summary_text() == "Rectangle 4 × 1 mm"
    editor.mode_combo.setCurrentIndex(1)
    assert editor.profile() == ((-0.002, -0.0005), (0.002, -0.0005), (0.002, 0.0005), (-0.002, 0.0005))
    assert "4 mm²" in editor.summary_text()


def test_unchanged_coordinates_keep_exact_si_values_and_origin(editor):
    points = (
        (0.001234567891234567, 0.004234567891234567),
        (0.003456789123456789, 0.004234567891234567),
        (0.003456789123456789, 0.006345678912345678),
        (0.001234567891234567, 0.006345678912345678),
    )
    signals = QSignalSpy(editor.changed)
    editor.set_profile(points)
    assert signals.count() == 0
    assert editor.profile() == points
    editor.table.item(0, 0).setText("1.1")
    expected = ((0.0011, points[0][1]), *points[1:])
    assert editor.profile() == expected
    editor.mode_combo.setCurrentIndex(0)
    assert editor.profile() == ((0.0, 0.0),)
    editor.set_rectangle_dimensions(6, 2)
    editor.mode_combo.setCurrentIndex(1)
    assert editor.profile() == expected


def test_nonconvex_or_crossing_draft_stays_visible_and_blocks_apply(editor):
    editor.mode_combo.setCurrentIndex(1)
    editor.table.item(1, 0).setText("-2")
    raw = editor.draft_value()
    assert editor.validation_message()
    with pytest.raises(ValueError):
        editor.profile()
    assert "Cannot apply" in editor.status_label.text()
    assert editor.preview._points
    assert not editor.preview._valid
    editor.hide()
    editor.show()
    assert editor.draft_value() == raw
    assert editor.table.item(1, 0).text() == "-2"


def test_active_cell_text_is_a_draft_before_focus_leaves_the_cell(editor, application):
    editor.mode_combo.setCurrentIndex(1)
    editor.table.setCurrentCell(0, 0)
    editor.table.editItem(editor.table.item(0, 0))
    application.processEvents()
    cell_editor = editor.table.findChild(QLineEdit)
    assert cell_editor is not None
    changes = QSignalSpy(editor.changed)
    QTest.keyClick(cell_editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(cell_editor, "-1e")
    assert cell_editor.isVisible()  # the cell editor remains open during typing
    assert changes.count() >= 3
    assert editor.draft_value()[1][0][0] == "-1e"
    assert editor.table.item(0, 0).text() == "-1e"
    with pytest.raises(ValueError, match="Point 1"):
        editor.profile()
    QTest.keyClicks(cell_editor, "0")
    assert editor.profile()[0] == (-0.001, -0.001)


def test_split_closing_segment_numbering_and_removal_keep_point_order(editor):
    editor.mode_combo.setCurrentIndex(1)
    original = editor.profile()
    editor.select_point(3)
    editor.insert_point()
    assert editor.table.rowCount() == 5
    assert editor.table.verticalHeaderItem(4).text() == "5"
    assert editor.table.item(4, 0).text() == "-1.0"
    assert editor.table.item(4, 1).text() == "0.0"
    assert editor.validation_message()  # midpoint must be moved into a convex corner
    editor._move_point(4, -1.2, 0)
    assert editor.profile() == (*original, (-0.0012, 0))
    editor.remove_point()
    assert editor.profile() == original
    editor.select_point(2)
    editor.remove_point()
    assert editor.table.rowCount() == 3
    assert not editor.remove_button.isEnabled()
    editor.remove_point()
    assert editor.table.rowCount() == 3


def test_inline_coordinate_typing_preserves_cursor_position(editor, application):
    editor.mode_combo.setCurrentIndex(1)
    editor.table.setCurrentCell(0, 0)
    editor.table.editItem(editor.table.item(0, 0))
    application.processEvents()
    cell_editor = editor.table.findChild(QLineEdit)
    assert cell_editor.text() == "-1.0"
    cell_editor.setCursorPosition(2)
    QTest.keyClicks(cell_editor, "25")
    assert cell_editor.text() == "-125.0"
    assert cell_editor.cursorPosition() == 4
    assert editor.draft_value()[1][0][0] == "-125.0"


def test_hidden_inline_invalid_editor_retains_text_but_load_replaces_it(editor, application):
    editor.mode_combo.setCurrentIndex(1)
    editor.table.setCurrentCell(0, 0)
    editor.table.editItem(editor.table.item(0, 0))
    application.processEvents()
    cell_editor = editor.table.itemDelegate().editor
    QTest.keyClick(cell_editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(cell_editor, "1e")
    editor.hide()
    editor.show()
    application.processEvents()
    assert editor.draft_value()[1][0][0] == "1e"
    assert editor.validation_message()
    rectangle = ((-0.003, -0.001), (0.003, -0.001), (0.003, 0.001), (-0.003, 0.001))
    editor.set_profile(rectangle)
    application.processEvents()
    assert editor.profile() == rectangle
    assert editor.table.itemDelegate().editor is None


def test_dragging_vertex_updates_numbered_table_and_si_profile(editor, application):
    editor.mode_combo.setCurrentIndex(1)
    original = editor.profile()
    preview = editor.preview
    view_bounds = preview._bounds
    start = preview.point_position((-1, -1)).toPoint()
    finish = preview.point_position((-1.35, -1)).toPoint()
    QTest.mousePress(preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    assert preview._drag == 0
    assert editor.table.currentRow() == 0
    QTest.mouseMove(preview, finish)
    application.processEvents()
    QTest.mouseRelease(preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, finish)
    assert preview._drag == -1
    assert preview._bounds == view_bounds
    moved = editor.profile()
    assert moved[0][0] < original[0][0]
    assert moved[0][1] == pytest.approx(original[0][1])
    assert moved[1:] == original[1:]
    assert float(editor.table.item(0, 0).text()) == pytest.approx(moved[0][0] * 1e3)


def test_valid_preview_is_equal_scale_copper_with_origin_axes(editor, application):
    editor.set_rectangle_dimensions(4, 1)
    editor.mode_combo.setCurrentIndex(1)
    application.processEvents()
    p0 = editor.preview.point_position((0, 0))
    pu = editor.preview.point_position((1, 0))
    pn = editor.preview.point_position((0, 1))
    assert pu.x() - p0.x() == pytest.approx(p0.y() - pn.y())
    image = editor.preview.grab().toImage()
    assert image.pixelColor(QPoint(int(p0.x() + 25), int(p0.y() - 15))) != image.pixelColor(3, 35)
    assert "u [mm]" == editor.table.horizontalHeaderItem(0).text()
    assert "n [mm]" == editor.table.horizontalHeaderItem(1).text()


@pytest.mark.parametrize("text", ["", "nan", "inf", "1e", "-inf"])
def test_invalid_numeric_drafts_never_return_previous_polygon(editor, text):
    editor.mode_combo.setCurrentIndex(1)
    editor.table.item(2, 1).setText(text)
    assert editor.validation_message()
    with pytest.raises(ValueError):
        editor.value()
    assert editor.draft_value()[1][2][1] == text
    assert not editor.preview._points


def test_custom_profile_is_independent_of_incomplete_rectangle_dimensions(editor):
    editor.mode_combo.setCurrentIndex(1)
    points = editor.profile()
    editor.set_rectangle_dimensions(None, "1e")
    editor.table.item(0, 0).setText("-1.2")
    assert not editor.validation_message()
    assert editor.profile()[0] == (-0.0012, points[0][1])
    editor.mode_combo.setCurrentIndex(0)
    assert "width and height" in editor.validation_message()


@pytest.mark.parametrize(
    "profile,rows,first_text",
    [
        (((float("nan"), 0), (0.002, 0), (0.002, 0.001)), 3, "nan"),
        (((float("inf"), 0), (0.002, 0), (0.002, 0.001)), 3, "inf"),
        ((("bad", 0), (0.002, 0), (0.002, 0.001)), 3, "bad"),
        (((None, 0), (0.002, 0), (0.002, 0.001)), 3, "None"),
        (((0.001, 0), (0.002,)), 0, None),
        (17, 0, None),
    ],
)
def test_invalid_imported_profile_opens_recoverably_without_defaulting_to_rectangle(editor, profile, rows, first_text):
    signals = QSignalSpy(editor.changed)
    editor.set_profile(profile)
    assert signals.count() == 0
    assert editor.mode_combo.currentData() == "polygon"
    assert editor.table.rowCount() == rows
    if first_text is not None:
        assert editor.table.item(0, 0).text() == first_text
    assert editor.validation_message()
    with pytest.raises(ValueError):
        editor.profile()
    editor.hide()
    editor.show()
    assert editor.validation_message()
    editor.start_rectangle()
    assert editor.profile() == ((-0.001, -0.001), (0.001, -0.001), (0.001, 0.001), (-0.001, 0.001))
    assert signals.count() == 1


def test_rectangle_hides_inactive_polygon_coordinates_and_reopens_them_exactly(editor):
    points = ((-0.002, -0.001), (0.002, -0.001), (0.0015, 0.001), (-0.0015, 0.001))
    editor.set_profile(points)
    assert editor._table_panel.isVisible()
    editor.mode_combo.setCurrentIndex(0)
    editor.set_rectangle_dimensions(5, 3)
    assert editor._table_panel.isHidden()
    assert editor.preview._points == ((-2.5, -1.5), (2.5, -1.5), (2.5, 1.5), (-2.5, 1.5))
    editor.mode_combo.setCurrentIndex(1)
    assert editor._table_panel.isVisible()
    assert editor.profile() == points


def test_compact_profile_height_keeps_preview_and_validation_visible(editor, application):
    editor.mode_combo.setCurrentIndex(1)
    editor.resize(578, 390)
    application.processEvents()
    assert editor.preview.height() >= 250
    assert editor.status_label.geometry().bottom() < editor.height()
    assert editor.rectangle_button.geometry().bottom() < editor.height()
    editor.table.item(0, 0).setText("1e")
    application.processEvents()
    assert editor.status_label.geometry().bottom() < editor.height()
    assert "Cannot apply" in editor.status_label.text()


def test_point_count_and_equal_area_drive_round_and_u_shape_presets(editor):
    editor.set_rectangle_dimensions(2, 3)
    editor.segments_spin.setValue(8)
    signals = QSignalSpy(editor.changed)
    editor.mode_combo.setCurrentIndex(editor.mode_combo.findData("round"))
    assert len(editor.profile()) == 8
    assert signals.count() == 1
    plain = editor.profile()
    editor.equal_area_check.setChecked(True)
    from coilgen.application.conductor_profiles import validate_conductor_profile

    assert validate_conductor_profile(editor.profile()).area_m2 == pytest.approx(3.141592653589793e-6)
    assert validate_conductor_profile(plain).area_m2 < 3.0e-6
    editor.u_shape_button.click()
    summary = validate_conductor_profile(editor.profile())
    assert editor.table.rowCount() == 7  # base corners, two shoulders, 3 interior arc points
    assert summary.width_m == pytest.approx(0.002) and summary.height_m == pytest.approx(0.003)
    editor.set_rectangle_dimensions(4, 1)  # height below the half-circle radius
    editor.u_shape_button.click()
    assert "at least half its width" in editor.status_label.text()
    assert editor.table.rowCount() == 7


def test_table_import_and_paste_replace_the_polygon(editor, tmp_path, application):
    path = tmp_path / "profile.csv"
    path.write_text("u;n\n-1;0\n1;0\n0;1,5\n", encoding="utf-8")
    assert editor.import_table(path)
    assert editor.mode_combo.currentData() == "polygon"
    assert editor.profile() == ((-0.001, 0.0), (0.001, 0.0), (0.0, 0.0015))
    bad = tmp_path / "bad.txt"
    bad.write_text("0 0\n1 x\n", encoding="utf-8")
    assert not editor.import_table(bad)
    assert "Cannot import table" in editor.status_label.text()
    assert editor.profile() == ((-0.001, 0.0), (0.001, 0.0), (0.0, 0.0015))
    application.clipboard().setText("0\t0\n2\t0\n2\t1\n0\t1\n")
    editor.table.setFocus()
    QTest.keyClick(editor.table, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert editor.profile() == ((0.0, 0.0), (0.002, 0.0), (0.002, 0.001), (0.0, 0.001))
    assert not editor.paste_table("1.5")  # a single copied cell is not a table
    assert editor.table.rowCount() == 4
