"""Physical profile drafts commit atomically through the existing model boundary."""

from dataclasses import replace

import numpy as np
import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.conductor_preview import ConductorPreview


application = usability.application
windows = usability.windows

PROFILE = ((-0.001, -0.001), (0.001, -0.001), (0.0007, 0.001), (-0.0007, 0.001))


def _window(windows, points=((0.0, 0.0),)):
    window = windows(ProjectDocument.from_config(replace(preset_config(), cross_sectional_points=points)))
    window.select_node("material")
    return window


def _editor(window):
    return window._draft_controls["cross_sectional_points"]


def _edit_cell(window, application, text):
    editor = _editor(window)
    window._conductor_profile_section.open()
    application.processEvents()
    editor.table.setCurrentCell(1, 0)
    editor.table.editItem(editor.table.item(1, 0))
    application.processEvents()
    entry = editor.table.findChild(QLineEdit)
    assert entry is not None
    entry.selectAll()
    QTest.keyClicks(entry, text)
    return editor, entry


def test_profile_window_keeps_original_coordinates_on_noop_and_close(windows, application):
    # Clockwise, off-center coordinates with precision beyond the mm display.
    profile = tuple(tuple(point) for point in np.asarray(PROFILE)[::-1] + (0.003123456789123456, 0.001234567891234567))
    window = _window(windows, profile)
    original = window.project
    section = window._conductor_profile_section
    assert not section.dialog.isVisible() and not section.dialog.isModal()
    QTest.mouseClick(section.button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert section.dialog.isVisible()
    assert _editor(window).profile() == profile
    section.dialog.close()
    section.open()
    assert window.apply_settings()
    assert window.project is original and window.undo_stack.count() == 0
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, profile)


def test_polygon_coordinate_apply_undo_redo_and_json_preserve_si(windows, application, tmp_path):
    window = _window(windows, PROFILE)
    original = window.project
    _edit_cell(window, application, "1.25")
    assert window._draft_dirty and window.project is original
    expected = list(PROFILE)
    expected[1] = (0.00125, expected[1][1])
    expected = tuple(expected)
    assert window.apply_settings()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, expected)
    assert window.project.revision == original.revision + 1
    assert window._conductor_profile_section.dialog.isVisible()
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, PROFILE)
    window.undo_stack.redo()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, expected)
    path = window.save_project(tmp_path / "profile.coilgen.json")
    restored = ProjectDocument.load(path)
    np.testing.assert_array_equal(restored.config.cross_sectional_points, expected)
    script = window.export_python(tmp_path / "profile.py")
    assert script.is_file() and "DesignService" in script.read_text(encoding="utf-8")
    np.testing.assert_array_equal(restored.config.cross_sectional_points, expected)


def test_pending_incomplete_cell_survives_close_blocks_compute_save_and_export(windows, application, tmp_path):
    window = _window(windows, PROFILE)
    original = window.project
    editor, _entry = _edit_cell(window, application, "1e")
    assert window._draft_dirty and editor.table.item(1, 0).text() == "1e"
    window._conductor_profile_section.dialog.close()
    window.compute()
    application.processEvents()
    assert window.project is original and not window.service.submissions
    assert window._conductor_profile_section.dialog.isVisible()
    assert editor.table.item(1, 0).text() == "1e"
    assert not window._field_errors["cross_sectional_points"].isHidden()
    assert "Point 2" in window.draft_error_label.text()
    with pytest.raises(ValueError, match="Correct pending settings"):
        window.save_project(tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="Correct the pending settings"):
        window.export_python(tmp_path / "invalid.py")
    assert not (tmp_path / "invalid.json").exists() and not (tmp_path / "invalid.py").exists()
    window.revert_draft()
    assert _editor(window).profile() == PROFILE and not window._draft_dirty
    assert window._conductor_profile_section.dialog.isVisible()
    assert window.undo_stack.count() == 0


def test_live_valid_cell_compute_submits_new_profile_without_enter(windows, application):
    window = _window(windows, PROFILE)
    _edit_cell(window, application, "1.3")
    window.compute()
    assert len(window.service.submissions) == 1
    project, _controls = window.service.submissions[0]
    assert project.config.cross_sectional_points[1][0] == pytest.approx(0.0013)
    np.testing.assert_array_equal(project.config.cross_sectional_points[0], PROFILE[0])


def test_rectangle_dimensions_do_not_overwrite_custom_polygon(windows):
    window = _window(windows, PROFILE)
    editor = _editor(window)
    window._draft_controls["wire_width"].setValue(7.5)
    window._draft_controls["wire_height"].setValue(4.0)
    assert editor.profile() == PROFILE
    assert editor.summary_text().startswith("Polygon")
    assert window.apply_settings()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, PROFILE)
    assert window.project.config.conductor_cross_section_width == 0.0075
    assert window._draft_controls["wire_width"].accessibleName() == "Nominal width"
    assert window.project.config.conductor_cross_section_height == 0.004
    editor = _editor(window)
    editor.mode_combo.setCurrentIndex(editor.mode_combo.findData("rectangle"))
    assert window.apply_settings()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, ((0.0, 0.0),))
    assert window.project.config.conductor_cross_section_width == 0.0075
    assert window._draft_controls["wire_width"].accessibleName() == "Conductor width"


def test_invalid_geometry_highlights_profile_even_when_other_fields_changed(windows):
    window = _window(windows, PROFILE)
    original = window.project
    editor = _editor(window)
    window._draft_controls["insulation"].setValue(0.15)
    editor.table.item(1, 0).setText("-2")
    assert not window.apply_settings()
    assert window.project is original
    assert not window._field_errors["cross_sectional_points"].isHidden()
    preview = window.settings.widget().findChild(ConductorPreview)
    assert preview.drawing._profile is None
    assert "Invalid profile draft" in preview.summary_label.text()


def test_rectangle_to_polygon_is_one_undoable_profile_change(windows):
    window = _window(windows)
    original = window.project
    editor = _editor(window)
    editor.mode_combo.setCurrentIndex(editor.mode_combo.findData("polygon"))
    assert window._draft_dirty
    profile = editor.profile()
    assert len(profile) == 4
    assert window.apply_settings()
    assert window.project.revision == original.revision + 1
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, profile)
    assert window.undo_stack.count() == 1


@pytest.mark.parametrize(
    "points",
    (
        ((0, 0), (0.002, 0), (0.001, 0.001), (0.002, 0.002), (0, 0.002)),
        ((0, 0), (0.001, 0)),
        ((0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)),
    ),
)
def test_imported_invalid_profile_compute_opens_editor_for_deliberate_correction(windows, points):
    window = windows(ProjectDocument.from_config(replace(preset_config(), cross_sectional_points=points)))
    original = window.project
    window.compute()
    assert window.project is original and not window.service.submissions
    assert window.tree.currentItem() is window.tree_items["material"]
    assert window._conductor_profile_section.dialog.isVisible()
    assert _editor(window).validation_message()
    _editor(window).start_rectangle()
    assert window.apply_settings()
    assert window.project.revision == original.revision + 1
    assert len(window.project.config.cross_sectional_points) == 4


def test_profile_launcher_summary_tracks_external_rectangle_dimension_drafts(windows):
    window = _window(windows)
    section = window._conductor_profile_section
    width = window._draft_controls["wire_width"]
    width.setValue(4)
    assert section.summary_label.text() == "Rectangle 4 × 2 mm"
    assert _editor(window).profile() == ((0.0, 0.0),)
    width.lineEdit().selectAll()
    QTest.keyClicks(width.lineEdit(), "1e")
    assert section.summary_label.text() == "Profile incomplete"
    assert _editor(window).validation_message()
