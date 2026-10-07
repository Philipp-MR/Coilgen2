"""Focused target windows operate over the same undoable scientific model."""

from dataclasses import replace

import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QTest
from shiboken6 import isValid

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config


application = usability.application
windows = usability.windows


def _target_window(windows):
    region = {
        "shape": "box",
        "size_m": [0.04123456789123456, 0.03234567891234567, 0.02345678912345678],
        "center_m": [0.001234567891234567, 0, 0],
    }
    project = ProjectDocument.from_config(replace(preset_config(), target_region_geometry=region))
    window = windows(project)
    window.select_node("target")
    return window


def test_target_placement_window_reopens_without_changing_exact_region(windows, application):
    window = _target_window(windows)
    original = window.project
    editor = window._target_region_editor
    section = editor.placement_section
    control = window._draft_controls["roi_center_m_x"]
    assert not section.dialog.isVisible()
    section.open()
    application.processEvents()
    assert section.dialog.isVisible()
    section.dialog.close()
    section.open()
    application.processEvents()
    assert window._draft_controls["roi_center_m_x"] is control
    assert window.apply_settings()
    assert window.project is original
    assert window.undo_stack.count() == 0


def test_target_placement_incomplete_text_survives_closing_and_fails_apply(windows, application):
    window = _target_window(windows)
    original = window.project
    section = window._target_region_editor.placement_section
    control = window._draft_controls["roi_center_m_x"]
    section.open()
    application.processEvents()
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "1e")
    section.dialog.close()
    section.open()
    application.processEvents()
    assert control.lineEdit().text().startswith("1e")
    assert not window.apply_settings()
    assert window.project is original and window.undo_stack.count() == 0
    assert not window._field_errors["roi_center_m_x"].isHidden()


def test_region_sample_count_reports_actual_preview_without_sampling_again(windows):
    window = _target_window(windows)
    original = window.project
    editor = window._target_region_editor
    editor.set_preflight({"target": {"available": True, "sample_count": 4913, "active_sample_count": 4813}})
    assert "4,913 target samples" in editor.sample_count_label.text()
    assert "4,813 active" in editor.sample_count_label.text()
    assert "current model" in editor.sample_count_label.text()
    assert window.project is original and window.undo_stack.count() == 0
    editor.set_preflight(None)
    assert "after preview" in editor.sample_count_label.text()


def test_navigating_away_disposes_owned_region_windows(windows, application):
    window = _target_window(windows)
    editor = window._target_region_editor
    placement, picker = editor.placement_section.dialog, editor.gallery.dialog
    editor.placement_section.open()
    editor.gallery.showPopup()
    application.processEvents()
    window.select_node("geometry")
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    assert not isValid(placement) and not isValid(picker)
