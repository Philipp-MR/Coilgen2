"""Workspace panels are recoverable views, without edits to scientific state."""

from copy import deepcopy
from pathlib import Path

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QDockWidget

from coilgen.application import ProjectDocument
from coilgen.application.system_presets import system_assembly


application = usability.application
windows = usability.windows


def _dock(window, name):
    return window.findChild(QDockWidget, f"{name}_dock")


def _show(window, application):
    window.resize(1120, 800)
    window.show()
    window.activateWindow()
    application.processEvents()


def _open_tabs(window):
    return [
        window.evidence.tabText(index)
        for index in range(window.evidence.count())
        if window.evidence.isTabVisible(index)
    ]


def test_new_workspace_preserves_a_clear_model_view(windows, application):
    window = windows()
    _show(window, application)
    assert not _dock(window, "model").isHidden()
    assert _dock(window, "settings").isHidden()
    assert _dock(window, "evidence") is None
    assert window.evidence.isHidden()  # Its backward-compatible page registry is never displayed.
    assert not window.evidence.any_open()
    assert not _open_tabs(window)
    assert window.view_menu in [action.menu() for action in window.menuBar().actions()]
    assert window.view_menu.isEnabled()
    assert not hasattr(window, "panels_button")


def test_clicking_the_already_selected_model_item_reopens_settings(windows, application):
    window = windows()
    _show(window, application)
    original = window.project
    selected = window.tree.currentItem()
    _dock(window, "settings").hide()
    centre = window.tree.visualItemRect(selected).center()
    QTest.mouseClick(window.tree.viewport(), Qt.MouseButton.LeftButton, pos=centre)
    application.processEvents()
    assert window.tree.currentItem() is selected
    assert not _dock(window, "settings").isHidden()
    assert window.project is original
    assert window.undo_stack.count() == 0


def test_closing_every_evidence_window_preserves_running_job_model_and_viewport(windows, application):
    window = windows()
    _show(window, application)
    job = window.compute()
    assert job is not None
    window.jobs.states[0]["status"] = "running"
    window.poll_jobs()
    for _ in range(3):
        application.processEvents()  # Settle progress/header layout before comparing window-only actions.
    original = window.project
    jobs = deepcopy(window.jobs.states)
    submissions = len(window.service.submissions)
    viewport_geometry = window.viewport.geometry()
    for index in range(window.evidence.count()):
        page = window.evidence.widget(index)
        window._open_evidence(page)
        application.processEvents()
        dialog = window.evidence.window_for(page)
        assert isinstance(dialog, QDialog)
        assert dialog.parentWidget() is window and not dialog.isModal()
        assert dialog.isVisible()
        assert dialog.content is page or dialog.content.isAncestorOf(page)
        assert window.evidence.isTabVisible(index)
        assert window.evidence.currentWidget() is page
        assert window.viewport.geometry() == viewport_geometry
        dialog.close()
        application.processEvents()
        assert dialog.isHidden()
        assert not window.evidence.isTabVisible(index)
        assert window.evidence.widget(index) is page
        window._open_evidence(page)
        assert window.evidence.window_for(page) is dialog  # Reuse preserves the exact evidence widget.
        dialog.close()
    window.evidence.close_all()
    application.processEvents()
    assert _dock(window, "evidence") is None
    assert not _open_tabs(window)
    assert window.project is original
    assert window.jobs.states == jobs
    assert len(window.service.submissions) == submissions
    assert window.undo_stack.count() == 0


def test_view_menu_reopens_closed_pages_with_context_controls(windows, application):
    window = windows()
    _show(window, application)
    original = window.project
    viewport_geometry = window.viewport.geometry()
    jobs_dialog = None
    for title, page in (
        ("Jobs", window.jobs_table),
        ("Design acceptance", window.acceptance_panel),
        ("Project history", window.history_panel),
    ):
        action = window.evidence_actions[title]
        assert not action.icon().isNull()
        action.trigger()
        application.processEvents()
        index = window.evidence.indexOf(page)
        dialog = window.evidence.window_for(page)
        assert dialog.isVisible()
        assert dialog.parentWidget() is window
        assert window.evidence.isTabVisible(index)
        assert window.evidence.currentWidget() is page
        if page is window.jobs_table:
            jobs_dialog = dialog
        assert jobs_dialog.isVisible()
        assert jobs_dialog.isAncestorOf(window.jobs_filter_bar)
        assert window.jobs_filter_bar.isVisible()  # Jobs keeps its controls while another window is inspected.
        assert window.viewport.geometry() == viewport_geometry
        if page is not window.jobs_table:
            dialog.close()
            assert dialog.isHidden()
            assert jobs_dialog.isVisible()
    jobs_dialog.close()
    assert not window.evidence.any_open()
    assert window.project is original
    assert not window.jobs.states
    assert window.undo_stack.count() == 0


def test_project_json_restores_open_tabs_and_panel_visibility(windows, application, tmp_path):
    source = windows()
    _show(source, application)
    source._inspect_node("target")
    source.evidence_actions["Jobs"].trigger()
    source.evidence_actions["Project history"].trigger()
    assert _open_tabs(source) == ["Jobs", "Project history"]
    original = source.project
    path = source.save_project(tmp_path / "workspace.coilgen.json")
    loaded = ProjectDocument.load(path)
    assert loaded.fingerprint == original.fingerprint
    assert loaded.revision == original.revision
    view = loaded.session["view_state"]
    assert view["model_node"] == "target"
    assert view["open_evidence_tabs"] == ["Jobs", "Project history"]
    assert view["evidence_tab"] == "Project history"
    assert view["settings_panel_visible"] and view["evidence_panel_visible"]
    restored = windows(loaded, tmp_path / "fresh-runtime" / "jobs")
    _show(restored, application)
    assert not _dock(restored, "settings").isHidden()
    assert restored.evidence.window_for(restored.jobs_table).isVisible()
    assert restored.evidence.window_for(restored.history_panel).isVisible()
    assert _dock(restored, "evidence") is None
    assert _open_tabs(restored) == ["Jobs", "Project history"]
    assert restored.evidence.currentWidget() is restored.history_panel
    assert restored.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "target"
    assert not restored.jobs.states


def test_project_json_preserves_deliberately_closed_panels(windows, application, tmp_path):
    source = windows()
    _show(source, application)
    source.evidence_actions["Jobs"].trigger()
    source.evidence.window_for(source.jobs_table).close()
    _dock(source, "settings").hide()
    path = source.save_project(tmp_path / "closed-workspace.coilgen.json")
    restored = windows(ProjectDocument.load(path), tmp_path / "fresh-runtime" / "jobs")
    _show(restored, application)
    assert _dock(restored, "settings").isHidden()
    assert not restored.evidence.any_open()
    assert not _open_tabs(restored)
    restored.evidence_actions["Jobs"].trigger()
    assert restored.evidence.window_for(restored.jobs_table).isVisible()
    assert restored.evidence.currentWidget() is restored.jobs_table


def test_channel_workspace_survives_switching_without_a_local_history_cache(windows, application, tmp_path):
    window = windows(system_assembly())
    window.history = None
    _show(window, application)
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("xz"))
    window._inspect_node("target")
    window._open_evidence(window.history_panel)
    before = window.project
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("gy"))
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("xz"))
    assert window.project is before
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "target"
    assert not _dock(window, "settings").isHidden()
    assert window.evidence.window_for(window.history_panel).isVisible()
    assert _open_tabs(window) == ["Project history"]
    window.channel_selector.setCurrentIndex(window.channel_selector.findData("gy"))
    path = window.save_project(tmp_path / "channels.json")
    restored = windows(directory=tmp_path / "fresh-runtime" / "jobs")
    restored.open_project(path)
    restored.channel_selector.setCurrentIndex(restored.channel_selector.findData("xz"))
    assert restored.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "target"
    assert _open_tabs(restored) == ["Project history"]


@pytest.mark.parametrize("view", [None, [], "legacy display preference"])
def test_invalid_display_preferences_do_not_prevent_opening_valid_inputs(windows, view):
    original = windows().project
    project = original.with_results((), session={"view_state": view})
    window = windows(project)
    assert window.project.fingerprint == original.fingerprint
    assert not _open_tabs(window)
    assert _dock(window, "settings").isHidden()
    assert not window.evidence.any_open()
    assert _dock(window, "evidence") is None


@pytest.mark.parametrize("width", (800, 900, 1040, 1160, 1280))
@pytest.mark.parametrize("multiple", (False, True))
def test_primary_compute_and_contextual_channel_controls_at_native_font(windows, application, width, multiple):
    window = windows(system_assembly()) if multiple else windows()
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            window.setFont(QFont(families[0], 9))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(width, 800)
    window.show()
    for _ in range(3):
        application.processEvents()
    assert window.channel_enabled.isVisible() is multiple
    assert window.channel_selector.isVisible() is multiple
    assert not hasattr(window, "panels_button")
    assert window.stage_selector.isHidden()
    assert not hasattr(window, "workflow")
    assert window.compute_button.isVisible()
