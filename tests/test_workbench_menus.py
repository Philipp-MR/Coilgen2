"""Menu navigation across empty, edited, busy and saved-result contexts."""

import asyncio
import time
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPlainTextEdit

from _coilgen_test_support.workbench_usability import (
    application as application,
    prepared_run as prepared_run,
    windows as windows,
)


def test_top_menus_remain_accessible_without_a_project(windows):
    window = windows()
    assert [a.text() for a in window.menuBar().actions()] == ["File", "Edit", "Model", "Run", "Results", "View", "Help"]
    assert window.resume_action in window.run_menu.actions() and window.resume_action.isVisible()
    window._clear_project()
    assert all(a.isEnabled() for a in window.menuBar().actions())
    assert window.open_run_action.isEnabled()
    assert window.about_action.isEnabled()
    assert not window.compute_action.isEnabled()
    assert not window.field_analysis_action.isEnabled()
    assert all(a.toolTip() for a in window._project_menu_actions if not a.isEnabled())
    window.new_project("cylinder", discard_checked=True)
    assert all(a.isEnabled() for a in window._project_menu_actions)


@pytest.mark.parametrize(
    "answer", [QMessageBox.StandardButton.Apply, QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Cancel]
)
def test_open_editor_resolves_setup_without_silent_loss(windows, monkeypatch, answer):
    window = windows()
    original = window.project
    window.setup_panel.draft.update({"cylinder_radius_mm": "93"})
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: answer)
    ready = window._settings_ready_for_editor()
    assert ready == (answer != QMessageBox.StandardButton.Cancel)
    if answer == QMessageBox.StandardButton.Apply:
        assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.093)
    else:
        assert window.project is original
    assert window.setup_panel.dirty == (answer == QMessageBox.StandardButton.Cancel)
    window.setup_panel.reset()


def test_busy_job_does_not_lock_model_or_view_menus(windows):
    window = windows()
    window.compute()
    window._update_execution_feedback()
    assert window.service.submissions
    assert window.model_menu.isEnabled() and window.view_menu.isEnabled() and window.open_run_action.isEnabled()
    assert window.gradient_actions["_gradient_requirements_dialog"].isEnabled()
    window.compute()
    assert len(window.service.submissions) == 1


def test_result_actions_explain_missing_evidence(windows):
    window = windows()
    window._update_execution_feedback()
    actions = [
        window.field_analysis_action,
        window.shielding_action,
        window.routing_analysis_action,
        window.routing_editor_action,
        window.gradient_actions["_gradient_circuit_dialog"],
        window.gradient_actions["_gradient_measured_dialog"],
    ]
    assert all(not action.isEnabled() and action.toolTip() for action in actions)


def test_open_completed_run_from_empty_workspace(windows, prepared_run, monkeypatch, application):
    _project, record = prepared_run
    window = windows()
    window._clear_project()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a: str(record.directory))
    window.open_run_action.trigger()
    assert window.project is None
    assert window.workspace_stack.currentWidget() is window.workspace
    assert window.viewport.displayed["project_id"] == record.metadata["project_id"]
    assert window.view_context.context.source == "run"
    assert "saved result only" in window.view_context.context.editing_text
    assert window.runs_table.rowCount() == 1
    assert window.close_project_action.isEnabled()
    assert not window.compute_action.isEnabled()
    window._inspect_selected_saved_settings("geometry")
    deadline = time.monotonic() + 10
    while window._saved_settings_task is not None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert any(
        editor.isReadOnly() and record.metadata["project_id"] in editor.toPlainText()
        for editor in window.findChildren(QPlainTextEdit)
    )
    assert window.close_project()
    assert window.workspace_stack.currentWidget() is window.welcome


@pytest.mark.parametrize("empty", [False, True])
def test_saved_review_finishes_without_applying_unrelated_edits(windows, prepared_run, application, empty):
    _project, record = prepared_run
    window = windows()
    if empty:
        window._clear_project()
    window.load_run(record.directory)
    original = window.project
    if not empty:
        window.setup_panel.draft.update({"cylinder_radius_mm": "91"})
    display = Mock()
    window._research_verified_review("summary", display, "Review saved evidence")
    deadline = time.monotonic() + 10
    while not display.called and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert display.call_count == 1
    assert window.project is original
    if not empty:
        assert window.setup_panel.dirty
        window.setup_panel.reset()


@pytest.fixture
def browser(tmp_path):
    from coilgen.web.app import create_app
    from uuid import uuid4

    server, instance = create_app(tmp_path, server_name="menu-audit-" + uuid4().hex)
    yield instance
    instance.plotter.close()
    instance.session.close()


def test_browser_menu_actions_are_present_and_missing_prerequisites_guide_user(browser):
    from coilgen.web.menus import MENUS, menu_action

    template = browser.server.state.trame__template_main
    for entries in MENUS.values():
        for key, label in entries:
            assert label in template
            if key not in {"new", "undo", "redo", "revert", "fit", "plot"}:
                asyncio.run(menu_action(browser, key))
    assert browser.session.project is None
    assert browser.server.state.file_dialog and browser.server.state.menu_help


def test_browser_menu_shielding_uses_selected_source(browser, monkeypatch):
    from coilgen.web.menus import menu_action

    browser.new_project()
    browser.record = object()
    state = browser.server.state
    state.selected_run = state.displayed_run = "saved"
    state.shielding_source, state.shielding_component = "surface", "normal"
    show = Mock()
    monkeypatch.setattr(browser, "show_shielding", show)
    asyncio.run(menu_action(browser, "shielding"))
    show.assert_called_once_with("surface", "normal", state.shielding_detail)


def test_browser_export_menu_opens_format_choices_before_packaging(browser, monkeypatch):
    from coilgen.web.menus import menu_action

    browser.record = object()
    state = browser.server.state
    state.selected_run = state.displayed_run = "saved"
    export = Mock()
    monkeypatch.setattr(browser, "prepare_geometry", export)
    asyncio.run(menu_action(browser, "export"))
    assert state.center_section == "export" and state.workspace_panel == "center"
    export.assert_not_called()


def test_browser_geometry_entry_applies_valid_setup_and_keeps_specialist_draft(browser, monkeypatch):
    browser.new_project()
    state = browser.server.state
    state.setup_values = {**state.setup_values, "cylinder_radius_mm": "92"}
    browser.setup._changed()
    monkeypatch.setattr(browser.surface, "_run", Mock(side_effect=lambda *_a: asyncio.sleep(0)))
    asyncio.run(browser.surface.open())
    assert browser.session.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.092)
    state.surface_open = False
    state.settings_values = {**state.settings_values, "cylinder_radius_mm": "94"}
    state.setup_values = {**state.setup_values, "cylinder_radius_mm": "96"}
    browser.setup._changed()
    original = browser.session.project
    asyncio.run(browser.surface.open())
    assert browser.session.project is original
    assert state.settings_dialog and not state.surface_open
    browser.setup.open_tool("target")
    assert state.node == "geometry" and state.settings_values["cylinder_radius_mm"] == "94"
    assert state.setup_dirty and state.setup_values["cylinder_radius_mm"] == "96"


def test_browser_undo_keeps_pending_setup(browser):
    from coilgen.web.menus import menu_action

    browser.new_project()
    state = browser.server.state
    state.setup_values = {**state.setup_values, "cylinder_radius_mm": "96"}
    browser.setup._changed()
    original = browser.session.project
    asyncio.run(menu_action(browser, "undo"))
    assert browser.session.project is original and state.setup_dirty
    assert state.setup_values["cylinder_radius_mm"] == "96"


def test_browser_reopens_advanced_draft_with_pending_setup(browser):
    from coilgen.web.menus import menu_action

    browser.new_project()
    browser.open_advanced()
    state = browser.server.state
    state.advanced_config = '{"levels":'
    state.setup_values = {**state.setup_values, "cylinder_radius_mm": "96"}
    browser.setup._changed()
    state.workspace_panel = "center"
    asyncio.run(menu_action(browser, "advanced"))
    assert state.advanced_dialog and state.workspace_panel == "model"
    assert state.advanced_config == '{"levels":' and state.setup_dirty
