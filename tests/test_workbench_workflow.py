"""Workflow navigation and submission contracts without a numerical solve."""

import pytest
from unittest.mock import Mock

import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDockWidget, QPushButton

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application import preflight
from coilgen.application.presets import preset_config


application = usability.application
windows = usability.windows


def _page(window):
    return window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)


def _show(window, application):
    window.show()
    window.activateWindow()
    application.processEvents()


def _selected_job(window):
    item = window.jobs_table.item(window.jobs_table.currentRow(), 0)
    return item.data(Qt.ItemDataRole.UserRole) if item else None


def test_model_builder_click_commits_valid_draft_once(windows, application):
    window = windows()
    _show(window, application)
    original = window.project
    window._draft_controls["radius_mm"].setValue(172)

    QTest.mouseClick(
        window.tree.viewport(),
        Qt.MouseButton.LeftButton,
        pos=window.tree.visualItemRect(window.tree_items["target"]).center(),
    )

    assert _page(window) == "target"
    assert window.tree.currentItem() is window.tree_items["target"]
    assert window.project.revision == original.revision + 1
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.172)
    assert window.undo_stack.count() == 1
    assert not window._draft_dirty


def test_model_builder_and_next_keep_invalid_draft_and_focus_error(windows, application):
    config = preset_config()
    config.coil_mesh_file = "create conical frustum mesh"
    config.surface_is_cylinder_flag = False
    window = windows(ProjectDocument.from_config(config))
    _show(window, application)
    original = window.project
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)

    QTest.mouseClick(
        window.tree.viewport(),
        Qt.MouseButton.LeftButton,
        pos=window.tree.visualItemRect(window.tree_items["target"]).center(),
    )
    application.processEvents()

    assert _page(window) == "geometry"
    assert window.tree.currentItem() is window.tree_items["geometry"]
    assert window.project is original
    assert window.undo_stack.count() == 0
    assert window._draft_dirty
    assert not window.draft_error_label.isHidden()
    focus = QApplication.focusWidget()
    assert any(
        focus is control or (focus is not None and control.isAncestorOf(focus))
        for key, control in window._draft_controls.items()
        if key in window._changed_controls
    )

    window.next_button.click()
    assert _page(window) == "geometry"
    assert window.project is original
    assert window._draft_controls["rotation_angle_deg"].value() == 45

    window.compute_button.click()
    assert not window.service.submissions
    assert not window.readiness_panel.isHidden()
    assert window.readiness_panel.issues[0].blocking
    assert QApplication.activeModalWidget() is None
    assert window.project is original

    window.readiness_panel.set_issues(())
    assert window.compute_through("prepared") is None
    assert not window.service.submissions
    assert not window.readiness_panel.isHidden()
    assert window.readiness_panel.issues[0].blocking
    assert window.project is original


def test_back_next_visit_all_sections_without_study_edits(windows):
    window = windows()
    original = window.project
    stage = window.stage_selector.currentData()
    pages = (
        "geometry",
        "target",
        "material",
        "winding",
        "optimization",
        "validation",
        f"stage:{stage}",
        "results",
        "export",
    )
    window.navigate_page(-1)
    assert _page(window) == "geometry"

    for index, page in enumerate(pages):
        assert _page(window) == page
        assert window.page_label.text() == f"Section: {window.tree.currentItem().text(0)}"
        back = window.back_button
        next_button = window.next_button
        assert back.isEnabled() is (index > 0)
        assert next_button.isEnabled() is (index < len(pages) - 1)
        if index < len(pages) - 1:
            next_button.click()

    window.navigate_page(1)
    assert _page(window) == "export"

    for page in reversed(pages[:-1]):
        window.back_button.click()
        assert _page(window) == page

    assert window.project is original
    assert window.project.study == original.study
    assert window.undo_stack.count() == 0


def test_section_navigation_preserves_stage_and_remembers_design_subpage(windows):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "routed"})
    window = windows(project)
    window.select_node("winding")

    for step, page in (("target", "target"), ("design", "winding"), ("compute", "stage:routed"), ("review", "results")):
        window.navigate_workflow(step)
        assert _page(window) == page
        assert window.page_label.text() == f"Section: {window.tree.currentItem().text(0)}"

    assert window.project is project
    assert window.stage_selector.currentData() == "routed"
    assert window.project.study["pipeline_stage"] == "routed"
    assert window.undo_stack.count() == 0


def test_model_builder_uses_the_scientific_model_and_study_groups(windows):
    window = windows()
    original = window.project
    for node, group in (
        ("geometry", "model"),
        ("target", "model"),
        ("material", "model"),
        ("winding", "model"),
        ("optimization", "study"),
        ("validation", "study"),
    ):
        window.select_node(node)
        assert window.tree.currentItem().parent().data(0, Qt.ItemDataRole.UserRole) == group
        assert window.tree.currentItem() is window.tree_items[node]
    assert window.tree_items["results"].parent() is None
    assert window.tree_items["export"].parent() is window.tree_items["results"]
    assert window.project is original
    assert window.undo_stack.count() == 0 and not window.service.submissions


def test_workflow_reopens_hidden_settings_and_review_evidence(windows, application):
    window = windows()
    _show(window, application)
    settings = window.findChild(QDockWidget, "settings_dock")
    settings.hide()
    window.evidence.close_all()

    window.navigate_workflow("target")
    assert not settings.isHidden()
    assert not window.evidence.any_open()

    settings.hide()
    window.navigate_workflow("review")
    assert not settings.isHidden()
    evidence = window.evidence.window_for(window.acceptance_panel)
    assert evidence.isVisible() and not evidence.isModal()
    assert evidence.parentWidget() is window
    assert window.evidence.currentWidget() is window.acceptance_panel


def test_blue_button_and_keyboard_share_submission_controls(windows, application):
    window = windows()
    _show(window, application)
    window.navigate_workflow("compute")
    window.stage_selector.setCurrentIndex(window.stage_selector.findData("contours"))

    window.compute_button.click()
    assert len(window.service.submissions) == 1
    _, button_controls = window.service.submissions[-1]
    assert button_controls["pipeline_stage"] == "contours"

    window.jobs.states[0]["status"] = "cancelled"
    window.poll_jobs()
    window.setFocus()
    QTest.keyClick(window, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    application.processEvents()

    assert len(window.service.submissions) == 2
    keyboard_project, keyboard_controls = window.service.submissions[-1]
    assert keyboard_controls == button_controls
    assert keyboard_project is window.project


@pytest.mark.parametrize("status", ("queued", "running", "cancelling"))
def test_repeated_compute_follows_identical_active_job_without_covering_plot(windows, application, status):
    window = windows()
    _show(window, application)
    first = window.compute()
    window.jobs.states[0]["status"] = status
    window.jobs_table.clearSelection()
    jobs_window = window.evidence.window_for(window.jobs_table)
    jobs_window.close()

    repeated = window.compute()

    assert repeated["id"] == first["id"]
    assert len(window.service.submissions) == 1
    assert _selected_job(window) == first["id"]
    assert window.evidence.currentWidget() is window.jobs_table
    assert window.evidence.window_for(window.jobs_table) is jobs_window
    assert not jobs_window.isVisible()
    assert window.live_preview.following
    assert window.live_preview.job_id == first["id"]


def test_changed_model_and_stage_submit_distinct_jobs(windows):
    window = windows()
    first = window.compute()
    assert window.apply_config({"levels": 9})
    changed_model = window.compute()
    window.stage_selector.setCurrentIndex(window.stage_selector.findData("contours"))
    changed_stage = window.compute()

    assert len({first["id"], changed_model["id"], changed_stage["id"]}) == 3
    assert len(window.service.submissions) == 3
    assert changed_model["project_fingerprint"] != first["project_fingerprint"]
    assert changed_stage["pipeline_stage"] == "contours"


def test_identical_designs_in_distinct_owned_channels_submit_independently(windows):
    first_project = ProjectDocument.from_config(preset_config(), name="First coil")
    second_project = first_project.duplicate("Second coil")
    assembly = ProjectAssembly.new("Independent coils", (first_project, second_project))
    window = windows(assembly)
    first = window.compute()
    window._set_assembly(assembly, assembly.channels[1].id)
    second = window.compute()

    assert first_project.fingerprint == second_project.fingerprint
    assert len(window.service.submissions) == 2
    assert first["id"] != second["id"]
    assert first["project_id"] == first_project.id
    assert second["project_id"] == second_project.id


def test_unavailable_physical_solver_blocks_nonmodally_and_method_change_recovers(windows, monkeypatch):
    window = windows()
    original = window.project
    monkeypatch.setattr(preflight, "find_spec", lambda _name: None)

    assert window.compute() is None
    assert not window.service.submissions
    assert window.project is original
    assert _page(window) == "optimization"
    assert window.tree.currentItem() is window.tree_items["optimization"]
    assert not window.readiness_panel.isHidden()
    issues = window.readiness_panel.issues
    assert any(issue.blocking and issue.node == "optimization" and "CVXPY" in issue.message for issue in issues)
    assert all(issue.action for issue in issues)
    assert QApplication.activeModalWidget() is None

    assert window.apply_config({"sf_opt_method": "tikhonov", "research_physics": None})
    assert window.compute() is not None
    assert len(window.service.submissions) == 1
    assert window.readiness_panel.isHidden() or not any(issue.blocking for issue in window.readiness_panel.issues)
    assert window.project.config.cylinder_mesh_parameter_list == original.config.cylinder_mesh_parameter_list


def test_unreadable_owned_support_blocks_compute_and_generated_support_recovers(windows, tmp_path):
    source = tmp_path / "unreadable_support.npz"
    source.write_bytes(b"This is not a mesh archive.")
    config = preset_config()
    config.coil_mesh_file = str(source)
    config.surface_is_cylinder_flag = False
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared"})
    window = windows(project)
    window.navigate_workflow("compute")

    assert window.compute() is None
    assert not window.service.submissions
    assert _page(window) == "geometry"
    assert window.tree.currentItem() is window.tree_items["geometry"]
    assert not window.readiness_panel.isHidden()
    assert any(
        issue.blocking and issue.key == "coil_mesh_file" and issue.node == "geometry"
        for issue in window.readiness_panel.issues
    )
    assert QApplication.activeModalWidget() is None

    assert window.apply_config({"coil_mesh_file": preset_config().coil_mesh_file, "surface_is_cylinder_flag": True})
    assert window.compute() is not None
    assert len(window.service.submissions) == 1
    assert window.readiness_panel.isHidden()


def test_missing_mesh_import_is_rejected_without_replacing_owned_model(windows, tmp_path):
    window = windows()
    project = window.project

    with pytest.raises(FileNotFoundError, match="Project input coil_mesh_file not found"):
        window.import_mesh(tmp_path / "missing_support.npz")

    assert window.project is project
    assert window.undo_stack.count() == 0
    assert not window.service.submissions


@pytest.mark.parametrize(
    ("missing_setting", "settings_node"),
    (("coil_mesh_file", "geometry"), ("field_shape_function", "target")),
)
def test_selected_study_step_primary_compute_checks_actual_missing_inputs(windows, missing_setting, settings_node):
    config = preset_config()
    setattr(config, missing_setting, "none")
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "sheet"})
    window = windows(project)
    window.navigate_workflow("compute")
    button = window.settings.widget().findChild(QPushButton, "compute_through_stage")

    button.click()
    assert window.stage_selector.currentData() == "sheet"
    assert not window.service.submissions and not window.jobs.states  # Selecting a boundary is not execution.
    assert window.compute_button.text() == "Optimize sheet"
    window.compute_button.click()

    assert not window.service.submissions
    assert not window.jobs.states
    assert _page(window) == settings_node
    assert window.project is project
    assert not window.readiness_panel.isHidden()
    assert any(
        issue.blocking and issue.key == missing_setting and issue.node == settings_node and issue.action
        for issue in window.readiness_panel.issues
    )
    assert QApplication.activeModalWidget() is None


def test_assembly_with_no_enabled_channels_shows_actionable_enable_focus(windows, application):
    project = ProjectDocument.from_config(preset_config())
    assembly = ProjectAssembly.new("Disabled assembly", (project,))
    assembly = assembly.update_channel(assembly.channels[0].id, enabled=False)
    window = windows(assembly)
    _show(window, application)
    window.service.submit_assembly = Mock(side_effect=AssertionError("Disabled channels must not be submitted."))

    assert window.compute_channels() is None
    window.service.submit_assembly.assert_not_called()
    assert not window.service.submissions
    assert not window.readiness_panel.isHidden()
    issues = window.readiness_panel.issues
    assert len(issues) == 1
    assert issues[0].blocking and issues[0].key == "channel_enabled" and issues[0].action

    window.readiness_panel.rows.itemAt(0).widget().findChild(QPushButton).click()
    application.processEvents()
    assert QApplication.focusWidget() is window.channel_enabled
    assert window.assembly is assembly
    assert not window.channel_enabled.isChecked()
    assert QApplication.activeModalWidget() is None


def test_missing_input_in_inactive_enabled_channel_blocks_all_and_focuses_owner(windows, application):
    ready = ProjectDocument.from_config(preset_config(), name="Ready channel")
    config = preset_config()
    config.field_shape_function = "none"
    blocked = ProjectDocument.from_config(config, name="Missing target channel")
    assembly = ProjectAssembly.new("All-or-none preflight", (ready, blocked))
    window = windows(assembly)
    _show(window, application)
    window.service.submit_assembly = Mock(side_effect=AssertionError("Every enabled channel must pass readiness."))
    assert window.project is ready

    assert window.compute_channels() is None
    application.processEvents()

    window.service.submit_assembly.assert_not_called()
    assert not window.service.submissions
    assert not window.jobs.states
    assert window.project is blocked
    assert window.channel_selector.currentData() == assembly.channels[1].id
    assert window.channel_enabled.isChecked()
    assert _page(window) == "target"
    assert window.tree.currentItem() is window.tree_items["target"]
    assert not window.readiness_panel.isHidden()
    assert any(
        issue.blocking and issue.key == "field_shape_function" and issue.node == "target"
        for issue in window.readiness_panel.issues
    )
    choice = window.settings.widget().findChild(QComboBox, "target_mode")
    section = next(section for section in window._focused_sections if choice in section.controls)
    assert section.dialog.isVisible()
    assert choice.scroll.isAncestorOf(section.dialog.focusWidget())
    assert choice.currentIndex() == -1
    assert "field_shape_function" not in window._draft_controls
    assert window.assembly.channels[0].project is ready
    assert window.assembly.channels[1].project is blocked
    assert QApplication.activeModalWidget() is None


def test_standard_gui_example_button_submits_complete_pipeline(windows):
    window = windows()
    project = window.project
    assert project.study["pipeline_stage"] == "evaluated"
    assert window.stage_selector.currentData() == "evaluated"
    window.select_node("winding")
    assert window._draft_controls["winding_intent"].currentText() == "Connected circuit"
    assert window._draft_controls["sweep"].isChecked()

    window.compute_button.click()

    assert len(window.service.submissions) == 1
    submitted, controls = window.service.submissions[0]
    assert submitted is project
    assert controls["pipeline_stage"] == "evaluated"
    assert submitted.config.levels > 0
    assert submitted.config.skip_postprocessing is False
    assert submitted.config.skip_sweep is False


@pytest.mark.parametrize("kind", ("cylinder", "biplanar"))
def test_new_example_resets_partial_study_and_enables_complete_winding_pipeline(windows, kind):
    config = preset_config()
    config.skip_postprocessing = True
    config.skip_sweep = True
    partial = ProjectDocument.from_config(config).update_study({"pipeline_stage": "sheet"})
    window = windows(partial)
    assert window.stage_selector.currentData() == "sheet"

    window.new_project(kind)

    assert window.project.id != partial.id
    assert window.project.study["pipeline_stage"] == "evaluated"
    assert window.stage_selector.currentData() == "evaluated"
    assert window.project.config.coil_mesh_file == preset_config(kind).coil_mesh_file
    window.select_node("winding")
    assert window._draft_controls["winding_intent"].currentText() == (
        "Separate circuits per support" if kind == "biplanar" else "Connected circuit"
    )
    assert window._draft_controls["sweep"].isChecked()

    window.compute_button.click()

    assert len(window.service.submissions) == 1
    submitted, controls = window.service.submissions[0]
    assert controls["pipeline_stage"] == "evaluated"
    assert submitted.study["pipeline_stage"] == "evaluated"
    assert submitted.config.levels > 0
    assert submitted.config.skip_postprocessing is False
    assert submitted.config.skip_sweep is False


@pytest.mark.parametrize("stage", ("prepared", "sheet", "contours", "routed"))
def test_explicit_saved_partial_stage_survives_construction_reopen_and_button_submission(windows, tmp_path, stage):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": stage})
    path = project.save(tmp_path / f"{stage}.coilgen.json")
    loaded = ProjectDocument.load(path)
    initial = windows(loaded)
    assert initial.project is loaded
    assert initial.project.study["pipeline_stage"] == stage
    assert initial.stage_selector.currentData() == stage
    assert (
        initial.compute_button.text()
        == {
            "prepared": "Prepare model",
            "sheet": "Optimize sheet",
            "contours": "Generate contours",
            "routed": "Connect / sweep",
        }[stage]
    )

    reopened = windows()
    reopened.open_project(path)
    assert reopened.project.fingerprint == project.fingerprint
    assert reopened.project.study == project.study
    assert reopened.stage_selector.currentData() == stage
    reopened.compute_button.click()

    assert len(reopened.service.submissions) == 1
    submitted, controls = reopened.service.submissions[0]
    assert submitted.fingerprint == project.fingerprint
    assert controls["pipeline_stage"] == stage
    assert controls["fabricate"] is False
