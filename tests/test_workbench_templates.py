"""Template navigation creates owned designs without submitting jobs or losing edits."""

import pytest

import test_workbench_usability as usability_tests
from coilgen.application import ProjectDocument
from coilgen.application.templates import TEMPLATE_SPECS
from coilgen.gui import workbench
from coilgen.gui.icons import ICON_NAMES
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDockWidget, QMessageBox, QToolButton

application = usability_tests.application
windows = usability_tests.windows


def template_actions(window):
    return [action for group in window.template_menu.actions() for action in group.menu().actions()]


def test_project_actions_and_templates_are_in_file_menu_without_toolbar_duplicates(windows):
    window = windows()
    file_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "File")
    assert [action.text() for action in file_menu.actions()[:3]] == ["New project…", "Open project…", "Load template"]
    assert file_menu.actions()[2].menu() is window.template_menu
    actions = template_actions(window)
    assert [action.data() for action in actions] == [spec.id for spec in TEMPLATE_SPECS]
    assert all(spec.icon in ICON_NAMES for spec in TEMPLATE_SPECS)
    assert all(not action.icon().isNull() and action.toolTip() for action in actions)
    assert {group.text() for group in window.template_menu.actions()} == {
        "Gradients",
        "Shim coils",
        "Coil systems",
        "Demonstrations",
    }
    button = window.findChild(QToolButton, "load_template_button")
    assert button is None


@pytest.mark.parametrize("kind", ("cylinder_gx", "biplanar_gz", "shim_z2", "shielded_gy"))
def test_clicking_template_loads_full_editable_study_without_computing(windows, kind):
    window = windows()
    old_id = window.project.id
    next(action for action in template_actions(window) if action.data() == kind).trigger()
    assert window.project.id != old_id
    assert window.project_path is None
    assert window.undo_stack.isClean() and not window._draft_dirty
    assert window.project.study["pipeline_stage"] == "evaluated"
    assert not window.project.config.skip_postprocessing and not window.project.config.skip_sweep
    assert window.stage_selector.currentData() == "evaluated"
    assert not window.service.submissions and not window.jobs.states
    if kind == "shielded_gy":
        config = window.project.config
        assert config.secondary_target_mesh_file != "none"
        before = window.project.model_fingerprint
        window.select_node("target")
        assert window.apply_settings()
        assert window.project.model_fingerprint == before
        window.select_node("optimization")
        assert window.apply_settings()
        assert window.project.model_fingerprint == before


@pytest.mark.parametrize("spec", TEMPLATE_SPECS, ids=lambda spec: spec.id)
def test_every_template_computes_immediately_without_visiting_settings(windows, spec):
    window = windows()
    # Even an open editor from the previous design is optional for the new template.
    window._inspect_node("geometry")
    assembly = window.load_template(spec.id)
    project = window.project
    assert project is assembly.channel(window._active_channel_id).project
    fingerprint, revision = project.fingerprint, project.revision
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "stage:evaluated"
    assert window.findChild(QDockWidget, "settings_dock").isHidden()
    assert not window._draft_dirty and window.undo_stack.isClean()
    assert window.compute_button.text() == "Compute full design" and window.compute_button.isEnabled()
    assert "Settings loaded" in window.statusBar().currentMessage()
    assert not window.service.submissions
    for channel in assembly.channels:
        assert channel.project.study["pipeline_stage"] == "evaluated"
        assert channel.project.study["fabricate"]
    window.compute_button.click()
    assert len(window.service.submissions) == 1
    submitted, controls = window.service.submissions[0]
    assert submitted is project and window.project is project
    assert (project.fingerprint, project.revision) == (fingerprint, revision)
    assert controls == project.study
    assert not window._draft_dirty and window.undo_stack.isClean()
    assert window.findChild(QDockWidget, "settings_dock").isHidden()


def test_loading_template_closes_previous_focused_windows_and_keeps_editing_optional(windows, application):
    window = windows()
    window._inspect_node("geometry")
    window._focused_sections[0].open()
    assert any(dialog.isVisible() for dialog in window.findChildren(QDialog))
    window.load_template("cylinder_gx")
    application.processEvents()
    assert not any(dialog.isVisible() for dialog in window.findChildren(QDialog))
    original = window.project
    window._inspect_node("geometry")
    assert not window.findChild(QDockWidget, "settings_dock").isHidden()
    assert window._draft_controls["radius_mm"].value() == original.config.cylinder_mesh_parameter_list[1] * 1000
    assert window.apply_settings() and window.project is original
    window._draft_controls["radius_mm"].setValue(95)
    assert window.apply_settings()
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.095)
    assert window.project.revision == original.revision + 1


def test_cancel_protects_pending_changes_when_loading_template(windows, monkeypatch):
    window = windows()
    original = window.project
    radius = window._draft_controls["radius_mm"]
    radius.setValue(radius.value() + 3)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Cancel)
    assert window.load_template("biplanar_gz") is None
    assert window.project is original and window._draft_dirty
    assert radius.value() == original.config.cylinder_mesh_parameter_list[1] * 1000 + 3


def test_failed_factory_keeps_current_project_and_undo(windows, monkeypatch):
    window = windows()
    original = window.project
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)

    def unavailable(*_args, **_kwargs):
        raise OSError("Input creation failed")

    monkeypatch.setattr(workbench, "template_assembly", unavailable)
    assert window.load_template("shielded_gy") is None
    assert window.project is original and window.undo_stack.isClean()
    assert errors == ["Cannot load template: Input creation failed"]


def test_discarded_expert_window_does_not_block_new_template(windows, monkeypatch):
    window = windows()
    window._advanced_dialog()
    expert = window.expert_window
    expert.fields["sweep_min_bend_radius"].control.setText("7")
    assert expert.pending_changes
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Discard)
    window.load_template("biplanar_gz")
    assert window.expert_window is None and not expert.isVisible()
    assert window._check_expert_draft()
    assert not window.service.submissions


def test_loaded_single_template_saves_and_reopens_as_one_json(windows, tmp_path):
    window = windows()
    window.load_template("shielded_gy")
    fingerprint = window.project.model_fingerprint
    path = window.save_project(tmp_path / "portable" / "shielded.coilgen.json")
    loaded = ProjectDocument.load(path)
    assert loaded.model_fingerprint == fingerprint
    assert loaded.config.secondary_target_mesh_file != "none"
    assert list(path.parent.iterdir()) == [path]
    reopened = windows()
    reopened.open_project(path)
    assert reopened.project.model_fingerprint == fingerprint
